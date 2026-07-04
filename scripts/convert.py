import argparse
import json
import sys
import tempfile
from pathlib import Path

PDF_CHUNK_PAGE_COUNT = 50
PDF_OCR_RENDER_SCALE = 0.5
PDF_OCR_LEAN_CONFIG = {
    "use_angle_cls": False,
    "det_model_path": "",
    "det_module_name": "rapidocr_onnxruntime.ch_ppocr_v3_det",
    "det_limit_side_len": 512,
    "det_limit_type": "min",
    "det_thresh": 0.25,
    "det_box_thresh": 0.3,
    "rec_model_path": "",
    "rec_module_name": "rapidocr_onnxruntime.ch_ppocr_v3_rec",
}


def result(ok, **kwargs):
    print(json.dumps({"ok": ok, **kwargs}, ensure_ascii=False))


def load_converter():
    try:
        from markitdown import MarkItDown
    except Exception:
        return None

    return MarkItDown()


def load_pdf_reader_writer():
    try:
        from pypdf import PdfReader, PdfWriter
    except Exception:
        return None, None

    return PdfReader, PdfWriter


def load_pdfium_and_ocr():
    try:
        import numpy as np
        import pypdfium2 as pdfium
        from rapidocr_onnxruntime import RapidOCR
    except Exception:
        return None, None, None

    return np, pdfium, RapidOCR(**PDF_OCR_LEAN_CONFIG)


def extract_markdown_text(markdown):
    text_content = getattr(markdown, "text_content", None)
    if text_content is None:
        return str(markdown)
    return text_content


def convert_local(converter, source):
    markdown = converter.convert_local(str(source))
    return extract_markdown_text(markdown)


def convert_pdf_in_chunks(converter, source):
    PdfReader, PdfWriter = load_pdf_reader_writer()
    if PdfReader is None or PdfWriter is None:
        return None

    try:
        reader = PdfReader(str(source))
    except Exception:
        return None

    pages = getattr(reader, "pages", [])
    if len(pages) <= PDF_CHUNK_PAGE_COUNT:
        return None

    parts = []
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        for chunk_index, start in enumerate(range(0, len(pages), PDF_CHUNK_PAGE_COUNT), start=1):
            writer = PdfWriter()
            for page in pages[start : start + PDF_CHUNK_PAGE_COUNT]:
                writer.add_page(page)

            chunk_path = temp_dir / f"{source.stem}-part-{chunk_index}.pdf"
            with chunk_path.open("wb") as chunk_file:
                writer.write(chunk_file)

            parts.append(convert_local(converter, chunk_path).rstrip())

    return "\n\n".join(part for part in parts if part)


def extract_text_from_ocr_result(result):
    lines = []
    for item in result or []:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue

        text = str(item[1]).strip()
        if text:
            lines.append(text)

    return "\n".join(lines).strip()


def convert_pdf_with_ocr(source):
    np, pdfium, ocr_engine = load_pdfium_and_ocr()
    if np is None or pdfium is None or ocr_engine is None:
        return None

    try:
        pdf = pdfium.PdfDocument(str(source))
    except Exception:
        return None

    try:
        page_count = len(pdf)
        if page_count <= 0:
            return None

        parts = []
        for page_index in range(page_count):
            page = pdf[page_index]
            bitmap = page.render(scale=PDF_OCR_RENDER_SCALE, grayscale=True)
            ocr_result, _ = ocr_engine(np.asarray(bitmap.to_numpy()))
            page_text = extract_text_from_ocr_result(ocr_result)
            if page_text:
                parts.append(page_text)

        return "\n\n".join(parts).strip()
    finally:
        close = getattr(pdf, "close", None)
        if callable(close):
            close()


def convert_source(converter, source):
    try:
        text_content = convert_local(converter, source)
    except Exception:
        if source.suffix.lower() != ".pdf":
            raise

        text_content = convert_pdf_in_chunks(converter, source)
        if text_content is None:
            raise

    if text_content and text_content.strip():
        return text_content

    if source.suffix.lower() != ".pdf":
        return text_content

    ocr_markdown = convert_pdf_with_ocr(source)
    if ocr_markdown and ocr_markdown.strip():
        return ocr_markdown

    raise RuntimeError("PDF 没有可提取文本，OCR 识别未返回内容，请安装 OCR 依赖或检查文件质量。")


def convert_with_converter(converter, input_path, output_path):
    source = Path(input_path)
    target = Path(output_path)

    if not source.exists():
        return {"inputPath": str(source), "ok": False, "errorCode": "INPUT_MISSING", "message": "源文件不存在，请重新选择文件。"}

    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        text_content = convert_source(converter, source)
        target.write_text(text_content, encoding="utf-8")
    except Exception as exc:
        return {"inputPath": str(source), "ok": False, "errorCode": "CONVERSION_FAILED", "message": f"转换失败：{exc}"}

    return {"inputPath": str(source), "ok": True, "outputPath": str(target)}


def convert(input_path, output_path):
    converter = load_converter()
    if converter is None:
        result(
            False,
            errorCode="MARKITDOWN_UNAVAILABLE",
            message="MarkItDown 环境不可用，请先安装 requirements.txt 中的 Python 依赖。",
        )
        return 1

    item_result = convert_with_converter(converter, input_path, output_path)
    result(item_result["ok"], **{key: value for key, value in item_result.items() if key != "ok"})
    return 0 if item_result["ok"] else 1


def convert_jobs(jobs_json):
    try:
        jobs = json.loads(jobs_json)
    except json.JSONDecodeError:
        result(False, errorCode="INVALID_JOBS_JSON", message="批量转换任务不是有效 JSON。")
        return 1

    if not isinstance(jobs, list):
        result(False, errorCode="INVALID_JOBS_JSON", message="批量转换任务必须是数组。")
        return 1

    converter = load_converter()
    if converter is None:
        result(
            False,
            errorCode="MARKITDOWN_UNAVAILABLE",
            message="MarkItDown 环境不可用，请先安装 requirements.txt 中的 Python 依赖。",
            results=[
                {
                    "inputPath": str(job.get("input", "")),
                    "ok": False,
                    "errorCode": "MARKITDOWN_UNAVAILABLE",
                    "message": "MarkItDown 环境不可用，请先安装 requirements.txt 中的 Python 依赖。",
                }
                for job in jobs
                if isinstance(job, dict)
            ],
        )
        return 1

    results = []
    for job in jobs:
        if not isinstance(job, dict) or not job.get("input") or not job.get("output"):
            results.append(
                {
                    "inputPath": str(job.get("input", "")) if isinstance(job, dict) else "",
                    "ok": False,
                    "errorCode": "INVALID_JOB",
                    "message": "批量转换任务缺少输入或输出路径。",
                }
            )
            continue

        results.append(convert_with_converter(converter, job["input"], job["output"]))

    result(any(item["ok"] for item in results), results=results)
    return 0 if any(item["ok"] for item in results) else 1


def main():
    parser = argparse.ArgumentParser(description="Convert a local file to Markdown with MarkItDown.")
    parser.add_argument("--input")
    parser.add_argument("--output")
    parser.add_argument("--jobs-json")
    args = parser.parse_args()

    if args.jobs_json:
        return convert_jobs(args.jobs_json)

    if not args.input or not args.output:
        parser.error("the following arguments are required together: --input, --output")

    return convert(args.input, args.output)


if __name__ == "__main__":
    sys.exit(main())
