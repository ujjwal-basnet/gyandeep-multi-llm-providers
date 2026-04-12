from __future__ import annotations

import argparse
from pathlib import Path

import pymupdf


def extract_pdf_text(pdf_path: str | Path, output_path: str | Path | None = None) -> str:
    """Extract OCR text from a PDF and optionally write it to disk."""
    pdf = pymupdf.open(str(pdf_path))
    total_pages = pdf.page_count

    text_parts: list[str] = []
    for idx, page in enumerate(pdf, start=1):
        print(f"On page: {idx} | {round((idx / total_pages) * 100)}%")
        text_parts.append(page.get_textpage_ocr().extractText())

    text = "".join(text_parts)
    if output_path:
        Path(output_path).write_text(text, encoding="utf-8")
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract OCR text from a PDF.")
    parser.add_argument("pdf", help="Path to the PDF file.")
    parser.add_argument("--out", default="totalBook.txt", help="Output text file path.")
    args = parser.parse_args()

    extract_pdf_text(args.pdf, args.out)


if __name__ == "__main__":
    main()
