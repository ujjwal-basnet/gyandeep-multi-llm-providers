from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pymupdf


def extract_pdf_text(
    pdf_path: str | Path,
    output_path: str | Path | None = None,
    max_workers: int | None = None,
) -> str:
    pdf_path = Path(pdf_path)
    pdf = pymupdf.open(str(pdf_path))
    try:
        total_pages = pdf.page_count
    finally:
        pdf.close()

    def _extract_page(idx: int) -> tuple[int, str]:
        # Open the document in each worker to avoid sharing page/document state across threads.
        local_pdf = pymupdf.open(str(pdf_path))
        try:
            page = local_pdf.load_page(idx)
            text = page.get_textpage_ocr().extractText()
        finally:
            local_pdf.close()
        print(f"On page: {idx + 1} | {round(((idx + 1) / total_pages) * 100)}%")
        return idx, text

    results: list[tuple[int, str]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_extract_page, i): i for i in range(total_pages)}
        for future in as_completed(futures):
            results.append(future.result())

    results.sort(key=lambda x: x[0])
    text = "".join(t for _, t in results)

    if output_path:
        Path(output_path).write_text(text, encoding="utf-8")
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract OCR text from a PDF.")
    parser.add_argument("pdf", help="Path to the PDF file.")
    parser.add_argument("--out", default="totalBook.txt", help="Output text file path.")
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel OCR threads (default: auto).",
    )
    args = parser.parse_args()

    extract_pdf_text(args.pdf, args.out, max_workers=args.workers)


if __name__ == "__main__":
    main()
