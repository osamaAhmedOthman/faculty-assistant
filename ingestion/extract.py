"""
extract.py — PDF text extraction

Responsibility: Convert raw PDFs into structured, page-by-page text blocks 
without applying cleaning, chunking, or text transformations.

Design notes:
- LAYOUT-AWARE EXTRACTION: Uses pdfplumber over standard text-stream extractors 
  to reliably preserve structure in tabular, credit-hour, and multi-column documents.
- ISOLATED PIPELINE BOUNDARY: Leaves text normalization and BiDi (Arabic RTL) 
  corrections entirely to preprocess.py, keeping extraction traceable and testable.
- AUTOMATIC OCR FALLBACK: Detects scanned/empty PDF pages and reruns them 
  through Tesseract OCR via page rasterization.
- AUDITABLE METADATA: Tags each page with its extraction source 
  (`extraction_method`: "text_layer" vs. "ocr") for downstream quality tracing.
"""

from dataclasses import dataclass, field
from pathlib import Path
import pdfplumber

try:
    import pytesseract
    from pdf2image import convert_from_path
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False


@dataclass
class PageText:
    """One page's raw extracted text, with provenance metadata."""
    source_file: str
    page_number: int  # 1-indexed, matches what a human would call "page N"
    text: str
    tables: list = field(default_factory=list)  # raw pdfplumber table extractions
    extraction_method: str = "text_layer"  # "text_layer" | "ocr" | "empty"


def _ocr_page(pdf_path: Path, page_number: int, lang: str = "ara+eng", dpi: int = 300) -> str:
    """
    Rasterize one page of the PDF and run Tesseract OCR on it.

    lang="ara+eng": these documents mix Arabic regulatory text with
    English course names/codes on the same page, so we run both
    language models together rather than guessing per-page.

    dpi=300: a reasonable default for scanned academic documents —
    high enough for OCR accuracy on printed text, without being so
    high that a 50-page document takes forever to process. Bump this
    if OCR output looks garbled on a specific document.
    """
    if not OCR_AVAILABLE:
        raise RuntimeError(
            "OCR dependencies not installed. Run: "
            "pip install pytesseract pdf2image --break-system-packages "
            "(and ensure tesseract-ocr + tesseract-ocr-ara are installed via apt)"
        )

    images = convert_from_path(
        str(pdf_path), dpi=dpi, first_page=page_number, last_page=page_number
    )
    if not images:
        return ""

    return pytesseract.image_to_string(images[0], lang=lang)


def extract_pdf(pdf_path: str, ocr_fallback: bool = True, ocr_lang: str = "ara+eng") -> list[PageText]:
    """
    Extract text and tables from every page of a PDF.

    Returns a list of PageText objects, one per page, in page order.
    Empty pages are still included — filtering those out is a
    preprocess.py concern, not this stage's job. We want extract.py
    to be a faithful mirror of the source document, just with a
    text layer attached (however that text was obtained).

    If a page's native text layer is empty and ocr_fallback=True,
    the page is re-extracted via OCR automatically.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    pages: list[PageText] = []

    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            tables = page.extract_tables() or []
            method = "text_layer"

            if not text.strip():
                if ocr_fallback:
                    text = _ocr_page(pdf_path, i, lang=ocr_lang)
                    method = "ocr" if text.strip() else "empty"
                else:
                    method = "empty"

            pages.append(
                PageText(
                    source_file=pdf_path.name,
                    page_number=i,
                    text=text,
                    tables=tables,
                    extraction_method=method,
                )
            )

    return pages


def extract_to_dict(pdf_path: str, ocr_fallback: bool = True, ocr_lang: str = "ara+eng") -> list[dict]:
    """Convenience wrapper returning plain dicts (for JSON serialization / debugging)."""
    return [
        {
            "source_file": p.source_file,
            "page_number": p.page_number,
            "text": p.text,
            "tables": p.tables,
            "extraction_method": p.extraction_method,
        }
        for p in extract_pdf(pdf_path, ocr_fallback=ocr_fallback, ocr_lang=ocr_lang)
    ]


if __name__ == "__main__":
    # Quick manual smoke test — run this file directly to sanity-check
    # extraction against a real PDF before trusting the pipeline.
    import sys
    import json

    if len(sys.argv) < 2:
        print("Usage: python extract.py <path_to_pdf>")
        sys.exit(1)

    result = extract_to_dict(sys.argv[1])
    print(f"Extracted {len(result)} pages from {sys.argv[1]}")

    methods = {}
    for p in result:
        methods[p["extraction_method"]] = methods.get(p["extraction_method"], 0) + 1
    print(f"Extraction methods used: {methods}")

    print(f"\nPage 1 preview (first 300 chars):\n{result[0]['text'][:300]}")
    print(f"\nTables found on page 1: {len(result[0]['tables'])}")

    out_path = Path("data/processed") / (Path(sys.argv[1]).stem + "_extracted.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nFull extraction saved to {out_path}")
