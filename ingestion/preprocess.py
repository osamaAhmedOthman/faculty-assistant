"""
preprocess.py — mechanical cleanup and text normalization

Responsibility: Strip positionally repetitive boilerplate (headers, footers, 
page numbers) and apply script-aware text normalization to raw page text.

CRITICAL BOUNDARY: Does NOT perform cross-document semantic deduplication; 
preserving program-specific provenance requires evaluating duplicates post-chunking.

Design notes:
- FREQUENCY-BASED BOILERPLATE DETECTION: Strips lines appearing on >60% of pages 
  to automatically eliminate headers and footers without hardcoded string patterns.
- SELECTIVE BIDI CORRECTION: Reverses visual-order Arabic lines extracted via text-layer 
  streaming while bypassing OCR outputs and English-majority tokens (e.g., "CS1001").
"""

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field


# Unicode block for Arabic script (covers standard Arabic letters).
# Used to decide, per line, whether it's "Arabic enough" to need
# bidi correction versus a line of English/numeric/code content.
def _is_arabic_char(ch: str) -> bool:
    return "\u0600" <= ch <= "\u06FF"


def _is_arabic_majority_line(line: str) -> bool:
    """A line needs bidi correction if most of its letter characters
    are Arabic script. Digits, punctuation, and whitespace don't count
    toward either side — a line like "Table 9: جدول" should still be
    treated as Arabic-majority since 'Table 9:' contributes no Arabic
    or Latin letters that dominate."""
    letters = [ch for ch in line if ch.isalpha()]
    if not letters:
        return False
    arabic_count = sum(1 for ch in letters if _is_arabic_char(ch))
    return arabic_count / len(letters) > 0.5


# A "run" is a maximal substring that is either Arabic-script or not.
# We use this to reverse word ORDER (to fix RTL/LTR mis-extraction)
# without reversing character order WITHIN non-Arabic runs like
# digits ("135"), course codes ("CS1001"), or embedded English words —
# those were already in correct internal order in the raw extraction;
# only their position in the sentence needs correcting.
_RUN_PATTERN = re.compile(r"[\u0600-\u06FF]+|[^\u0600-\u06FF]+")


def fix_bidi_line(line: str) -> str:
    """
    Restore logical reading order for visual-order Arabic lines extracted by pdfplumber.

    Responsibility: Fix mirrored RTL text while preserving embedded LTR tokens (digits, course codes).

    Design notes:
    - RUN-BASED PARSING: Splits lines into distinct Arabic and non-Arabic token segments.
    - SEGMENT REVERSAL: Reverses sentence-level run order and character order within Arabic segments.
    - LTR PRESERVATION: Leaves digit and English character sequences (e.g., "135", "CS100") 
    unmodified to prevent corrupting credit-hour figures and course codes.
    """ 
    if not _is_arabic_majority_line(line):
        return line

    runs = _RUN_PATTERN.findall(line)
    fixed_runs = []
    for run in runs:
        if run and _is_arabic_char(run.strip()[0] if run.strip() else " "):
            fixed_runs.append(run[::-1])
        else:
            fixed_runs.append(run)

    return "".join(reversed(fixed_runs))


@dataclass
class CleanedPage:
    source_file: str
    page_number: int
    text: str
    tables: list = field(default_factory=list)


def _normalize_line(line: str) -> str:
    """Collapse whitespace/digits so near-identical headers (which often
    contain a changing page number) still hash to the same bucket."""
    line = _strip_invisible_controls(line)
    line = re.sub(r"\d+", "#", line.strip())
    line = re.sub(r"\s+", " ", line)
    return line


"""
Strip invisible Unicode BiDi and formatting control characters from extracted text streams.

Responsibility: Eliminate non-printing directional marks (e.g., U+200F RLM, LRM, directional isolates) 
that corrupt string anchor matching in downstream regex operations.

Design notes:
- REGEX BOUNDARY PRESERVATION: Removes embedded control characters that sit between whitespace 
  and structural markers (e.g., article boundaries like "مادة") to ensure clean line-start anchors.
- NORMALIZATION ISOLATION: Normalizes control characters during cleanup to keep downstream 
  regex patterns clean and prevent repetitive inline character stripping.
"""
_INVISIBLE_CONTROLS = re.compile(
    "[\u200b-\u200f\u202a-\u202e\u2066-\u2069\ufeff]"
)


def _strip_invisible_controls(text: str) -> str:
    return _INVISIBLE_CONTROLS.sub("", text)


def detect_boilerplate_lines(pages: list[dict], min_frequency: float = 0.6) -> set[str]:
    """
    Scan all pages of a document and find lines that repeat often enough
    to be considered running headers/footers rather than content.

    min_frequency: fraction of pages a (normalized) line must appear on
    to be flagged as boilerplate. 0.6 is a reasonable starting point —
    tune down if a document has genuinely short pages with little
    unique text, tune up if you're over-stripping legitimate repeated
    section titles.
    """
    total_pages = len(pages)
    if total_pages == 0:
        return set()

    line_counts: Counter[str] = Counter()
    for page in pages:
        seen_this_page = set()
        for raw_line in page["text"].splitlines():
            stripped = raw_line.strip()
            # Guard against empty/trivial lines using the RAW line length,
            # not the normalized one. A bare page-number line like "30"
            # normalizes to "#" (a single character) — checking the
            # normalized length would discard it before it's ever counted,
            # which would make pure-digit page-number lines undetectable
            # as boilerplate no matter how often they repeat across pages.
            if len(stripped) < 1:
                continue
            norm = _normalize_line(raw_line)
            if not norm:
                continue
            if norm not in seen_this_page:
                line_counts[norm] += 1
                seen_this_page.add(norm)

    threshold = total_pages * min_frequency
    boilerplate = {line for line, count in line_counts.items() if count >= threshold}
    return boilerplate


def clean_page_text(text: str, boilerplate: set[str], apply_bidi_fix: bool) -> str:
    """
    Remove boilerplate lines, normalize whitespace, and (if this page
    came from the native text layer, not OCR) fix Arabic line order.

    apply_bidi_fix should be False for OCR-extracted pages — Tesseract
    already outputs correct logical order, so reversing again would
    corrupt it. See module docstring for the full reasoning.
    """
    kept_lines = []
    for raw_line in text.splitlines():
        norm = _normalize_line(raw_line)
        if norm in boilerplate:
            continue
        stripped = _strip_invisible_controls(raw_line).strip()
        if not stripped:
            continue
        if apply_bidi_fix:
            stripped = fix_bidi_line(stripped)
        kept_lines.append(stripped)

    cleaned = "\n".join(kept_lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)  # collapse excess blank lines
    return cleaned


def clean_table_cells(table: list, apply_bidi_fix: bool) -> list:
    """
    Pass raw pdfplumber table rows through, applying the same bidi
    correction used on page text to any Arabic-majority cell strings.

    pdfplumber's extract_tables() returns raw cell text completely
    independent of the page-text extraction path, so it needs the same
    per-cell correction fix_bidi_line applies to page lines. Numeric
    and English cells (course codes, credit-hour numbers) pass through
    unchanged, same logic as fix_bidi_line's line-level handling. This
    preserves table structure end-to-end so chunk.py's table-aware
    course parser has clean rows to work with.
    """
    cleaned_table = []
    for row in table:
        cleaned_row = []
        for cell in row:
            if isinstance(cell, str) and cell.strip():
                fixed = fix_bidi_line(cell.strip()) if apply_bidi_fix else cell.strip()
                cleaned_row.append(fixed)
            else:
                cleaned_row.append(cell)
        cleaned_table.append(cleaned_row)
    return cleaned_table


def preprocess_document(pages: list[dict], min_frequency: float = 0.6) -> list[CleanedPage]:
    """
    Full preprocessing pass for a single document's extracted pages.

    Responsibility: Strip boilerplate, fix Arabic reading order, and preserve 
    structured table data across extracted pages on a per-document basis.

    Design notes:
    - DOCUMENT-SCOPED BOILERPLATE DETECT: Scopes header/footer frequency analysis 
    to individual documents to prevent cross-program pattern pollution.
    - PRESERVED TABLE DATA: Carries forward extracted tables in `CleanedPage.tables` 
    with cell-level BiDi corrections for downstream parsing in chunk.py.
    - SAFE METHOD FALLBACK: Evaluates `extraction_method` per page, defaulting to BiDi 
    correction when the flag is absent.
    """
    boilerplate = detect_boilerplate_lines(pages, min_frequency=min_frequency)

    cleaned_pages = [
        CleanedPage(
            source_file=page["source_file"],
            page_number=page["page_number"],
            text=clean_page_text(
                page["text"],
                boilerplate,
                apply_bidi_fix=(page.get("extraction_method", "text_layer") == "text_layer"),
            ),
            tables=clean_table_cells(
                page.get("tables", []),
                apply_bidi_fix=(page.get("extraction_method", "text_layer") == "text_layer"),
            ),
        )
        for page in pages
    ]

    return cleaned_pages


if __name__ == "__main__":
    import sys
    import json
    from pathlib import Path

    if len(sys.argv) < 2:
        print("Usage: python preprocess.py <path_to_extracted_json>")
        sys.exit(1)

    pages = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    boilerplate = detect_boilerplate_lines(pages)

    print(f"Detected {len(boilerplate)} boilerplate line(s) across {len(pages)} pages:")
    for line in boilerplate:
        print(f"  - {line!r}")

    cleaned = preprocess_document(pages)
    out_path = Path("data/processed") / (Path(sys.argv[1]).stem.replace("_extracted", "") + "_cleaned.json")
    out_path.write_text(
        json.dumps([c.__dict__ for c in cleaned], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nCleaned pages saved to {out_path}")
    print(f"\nPage 1 preview after cleaning:\n{cleaned[0].text[:300]}")
