"""
preprocess.py — mechanical cleanup only

Responsibility: strip content that is positionally repetitive and
carries zero retrieval value (headers, footers, page-number lines,
watermark text, university seal captions) and normalize whitespace.

CRITICAL BOUNDARY: this module does NOT deduplicate semantically
similar content across documents (e.g. the GPA article appearing in
both the AI and SWE docs). That is a downstream, embedding-based
step that runs AFTER chunking, because you need chunk-level units to
compare, not raw page text. Conflating the two here would make this
stage untestable and would throw away information the dedup step
needs (i.e. which program each chunk came from).

Design notes:
- We detect headers/footers by FREQUENCY, not by hardcoding strings.
  A line that appears near-identically on >60% of pages is almost
  certainly boilerplate (running header, page number, seal caption),
  regardless of language. This generalizes across your AI, SWE, and
  biomedical docs without you having to hand-write Arabic string
  matches for each new document.
- We deliberately do NOT strip anything that appears in <60% of
  pages — a GPA formula that happens to repeat because it's discussed
  in 3 different articles is not the same thing as a running header
  on every single page. Frequency threshold is the safety margin.

- BIDI (Arabic reading-order) CORRECTION, applied ONLY to pages where
  extraction_method == "text_layer": pdfplumber streams PDF content
  in the order glyphs appear in the file's content stream, which for
  Arabic text is often visual (RTL-rendered) order rather than
  logical reading order. The result is that a correct sentence like
  "جامعة المنصورة كلية الحاسبات والمعلومات" comes out reversed. A
  full-line character reversal fixes this reliably for these
  documents (confirmed against real extracted output).
  We do NOT apply this to OCR-extracted pages — Tesseract's layout
  analysis already outputs correct logical reading order, so
  reversing it again would re-break it.
  We do NOT blanket-reverse every line either, since these documents
  mix Arabic with English (course codes like "CS1001", table
  headers). Reversing "CS1001" would corrupt it. Instead we classify
  each line by its majority script and only reverse Arabic-majority
  lines.
"""

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass


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
    Restore logical reading order for an Arabic-majority line that was
    extracted in visual (mirrored) order by pdfplumber.

    Verified against real output: naive line[::-1] corrupts embedded
    digit/English runs (e.g. credit-hour figure "135" became "531").
    This version instead:
      1. Splits the line into runs of (Arabic-script | everything else)
      2. Reverses the ORDER of runs (fixes RTL sentence-level ordering)
      3. Within each Arabic run, reverses characters (Arabic words were
         individually mirrored too, since pdfplumber has no bidi logic)
      4. Within each non-Arabic run, leaves character order untouched
         (digits/English were already correct internally)

    This is still not a full Unicode BiDi algorithm implementation
    (that would need to handle nested embedding levels, numbers with
    embedded commas/decimals, mixed punctuation direction, etc. — see
    the `python-bidi` package if this ever proves insufficient). It's
    a targeted fix for the specific mis-extraction pattern these
    documents exhibit, verified against real extracted text.
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


def _normalize_line(line: str) -> str:
    """Collapse whitespace/digits so near-identical headers (which often
    contain a changing page number) still hash to the same bucket."""
    line = _strip_invisible_controls(line)
    line = re.sub(r"\d+", "#", line.strip())
    line = re.sub(r"\s+", " ", line)
    return line


# Unicode bidi/formatting control characters that PDF text streams for
# Arabic documents frequently embed (RLM/LRM marks, directional
# embedding/override/isolate marks). These are INVISIBLE when rendered
# but are real characters in the extracted string, and Python's regex
# \s does NOT match them — confirmed against real output: a U+200F
# (Right-to-Left Mark) sitting between a newline and "مادة" silently
# broke our article-boundary regex's start-of-line anchor, causing
# article 3 (and others) to be missed even though the bracket text
# itself was perfectly intact. This is a normalization concern, not a
# chunking concern, so it's fixed here rather than patched into every
# downstream regex that needs a clean line start.
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
            # not the normalized one. Bug found via real output: a bare
            # page-number line like "30" normalizes to "#" (single char),
            # which the OLD guard (checking normalized length) discarded
            # before it ever got counted — meaning pure-digit page-number
            # lines could never be detected as boilerplate no matter how
            # often they repeated. Confirmed cause of stray "30"/"31"
            # page-number fragments leaking into course Prerequisites
            # fields in real chunk output.
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


def preprocess_document(pages: list[dict], min_frequency: float = 0.6) -> list[CleanedPage]:
    """
    Full preprocessing pass for one document's extracted pages.

    Takes the output of extract.extract_to_dict() and returns cleaned
    pages with boilerplate stripped and Arabic reading order corrected
    where needed. Run this per-document (not across all 3 program PDFs
    at once) since header/footer text differs between the AI, SWE, and
    biomedical documents.

    Expects each page dict to include "extraction_method" (added by
    extract.py) so we know whether to apply the bidi fix. Pages missing
    this key default to applying the fix (safe default: most of these
    documents' native PDFs need it).
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
