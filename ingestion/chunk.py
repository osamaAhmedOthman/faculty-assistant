"""
chunk.py — zone-aware chunking

Responsibility: Transform cleaned page text into semantically isolated chunks 
using domain-specific boundary strategies across document zones.

Design notes:
- STRATEGIC ZONE HANDLING: Applies tailored regex boundaries for regulatory 
  articles, extracts structured program tables intact, and splits course 
  catalogs into individual course units.
- PRESERVATION OVER GENERIC SPLITTING: Replaces fixed-size sliding windows with 
  pattern-based boundaries to prevent splitting mathematical formulas, course codes, 
  or article bodies.
- BILINGUAL CATALOG ADAPTATION: Implements robust parsing across distinct document 
  layouts (e.g., freeform prose vs. structured attribute blocks) with explicit 
  error reporting for unparsed segments.
"""

import re
import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Chunk:
    chunk_id: str
    zone_type: str  # "regulation" | "table" | "course"
    text: str
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Zone 1: Regulatory articles
# ---------------------------------------------------------------------------

# Matches: "مادة [13]:", "مادة (9)", "Article (12)", and tolerates three
# confirmed real-world corruption patterns:
#   1. OCR bracket-garbling, e.g. "مادة 8[1[" instead of "[8]"
#      (Tesseract misreading small bracket glyphs on scanned pages)
#   2. Bidi-fix punctuation displacement, e.g. "مادة: )1(" instead of
#      "مادة (1):" (colon lands in the wrong place after run-reversal)
#   3. Bidi-fix bracket MIRRORING, e.g. "مادة: )1(" where the actual
#      bracket glyphs are swapped (closing paren before the digit,
#      opening paren after) — confirmed identical in both the SWE and
#      biomedical docs, so this is systematic, not one-off noise.
# Given brackets are this unreliable, the regex stops trying to
# validate bracket structure at all and just tolerates any mix of
# bracket-like characters around the digit. Precision is instead
# enforced by requiring the marker at START OF LINE (the (?:^|\n)
# anchor), which is what actually distinguishes a real article header
# from an in-body reference like "طبقا للمادة 8" (mid-sentence, so
# never line-initial).
ARTICLE_PATTERN = re.compile(
    r"(?:^|\n)\s*(?:مادة|Article)\s*:?\s*[\[\(\)\]]*\s*(\d{1,2})\s*[\[\(\)\]]*",
    re.MULTILINE,
)


def chunk_regulatory_articles(full_text: str, program: str, source_file: str) -> list[Chunk]:
    """
    Split full document text into one chunk per regulatory article.

    Strategy: find every article marker position, then each chunk spans
    from one marker to the start of the next. The LAST article's chunk
    will run into whatever follows it in the document (tables/catalog) —
    callers should only pass this function the text up through the end
    of the regulatory section, not the whole document. (In practice: run
    zone detection first, see classify_zones() below.)
    """
    matches = list(ARTICLE_PATTERN.finditer(full_text))
    chunks = []

    for i, match in enumerate(matches):
        article_num = match.group(1)
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)
        article_text = full_text[start:end].strip()

        if len(article_text) < 20:  # guard against spurious matches
            continue

        chunks.append(
            Chunk(
                chunk_id=f"{source_file}_article_{article_num}",
                zone_type="regulation",
                text=article_text,
                metadata={
                    "program": program,
                    "source_file": source_file,
                    "article_num": int(article_num),
                },
            )
        )

    return chunks


# ---------------------------------------------------------------------------
# Zone 2: Program tables (structured, not chunked as prose)
# ---------------------------------------------------------------------------

def chunk_tables(pages: list[dict], program: str, source_file: str) -> list[Chunk]:
    """
    Turn pdfplumber's raw table extractions into structured chunks.

    Each table becomes ONE chunk containing:
      - a short auto-generated text summary (for embedding/retrieval)
      - the raw row/column data as structured JSON (for exact lookups,
        e.g. "how many credit hours is Data Structures")

    This deliberately does NOT try to prose-ify entire tables into
    paragraphs — that destroys the row/column relationships that make
    the data useful for precise queries.
    """
    chunks = []
    for page in pages:
        for t_idx, table in enumerate(page.get("tables", [])):
            if not table or len(table) < 2:  # need at least header + 1 row
                continue

            header = table[0]
            rows = table[1:]
            summary = f"Table from page {page['page_number']}: columns = {header}"

            chunks.append(
                Chunk(
                    chunk_id=f"{source_file}_p{page['page_number']}_table{t_idx}",
                    zone_type="table",
                    text=summary,
                    metadata={
                        "program": program,
                        "source_file": source_file,
                        "page_number": page["page_number"],
                        "header": header,
                        "rows": rows,
                    },
                )
            )
    return chunks


# ---------------------------------------------------------------------------
# Zone 3: Course catalog
# ---------------------------------------------------------------------------

# SWE/BIO-style structured block. Confirmed against real extracted text:
# the PDF's vertical 2-word label "Course / Description" (narrow left
# column) gets INTERLEAVED with the actual description paragraph by
# pdfplumber's line-by-line left-to-right reading — e.g. "Course" attaches
# to the description's first sentence, "Description" attaches to the
# second sentence, one full line later. The two label words are NOT
# adjacent in the extracted text, so we can't require them consecutively.
# Instead: capture everything between "Credit hours ...\n" and the next
# "Prerequisites" as one raw block, then strip stray standalone "Course"
# / "Description" label tokens out of it afterward (see _strip_label_noise).
COURSE_BLOCK_PATTERN = re.compile(
    r"Course Code\s+([A-Z]{2,5}\d{2,4})\s*"
    r"Course Name\s+(.+?)\s*\n"
    r"Credit hours\s+(.+?)\n"
    r"(.+?)"
    r"Prerequisites\s+(.+?)(?=Course Code|\Z)",
    re.DOTALL,
)

# Matches a standalone "Course" or "Description" label word sitting at
# the start of a line (i.e. where the interleaved table label landed),
# so we can strip it without touching those words if they legitimately
# appear mid-sentence inside real description prose.
_LABEL_NOISE_PATTERN = re.compile(r"(?m)^(Course|Description)\s+")


def _strip_label_noise(text: str) -> str:
    return _LABEL_NOISE_PATTERN.sub("", text).strip()

# Fallback for freeform catalog (AI doc): bare course-code anchors.
# We anchor on the code itself since labels aren't consistent.
BARE_COURSE_CODE_PATTERN = re.compile(r"\b([A-Z]{2,4}\d{3,4})\b")


def chunk_course_catalog_structured(catalog_text: str, program: str, source_file: str) -> list[Chunk]:
    """Parser for the SWE/BIO-style labeled catalog format."""
    chunks = []
    for match in COURSE_BLOCK_PATTERN.finditer(catalog_text):
        code, name, credits, raw_description, prereqs = (g.strip() for g in match.groups())
        description = _strip_label_noise(raw_description)
        chunks.append(
            Chunk(
                chunk_id=f"{source_file}_course_{code}",
                zone_type="course",
                text=f"{name} ({code})\nCredit hours: {credits}\nPrerequisites: {prereqs}\n\n{description}",
                metadata={
                    "program": program,
                    "source_file": source_file,
                    "course_code": code,
                    "course_name": name,
                    "prerequisites": prereqs,
                },
            )
        )
    return chunks


def chunk_course_catalog_freeform(catalog_text: str, program: str, source_file: str) -> tuple[list[Chunk], list[str]]:
    """
    Best-effort parser for the AI-doc-style freeform catalog: description
    paragraphs precede or follow a bare course code with no consistent
    labeling. We anchor on code positions and take a window of text
    around each as the chunk.

    Returns (chunks, unparsed_warnings) — the warnings list surfaces
    anything that looked like it should be a course entry but didn't
    fit the expected shape, so you can inspect rather than silently
    losing content. This is the "flag it, don't fix it in the dark"
    principle applied to messy source data.
    """
    matches = list(BARE_COURSE_CODE_PATTERN.finditer(catalog_text))
    chunks = []
    warnings = []

    for i, match in enumerate(matches):
        code = match.group(1)
        start = max(0, match.start() - 50)  # small lookback for course name
        end = matches[i + 1].start() if i + 1 < len(matches) else len(catalog_text)
        window = catalog_text[start:end].strip()

        if len(window) < 40:
            warnings.append(f"Suspiciously short entry near code {code}: {window!r}")
            continue

        chunks.append(
            Chunk(
                chunk_id=f"{source_file}_course_{code}_{i}",
                zone_type="course",
                text=window,
                metadata={
                    "program": program,
                    "source_file": source_file,
                    "course_code": code,
                    "parse_confidence": "low",  # freeform parse — flag for review
                },
            )
        )

    return chunks, warnings


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def chunk_document(cleaned_pages: list[dict], program: str, source_file: str) -> dict:
    """
    Run the full zone-aware chunking pipeline on one document's cleaned
    pages. Returns a dict with chunks grouped by zone plus any warnings,
    so you can inspect quality per zone rather than one flat list.
    """
    full_text = "\n".join(p["text"] for p in cleaned_pages)

    article_chunks = chunk_regulatory_articles(full_text, program, source_file)
    table_chunks = chunk_tables(cleaned_pages, program, source_file)

    # Catalog-zone isolation.
    #
    # KNOWN GAP #1 (AI doc): the last regulatory article is immediately
    # followed by OCR'd study-plan tables containing bare course codes,
    # which the freeform parser would match before reaching real
    # descriptions. Fixed by anchoring catalog start on the first
    # "This course" occurrence rather than "right after last article".
    #
    # KNOWN GAP #2 (SWE/BIO docs), confirmed against real data: these
    # documents do NOT follow strict [all regulations] -> [catalog]
    # ordering. A few short administrative articles (e.g. "Article 24:
    # appointing graduates as teaching assistants") are positioned
    # AFTER the course catalog in the actual text, not before it. Using
    # "last article position" as the catalog start reference is
    # therefore unreliable — it can land past the entire catalog.
    #
    # Combined fix: decouple catalog-zone detection from article
    # positions entirely. Search the FULL document text for the first
    # "This course" occurrence and take everything from there to the
    # end as candidate catalog text. This is safe even though a few
    # regulatory articles may fall inside that range, because the
    # structured course-block parser requires specific field labels
    # ("Course Code", "Course Name", etc.) and won't misfire on plain
    # regulation prose. Regulation chunking is unaffected by this —
    # it always scans the full original text independently, so those
    # trailing articles are still correctly captured as regulation
    # chunks regardless of where the catalog boundary falls.
    description_anchor = re.search(r"This course", full_text)
    if description_anchor:
        catalog_text = full_text[description_anchor.start():]
    else:
        # fallback: no description marker found at all (shouldn't
        # happen given real data so far, but fail toward "process
        # everything" rather than silently producing zero course
        # chunks with no warning)
        catalog_text = full_text

    structured_course_chunks = chunk_course_catalog_structured(catalog_text, program, source_file)
    if structured_course_chunks:
        course_chunks = structured_course_chunks
        warnings = []
    else:
        course_chunks, warnings = chunk_course_catalog_freeform(catalog_text, program, source_file)

    return {
        "program": program,
        "source_file": source_file,
        "regulation_chunks": [c.__dict__ for c in article_chunks],
        "table_chunks": [c.__dict__ for c in table_chunks],
        "course_chunks": [c.__dict__ for c in course_chunks],
        "warnings": warnings,
        "counts": {
            "regulations": len(article_chunks),
            "tables": len(table_chunks),
            "courses": len(course_chunks),
        },
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python chunk.py <path_to_cleaned_json> <program_name>")
        sys.exit(1)

    cleaned_pages = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    program = sys.argv[2]
    source_file = cleaned_pages[0]["source_file"] if cleaned_pages else "unknown"

    result = chunk_document(cleaned_pages, program, source_file)

    print(f"Program: {result['program']}")
    print(f"Counts: {result['counts']}")
    if result["warnings"]:
        print(f"\n{len(result['warnings'])} warning(s):")
        for w in result["warnings"][:10]:
            print(f"  - {w}")

    out_path = Path("data/processed") / f"{Path(sys.argv[1]).stem.replace('_cleaned', '')}_chunks.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nChunks saved to {out_path}")
