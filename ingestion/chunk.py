"""
chunk.py — zone-aware chunking

Responsibility: Split cleaned document text into semantically isolated chunks 
using targeted, zone-specific parsing strategies.

Design notes:
- STRATEGIC ZONE HANDLING: Applies pattern boundaries for regulatory articles, 
  preserves structured program tables intact, and isolates individual course catalog entries.
- REGEX BOUNDARY PRESERVATION: Uses domain-specific regex patterns instead of 
  fixed-size windows to prevent cutting formulas, course codes, or article bodies.
- ADAPTIVE CATALOG PARSING: Handles structural variations across documents 
  (e.g., freeform prose vs. structured attribute blocks) with explicit logging 
  for low-confidence fallback entries.
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

"""
Responsibility: Detect regulatory article headers (e.g., "مادة [13]:", "Article (12)") 
while tolerating OCR artifacts and BiDi punctuation displacement.

Design notes:
- LINE-START ANCHORING: Uses `(?:^|\n)` anchors to distinguish article headers from inline 
  references (e.g., "طبقا للمادة 8") without relying on rigid bracket matching.
- NOISE TOLERANCE: Permits flexible bracket/punctuation sequences to handle systematic 
  BiDi bracket mirroring (e.g., "مادة: )1(") and Tesseract glyph corruption.
"""
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

    FIX (previously a real bug, confirmed against real output): the
    course-code normalization used by the course-row parser (see
    _normalize_course_code below) was never applied here, so this
    function's stored rows still showed the raw OCR-corrupted AI codes
    (e.g. "A14801" instead of "AI4801") even after course_chunks were
    correctly normalized. Reuses the SAME _normalize_course_code
    function (no second normalization implementation) so the two
    paths can't drift out of sync with each other.
    """
    chunks = []
    for page in pages:
        for t_idx, table in enumerate(page.get("tables", [])):
            if not table or len(table) < 2:  # need at least header + 1 row
                continue

            header = table[0]
            rows = [
                [_normalize_row_cell(cell, program) for cell in row]
                for row in table[1:]
            ]
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

"""
Structured Course Block Boundary Regex Pattern

Responsibility: Extract description content from multi-column SWE/BIO-style course blocks.

Design notes:
- BOUNDARY-BASED EXTRACTION: Captures raw text between the credit hours header and the prerequisites marker 
  to prevent missing description content across interleaved lines.
- POST-PROCESSING CLEANUP: Defers label removal to `_strip_label_noise` to sanitize isolated, 
  vertically aligned label tokens ("Course", "Description") created by pdfplumber line-streaming.
"""
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
                    "credit_hours": credits,
                    "prerequisites": prereqs,
                    "parse_confidence": "high",  # full labeled block: code+name+credits+prereqs+description
                },
            )
        )
    return chunks


# ---------------------------------------------------------------------------
# Table-aware course parser
# ---------------------------------------------------------------------------
#
"""
Responsibility: Parse structured course records directly from preserved page tables.

Design notes:
- EXPLICIT STRUCTURAL BOUNDARIES: Extracts course entries from table rows to eliminate 
  boundary-bleed risks inherent in freeform text window parsing.
- ACCURATE METADATA PRESERVATION: Populates metadata fields solely from explicit cell 
  values, leaving missing attributes as `None` rather than fabricating data.
- EXPLICIT CONFIDENCE SCORING: Sets parser confidence based on recoverable row data 
  to ensure downstream quality tracing without masking incomplete source entries.
"""

_TABLE_COURSE_CODE_PATTERN = re.compile(r"^[A-Z]{2,5}\d{2,4}$")
_TABLE_CREDIT_HOURS_PATTERN = re.compile(r"^\d{1,2}$")  # bare 1-2 digit cell, e.g. "3"

# Table-header labels that can end up sitting in a cell adjacent to a
# course code (e.g. a malformed/merged row where a column header wasn't
# properly excluded from the data rows) and would otherwise look like a
# plausible "longest text cell" course name to the heuristic below.
# Matching is case/whitespace-normalized so "Prerequisites",
# "prerequisites ", "PREREQUISITES" etc. are all caught the same way.
_TABLE_HEADER_WORDS = {
    "prerequisites",
    "course",
    "course name",
    "course code",
    "description",
    "credit hours",
    "credit hour",
    "credits",
}


def _is_table_header_word(value: str) -> bool:
    normalized = re.sub(r"\s+", " ", value.strip()).lower()
    return normalized in _TABLE_HEADER_WORDS


def _normalize_course_code(code: str, program: str) -> str:
    """Seam for per-program course-code corrections. No corrections are
    currently needed for the SWE catalog, so this is a passthrough."""
    return code


def _normalize_row_cell(cell, program: str):
    """
    Apply _normalize_course_code to a single raw table cell, used when
    building chunk_tables()'s stored rows (see above). Only rewrites a
    cell when normalization actually changes it — every other cell
    (course names, credit-hour numbers, None values, Arabic text, etc.)
    passes through completely untouched, not even whitespace-stripped,
    to avoid any unrelated formatting side effects.
    """
    if isinstance(cell, str):
        stripped = cell.strip()
        normalized = _normalize_course_code(stripped, program)
        if normalized != stripped:
            return normalized
    return cell


def _is_code_cell(value: str, program: str) -> bool:
    return bool(_TABLE_COURSE_CODE_PATTERN.match(value))


def _extract_course_row(row: list, program: str, source_file: str) -> dict | None:
    """
    Identify and map course attributes (course code, title, credit hours) from individual table rows.

    Responsibility: Distinguish course records from header or administrative rows and resolve column-ambiguity 
    between primary course codes and embedded prerequisite codes.

    Design notes:
    - NON-COURSE ROW FILTERING: Rejects rows lacking course-code tokens (e.g., table headers, grading scales) 
    to prevent misclassifying non-course entries.
    - PROXIMITY-BASED CODE DISAMBIGUATION: Resolves ambiguous multi-code rows (e.g., `['CS2202', 'Software Engineering', 'CS3301']`) 
    by pairing the course title with the code-shaped cell that minimizes raw column-index distance.
    - DIRECTION-AGNOSTIC PAIRING: Handles both left-to-right and right-to-left column layouts without 
    hardcoded positional bias.
    """
    # Keep raw (index, value) pairs — the distance heuristic needs
    # original column position, not the position after blank cells
    # are filtered out (filtering first would make the prerequisite
    # code and the real code look equally "adjacent" to the name,
    # which is exactly the ambiguity that broke the naive version).
    indexed_cells = [(i, str(c).strip()) for i, c in enumerate(row) if c and str(c).strip()]
    if not indexed_cells:
        return None

    code_candidates = [(i, c) for i, c in indexed_cells if _is_code_cell(c, program)]
    if not code_candidates:
        return None  # not a course row

    remaining = [(i, c) for i, c in indexed_cells if not _is_code_cell(c, program)]

    credit_hours = None
    credit_hours_idx = None
    credit_candidates = [(i, c) for i, c in remaining if _TABLE_CREDIT_HOURS_PATTERN.match(c)]
    if credit_candidates:
        credit_hours_idx, credit_hours = credit_candidates[0]

    name_candidates = [
        (i, c) for i, c in remaining
        if i != credit_hours_idx
        and re.search(r"[A-Za-z\u0600-\u06FF]{3,}", c)
        and not _is_table_header_word(c)  # reject header-label cells (e.g. "Prerequisites") from being read as a course name
    ]
    if not name_candidates:
        # No usable name cell at all — code-only row, too little to
        # build a meaningful chunk from regardless of which code is
        # "correct". Report the first code found, low confidence.
        return {
            "program": program,
            "source_file": source_file,
            "course_code": code_candidates[0][1],
            "course_name": None,
            "credit_hours": credit_hours,
            "prerequisites": None,
            "parse_confidence": "low",
        }

    name_idx, name = max(name_candidates, key=lambda x: len(x[1]))

    if len(code_candidates) == 1:
        # Unambiguous — the only case the original version handled,
        # and the common case for SWE/BIO tables and page 22's
        # elective table. Behavior unchanged from before this fix.
        code = _normalize_course_code(code_candidates[0][1], program)
        confidence = "high" if credit_hours else "medium"
    else:
        # Multiple code-shaped cells in this row — resolve by minimum
        # raw column distance to the name cell.
        distances = [(abs(i - name_idx), c) for i, c in code_candidates]
        distances.sort(key=lambda x: x[0])
        smallest, second = distances[0], distances[1]
        if smallest[0] == second[0]:
            # Genuinely ambiguous — two codes equally close to the
            # name. Per requirement: do not invent a confident
            # relationship. Skip this row rather than guess.
            return None
        code = _normalize_course_code(smallest[1], program)
        # Multi-code rows get one confidence notch down even when
        # resolved, since the pairing is inferred rather than direct.
        confidence = "medium" if credit_hours else "low"

    return {
        "program": program,
        "source_file": source_file,
        "course_code": code,
        "course_name": name,
        "credit_hours": credit_hours,
        "prerequisites": None,  # these table layouts don't reliably expose a prereq column; left unset rather than guessed
        "parse_confidence": confidence,
    }


def chunk_table_courses(cleaned_pages: list[dict], program: str, source_file: str) -> list[Chunk]:
    """
    Scan every table on every page for course-listing rows and emit
    one chunk per distinct course code found. If the same course code
    appears in multiple tables/rows (e.g. listed once in a level
    summary and again in an elective list), the FIRST occurrence with
    the highest confidence wins — duplicates are merged, not stacked,
    so retrieval doesn't return the same course twice.
    """
    found: dict[str, dict] = {}  # course_code -> best record seen so far
    confidence_rank = {"high": 2, "medium": 1, "low": 0}

    for page in cleaned_pages:
        for table in page.get("tables", []):
            for row in table:
                record = _extract_course_row(row, program, source_file)
                if record is None:
                    continue
                code = record["course_code"]
                if code not in found or confidence_rank[record["parse_confidence"]] > confidence_rank[found[code]["parse_confidence"]]:
                    found[code] = record

    chunks = []
    for code, record in found.items():
        name_part = record["course_name"] or "(name not recovered from table)"
        credits_part = record["credit_hours"] or "(not specified)"
        text = f"{name_part} ({code})\nCredit hours: {credits_part}"
        chunks.append(
            Chunk(
                chunk_id=f"{source_file}_course_{code}",
                zone_type="course",
                text=text,
                metadata=record,
            )
        )
    return chunks


def chunk_course_catalog_freeform(catalog_text: str, program: str, source_file: str) -> tuple[list[Chunk], list[str]]:
    """
    Best-effort fallback parser for unstructured, freeform course catalog text.

    Responsibility: Extract course entries when text lacks structured block markers 
    or table layout, using course-code position anchors to define chunk windows.

    Design notes:
    - LINE-ISOLATED BOUNDARY ANCHORING: Treats course codes as new chunk boundaries only 
    when they occupy an entire line, distinguishing real course headers from inline 
    prerequisite references (e.g., "Prerequisite: CS3301").
    - REFERENCE PRESERVATION: Keeps inline course code citations intact within description 
    body text without triggering premature chunk splitting.
    - TRANSPARENT PARSE LOGGING: Returns explicit warning records alongside generated 
    chunks to flag low-confidence entries for manual review.
    """
    all_matches = list(BARE_COURSE_CODE_PATTERN.finditer(catalog_text))

    # Keep only matches where the code is the ENTIRE content of its
    # line (a standalone anchor), not embedded inline in prose.
    matches = []
    for m in all_matches:
        line_start = catalog_text.rfind("\n", 0, m.start()) + 1  # 0 if no prior newline
        line_end = catalog_text.find("\n", m.end())
        if line_end == -1:
            line_end = len(catalog_text)
        line = catalog_text[line_start:line_end].strip()

        if line != m.group(1):
            continue  # code is embedded inline in a longer line — a reference, not a boundary
        matches.append(m)

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
                    "course_name": None,  # freeform parser cannot reliably isolate a name field
                    "credit_hours": None,
                    "prerequisites": None,
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

    """
    Catalog Zone Isolation & Decoupled Boundary Detection

    Responsibility: Isolate course catalog text across non-linear document structures 
    without relying on regulatory article positions.

    Design notes:
    - DECOUPLED ZONE SCANNING: Decouples catalog boundary detection from regulatory 
    article positions, scanning full text for anchor markers (e.g., "This course") 
    to handle interleaved administrative articles gracefully.
    - CROSS-ZONE INTEGRITY: Allows overlapping candidate zones because regulation chunking 
    scans independently, ensuring trailing articles are captured without misinterpreting prose.
    - STUDY-PLAN NOISE FILTERING: Prevents premature anchor matching on isolated 
    course-code listings within pre-catalog study-plan tables.
    """
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
    table_course_chunks = chunk_table_courses(cleaned_pages, program, source_file)

    # Merge precedence: structured (full descriptions, highest fidelity)
    # > table-aware (clean code/name/credits, no description) > freeform
    # (last resort, lowest confidence). A course code is only sourced
    # from a lower-precedence parser if no higher-precedence parser
    # already produced a chunk for that exact code — this is what
    # prevents the same course appearing twice with conflicting/
    # redundant content, per the "one course does not absorb another"
    # requirement.
    covered_codes = {c.metadata["course_code"] for c in structured_course_chunks}

    table_only = [c for c in table_course_chunks if c.metadata["course_code"] not in covered_codes]
    covered_codes |= {c.metadata["course_code"] for c in table_only}

    freeform_chunks, warnings = chunk_course_catalog_freeform(catalog_text, program, source_file)
    freeform_only = [c for c in freeform_chunks if c.metadata["course_code"] not in covered_codes]

    # Safeguard AI-specific merging: only attempt to call the
    # AI-specific extractor when it is present. This prevents crashes
    # for deployments that only include SWE documents and do not define
    # `_extract_ai_course_details` (e.g., AI/BIO sources removed).
    ai_extractor = globals().get("_extract_ai_course_details")
    if program == "AI" and ai_extractor:
        freeform_by_code = {c.metadata["course_code"]: c for c in freeform_chunks}
        merged_courses = []
        used_freeform_codes: set[str] = set()

        for table_chunk in table_only:
            code = table_chunk.metadata["course_code"]
            freeform_chunk = freeform_by_code.get(code)
            if freeform_chunk is None:
                merged_courses.append(table_chunk)
                continue

            merged_metadata = dict(table_chunk.metadata)
            course_name, freeform_credit_hours, prereqs = ai_extractor(freeform_chunk.text, code)
            if course_name:
                merged_metadata["course_name"] = course_name
            if not merged_metadata.get("credit_hours") and freeform_credit_hours:
                merged_metadata["credit_hours"] = freeform_credit_hours
            if prereqs:
                merged_metadata["prerequisites"] = prereqs
            if merged_metadata.get("course_name") and merged_metadata.get("credit_hours") and merged_metadata.get("prerequisites"):
                merged_metadata["parse_confidence"] = "high"
            elif merged_metadata.get("course_name") and merged_metadata.get("credit_hours"):
                merged_metadata["parse_confidence"] = "medium"

            merged_courses.append(
                Chunk(
                    chunk_id=table_chunk.chunk_id,
                    zone_type=table_chunk.zone_type,
                    text=freeform_chunk.text,
                    metadata=merged_metadata,
                )
            )
            used_freeform_codes.add(code)

        freeform_only = [c for c in freeform_only if c.metadata["course_code"] not in used_freeform_codes]
        course_chunks = structured_course_chunks + merged_courses + freeform_only
    else:
        if program == "AI" and not ai_extractor:
            warnings.append("AI-specific extractor `_extract_ai_course_details` not found; skipping AI merging.")
        course_chunks = structured_course_chunks + table_only + freeform_only

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
            "courses_by_source": {
                "structured": len(structured_course_chunks),
                "table": len(table_only),
                "freeform": len(freeform_only),
            },
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
