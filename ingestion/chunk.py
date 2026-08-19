"""chunk.py — zone-aware chunking

Split cleaned page text into three zone types: regulatory
articles, program tables (kept structured), and course catalog
entries. Uses regex and table-aware heuristics and flags low-
confidence parses for manual review.
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

# An article's chunk beyond this length likely inline-references table
# content rather than being pure regulatory prose (see
# chunk_regulatory_articles). Chosen as a generous multiple of a
# typical single-article chunk length, not tuned to any specific
# article number.
MAX_ARTICLE_CHARS = 1500

# Match article headers like "مادة 13" or "Article (12)".
# Tolerant to OCR/BiDi noise; anchored at line start for precision.
ARTICLE_PATTERN = re.compile(
    r"(?:^|\n)\s*(?:مادة|Article)\s*:?\s*[\[\(\)\]]*\s*(\d{1,2})\s*[\[\(\)\]]*",
    re.MULTILINE,
)


def chunk_regulatory_articles(full_text: str, program: str, source_file: str) -> tuple[list[Chunk], list[str]]:
    """
    Split full document text into one chunk per regulatory article.

    Strategy: find every article marker position, then each chunk spans
    from one marker to the start of the next. An article positioned
    immediately before the course catalog is additionally bounded at
    the catalog's start (first "Course Code" occurrence) — otherwise,
    since the next article marker may not appear again until after the
    entire catalog, that article's span would swallow the whole
    catalog while waiting for it.

    An article can still legitimately reference several tables inline
    (e.g. "Table 4 shows the credit hours required...") without a text
    marker separating its own prose from that table content, so the
    catalog-start bound alone doesn't guarantee a small chunk. Rather
    than guess where an article's prose ends and referenced table
    content begins, an oversized result is truncated at
    MAX_ARTICLE_CHARS and flagged in the returned warnings list —
    consistent with this file's "flag it, don't fix it in the dark"
    approach elsewhere. The full table data is not lost: chunk_tables()
    captures it separately and correctly regardless of this bound.

    Returns (chunks, warnings).
    """
    matches = list(ARTICLE_PATTERN.finditer(full_text))
    catalog_marker = re.search(r"Course Code", full_text)
    catalog_start = catalog_marker.start() if catalog_marker else None
    chunks = []
    warnings = []

    for i, match in enumerate(matches):
        article_num = match.group(1)
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)
        if catalog_start is not None and start < catalog_start < end:
            end = catalog_start
        article_text = full_text[start:end].strip()

        if len(article_text) < 20:  # guard against spurious matches
            continue

        if len(article_text) > MAX_ARTICLE_CHARS:
            warnings.append(
                f"Article {article_num} truncated at {MAX_ARTICLE_CHARS} chars "
                f"(was {len(article_text)}) — likely inline-references table content; "
                f"see table_chunks for the full table data."
            )
            article_text = article_text[:MAX_ARTICLE_CHARS].strip()

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

    return chunks, warnings


# ---------------------------------------------------------------------------
# Zone 2: Program tables (structured, not chunked as prose)
# ---------------------------------------------------------------------------

def chunk_tables(pages: list[dict], program: str, source_file: str) -> list[Chunk]:
    """Convert pdfplumber tables into structured chunks.

    Each table chunk includes a short summary and the raw rows
    (normalized via _normalize_row_cell). Preserves table structure
    for precise lookups rather than flattening to prose.
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

# Parse SWE/BIO-style labeled course blocks. Capture code, name,
# credit hours, description, and optional prerequisites. Strip
# stray "Course"/"Description" label tokens after extraction.
COURSE_BLOCK_PATTERN = re.compile(
    r"Course Code\s+([A-Z]{2,5}\d{2,4})\s*"
    r"Course Name\s+(.+?)\s*\n"
    r"Credit hours\s+(.+?)\n"
    r"(.+?)"
    r"(?:Prerequisites\s+(.+?))?(?=Course Code|\n\s*(?:مادة|Article)\s*:?\s*[\[\(\)\]]*\s*\d|\Z)",
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
        code, name, credits, raw_description = (g.strip() for g in match.groups()[:4])
        prereqs = match.group(5).strip() if match.group(5) is not None else None
        description = _strip_label_noise(raw_description)
        prereqs_line = f"Prerequisites: {prereqs}\n" if prereqs is not None else ""
        chunks.append(
            Chunk(
                chunk_id=f"{source_file}_course_{code}",
                zone_type="course",
                text=f"{name} ({code})\nCredit hours: {credits}\n{prereqs_line}\n{description}",
                metadata={
                    "program": program,
                    "source_file": source_file,
                    "course_code": code,
                    "course_name": name,
                    "credit_hours": credits,
                    "prerequisites": prereqs,
                    "parse_confidence": "high",  # full labeled block: code+name+credits+description (prerequisites present when the source states one)
                },
            )
        )
    return chunks


"""
Table-aware course extraction parser.

Architecture & Design Notes:
- Structural Preservation: Directly processes structured page tables (`CleanedPage.tables`) 
  to prevent discarding critical tabular course data captured during extraction.
- Deterministic Parsing: Treats individual table rows as explicit, unambiguous records, avoiding 
  the text-boundary bleed and chunk-truncation risks inherent in freeform prose windowing.
- Honest Completeness & Confidence: Leaves unprovided or ambiguous row fields as `None` rather than 
  guessing defaults, setting explicit confidence scores based strictly on recoverable data.
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
    Inspect one table row and try to identify a course-code cell, a
    course-name cell, and a credit-hours cell. Returns None if no
    course-code-shaped cell is found at all (row isn't a course row —
    e.g. it's a header row, or a grading-scale/admin table row, both
    of which are common in these documents and must NOT be misread
    as course entries).

    DISAMBIGUATION: some rows contain MULTIPLE code-shaped cells — one
    is the actual course code, another is a prerequisite/reference
    code embedded in the same row. E.g. the raw row
        ['CS2202', '', '', '', '', 'Software Engineering', 'CS3301']
    has CS2202 (a prerequisite) in the first cell and CS3301 (the real
    code) in the last — taking "the first code-shaped cell found"
    would get this backwards. The real code is consistently the
    code-shaped cell with the SMALLEST raw column-index distance to
    the course-name cell. This is a distance metric rather than a
    "prefer left" or "prefer right" rule, so it also handles
    code-before-name row layouts correctly, and it's a no-op for rows
    with only one code-shaped cell (the common case), leaving
    already-correct parsing unaffected.
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

    # Bilingual rows can have both an Arabic and an English label as
    # separate cells. Picking purely by raw character length is not
    # script-aware and can pick the Arabic cell over a shorter English
    # name (or vice versa). Prefer a Latin-script candidate when one
    # exists, since course_name elsewhere in this corpus is recorded
    # in English; fall back to the longest candidate only when no
    # Latin-script candidate is present.
    latin_candidates = [(i, c) for i, c in name_candidates if re.search(r"[A-Za-z]{3,}", c)]
    name_idx, name = max(latin_candidates or name_candidates, key=lambda x: len(x[1]))

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
    Best-effort fallback parser for freeform catalog text: used when a
    course entry doesn't match the structured "Course Code / Course
    Name / ..." block format or a table row. Description paragraphs
    can precede or follow a bare course code with no consistent
    labeling, so we anchor on code positions and take a window of text
    around each as the chunk.

    BOUNDARY DETECTION: a course code can also appear as a
    PREREQUISITE reference inside another course's description (e.g.
    "Software Engineering (CS3301)" cited as a prerequisite), and that
    must not be mistaken for a new course boundary. The structural
    signal that distinguishes a real course-code heading from an
    inline reference is line placement: a real course-code anchor
    sits ALONE on its own line, while a reference is embedded inline
    within a longer line alongside other text. A code is only treated
    as a course boundary if the ENTIRE line it appears on (after
    stripping whitespace) is exactly that code. This covers
    parenthesized and unparenthesized reference styles uniformly. It
    only changes which matches are treated as chunk BOUNDARIES — an
    inline reference still remains exactly as-is inside whichever
    course's description text contains it.

    Returns (chunks, unparsed_warnings) — the warnings list surfaces
    anything that looked like it should be a course entry but didn't
    fit the expected shape, so it can be inspected rather than
    silently losing content.
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

def enrich_prerequisite_names(course_chunks: list[Chunk]) -> None:
    """
    Second-pass enrichment: attach a resolved course_name for each
    prerequisite code, without touching the existing prerequisites
    field. Mutates each chunk's metadata in place.

    Runs after all course chunks exist, since resolving a code to a
    name requires the full code -> name map, which no single-course
    parser has access to while it's still parsing one block/row.
    Codes with no match in this document's catalog (cross-department
    prerequisites, for example) are left unresolved rather than
    treated as an error.
    """
    name_by_code = {c.metadata["course_code"]: c.metadata["course_name"] for c in course_chunks}

    for chunk in course_chunks:
        prereqs = chunk.metadata.get("prerequisites")
        if not prereqs or prereqs == "---":
            chunk.metadata["prerequisite_names"] = []
            continue

        codes = [code.strip() for code in prereqs.split(",") if code.strip()]
        chunk.metadata["prerequisite_names"] = [name_by_code.get(code, code) for code in codes]


def chunk_document(cleaned_pages: list[dict], program: str, source_file: str) -> dict:
    """
    Run the full zone-aware chunking pipeline on one document's cleaned
    pages. Returns a dict with chunks grouped by zone plus any warnings,
    so you can inspect quality per zone rather than one flat list.
    """
    full_text = "\n".join(p["text"] for p in cleaned_pages)

    article_chunks, article_warnings = chunk_regulatory_articles(full_text, program, source_file)
    table_chunks = chunk_tables(cleaned_pages, program, source_file)

    # Catalog-zone isolation. Course descriptions in this document
    # follow the regulatory articles, but a few short administrative
    # articles can appear after the catalog rather than before it — so
    # "last article position" isn't a reliable catalog-start reference.
    # Instead, search the FULL document text for the first "This
    # course" occurrence and take everything from there to the end as
    # candidate catalog text. This is safe even if a trailing article
    # falls inside that range, because the structured course-block
    # parser requires specific field labels ("Course Code", "Course
    # Name", etc.) and won't misfire on plain regulation prose.
    # Regulation chunking is unaffected by this boundary — it always
    # scans the full original text independently, so trailing articles
    # are still correctly captured as regulation chunks regardless of
    # where the catalog boundary falls.
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

    freeform_chunks, freeform_warnings = chunk_course_catalog_freeform(catalog_text, program, source_file)
    warnings = article_warnings + freeform_warnings
    freeform_only = [c for c in freeform_chunks if c.metadata["course_code"] not in covered_codes]

    course_chunks = structured_course_chunks + table_only + freeform_only
    enrich_prerequisite_names(course_chunks)

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
