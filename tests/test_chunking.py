"""
Unit tests for the chunking module (`ingestion/chunk.py`).

Architecture & Design Notes:
- Isolated Synthetic Testing: Tests chunking heuristics against lightweight string fixtures without 
  reading PDFs or running upstream pipeline stages (extraction/preprocessing), keeping execution fast.
- Empirical Regression Coverage: Directly encodes historical regression cases (catalog-text bleed, 
  course-boundary swallowing, Arabic name-resolution priority) using minimal synthetic fixtures to prevent bug reintroduction.
- Refactor Safety Net: Serves as the primary validation harness for regex and heuristic updates within `ingestion/chunk.py`.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest

from ingestion.chunk import (
    chunk_regulatory_articles,
    chunk_tables,
    chunk_course_catalog_structured,
    chunk_table_courses,
    chunk_course_catalog_freeform,
    enrich_prerequisite_names,
    chunk_document,
    _extract_course_row,
    _is_table_header_word,
    Chunk,
)


# ---------------------------------------------------------------------------
# Zone 1: regulatory articles
# ---------------------------------------------------------------------------

def test_single_article_extracted():
    text = "Article (9) Graduation requirements text here, quite short."
    chunks, warnings = chunk_regulatory_articles(text, program="SWE", source_file="swe.pdf")
    assert len(chunks) == 1
    assert chunks[0].zone_type == "regulation"
    assert chunks[0].metadata["article_num"] == 9
    assert warnings == []


def test_article_boundary_stops_at_next_article_marker():
    """
    Regression test for the SWE137 bug: the last catalog course's
    prerequisites field was bleeding regulation text because an
    article chunk wasn't stopping where it should. This test checks
    the article-splitting side of that fix directly — Article 9's
    chunk must stop exactly where Article 10 begins, not run past it.
    """
    text = (
        "Article (9) First article body text that is reasonably long.\n"
        "Article (10) Second article body text, completely separate."
    )
    chunks, _ = chunk_regulatory_articles(text, program="SWE", source_file="swe.pdf")
    assert len(chunks) == 2
    assert "Second article" not in chunks[0].text
    assert "First article" not in chunks[1].text


def test_article_bounded_by_catalog_start():
    """An article immediately preceding the course catalog must stop at
    'Course Code', not swallow the whole catalog while waiting for a
    next article marker that may not appear until after it."""
    text = (
        "Article (20) Final article before the catalog begins here.\n"
        "Course Code SWE999\nCourse Name Some Course\nCredit hours 3\nDescription text."
    )
    chunks, _ = chunk_regulatory_articles(text, program="SWE", source_file="swe.pdf")
    assert len(chunks) == 1
    assert "Course Code" not in chunks[0].text


def test_oversized_article_is_truncated_and_warned():
    long_body = "x" * 2000
    text = f"Article (5) {long_body}"
    chunks, warnings = chunk_regulatory_articles(text, program="SWE", source_file="swe.pdf")
    assert len(chunks[0].text) <= 1500
    assert len(warnings) == 1
    assert "Article 5" in warnings[0]


def test_spurious_short_match_is_dropped():
    """A stray 'Article' mention too short to be a real article body
    (< 20 chars) should be silently dropped, not emitted as a chunk."""
    text = "See Article (3) above.\nArticle (4) A real article with a proper amount of body text here."
    chunks, _ = chunk_regulatory_articles(text, program="SWE", source_file="swe.pdf")
    article_nums = [c.metadata["article_num"] for c in chunks]
    assert 4 in article_nums
    # The "Article (3) above." fragment is far under 20 chars once
    # bounded by the next marker — should not produce its own chunk.
    assert len([c for c in chunks if c.metadata["article_num"] == 3]) == 0


# ---------------------------------------------------------------------------
# Zone 2: tables
# ---------------------------------------------------------------------------

def test_table_chunk_built_from_pdfplumber_style_rows():
    pages = [
        {
            "source_file": "swe.pdf",
            "page_number": 12,
            "text": "",
            "tables": [
                [
                    ["Grade", "Points"],
                    ["A", "4.0"],
                    ["B", "3.0"],
                ]
            ],
        }
    ]
    chunks = chunk_tables(pages, program="SWE", source_file="swe.pdf")
    assert len(chunks) == 1
    assert chunks[0].zone_type == "table"
    assert chunks[0].metadata["page_number"] == 12
    assert chunks[0].metadata["rows"] == [["A", "4.0"], ["B", "3.0"]]


def test_table_with_fewer_than_two_rows_is_skipped():
    """Needs header + at least one data row; a header-only table isn't
    useful as a chunk and should be silently skipped, not error."""
    pages = [{"source_file": "swe.pdf", "page_number": 1, "text": "", "tables": [[["Grade", "Points"]]]}]
    chunks = chunk_tables(pages, program="SWE", source_file="swe.pdf")
    assert chunks == []


# ---------------------------------------------------------------------------
# Zone 3: structured course catalog
# ---------------------------------------------------------------------------

def test_structured_course_with_prerequisites():
    text = (
        "Course Code SWE145\n"
        "Course Name Estimating Software Development\n"
        "Credit hours 3\n"
        "Description Some description text.\n"
        "Prerequisites SWE131\n"
    )
    chunks = chunk_course_catalog_structured(text, program="SWE", source_file="swe.pdf")
    assert len(chunks) == 1
    c = chunks[0]
    assert c.metadata["course_code"] == "SWE145"
    assert c.metadata["prerequisites"] == "SWE131"
    assert c.metadata["parse_confidence"] == "high"


def test_structured_course_without_prerequisites_field():
    """
    Regression test for the UNI013 bug: a course with NO 'Prerequisites'
    line in the source (common for foundational courses with none)
    must not swallow the following course's entire block into its own
    text field. group(5) should come back None, not consume the next
    course's 'Course Code ...' header.
    """
    text = (
        "Course Code UNI013\n"
        "Course Name Foundations of Communication\n"
        "Credit hours 3\n"
        "Description Some description with no prerequisites line at all.\n"
        "Course Code MATH011\n"
        "Course Name Linear Algebra\n"
        "Credit hours 3\n"
        "Description A completely separate course description.\n"
    )
    chunks = chunk_course_catalog_structured(text, program="SWE", source_file="swe.pdf")
    codes = {c.metadata["course_code"]: c for c in chunks}
    assert set(codes.keys()) == {"UNI013", "MATH011"}
    assert codes["UNI013"].metadata["prerequisites"] is None
    # The critical assertion: UNI013's own text must not contain
    # MATH011's course block.
    assert "MATH011" not in codes["UNI013"].text
    assert "Linear Algebra" not in codes["UNI013"].text


def test_label_noise_stripped_from_description():
    """
    Label noise (a stray 'Course' or 'Description' word from an
    interleaved table label) lands at the START of its OWN line — per
    _strip_label_noise's docstring, only a label word sitting alone at
    true line-start is stripped; the pattern deliberately does NOT
    chain-strip multiple label words concatenated on one line, since
    that could over-strip real prose starting with those common words.
    """
    text = (
        "Course Code SWE999\n"
        "Course Name Test Course\n"
        "Credit hours 3\n"
        "Description\n"
        "Some real description text here.\n"
    )
    chunks = chunk_course_catalog_structured(text, program="SWE", source_file="swe.pdf")
    assert "Description\n" not in chunks[0].text
    assert "Some real description text here." in chunks[0].text


# ---------------------------------------------------------------------------
# Table-aware course parser
# ---------------------------------------------------------------------------

def test_extract_course_row_single_code_high_confidence():
    row = ["SWE145", "Estimating Software Development", "3"]
    record = _extract_course_row(row, program="SWE", source_file="swe.pdf")
    assert record["course_code"] == "SWE145"
    assert record["course_name"] == "Estimating Software Development"
    assert record["credit_hours"] == "3"
    assert record["parse_confidence"] == "high"


def test_extract_course_row_no_course_code_returns_none():
    row = ["Prerequisites", "Credit hours", "Description"]
    assert _extract_course_row(row, program="SWE", source_file="swe.pdf") is None


def test_extract_course_row_header_word_rejected_as_name():
    """A malformed row where a header label ('Prerequisites') sits next
    to a real code shouldn't be picked as the course name."""
    row = ["SWE145", "Prerequisites", "3"]
    record = _extract_course_row(row, program="SWE", source_file="swe.pdf")
    # No usable name candidate once "Prerequisites" is excluded as a
    # header word -> falls to the code-only / low-confidence path.
    assert record["course_name"] is None
    assert record["parse_confidence"] == "low"


def test_extract_course_row_disambiguates_multiple_codes_by_distance():
    """A prerequisite code and the real code both appear in one row;
    the real code should be the one closest to the name cell."""
    row = ["CS2202", "", "", "", "", "Software Engineering", "CS3301"]
    record = _extract_course_row(row, program="SWE", source_file="swe.pdf")
    assert record["course_code"] == "CS3301"  # closer to the name cell (index 5) than CS2202 is
    assert record["course_name"] == "Software Engineering"


def test_extract_course_row_equally_ambiguous_codes_returns_none():
    """Two code-shaped cells equidistant from the name cell — must not
    guess; returns None per the 'don't invent a relationship' rule."""
    row = ["CS1001", "Some Course Name", "CS1002"]
    record = _extract_course_row(row, program="SWE", source_file="swe.pdf")
    assert record is None


def test_is_table_header_word_case_and_whitespace_insensitive():
    assert _is_table_header_word("Prerequisites")
    assert _is_table_header_word("  prerequisites  ")
    assert _is_table_header_word("PREREQUISITES")
    assert not _is_table_header_word("Software Engineering")


def test_table_course_prefers_latin_script_name():
    """
    Regression test for the MATH011 bug: when a table row has both an
    Arabic-script and a Latin-script name candidate, course_name must
    resolve to the Latin (English) candidate, matching how course_name
    is recorded everywhere else in this corpus — not the longest
    candidate by raw character count regardless of script.
    """
    row = ["MATH011", "الجبر الخطي والتفاضل", "Linear Algebra", "3"]
    record = _extract_course_row(row, program="SWE", source_file="swe.pdf")
    assert record["course_code"] == "MATH011"
    assert record["course_name"] == "Linear Algebra"


def test_chunk_table_courses_merges_duplicate_codes_keeping_highest_confidence():
    pages = [
        {
            "source_file": "swe.pdf",
            "page_number": 1,
            "tables": [
                [["SWE145", "Estimating Software Dev"]],  # no credit hours -> medium confidence
            ],
        },
        {
            "source_file": "swe.pdf",
            "page_number": 2,
            "tables": [
                [["SWE145", "Estimating Software Dev", "3"]],  # with credit hours -> high confidence
            ],
        },
    ]
    chunks = chunk_table_courses(pages, program="SWE", source_file="swe.pdf")
    assert len(chunks) == 1
    assert chunks[0].metadata["parse_confidence"] == "high"
    assert chunks[0].metadata["credit_hours"] == "3"


# ---------------------------------------------------------------------------
# Freeform fallback parser
# ---------------------------------------------------------------------------

def test_freeform_parser_ignores_inline_reference_codes():
    """A course code embedded inline in prose (a prerequisite reference)
    must not be treated as a new course boundary — only a code that is
    the ENTIRE content of its own line counts as a boundary."""
    text = (
        "SWE131\n"
        "Software Requirements Analysis. This course requires knowledge from "
        "SWE021 as background but that is only a reference, not a boundary. "
        "The description continues for a while to clear the length guard.\n"
        "SWE145\n"
        "Estimating Software Development and Maintenance Projects, a separate "
        "course entirely with its own sufficiently long description text.\n"
    )
    chunks, warnings = chunk_course_catalog_freeform(text, program="SWE", source_file="swe.pdf")
    codes = [c.metadata["course_code"] for c in chunks]
    assert codes == ["SWE131", "SWE145"]
    # SWE021 must never appear as its own boundary/chunk — it was inline.
    assert "SWE021" not in codes


def test_freeform_parser_warns_on_suspiciously_short_entry():
    text = "SWE999\nSWE998\n"  # two bare codes back-to-back, no real description between them
    chunks, warnings = chunk_course_catalog_freeform(text, program="SWE", source_file="swe.pdf")
    assert len(warnings) >= 1


# ---------------------------------------------------------------------------
# Enrichment
# ---------------------------------------------------------------------------

def test_enrich_prerequisite_names_resolves_known_codes():
    chunks = [
        Chunk("c1", "course", "text", {"course_code": "SWE131", "course_name": "Software Requirements Analysis", "prerequisites": None}),
        Chunk("c2", "course", "text", {"course_code": "SWE145", "course_name": "Estimating Software Dev", "prerequisites": "SWE131"}),
    ]
    enrich_prerequisite_names(chunks)
    assert chunks[1].metadata["prerequisite_names"] == ["Software Requirements Analysis"]
    assert chunks[0].metadata["prerequisite_names"] == []


def test_enrich_prerequisite_names_handles_dash_as_no_prereqs():
    chunks = [
        Chunk("c1", "course", "text", {"course_code": "MATH013", "course_name": "Probability", "prerequisites": "---"}),
    ]
    enrich_prerequisite_names(chunks)
    assert chunks[0].metadata["prerequisite_names"] == []


def test_enrich_prerequisite_names_leaves_unresolved_codes_as_bare_code():
    """A prerequisite from outside this document's own catalog (e.g. a
    cross-department code) has no name to resolve to — should fall back
    to the bare code rather than erroring or dropping it."""
    chunks = [
        Chunk("c1", "course", "text", {"course_code": "SWE145", "course_name": "Estimating Software Dev", "prerequisites": "XYZ999"}),
    ]
    enrich_prerequisite_names(chunks)
    assert chunks[0].metadata["prerequisite_names"] == ["XYZ999"]


# ---------------------------------------------------------------------------
# Full orchestration
# ---------------------------------------------------------------------------

def test_chunk_document_merge_precedence_structured_over_table_over_freeform():
    """A course appearing in both the structured catalog text AND a
    table must be sourced from the structured parser only — merge
    precedence should prevent the same course_code appearing twice."""
    cleaned_pages = [
        {
            "source_file": "swe.pdf",
            "page_number": 1,
            "text": (
                "Article (1) Some short intro article text that is long enough to pass the guard.\n"
                "This course catalog begins now with the following entries.\n"
                "Course Code SWE145\n"
                "Course Name Estimating Software Development\n"
                "Credit hours 3\n"
                "This course covers estimation techniques in real depth for the program.\n"
                "Prerequisites SWE131\n"
            ),
            "tables": [
                [["SWE145", "Estimating Software Development", "3"]],  # same course, table form
            ],
        }
    ]
    result = chunk_document(cleaned_pages, program="SWE", source_file="swe.pdf")
    codes = [c["metadata"]["course_code"] for c in result["course_chunks"]]
    assert codes.count("SWE145") == 1
    assert result["counts"]["courses_by_source"]["structured"] == 1
    assert result["counts"]["courses_by_source"]["table"] == 0


def test_chunk_document_counts_match_chunk_lists():
    cleaned_pages = [
        {
            "source_file": "swe.pdf",
            "page_number": 1,
            "text": (
                "Article (1) Short intro article with enough body text to pass the length guard.\n"
                "This course catalog begins now.\n"
                "Course Code SWE999\n"
                "Course Name Sample Course\n"
                "Credit hours 3\n"
                "This course has a sufficiently long description for testing purposes here.\n"
            ),
            "tables": [],
        }
    ]
    result = chunk_document(cleaned_pages, program="SWE", source_file="swe.pdf")
    assert result["counts"]["regulations"] == len(result["regulation_chunks"])
    assert result["counts"]["tables"] == len(result["table_chunks"])
    assert result["counts"]["courses"] == len(result["course_chunks"])


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
