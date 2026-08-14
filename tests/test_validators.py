"""
tests/test_validators.py — unit tests for guardrails/validators.py

Responsibility: verify verify_citations() against constructed
generator-result dicts. No real Pinecone/Groq calls — same injectable/
fake-input pattern already used for retriever.py and embed.py in this
project, since the guardrail only operates on the dict shape
generator.answer() returns, not on retrieval or generation themselves.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest

from guardrails.validators import verify_citations


def _chunk(zone_type: str, **metadata) -> dict:
    """Build a fake retrieved-chunk dict matching retriever.py's shape."""
    return {
        "text": "irrelevant for this test",
        "score": 0.7,
        "metadata": {"zone_type": zone_type, **metadata},
    }


def test_valid_citations_pass_unflagged():
    result = {
        "answer": "SWE142 requires SWE101 as a prerequisite, per Article 9.",
        "sources": ["SWE142", "Article 9"],
        "confidence": "high",
        "retrieved_chunks": [
            _chunk("course", course_code="SWE142"),
            _chunk("regulation", article_num=9),
        ],
    }
    report = verify_citations(result)
    assert report["citations_valid"] is True
    assert report["citation_warnings"] == []
    assert report["confidence"] == "high"  # untouched — nothing to downgrade


def test_hallucinated_course_code_is_flagged_and_downgraded():
    result = {
        "answer": "SWE999 is a required course.",
        "sources": ["SWE999"],  # never retrieved
        "confidence": "high",
        "retrieved_chunks": [
            _chunk("course", course_code="SWE142"),
        ],
    }
    report = verify_citations(result)
    assert report["citations_valid"] is False
    assert len(report["citation_warnings"]) == 1
    assert "SWE999" in report["citation_warnings"][0]
    assert report["confidence"] == "low"  # downgraded from high


def test_case_and_whitespace_drift_not_flagged():
    result = {
        "answer": "See article 9 for details.",
        "sources": ["article   9"],  # lowercase, extra whitespace
        "confidence": "medium",
        "retrieved_chunks": [
            _chunk("regulation", article_num=9),
        ],
    }
    report = verify_citations(result)
    assert report["citations_valid"] is True
    assert report["citation_warnings"] == []


def test_parse_error_result_passes_through_unflagged():
    result = {
        "answer": "raw model text, not valid JSON",
        "sources": [],
        "confidence": "low",
        "retrieved_chunks": [_chunk("course", course_code="SWE142")],
        "parse_error": True,
    }
    report = verify_citations(result)
    assert report["citations_valid"] is True
    assert report["citation_warnings"] == []
    assert report["parse_error"] is True


def test_no_match_case_empty_sources_empty_chunks():
    result = {
        "answer": "I don't have information about that in the SWE program regulations.",
        "sources": [],
        "confidence": "low",
        "retrieved_chunks": [],
    }
    report = verify_citations(result)
    assert report["citations_valid"] is True
    assert report["citation_warnings"] == []


def test_table_chunk_is_not_a_valid_citation_label():
    """A table chunk being retrieved doesn't make a citation shaped
    like a table reference valid — system.txt never asks the model to
    cite tables directly, so this should be flagged same as any other
    unmatched source."""
    result = {
        "answer": "See the table for details.",
        "sources": ["Table, page 12"],
        "confidence": "medium",
        "retrieved_chunks": [
            _chunk("table", page_number=12),
        ],
    }
    report = verify_citations(result)
    assert report["citations_valid"] is False
    assert len(report["citation_warnings"]) == 1


def test_multiple_sources_one_bad_one_good():
    """Confirms partial failure: a real hallucination alongside a
    legitimate citation still flags only the bad one, and the answer
    itself is preserved (see validators.py's fail-open design note)."""
    result = {
        "answer": "SWE142 and SWE999 are both relevant.",
        "sources": ["SWE142", "SWE999"],
        "confidence": "high",
        "retrieved_chunks": [
            _chunk("course", course_code="SWE142"),
        ],
    }
    report = verify_citations(result)
    assert report["citations_valid"] is False
    assert len(report["citation_warnings"]) == 1
    assert "SWE999" in report["citation_warnings"][0]
    assert report["answer"] == "SWE142 and SWE999 are both relevant."  # not discarded
    assert report["confidence"] == "low"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
