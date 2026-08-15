"""
tests/test_guardrails.py — unit tests for guardrails/schemas.py

Responsibility: exercise the STRUCTURAL contract — GeneratorResult and
GuardrailReport's pydantic validation — in isolation from the citation
business logic. tests/test_validators.py already covers verify_citations'
BEHAVIOR (does it correctly flag a hallucinated source); this file
covers the layer underneath it: does the schema actually reject a
malformed shape loudly, the way schemas.py's own module docstring says
it's meant to ("a malformed response... fails fast and loudly at the
schema layer").

Why this is a separate file and not folded into test_validators.py:
schemas.py and validators.py have genuinely different jobs (shape vs.
business rule — see schemas.py's own docstring), and a schema
regression (e.g. someone loosens a field's type "to fix a test" without
realizing why it was strict) should be caught by a test that's ABOUT
the schema, not buried inside a citation-logic test where the failure
would be confusing to diagnose.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
from pydantic import ValidationError

from guardrails.schemas import RetrievedChunk, GeneratorResult, GuardrailReport
from guardrails.validators import verify_citations


# ---------------------------------------------------------------------------
# RetrievedChunk
# ---------------------------------------------------------------------------

def test_retrieved_chunk_accepts_valid_shape():
    chunk = RetrievedChunk(text="some chunk text", score=0.42, metadata={"zone_type": "course"})
    assert chunk.text == "some chunk text"
    assert chunk.score == 0.42
    assert chunk.metadata == {"zone_type": "course"}


def test_retrieved_chunk_metadata_defaults_to_empty_dict():
    chunk = RetrievedChunk(text="text", score=0.5)
    assert chunk.metadata == {}


def test_retrieved_chunk_requires_text():
    with pytest.raises(ValidationError):
        RetrievedChunk(score=0.5)


def test_retrieved_chunk_requires_score():
    with pytest.raises(ValidationError):
        RetrievedChunk(text="text")


def test_retrieved_chunk_rejects_non_numeric_score():
    with pytest.raises(ValidationError):
        RetrievedChunk(text="text", score="not a number")


# ---------------------------------------------------------------------------
# GeneratorResult
# ---------------------------------------------------------------------------

def _valid_generator_dict(**overrides) -> dict:
    base = {
        "answer": "SWE145 requires SWE131.",
        "sources": ["SWE145"],
        "confidence": "high",
        "retrieved_chunks": [{"text": "chunk text", "score": 0.6, "metadata": {"course_code": "SWE145"}}],
    }
    base.update(overrides)
    return base


def test_generator_result_accepts_valid_shape():
    result = GeneratorResult.model_validate(_valid_generator_dict())
    assert result.answer == "SWE145 requires SWE131."
    assert result.confidence == "high"
    assert len(result.retrieved_chunks) == 1
    assert isinstance(result.retrieved_chunks[0], RetrievedChunk)


def test_generator_result_requires_confidence():
    """confidence has no default (unlike sources/retrieved_chunks) —
    generator.py always sets it explicitly, and a result missing it
    entirely signals something upstream is broken; this should fail
    the schema, not silently default to some guessed value."""
    bad = _valid_generator_dict()
    del bad["confidence"]
    with pytest.raises(ValidationError):
        GeneratorResult.model_validate(bad)


def test_generator_result_rejects_invalid_confidence_value():
    """confidence is a Literal["high","medium","low"] — anything else
    (a typo, a model hallucinating a different word) must be rejected
    at the schema layer rather than silently accepted and causing
    confusing behavior downstream (e.g. in print_summary's formatting)."""
    with pytest.raises(ValidationError):
        GeneratorResult.model_validate(_valid_generator_dict(confidence="very high"))


def test_generator_result_requires_answer():
    bad = _valid_generator_dict()
    del bad["answer"]
    with pytest.raises(ValidationError):
        GeneratorResult.model_validate(bad)


def test_generator_result_sources_defaults_to_empty_list():
    bad = _valid_generator_dict()
    del bad["sources"]
    result = GeneratorResult.model_validate(bad)
    assert result.sources == []


def test_generator_result_retrieved_chunks_defaults_to_empty_list():
    bad = _valid_generator_dict()
    del bad["retrieved_chunks"]
    result = GeneratorResult.model_validate(bad)
    assert result.retrieved_chunks == []


def test_generator_result_parse_error_defaults_to_false():
    result = GeneratorResult.model_validate(_valid_generator_dict())
    assert result.parse_error is False


def test_generator_result_rejects_malformed_retrieved_chunk():
    """A retrieved_chunks entry missing 'score' should fail the whole
    GeneratorResult validation, not silently produce a chunk with a
    missing field that downstream code (e.g. extract_valid_citation_labels,
    which reads chunk.metadata) would then error on less clearly."""
    bad = _valid_generator_dict(retrieved_chunks=[{"text": "no score here"}])
    with pytest.raises(ValidationError):
        GeneratorResult.model_validate(bad)


# ---------------------------------------------------------------------------
# GuardrailReport
# ---------------------------------------------------------------------------

def test_guardrail_report_defaults():
    report = GuardrailReport(
        answer="text",
        sources=[],
        confidence="low",
        retrieved_chunks=[],
    )
    assert report.citation_warnings == []
    assert report.citations_valid is True
    assert report.parse_error is False


def test_guardrail_report_rejects_invalid_confidence():
    with pytest.raises(ValidationError):
        GuardrailReport(answer="text", sources=[], confidence="super-high", retrieved_chunks=[])


# ---------------------------------------------------------------------------
# Integration: verify_citations surfaces schema failures loudly
# ---------------------------------------------------------------------------

def test_verify_citations_raises_on_schema_violation_not_silently_passes():
    """
    Per verify_citations' own docstring: a shape-level failure should
    surface immediately as a ValidationError, not be swallowed and
    treated as "no citation problems found". This is the one place
    schemas.py and validators.py's behaviors actually meet — worth a
    test at the integration point, not just each file in isolation.
    """
    malformed = {"answer": "text", "sources": [], "retrieved_chunks": []}  # missing required "confidence"
    with pytest.raises(ValidationError):
        verify_citations(malformed)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
