"""
tests/test_retrieval.py — unit tests for rag/retriever.py

Responsibility: exercise Retriever's filter-building, result-shaping,
and input validation logic using injected FAKE embedding/Pinecone
clients — never a real sentence-transformers model download or a real
Pinecone connection. Retriever.__init__ accepts both clients as
constructor args specifically to make this possible (see its own
docstring comment), matching the same injectable-client pattern
already used for GroqClient in generator.py.

WHAT THIS DOES NOT TEST: whether the real EmbeddingClient produces
correct vectors, or whether the real PineconeClient's query() call
actually talks to Pinecone correctly. Those are external-dependency
concerns exercised instead by retriever.py's own __main__ smoke test
(python rag/retriever.py "a real question") against the live index —
this file is about Retriever's OWN logic (filter construction, result
reshaping, validation) in isolation from both.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest

from rag.retriever import Retriever


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class FakeEmbeddingClient:
    """Returns a fixed vector regardless of input text — Retriever
    never inspects the vector's actual values, only passes it through
    to the Pinecone client, so a constant stub is sufficient."""

    def __init__(self):
        self.last_query = None

    def embed_query(self, text: str) -> list[float]:
        self.last_query = text
        return [0.1, 0.2, 0.3]


class FakePineconeClient:
    """Records the exact kwargs it was called with (so tests can assert
    on filter construction) and returns a canned match list shaped like
    PineconeClient.query()'s real return value."""

    def __init__(self, matches: list[dict] | None = None):
        self.matches = matches if matches is not None else []
        self.last_call_kwargs = None

    def query(self, vector, top_k, filter=None, **kwargs):
        self.last_call_kwargs = {"vector": vector, "top_k": top_k, "filter": filter, **kwargs}
        return self.matches


def _match(chunk_id: str, score: float, metadata: dict) -> dict:
    return {"id": chunk_id, "score": score, "metadata": metadata}


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def test_empty_query_raises_value_error():
    retriever = Retriever(embedding_client=FakeEmbeddingClient(), pinecone_client=FakePineconeClient())
    with pytest.raises(ValueError):
        retriever.retrieve("")


def test_whitespace_only_query_raises_value_error():
    retriever = Retriever(embedding_client=FakeEmbeddingClient(), pinecone_client=FakePineconeClient())
    with pytest.raises(ValueError):
        retriever.retrieve("   \n\t  ")


# ---------------------------------------------------------------------------
# Filter construction
# ---------------------------------------------------------------------------

def test_no_filters_passes_none_to_pinecone():
    fake_pc = FakePineconeClient()
    retriever = Retriever(embedding_client=FakeEmbeddingClient(), pinecone_client=fake_pc)
    retriever.retrieve("What is the minimum GPA?")
    assert fake_pc.last_call_kwargs["filter"] is None


def test_program_filter_only():
    fake_pc = FakePineconeClient()
    retriever = Retriever(embedding_client=FakeEmbeddingClient(), pinecone_client=fake_pc)
    retriever.retrieve("query", program_filter="SWE")
    assert fake_pc.last_call_kwargs["filter"] == {"program": {"$eq": "SWE"}}


def test_zone_filter_only():
    fake_pc = FakePineconeClient()
    retriever = Retriever(embedding_client=FakeEmbeddingClient(), pinecone_client=fake_pc)
    retriever.retrieve("query", zone_filter="course")
    assert fake_pc.last_call_kwargs["filter"] == {"zone_type": {"$eq": "course"}}


def test_both_filters_combined():
    fake_pc = FakePineconeClient()
    retriever = Retriever(embedding_client=FakeEmbeddingClient(), pinecone_client=fake_pc)
    retriever.retrieve("query", program_filter="SWE", zone_filter="regulation")
    assert fake_pc.last_call_kwargs["filter"] == {
        "program": {"$eq": "SWE"},
        "zone_type": {"$eq": "regulation"},
    }


def test_build_filter_static_method_directly():
    """_build_filter is a @staticmethod — worth testing directly too,
    not just through retrieve(), since it's the one piece of Retriever
    with real conditional logic (everything else is pass-through)."""
    assert Retriever._build_filter(None, None) is None
    assert Retriever._build_filter("SWE", None) == {"program": {"$eq": "SWE"}}
    assert Retriever._build_filter(None, "table") == {"zone_type": {"$eq": "table"}}
    assert Retriever._build_filter("SWE", "table") == {
        "program": {"$eq": "SWE"},
        "zone_type": {"$eq": "table"},
    }


# ---------------------------------------------------------------------------
# top_k and query pass-through
# ---------------------------------------------------------------------------

def test_top_k_passed_through_to_pinecone():
    fake_pc = FakePineconeClient()
    retriever = Retriever(embedding_client=FakeEmbeddingClient(), pinecone_client=fake_pc)
    retriever.retrieve("query", top_k=3)
    assert fake_pc.last_call_kwargs["top_k"] == 3


def test_default_top_k_used_when_not_specified():
    from core.config import RETRIEVAL_TOP_K

    fake_pc = FakePineconeClient()
    retriever = Retriever(embedding_client=FakeEmbeddingClient(), pinecone_client=fake_pc)
    retriever.retrieve("query")
    assert fake_pc.last_call_kwargs["top_k"] == RETRIEVAL_TOP_K


def test_query_text_reaches_embedding_client_unmodified():
    fake_embed = FakeEmbeddingClient()
    retriever = Retriever(embedding_client=fake_embed, pinecone_client=FakePineconeClient())
    retriever.retrieve("What are the prerequisites for SWE145?")
    assert fake_embed.last_query == "What are the prerequisites for SWE145?"


def test_embedding_vector_passed_through_to_pinecone_query():
    fake_pc = FakePineconeClient()
    retriever = Retriever(embedding_client=FakeEmbeddingClient(), pinecone_client=fake_pc)
    retriever.retrieve("query")
    assert fake_pc.last_call_kwargs["vector"] == [0.1, 0.2, 0.3]


# ---------------------------------------------------------------------------
# Result shaping
# ---------------------------------------------------------------------------

def test_results_reshaped_to_text_score_metadata():
    matches = [
        _match("swe.pdf_course_SWE145", 0.42, {"text": "Estimating Software Dev...", "course_code": "SWE145", "zone_type": "course"}),
    ]
    retriever = Retriever(embedding_client=FakeEmbeddingClient(), pinecone_client=FakePineconeClient(matches))
    results = retriever.retrieve("query")
    assert len(results) == 1
    assert set(results[0].keys()) == {"text", "score", "metadata"}
    assert results[0]["text"] == "Estimating Software Dev..."
    assert results[0]["score"] == 0.42
    assert results[0]["metadata"]["course_code"] == "SWE145"


def test_results_preserve_pinecone_return_order():
    """Retriever must not re-sort — Pinecone already returns matches
    ordered by relevance for cosine similarity, and re-sorting here
    would be redundant at best and a bug source if a different
    ordering were ever introduced upstream."""
    matches = [
        _match("a", 0.9, {"text": "highest"}),
        _match("b", 0.5, {"text": "middle"}),
        _match("c", 0.1, {"text": "lowest"}),
    ]
    retriever = Retriever(embedding_client=FakeEmbeddingClient(), pinecone_client=FakePineconeClient(matches))
    results = retriever.retrieve("query")
    assert [r["text"] for r in results] == ["highest", "middle", "lowest"]


def test_missing_text_in_metadata_defaults_to_empty_string():
    """A chunk with no 'text' key in its metadata (shouldn't happen
    with real uploaded data, but Retriever shouldn't crash on it)
    should default to '' rather than raising a KeyError."""
    matches = [_match("x", 0.5, {"course_code": "SWE145"})]  # no "text" key
    retriever = Retriever(embedding_client=FakeEmbeddingClient(), pinecone_client=FakePineconeClient(matches))
    results = retriever.retrieve("query")
    assert results[0]["text"] == ""


def test_empty_matches_returns_empty_list():
    retriever = Retriever(embedding_client=FakeEmbeddingClient(), pinecone_client=FakePineconeClient(matches=[]))
    results = retriever.retrieve("a query with no relevant matches at all")
    assert results == []


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
