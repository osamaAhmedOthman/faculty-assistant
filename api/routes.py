"""
api/routes.py — HTTP routes

Responsibility: translate an HTTP request into a Pipeline.run() call
and reshape the result into QueryResponse. No retrieval, generation,
or guardrail logic lives here — this file is a thin adapter between
FastAPI and rag/pipeline.py, matching the project's existing
one-concern-per-file convention (see pipeline.py's own docstring).

Design notes:
- ONE Pipeline INSTANCE, MODULE-SCOPED: instantiated once at import
  time, not per-request. Pipeline.__init__ builds a Generator, which
  builds a Retriever, which loads the sentence-transformers embedding
  model and opens a Pinecone connection — doing that per-request would
  be a multi-second cost on every single call, and would also reset
  the shared circuit breaker's state (see pipeline.py's
  _GROQ_CIRCUIT_BREAKER) between requests, defeating its purpose of
  tracking failures ACROSS calls.
- NO TRY/EXCEPT AROUND pipeline.run(): Pipeline.run() already has its
  own internal fallback path for generation failures (retry exhaustion,
  circuit breaker open — see pipeline.py's _call_generator_with_reliability)
  and returns a normal, valid response shape even in that case. There
  is no expected exception type left for this layer to catch; letting
  something genuinely unexpected propagate to FastAPI's default 500
  handler is more honest than swallowing it into a fake 200.
- CHUNK RESHAPING LIVES HERE, NOT IN models.py: building a display
  label from a chunk's metadata (course_code vs. "Article N") is
  presentation logic specific to this HTTP layer, not a schema
  concern — models.py only defines the shape, this file decides how
  to populate it.
"""

from fastapi import APIRouter

from api.models import HealthResponse, QueryRequest, QueryResponse, RetrievedChunkResponse
from rag.pipeline import Pipeline

router = APIRouter()

# Module-scoped singleton — see design notes above for why this must
# NOT be constructed inside query_endpoint().
_pipeline = Pipeline()


def _label_for_chunk(metadata: dict) -> str | None:
    """
    Resolve a display label for one retrieved chunk's metadata,
    mirroring the exact label format rag/generator.py's format_context()
    shows the model (course code, or "Article N") — see
    guardrails/validators.py's extract_valid_citation_labels for the
    same mapping used internally for citation matching. Kept as its
    own small function rather than inlined so the course/regulation/
    table cases are each a single readable branch.
    """
    zone = metadata.get("zone_type")
    if zone == "course":
        return metadata.get("course_code")
    if zone == "regulation":
        article_num = metadata.get("article_num")
        return f"Article {article_num}" if article_num is not None else None
    if zone == "table":
        page_number = metadata.get("page_number")
        return f"Table, page {page_number}" if page_number is not None else "Table"
    return None


def _to_chunk_response(chunk: dict) -> RetrievedChunkResponse:
    metadata = chunk.get("metadata", {})
    return RetrievedChunkResponse(
        text=chunk["text"],
        score=chunk["score"],
        zone_type=metadata.get("zone_type"),
        label=_label_for_chunk(metadata),
    )


@router.post("/query", response_model=QueryResponse)
def query_endpoint(request: QueryRequest) -> QueryResponse:
    result = _pipeline.run(request.question, top_k=request.top_k, zone_filter=request.zone_filter)

    return QueryResponse(
        answer=result["answer"],
        sources=result["sources"],
        confidence=result["confidence"],
        retrieved_chunks=[_to_chunk_response(c) for c in result["retrieved_chunks"]],
        parse_error=result.get("parse_error", False),
        citation_warnings=result.get("citation_warnings", []),
        citations_valid=result.get("citations_valid", True),
    )


@router.get("/health", response_model=HealthResponse)
def health_endpoint() -> HealthResponse:
    """
    Deliberately does NOT touch Pinecone/Groq — a liveness check for
    "is the process up and able to handle a request" should not itself
    depend on external services being reachable, or a Groq outage would
    make the health check fail too, which is a misleading signal for
    an orchestrator (e.g. docker-compose, k8s) deciding whether to
    restart the container.
    """
    return HealthResponse(status="ok")
