"""
api/models.py — request/response schemas for the FastAPI layer

Responsibility: define the HTTP-facing contract only. No pipeline
logic, no guardrail logic — those stay in rag/pipeline.py and
guardrails/. This file's only job is the shape of what crosses the
wire, so routes.py and dashboard/app.py (and FastAPI's own
auto-generated OpenAPI docs) all agree on it.

Design notes:
- MIRRORS GuardrailReport, DOESN'T IMPORT IT: QueryResponse's fields
  are deliberately kept in lockstep with guardrails/schemas.py's
  GuardrailReport (same field names, same types) rather than importing
  and reusing that model directly. Reusing it would couple the HTTP
  contract to an internal pipeline type — a future internal refactor
  of GuardrailReport (e.g. adding a field only pipeline.py cares about)
  would then silently change the public API response shape too. Two
  separate models, kept in sync deliberately, means the HTTP contract
  only changes when someone decides it should.
- RetrievedChunkResponse OMITS raw metadata: the internal
  RetrievedChunk.metadata dict is a grab-bag (course_code, article_num,
  page_number, prerequisite_names, ...) meant for internal guardrail
  matching, not a stable public field. Exposing the handful of fields
  a dashboard/API consumer actually wants (zone_type, a resolved
  label) is a clearer contract than passing the whole internal dict
  through unfiltered.
- top_k / zone_filter ARE OPTIONAL ON THE REQUEST: mirrors
  Pipeline.run()'s own optional args with the same defaults, so a
  client that sends neither gets identical behavior to calling
  Pipeline.run(query) directly.
"""

from typing import Literal

from pydantic import BaseModel, Field

from core.config import RETRIEVAL_TOP_K


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The user's question about SWE program regulations.")
    top_k: int = Field(default=RETRIEVAL_TOP_K, ge=1, le=20, description="Number of chunks to retrieve.")
    zone_filter: Literal["course", "regulation", "table"] | None = Field(
        default=None, description="Restrict retrieval to one zone type, if known in advance."
    )


class RetrievedChunkResponse(BaseModel):
    """
    The subset of a retrieved chunk worth surfacing to a client — see
    module docstring's note on why this isn't just RetrievedChunk's
    raw metadata dict passed through unfiltered.
    """

    text: str
    score: float
    zone_type: str | None = None
    label: str | None = None  # e.g. "SWE145" or "Article 9" — resolved for display, not for re-parsing


class QueryResponse(BaseModel):
    """
    Mirrors guardrails.schemas.GuardrailReport field-for-field (see
    module docstring) — this is the exact shape Pipeline.run() returns,
    reshaped into the client-facing chunk representation above.
    """

    answer: str
    sources: list[str] = Field(default_factory=list)
    confidence: Literal["high", "medium", "low"]
    retrieved_chunks: list[RetrievedChunkResponse] = Field(default_factory=list)
    parse_error: bool = False
    citation_warnings: list[str] = Field(default_factory=list)
    citations_valid: bool = True


class HealthResponse(BaseModel):
    status: Literal["ok"]
