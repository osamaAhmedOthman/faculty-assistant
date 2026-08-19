"""
Request/Response schemas for the FastAPI layer (Pydantic contract).

Architecture & Design Notes:
- Contract Isolation: Defines public-facing HTTP data shapes without importing internal pipeline types 
  (e.g., mirrors `GuardrailReport`), preventing internal refactors from accidentally breaking the public API.
- Clean Payload Design: Filters out internal metadata dicts from `RetrievedChunkResponse`, exposing only 
  essential consumer fields (`zone_type`, resolved labels) for a cleaner UI/API interface.
- Flexible Querying: Configures `top_k` and `zone_filter` as optional parameters on requests, matching 
  `Pipeline.run()` defaults for seamless client execution.
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
