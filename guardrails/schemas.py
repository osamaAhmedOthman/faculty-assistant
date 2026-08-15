"""
guardrails/schemas.py — Pydantic output-validation models

Responsibility: define the structural contract for what a Generator
result and a guardrail-checked result look like. Validation here is
purely about SHAPE — types, required fields — not business rules
(citation-vs-retrieval consistency lives in validators.py). Keeping
these separate means a malformed response (wrong types, missing
fields) fails fast and loudly at the schema layer, while a
well-formed-but-untrustworthy response (correct shape, hallucinated
citation) is caught by the semantic checks in validators.py instead.
"""

from typing import Literal

from pydantic import BaseModel, Field


class RetrievedChunk(BaseModel):
    """
    Mirrors the dict shape retriever.py / generator.py already return.
    Deliberately NOT chunk.py's Chunk dataclass — this only needs to
    validate the subset of fields the guardrail layer actually reads
    (text/score/metadata), so it stays a lightweight structural check
    rather than duplicating chunk.py's model or coupling guardrails/
    to ingestion/'s internal representation.
    """

    text: str
    score: float
    metadata: dict = Field(default_factory=dict)


class GeneratorResult(BaseModel):
    """
    The exact dict shape rag/generator.py's Generator.answer() returns.
    Validating incoming results against this at the guardrails boundary
    means a future change to generator.py's output shape fails LOUDLY
    here, instead of validators.py silently no-op'ing on fields that
    no longer exist.
    """

    answer: str
    sources: list[str] = Field(default_factory=list)
    confidence: Literal["high", "medium", "low"]
    retrieved_chunks: list[RetrievedChunk] = Field(default_factory=list)
    parse_error: bool = False


class GuardrailReport(BaseModel):
    """
    Output of the guardrail layer: the original result fields plus
    what the guardrail found. Kept as a SEPARATE model rather than
    mutating GeneratorResult in place, so pipeline.py's return shape
    always makes explicit whether guardrail checks ran, and what they
    found — a caller can distinguish "no warnings because nothing was
    checked" from "no warnings because checks passed" is not possible
    if this were the same model as GeneratorResult.
    """

    answer: str
    sources: list[str]
    confidence: Literal["high", "medium", "low"]
    retrieved_chunks: list[RetrievedChunk]
    parse_error: bool = False
    citation_warnings: list[str] = Field(default_factory=list)
    citations_valid: bool = True


class InputValidationResult(BaseModel):
    """
    Output of the INPUT-side guardrail (validators.validate_input),
    run before retrieval/generation ever happen. Deliberately a
    separate model from GuardrailReport, which validates the OUTPUT
    side — input validation runs before there's any answer, sources,
    or retrieved_chunks to speak of, so reusing GuardrailReport's shape
    here would mean either fabricating those fields or making them
    optional everywhere, weakening the schema for every other caller.

    FAIL CLOSED, NOT OPEN — the opposite policy from the citation
    guardrail: `blocked=True` means the query never reaches retrieval
    or the LLM at all. This asymmetry is deliberate (see
    validators.py's module docstring for the full reasoning) — a
    false-positive block costs the user one rephrase; a missed prompt
    injection could get the model to ignore system.txt entirely.
    """

    query: str
    blocked: bool = False
    reason: str | None = None
