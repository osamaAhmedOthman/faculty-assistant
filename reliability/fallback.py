"""
Degraded-but-valid response construction module.

Architecture & Design Notes:
- Single-Concern Isolation: Decouples fallback payload construction from orchestration (`pipeline.py`), 
  ensuring isolated testability for degraded execution paths.
- Interface Parity: Mirrors the standard `GeneratorResult` schema (`confidence="low"`, empty sources) to allow 
  downstream guardrail checks (`verify_citations`) to process fallback outputs without special-case logic.
- Secure Degradation: Logs specific failure causes (`reason`) internally for debugging while returning a clean, 
  user-safe default message (`FALLBACK_ANSWER`) to prevent leaking internal stack traces or breaker states.
"""

import logging

logger = logging.getLogger(__name__)

FALLBACK_ANSWER = (
    "The assistant is temporarily unable to reach the language model service. "
    "Please try again in a moment."
)


def build_fallback_result(reason: str) -> dict:
    """
    Construct a degraded-but-valid result matching the shape
    Generator.answer() / guardrails.schemas.GeneratorResult expect:
    {answer, sources, confidence, retrieved_chunks}.

    `reason` is logged at WARNING level for operator visibility (e.g.
    "circuit breaker open" vs. "generation failed after retries: ..."
    are very different situations to be debugging) but never appears
    in the returned answer text itself.
    """
    logger.warning("Pipeline falling back to a degraded response: %s", reason)
    return {
        "answer": FALLBACK_ANSWER,
        "sources": [],
        "confidence": "low",
        "retrieved_chunks": [],
    }
