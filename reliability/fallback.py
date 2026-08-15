"""
reliability/fallback.py — degraded-but-valid response construction

Responsibility: build the result dict Pipeline.run() returns when
generation fails even after retry.py's retries are exhausted and
circuit_breaker.py's breaker won't allow another attempt. Nothing
about WHEN to fall back lives here — that decision is pipeline.py's
(see _call_generator_with_reliability). This file only answers "what
does a valid-but-degraded response look like."

Design notes:
- SEPARATE FILE, NOT INLINE IN PIPELINE.PY: matches this project's own
  established convention — retry.py owns retry, circuit_breaker.py
  owns breaker tracking, each a single reliability concern in its own
  file. Burying fallback construction inside pipeline.py as a private
  method broke that pattern and made the fallback path easy to forget
  to test (which is exactly what happened — see
  tests/test_reliability.py's fallback tests and test_pipeline.py's
  total-failure test, neither of which existed before this file did).
- MATCHES GeneratorResult'S SHAPE ON PURPOSE: confidence="low" and
  empty sources/retrieved_chunks mean guardrails/validators.py's
  citation guardrail runs on a fallback result exactly like it would
  on a real one — no sources to check against no retrieved chunks is
  a match, not a mismatch (see verify_citations' NO_MATCH_ANSWER
  handling) — so pipeline.py never needs a special case in
  _validate_output for "this was a fallback, skip the guardrail."
- REASON IS LOGGED, NEVER SHOWN TO THE USER: `reason` carries internal
  detail (an exception repr, a breaker-state message) that's useful
  for debugging and worth logging, but FALLBACK_ANSWER — not `reason`
  — is what actually goes in the returned `answer` field. An internal
  exception message or breaker-state string isn't something api/ or
  dashboard/ should ever render as-is to an end user.
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
