"""
Business-rule and semantic input/output guardrails.

Architecture & Design Notes:
- Asymmetric Safety Policy: Fail-open on output validation (flags bad citations and downgrades 
  confidence without discarding the answer); fail-closed on input validation (blocks suspected prompt injections early).
- Citation Verification: Checks generated citations against retrieved context labels, applying case-insensitive 
  and whitespace normalization to distinguish minor formatting drift from true hallucinations.
- Exclusion Rules: Excludes table chunks from valid citation targets to align strictly with `system.txt` citation requirements.
- Lightweight Input Filtering: Performs heuristic pattern-matching and bounds checks on incoming queries before 
  retrieval, providing a zero-latency defense-in-depth layer against common prompt injections.
"""

import logging
import re

from guardrails.schemas import GeneratorResult, GuardrailReport, InputValidationResult

logger = logging.getLogger(__name__)

_WHITESPACE = re.compile(r"\s+")


def _normalize_label(label: str) -> str:
    return _WHITESPACE.sub(" ", label.strip()).lower()


def extract_valid_citation_labels(retrieved_chunks: list) -> set[str]:
    """
    Build the set of citation labels that legitimately correspond to
    chunks that were actually retrieved, in normalized form. Mirrors
    the exact label format rag/generator.py's format_context() shows
    the model — a course chunk is citable by its course_code, a
    regulation chunk by "Article {article_num}" — since that's the
    format the model is instructed to reuse.
    """
    labels = set()
    for chunk in retrieved_chunks:
        metadata = chunk.metadata
        zone = metadata.get("zone_type")
        if zone == "course" and metadata.get("course_code"):
            labels.add(_normalize_label(metadata["course_code"]))
        elif zone == "regulation" and metadata.get("article_num") is not None:
            labels.add(_normalize_label(f"Article {metadata['article_num']}"))
    return labels


def verify_citations(result: dict) -> dict:
    """
    Check every entry in result["sources"] against the labels actually
    present in result["retrieved_chunks"]. Returns a new dict (does
    not mutate the input) matching GuardrailReport's shape.

    On a parse_error result (generator.py already flagged malformed
    JSON from the model), sources is always empty by construction —
    there is nothing to check, and this passes through with
    citations_valid=True and no warnings, since "no citations given"
    is not itself a citation problem.

    Raises pydantic.ValidationError if `result` doesn't match
    GeneratorResult's shape at all — a schema-level failure, which
    should surface immediately rather than be swallowed here (see
    schemas.py's design note on why this validation is separate).
    """
    validated = GeneratorResult.model_validate(result)

    valid_labels = extract_valid_citation_labels(validated.retrieved_chunks)

    warnings = []
    for source in validated.sources:
        if _normalize_label(source) not in valid_labels:
            warnings.append(
                f"Cited source {source!r} does not match any retrieved chunk "
                f"— possible hallucinated citation."
            )

    confidence = validated.confidence
    if warnings and confidence != "low":
        # Downgrade rather than silently leave as-is — the warning
        # list is the record of WHY, so a caller (or the evaluation
        # harness) can tell "genuinely low-confidence retrieval" apart
        # from "guardrail downgraded this after generation."
        confidence = "low"

    report = GuardrailReport(
        answer=validated.answer,
        sources=validated.sources,
        confidence=confidence,
        retrieved_chunks=validated.retrieved_chunks,
        parse_error=validated.parse_error,
        citation_warnings=warnings,
        citations_valid=not warnings,
    )
    return report.model_dump()


# ---------------------------------------------------------------------------
# Input-side guardrail
# ---------------------------------------------------------------------------

# Chosen generously — this document's real regulation/course questions
# observed in practice are well under a few hundred characters. A few
# thousand is enough room for a legitimately detailed question (e.g.
# quoting a whole paragraph and asking about it) while still blocking
# the kind of oversized input that's either abuse or a cost/latency
# concern for retrieval and generation, not a real question.
MAX_QUERY_LENGTH = 2000

# Known prompt-injection phrasings, compiled case-insensitively. This
# is a starter set covering common, publicly-documented injection
# patterns — not an exhaustive or adversarially-hardened list (see
# module docstring's "INPUT VALIDATION IS HEURISTIC" note). Grouped by
# what the phrase is trying to do, so the list can be extended in the
# same style as new attack patterns are observed.
_INJECTION_PATTERNS = [
    # Trying to override/discard the system prompt or prior instructions
    re.compile(r"ignore (all |any |the )?(previous|above|prior)?\s*instructions", re.IGNORECASE),
    re.compile(r"disregard (the |your )?(system|previous)?\s*(prompt|instructions)", re.IGNORECASE),
    re.compile(r"forget (everything|all)\s*(you (were|are) told|previous instructions)", re.IGNORECASE),
    re.compile(r"new instructions\s*:", re.IGNORECASE),
    re.compile(r"override (your )?instructions", re.IGNORECASE),
    # Trying to make the model role-play out of its defined role.
    # NOTE: deliberately requires "act as IF you..." rather than bare
    # "act as" — "act as a tutor" / "act as a study planner" are
    # common, entirely legitimate phrasings for an ordinary request,
    # and a pattern matching bare "act as" would false-positive on
    # them constantly. "act as if you" is a much stronger, more
    # specific signal of an actual role-override attempt.
    re.compile(r"act as if you\b", re.IGNORECASE),
    re.compile(r"you are now\s", re.IGNORECASE),
    re.compile(r"pretend (that )?you are", re.IGNORECASE),
    re.compile(r"developer mode", re.IGNORECASE),
    # Trying to extract the system prompt itself
    re.compile(r"(reveal|print|show|repeat) (your |the )?(system )?prompt", re.IGNORECASE),
    re.compile(r"what (are|is) your (system )?instructions", re.IGNORECASE),
]


def validate_input(query: str) -> InputValidationResult:
    """
    Cheap, pre-retrieval checks on the raw user query. Returns an
    InputValidationResult with blocked=True and a reason if the query
    should not proceed to retrieval/generation at all — see module
    docstring's "FAIL CLOSED" note for why this policy is the opposite
    of verify_citations' fail-open behavior.

    Order matters: emptiness and length are checked before pattern
    matching since they're cheaper and more certain — no point running
    every injection pattern against a query that's already going to be
    rejected for being empty.
    """
    if not query or not query.strip():
        return InputValidationResult(query=query, blocked=True, reason="empty query")

    if len(query) > MAX_QUERY_LENGTH:
        return InputValidationResult(
            query=query,
            blocked=True,
            reason=f"query exceeds {MAX_QUERY_LENGTH} character limit ({len(query)} chars)",
        )

    for pattern in _INJECTION_PATTERNS:
        if pattern.search(query):
            logger.warning("Blocked input matching injection pattern %r: %.200s", pattern.pattern, query)
            return InputValidationResult(
                query=query,
                blocked=True,
                reason="query matched a known prompt-injection pattern",
            )

    return InputValidationResult(query=query, blocked=False)


BLOCKED_INPUT_ANSWER = (
    "I can only help with questions about the SWE program regulations, "
    "and couldn't process that request. Please rephrase your question."
)


def build_blocked_input_result(reason: str) -> dict:
    """
    Construct a result dict matching GeneratorResult's shape for a
    query that validate_input blocked — same shape reliability/
    fallback.py's build_fallback_result returns, so pipeline.py's
    run() has one uniform response shape regardless of WHY generation
    never happened (input blocked vs. a total Groq outage), and
    _validate_output's citation guardrail runs on it unmodified (empty
    sources against empty retrieved_chunks is a match, not a
    mismatch — nothing to flag).

    `reason` is logged for operator visibility, never shown to the
    user — matches build_fallback_result's same "internal detail is
    not user-facing text" rule, for the same reason: reason strings
    here can include a slice of the raw blocked query itself.
    """
    logger.warning("Blocking input: %s", reason)
    return {
        "answer": BLOCKED_INPUT_ANSWER,
        "sources": [],
        "confidence": "low",
        "retrieved_chunks": [],
    }
