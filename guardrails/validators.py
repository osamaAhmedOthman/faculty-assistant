"""
guardrails/validators.py — business-rule / semantic guardrails

Responsibility: catch cases where a Generator result is *structurally*
fine (valid JSON, right types — schemas.py's job) but not
*trustworthy*. The first and highest-value check here: the model
citing a source that was never actually retrieved (hallucinated
grounding) — the failure mode strict-grounding prompting in system.txt
is meant to prevent, but prompting alone doesn't guarantee.

Design notes:
- FAIL OPEN, NOT CLOSED: an invalid citation downgrades confidence and
  attaches a warning rather than discarding the answer outright. The
  rest of the answer may still be entirely correct — the model may
  have cited one extra course code from pattern-matching alongside
  otherwise-grounded claims. Blanking the whole answer on one bad
  citation throws away real signal; silently stripping the bad
  citation would hide the hallucination from both the eval harness and
  the user. Flagging is the only option that does both: keeps what's
  usable, surfaces what isn't.
- LABEL NORMALIZATION: system.txt asks the model to cite course codes
  and "Article N" exactly as shown in context labels, but LLMs aren't
  perfectly literal about formatting — whitespace/case drift
  ("article 9" vs "Article 9") is expected and shouldn't be reported
  as a hallucination. Comparison is case-insensitive and
  whitespace-normalized.
- TABLE CHUNKS EXCLUDED FROM VALID LABELS: system.txt never asks the
  model to cite a table directly (courses and articles only), so a
  citation shaped like a table reference isn't a legitimate match
  regardless of whether a table chunk was retrieved.
"""

import re

from guardrails.schemas import GeneratorResult, GuardrailReport

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
