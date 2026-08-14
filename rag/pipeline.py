"""
pipeline.py — single orchestration entry point

Responsibility: the one place api/ and dashboard/ call into. Wires
retriever -> generator (which internally builds prompts and calls
Groq) -> guardrails -> retry/fallback, and returns one structured
response shape. No retrieval, prompt, or LLM logic lives here — this
file only sequences calls to the modules that already own that logic.

Design notes:
- SINGLE ENTRY POINT: api/routes.py and dashboard/app.py should both
  call Pipeline.run(), never Retriever or Generator directly. This
  keeps guardrail/retry behavior consistent across every caller
  instead of each caller having to remember to apply it.
- GUARDRAILS AND RELIABILITY ARE NOT YET IMPLEMENTED. The hooks below
  are structured as no-op pass-throughs so this file's shape matches
  the intended final architecture now, rather than being rewritten
  once guardrails/ and reliability/ exist. Each hook is marked with
  what it will do once implemented. Building the wiring now avoids
  discovering integration problems only after guardrails/reliability
  are written in isolation.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.config import RETRIEVAL_TOP_K
from rag.generator import Generator
from guardrails.validators import verify_citations


class Pipeline:
    def __init__(self, generator: Generator | None = None):
        self.generator = generator or Generator()

    def run(self, query: str, top_k: int = RETRIEVAL_TOP_K, zone_filter: str | None = None) -> dict:
        """
        Run the full query -> answer flow and return one structured
        response. This is the shape api/ and dashboard/ should depend
        on; it's stable even as guardrails/reliability are filled in
        below, since those stages currently pass their input through
        unchanged.
        """
        # TODO(guardrails): validate/sanitize `query` before it reaches
        # retrieval — reject prompt-injection attempts, flag abusive
        # input. Currently a no-op pass-through.
        validated_query = self._validate_input(query)

        # TODO(reliability): wrap this call with retry.py's retry
        # policy (transient Groq/Pinecone failures) and
        # circuit_breaker.py's breaker (stop hammering a failing
        # dependency). Currently a direct, unprotected call.
        result = self.generator.answer(validated_query, top_k=top_k, zone_filter=zone_filter)


        result = self._validate_output(result)

        return result

    @staticmethod
    def _validate_input(query: str) -> str:
        """Placeholder for guardrails/validators.py input checks."""
        return query

    @staticmethod
    def _validate_output(result: dict) -> dict:
        """
        Run the citation-verification guardrail: catches the model
        citing a course code or article that was never actually
        retrieved. Does not drop the answer or the bad citation on a
        failure — see guardrails/validators.py's module docstring for
        why flag-and-downgrade is the chosen behavior over discard-
        or-silently-strip. Input-side guardrails (prompt-injection /
        off-topic rejection before retrieval) are still a TODO in
        _validate_input above.
        """
        return verify_citations(result)


if __name__ == "__main__":
    # Manual smoke test — same interface api/ will eventually use.
    if len(sys.argv) < 2:
        print('Usage: python -m rag.pipeline "your question here"')
        sys.exit(1)

    query = sys.argv[1]
    pipeline = Pipeline()
    result = pipeline.run(query)

    print(f"Query: {query!r}\n")
    print(f"Answer: {result['answer']}")
    print(f"Sources: {result['sources']}")
    print(f"Confidence: {result['confidence']}")
