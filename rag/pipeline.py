"""
Orchestration pipeline for end-to-end query processing.

Architecture & Design Notes:
- Unified Facade: Serves as the single orchestration entry point (`Pipeline.run()`) for HTTP endpoints (`api/`) 
  and UI dashboards (`dashboard/`), ensuring identical retrieval, generation, guardrail, and retry logic across all callers.
- Multi-Layered Reliability: Wraps LLM calls with exponential backoff retries for transient failures and a circuit 
  breaker for sustained provider outages, preventing repeated timeout delays during downtime.
- Dual-Phase Guardrails: Runs pre-retrieval input validation (fail-closed) to block empty queries and prompt injections early, 
  and post-generation output validation (fail-open) to flag hallucinated or ungrounded citations.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.config import RETRIEVAL_TOP_K
from rag.generator import Generator
from guardrails.validators import verify_citations, validate_input, build_blocked_input_result
from reliability.retry import call_with_retry
from reliability.circuit_breaker import CircuitBreaker, CircuitBreakerOpenError
from reliability.fallback import build_fallback_result

# Shared across all Pipeline instances by default: "Groq is down" is a
# fact about Groq, not about which Pipeline object happens to be
# calling it — see circuit_breaker.py's module docstring for why this
# is one instance per dependency rather than one per call site.
_GROQ_CIRCUIT_BREAKER = CircuitBreaker(name="groq", failure_threshold=5, recovery_timeout=30.0)


class Pipeline:
    def __init__(
        self,
        generator: Generator | None = None,
        circuit_breaker: CircuitBreaker | None = None,
        max_retry_attempts: int = 3,
    ):
        self.generator = generator or Generator()
        # Injectable for testing (a fresh breaker per test avoids one
        # test's failures tripping another test's breaker state via
        # the shared default instance) and so a caller could someday
        # run multiple independently-tracked breakers if that ever
        # becomes useful.
        self.circuit_breaker = circuit_breaker or _GROQ_CIRCUIT_BREAKER
        self.max_retry_attempts = max_retry_attempts

    def run(self, query: str, top_k: int = RETRIEVAL_TOP_K, zone_filter: str | None = None) -> dict:
        """
        Run the full query -> answer flow and return one structured
        response. This is the shape api/ and dashboard/ should depend
        on; it stays stable even when generation fails entirely (see
        _call_generator_with_reliability's fallback) or when the input
        guardrail blocks the query before retrieval ever runs (see
        _validate_input) — both paths converge back through
        _validate_output so callers only ever see one response shape.
        """
        validation = self._validate_input(query)

        if validation.blocked:
            # Blocked queries skip retrieval AND generation entirely —
            # the whole point of an input-side guardrail is avoiding
            # the cost (and risk) of a call that was never going to be
            # answered, not just flagging it after the fact.
            result = build_blocked_input_result(validation.reason)
        else:
            result = self._call_generator_with_reliability(validation.query, top_k=top_k, zone_filter=zone_filter)

        result = self._validate_output(result)

        return result

    def _call_generator_with_reliability(self, query: str, top_k: int, zone_filter: str | None) -> dict:
        """
        Call self.generator.answer() through the circuit breaker, with
        retry attempted INSIDE each breaker-tracked call — so a
        transient blip that recovers within max_retry_attempts never
        counts as a breaker failure at all, and only a call that fails
        even after retries counts as the ONE failure the breaker
        tracks. This ordering matters: retrying outside the breaker
        would let each individual retry attempt count separately
        toward the failure threshold, tripping the breaker after far
        fewer real outages than failure_threshold actually describes.

        On CircuitBreakerOpenError (sustained outage, failing fast) or
        on final exhaustion of retries (a real, non-transient failure,
        or a transient one that outlasted the retry budget), returns a
        fallback result matching the normal response shape rather than
        raising — api/ and dashboard/ should never have to catch a
        pipeline-internal exception type to handle "the LLM is down";
        a low-confidence fallback answer is a valid response for them
        to render as-is.
        """
        try:
            return self.circuit_breaker.call(
                call_with_retry,
                self.generator.answer,
                query,
                top_k=top_k,
                zone_filter=zone_filter,
                max_attempts=self.max_retry_attempts,
            )
        except CircuitBreakerOpenError:
            return build_fallback_result(
                "circuit breaker open — the Groq dependency has failed repeatedly "
                "and is being given a cooldown window before the next attempt."
            )
        except Exception as exc:
            return build_fallback_result(f"generation failed after retries: {exc!r}")

    @staticmethod
    def _validate_input(query: str):
        """Runs guardrails/validators.py's input-side checks (emptiness,
        length, known injection patterns) — see validate_input's own
        docstring for the fail-closed policy and its reasoning."""
        return validate_input(query)

    @staticmethod
    def _validate_output(result: dict) -> dict:
        """
        Run the citation-verification guardrail: catches the model
        citing a course code or article that was never actually
        retrieved. Does not drop the answer or the bad citation on a
        failure — see guardrails/validators.py's module docstring for
        why flag-and-downgrade is the chosen behavior over discard-
        or-silently-strip.
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
