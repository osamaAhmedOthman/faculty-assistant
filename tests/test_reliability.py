"""
tests/test_reliability.py — unit tests for reliability/retry.py and
reliability/circuit_breaker.py

Responsibility: exercise retry and circuit-breaker logic in isolation,
using fake functions that raise on a schedule rather than a real Groq
call. WAIT TIMES ARE DELIBERATELY TINY (min_wait/max_wait in the
0.001-0.05s range, recovery_timeout in the 0.05-0.1s range) — real
retry/backoff timing (1-20s) would make this file slow to run
repeatedly during development; what's being tested is the LOGIC
(does it retry the right number of times, does the breaker open/close
at the right moments), not the actual production timing constants,
which are just arguments the logic happens to be parameterized by.
"""

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest

from reliability.retry import call_with_retry
from reliability.circuit_breaker import CircuitBreaker, CircuitBreakerOpenError
from reliability.fallback import build_fallback_result, FALLBACK_ANSWER


# ---------------------------------------------------------------------------
# retry.py
# ---------------------------------------------------------------------------

class FlakyFunction:
    """Raises a given exception type for the first `fail_times` calls,
    then returns `return_value` on every call after that. Tracks call
    count so tests can assert exactly how many attempts were made."""

    def __init__(self, exception_type, fail_times: int, return_value="ok"):
        self.exception_type = exception_type
        self.fail_times = fail_times
        self.return_value = return_value
        self.call_count = 0

    def __call__(self, *args, **kwargs):
        self.call_count += 1
        if self.call_count <= self.fail_times:
            raise self.exception_type("simulated transient failure")
        return self.return_value


def test_retry_succeeds_after_transient_failures():
    fn = FlakyFunction(TimeoutError, fail_times=2)
    result = call_with_retry(fn, max_attempts=3, min_wait=0.001, max_wait=0.01)
    assert result == "ok"
    assert fn.call_count == 3  # 2 failures + 1 success


def test_retry_gives_up_after_max_attempts_and_reraises():
    fn = FlakyFunction(TimeoutError, fail_times=10)  # never succeeds
    with pytest.raises(TimeoutError):
        call_with_retry(fn, max_attempts=3, min_wait=0.001, max_wait=0.01)
    assert fn.call_count == 3  # exactly max_attempts, not more


def test_non_retryable_exception_fails_immediately_no_retry():
    """A ValueError (e.g. Retriever's empty-query guard) is not in
    RETRYABLE_EXCEPTIONS — it must propagate on the FIRST call, not
    be retried and masked behind a delay before failing anyway."""
    fn = FlakyFunction(ValueError, fail_times=10)
    with pytest.raises(ValueError):
        call_with_retry(fn, max_attempts=3, min_wait=0.001, max_wait=0.01)
    assert fn.call_count == 1  # no retries attempted at all


def test_retry_passes_args_and_kwargs_through():
    received = {}

    def fn(a, b, keyword=None):
        received["a"] = a
        received["b"] = b
        received["keyword"] = keyword
        return "done"

    result = call_with_retry(fn, "x", "y", keyword="z", max_attempts=1, min_wait=0.001, max_wait=0.01)
    assert result == "done"
    assert received == {"a": "x", "b": "y", "keyword": "z"}


def test_retry_succeeds_first_try_no_failures():
    fn = FlakyFunction(TimeoutError, fail_times=0)
    result = call_with_retry(fn, max_attempts=3, min_wait=0.001, max_wait=0.01)
    assert result == "ok"
    assert fn.call_count == 1


# ---------------------------------------------------------------------------
# circuit_breaker.py
# ---------------------------------------------------------------------------

def test_breaker_starts_closed():
    breaker = CircuitBreaker(name="test", failure_threshold=3, recovery_timeout=0.05)
    assert breaker.state == CircuitBreaker.CLOSED


def test_breaker_stays_closed_below_failure_threshold():
    breaker = CircuitBreaker(name="test", failure_threshold=3, recovery_timeout=0.05)

    def failing_fn():
        raise TimeoutError("boom")

    for _ in range(2):  # threshold is 3 — 2 failures should not trip it
        with pytest.raises(TimeoutError):
            breaker.call(failing_fn)
    assert breaker.state == CircuitBreaker.CLOSED


def test_breaker_opens_at_failure_threshold():
    breaker = CircuitBreaker(name="test", failure_threshold=3, recovery_timeout=0.05)

    def failing_fn():
        raise TimeoutError("boom")

    for _ in range(3):
        with pytest.raises(TimeoutError):
            breaker.call(failing_fn)
    assert breaker.state == CircuitBreaker.OPEN


def test_breaker_open_fails_fast_without_calling_fn():
    breaker = CircuitBreaker(name="test", failure_threshold=1, recovery_timeout=10.0)  # long timeout — won't recover during this test

    def failing_fn():
        raise TimeoutError("boom")

    with pytest.raises(TimeoutError):
        breaker.call(failing_fn)
    assert breaker.state == CircuitBreaker.OPEN

    call_count = {"n": 0}

    def should_not_be_called():
        call_count["n"] += 1
        return "should never get here"

    with pytest.raises(CircuitBreakerOpenError):
        breaker.call(should_not_be_called)
    assert call_count["n"] == 0  # the dependency was never actually touched


def test_breaker_transitions_to_half_open_after_recovery_timeout():
    breaker = CircuitBreaker(name="test", failure_threshold=1, recovery_timeout=0.05)

    def failing_fn():
        raise TimeoutError("boom")

    with pytest.raises(TimeoutError):
        breaker.call(failing_fn)
    assert breaker.state == CircuitBreaker.OPEN

    time.sleep(0.07)  # exceed recovery_timeout
    assert breaker.state == CircuitBreaker.HALF_OPEN


def test_breaker_successful_probe_closes_breaker_and_resets_count():
    breaker = CircuitBreaker(name="test", failure_threshold=1, recovery_timeout=0.05)

    def failing_fn():
        raise TimeoutError("boom")

    with pytest.raises(TimeoutError):
        breaker.call(failing_fn)
    time.sleep(0.07)
    assert breaker.state == CircuitBreaker.HALF_OPEN

    result = breaker.call(lambda: "recovered")
    assert result == "recovered"
    assert breaker.state == CircuitBreaker.CLOSED


def test_breaker_failed_probe_reopens_immediately_not_after_threshold_again():
    """A failed HALF_OPEN probe should reopen the breaker right away —
    not require accumulating failure_threshold failures again."""
    breaker = CircuitBreaker(name="test", failure_threshold=5, recovery_timeout=0.05)

    def failing_fn():
        raise TimeoutError("boom")

    for _ in range(5):
        with pytest.raises(TimeoutError):
            breaker.call(failing_fn)
    assert breaker.state == CircuitBreaker.OPEN

    time.sleep(0.07)
    assert breaker.state == CircuitBreaker.HALF_OPEN

    with pytest.raises(TimeoutError):
        breaker.call(failing_fn)  # the probe itself fails
    assert breaker.state == CircuitBreaker.OPEN  # reopened after just ONE failed probe


def test_breaker_reset_forces_closed():
    breaker = CircuitBreaker(name="test", failure_threshold=1, recovery_timeout=10.0)

    def failing_fn():
        raise TimeoutError("boom")

    with pytest.raises(TimeoutError):
        breaker.call(failing_fn)
    assert breaker.state == CircuitBreaker.OPEN

    breaker.reset()
    assert breaker.state == CircuitBreaker.CLOSED


def test_breaker_call_returns_fn_result_on_success():
    breaker = CircuitBreaker(name="test", failure_threshold=3, recovery_timeout=0.05)
    result = breaker.call(lambda x: x * 2, 21)
    assert result == 42


# ---------------------------------------------------------------------------
# fallback.py
# ---------------------------------------------------------------------------

def test_build_fallback_result_shape_matches_generator_result():
    """Must match the same shape GeneratorResult expects (see
    guardrails/schemas.py) so _validate_output's citation guardrail can
    run on a fallback result unmodified — no special-casing needed."""
    result = build_fallback_result("some internal reason")
    assert set(result.keys()) == {"answer", "sources", "confidence", "retrieved_chunks"}
    assert result["answer"] == FALLBACK_ANSWER
    assert result["sources"] == []
    assert result["confidence"] == "low"
    assert result["retrieved_chunks"] == []


def test_build_fallback_result_never_leaks_reason_into_answer():
    """The internal `reason` string (an exception repr, a breaker-state
    message) must never appear in the user-facing answer field — it's
    for logging only."""
    secret_detail = "APIConnectionError: connection reset by peer at 10.0.0.5:443"
    result = build_fallback_result(secret_detail)
    assert secret_detail not in result["answer"]


def test_build_fallback_result_passes_schema_validation():
    """The fallback result should validate cleanly against
    guardrails.schemas.GeneratorResult — proves the shape contract
    actually holds, not just that this test's own assumptions about
    the shape are self-consistent."""
    from guardrails.schemas import GeneratorResult

    result = build_fallback_result("reason")
    validated = GeneratorResult.model_validate(result)
    assert validated.confidence == "low"


def test_build_fallback_result_passes_citation_guardrail_unflagged():
    """End-to-end proof the fallback path and the citation guardrail
    compose correctly: empty sources against empty retrieved_chunks is
    a match, not a hallucination — a fallback response should never
    itself trip the guardrail."""
    from guardrails.validators import verify_citations

    result = build_fallback_result("reason")
    report = verify_citations(result)
    assert report["citations_valid"] is True
    assert report["citation_warnings"] == []


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
