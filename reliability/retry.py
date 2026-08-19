"""
Retry policy wrapper for transient external-dependency failures.

Architecture & Design Notes:
- Production-Grade Library Integration: Leverages `tenacity` for exponential backoff and randomized jitter, 
  preventing synchronized retry storms across concurrent callers while eliminating hand-rolled sleep loops.
- Targeted Transient Exception Filtering: Restricts retries strictly to network, timeout, and rate-limit errors, 
  failing fast on programmatic bugs (`ValueError`) to prevent masking errors behind artificial delays.
- Defensive SDK Exception Wiring: Dynamically imports Groq-specific error types (`RateLimitError`, `APIConnectionError`, 
  `APITimeoutError`) so real vendor SDK exceptions are reliably caught and retried.
"""
import logging

from tenacity import (
    Retrying,
    stop_after_attempt,
    wait_exponential_jitter,
    retry_if_exception_type,
    before_sleep_log,
)

logger = logging.getLogger(__name__)

# Base retryable set: standard transient-failure shapes.
RETRYABLE_EXCEPTIONS: tuple = (TimeoutError, ConnectionError)

# Extend with Groq's actual exception types when available, so this
# policy matches what a real Groq call can actually raise (confirmed
# against this project's own RateLimitError traceback from the RAGAS
# runs) rather than only generic built-in exception types that a real
# API client may never raise.
try:
    from groq import (
        RateLimitError as _GroqRateLimitError,
        APIConnectionError as _GroqAPIConnectionError,
        APITimeoutError as _GroqAPITimeoutError,
    )

    RETRYABLE_EXCEPTIONS = RETRYABLE_EXCEPTIONS + (
        _GroqRateLimitError,
        _GroqAPIConnectionError,
        _GroqAPITimeoutError,
    )
except ImportError:
    # groq not installed in this environment (e.g. a test sandbox that
    # only needs retry.py's own logic) — fall back to the base set
    # rather than failing to import this module at all.
    pass


def call_with_retry(
    fn,
    *args,
    max_attempts: int = 3,
    min_wait: float = 1.0,
    max_wait: float = 20.0,
    **kwargs,
):
    """
    Call fn(*args, **kwargs), retrying on RETRYABLE_EXCEPTIONS with
    exponential backoff + jitter between attempts.

    A plain function (not a decorator) so call sites (pipeline.py) can
    pass runtime-determined arguments straight through without needing
    a pre-decorated function object — and so tests can override
    min_wait/max_wait to keep retry tests fast without real multi-
    second sleeps (see tests/test_reliability.py).

    Re-raises the final exception unchanged after max_attempts is
    exhausted — the caller (pipeline.py) decides what a total failure
    means for the response shape it returns, not this file.
    """
    retrying = Retrying(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential_jitter(initial=min_wait, max=max_wait),
        retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
        reraise=True,
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    return retrying(fn, *args, **kwargs)
