"""
reliability/retry.py — retry policy for transient external-dependency
failures

Responsibility: retry a call to an external dependency (Groq,
Pinecone) with exponential backoff and jitter, ONLY for exceptions
that are actually transient. Nothing about circuit-breaking or what
happens after retries are exhausted lives here — that's
circuit_breaker.py and pipeline.py's job respectively. This file only
answers "should this specific failure be retried, and how long should
we wait before trying again."

Design notes:
- BUILT ON TENACITY, NOT HAND-ROLLED: tenacity is already in this
  project's requirements.txt. Writing a bespoke retry loop (a while
  loop with time.sleep and a counter) would be reinventing something
  tenacity already does correctly — including the harder-to-get-right
  parts like jitter (avoiding synchronized retry storms across
  multiple callers) and clean separation between "should retry" and
  "how long to wait." Demonstrating correct use of an existing
  reliability library is a stronger signal for this project's stated
  goals than a from-scratch implementation would be.
- NARROW RETRYABLE EXCEPTION SET, DELIBERATELY: only transient/
  network/rate-limit-shaped exceptions are retried. A ValueError from
  Retriever.retrieve() being called with an empty query, or a genuine
  bug elsewhere, must fail immediately and loudly — retrying those
  would silently mask a real problem behind a few seconds of delay
  before it fails anyway, which is strictly worse for debugging than
  an immediate failure.
- GROQ EXCEPTION TYPES IMPORTED DEFENSIVELY: the retryable set
  includes Groq's own exception types (RateLimitError,
  APIConnectionError, APITimeoutError) when the groq package is
  installed, so real API failures like the RateLimitError seen during
  this project's own RAGAS runs are actually caught by this policy —
  not just generic TimeoutError/ConnectionError, which wouldn't have
  matched Groq's actual exception types.
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
