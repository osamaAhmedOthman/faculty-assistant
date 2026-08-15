"""
reliability/circuit_breaker.py — stop hammering a failing dependency

Responsibility: track consecutive failures of calls to ONE external
dependency, and once a failure threshold is crossed, fail fast (no
network call at all) for a cooldown window rather than continuing to
hit an already-failing service. Nothing about retrying an individual
call lives here — that's retry.py's job. This file only decides
whether to attempt the call at all.

Design notes:
- ONE BREAKER PER DEPENDENCY, NOT PER CALL SITE: matches how clients/
  already isolates each external dependency (GroqClient,
  PineconeClient) behind its own module. "Groq is down" is a fact
  about Groq, not about which function happened to call it — sharing
  one breaker instance across every Groq call site means a failure
  detected via one path (e.g. generation) correctly fails fast on
  every other path too, instead of each call site discovering the
  same outage independently and burning its own failure budget before
  tripping.
- THREE-STATE MACHINE (CLOSED / OPEN / HALF_OPEN), STANDARD PATTERN:
    CLOSED     — normal operation; failures are counted.
    OPEN       — failure_threshold was crossed; calls fail immediately
                 via CircuitBreakerOpenError, without touching the
                 dependency, until recovery_timeout has elapsed.
    HALF_OPEN  — recovery_timeout has elapsed; the NEXT call is let
                 through as a probe. Success closes the breaker and
                 resets the failure count; failure reopens it
                 immediately (not after threshold failures again —
                 one failed probe is enough evidence the dependency is
                 still down).
- THREAD SAFETY: a lock guards state transitions. Not currently load-
  bearing for this project's synchronous single-request pipeline, but
  api/'s FastAPI server will handle concurrent requests, and a breaker
  that isn't thread-safe would be a real bug there — cheaper to build
  correctly now than to retrofit once api/ exists.
"""

import threading
import time


class CircuitBreakerOpenError(Exception):
    """Raised when the breaker is OPEN and recovery_timeout hasn't
    elapsed yet — the caller should treat this as 'dependency
    currently unavailable', not as a normal call failure."""


class CircuitBreaker:
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

    def __init__(self, name: str, failure_threshold: int = 5, recovery_timeout: float = 30.0):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout

        self._state = self.CLOSED
        self._failure_count = 0
        self._opened_at: float | None = None
        self._lock = threading.Lock()

    @property
    def state(self) -> str:
        """
        Read the current state, applying the OPEN -> HALF_OPEN
        transition lazily (on read) rather than via a background
        timer — there's no clock ticking in the background, so the
        transition only needs to be correct at the moments the state
        is actually checked (i.e. right before a call).
        """
        with self._lock:
            if self._state == self.OPEN and self._opened_at is not None:
                if time.monotonic() - self._opened_at >= self.recovery_timeout:
                    self._state = self.HALF_OPEN
            return self._state

    def call(self, fn, *args, **kwargs):
        """
        Call fn(*args, **kwargs) if the breaker allows it, tracking
        the outcome. Raises CircuitBreakerOpenError immediately
        (without calling fn at all) if the breaker is OPEN.
        """
        current_state = self.state  # applies OPEN -> HALF_OPEN transition if due
        if current_state == self.OPEN:
            raise CircuitBreakerOpenError(
                f"Circuit breaker '{self.name}' is open — failing fast without "
                f"calling the dependency. Will allow a probe call again once "
                f"the {self.recovery_timeout}s recovery window has elapsed."
            )

        try:
            result = fn(*args, **kwargs)
        except Exception:
            self._record_failure()
            raise
        else:
            self._record_success()
            return result

    def _record_failure(self) -> None:
        with self._lock:
            self._failure_count += 1
            if self._state == self.HALF_OPEN or self._failure_count >= self.failure_threshold:
                # A failed HALF_OPEN probe reopens immediately — no
                # need to re-accumulate failure_threshold failures
                # again, one failed probe is sufficient evidence.
                self._state = self.OPEN
                self._opened_at = time.monotonic()

    def _record_success(self) -> None:
        with self._lock:
            self._failure_count = 0
            self._state = self.CLOSED
            self._opened_at = None

    def reset(self) -> None:
        """Force the breaker back to CLOSED. Not used by normal
        operation — exists for tests and for manual operator
        intervention (e.g. a dashboard 'reset breaker' action) if
        that's ever built."""
        with self._lock:
            self._state = self.CLOSED
            self._failure_count = 0
            self._opened_at = None
