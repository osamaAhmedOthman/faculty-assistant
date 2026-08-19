"""
Circuit breaker mechanism for dependency failure isolation.

Architecture & Design Notes:
- Per-Dependency Instance: Tracks failure states at the dependency level (e.g., Groq, Pinecone) rather 
  than per call site, ensuring outage discoveries immediately protect all execution paths.
- Three-State Finite State Machine: Implements standard Closed (normal), Open (fail-fast cooldown), 
  and Half-Open (probe recovery) states to manage outage lifecycles deterministically.
- Strict Probe Policy: Re-opens immediately upon a single failed probe in the Half-Open state, 
  preventing recurring retry loops during ongoing downstream outages.
- Concurrency Safe: Uses thread locks around state transitions to safely handle concurrent HTTP requests 
  within asynchronous/multi-threaded application entry points (`FastAPI`).
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
