"""
Centralized logging configuration module.

Architecture & Design Notes:
- Standard Library-vs-Application Split: Entry points (`api/main.py`, `app.py`) invoke 
  `configure_logging()`, while internal library modules call `logging.getLogger(__name__)` 
  to emit formatted logs cleanly without configuring handlers.
- Idempotent Design: Guards against duplicate handler attachment, allowing safe invocation across 
  multiple test sessions or process entry points without duplicating log output.
- Configurable Runtime Levels: Reads `LOG_LEVEL` from environment variables, defaulting to `INFO` 
  with easy toggling for local debugging.
- Traceable Formatting: Includes `%(name)s` in log formats to pinpoint the originating module 
  (e.g., `reliability.circuit_breaker`) for easy log filtering.
"""

import logging
import os
import sys

_DEFAULT_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def configure_logging(level: str | None = None) -> None:
    """
    Configure the root logger once. Subsequent calls are no-ops (see
    module docstring's IDEMPOTENT note) so this is safe to call
    defensively from more than one entry point.

    level: an explicit override (e.g. "DEBUG"). Falls back to the
    LOG_LEVEL env var, then to "INFO" if neither is set.
    """
    root = logging.getLogger()
    if root.handlers:
        return  # already configured — don't attach a second handler

    resolved_level = (level or os.getenv("LOG_LEVEL", "INFO")).upper()

    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(logging.Formatter(_DEFAULT_FORMAT, datefmt=_DEFAULT_DATE_FORMAT))

    root.setLevel(resolved_level)
    root.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """
    Convenience wrapper so call sites can do
    `from core.logger import get_logger; logger = get_logger(__name__)`
    instead of importing the stdlib `logging` module directly. Purely
    a naming/discoverability convenience — behaves identically to
    `logging.getLogger(name)`, since configuration lives entirely in
    configure_logging(), not here.
    """
    return logging.getLogger(name)
