"""
core/logger.py — centralized logging configuration

Responsibility: configure Python's logging ONCE, with one consistent
format/level, so every other module (fallback.py, validators.py,
retry.py, and eventually api/ and dashboard/) can just call
`logging.getLogger(__name__)` and get correctly-formatted output
without each file reinventing basicConfig() or — worse — some files
configuring it and others silently relying on the unconfigured root
logger's default (which prints WARNING+ only, with no timestamp or
module name, to stderr).

Design notes:
- CALLED ONCE, AT PROCESS ENTRY: configure_logging() should be called
  exactly once, from each process's actual entry point (api/main.py,
  dashboard/app.py, or a script's own __main__ block) — NOT imported
  and called from inside library modules like fallback.py or
  validators.py. Those files already do the right thing:
  `logger = logging.getLogger(__name__)` at import time, with no
  handler/level configuration of their own. That's the standard
  library-vs-application split: library code gets a logger and emits
  through it; only the application's entry point decides how those
  records are actually formatted and where they go.
- IDEMPOTENT: safe to call more than once (e.g. once from pipeline.py's
  own __main__ smoke test AND once from api/main.py during the same
  test session) without duplicating handlers and therefore duplicating
  every log line. Guards on whether the root logger already has
  handlers attached.
- LEVEL VIA ENV VAR: LOG_LEVEL follows the same "config lives in env
  vars with a sane default" convention as core/config.py — no need to
  edit code to turn on DEBUG logging locally vs. INFO in a deployed
  container.
- FORMAT INCLUDES MODULE NAME: `%(name)s` is the logger name passed to
  getLogger(__name__), so a log line is traceable to the exact module
  that emitted it (e.g. "reliability.circuit_breaker") without needing
  to grep for a message string.
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
