"""
prompts.py — prompt template loading

Responsibility: load the .txt templates in prompts/ and assemble the
system prompt actually sent to the LLM. Nothing else — no retrieval,
no LLM call, no context formatting (that stays in generator.py).

Design notes:
- FILES, NOT PYTHON STRINGS: prompt wording lives in prompts/*.txt so
  it can be edited, diffed, and referenced by evaluation/ without
  touching generator.py's logic.
- ZONE-CONDITIONAL ASSEMBLY: regulations.txt and subjects.txt are only
  appended when the retrieved context actually contains that zone
  type, so the model isn't given instructions about course-citation
  formatting on a purely regulation-only query, and vice versa.
- LOADED ONCE, CACHED: template files are read from disk once and
  reused across calls — they don't change during a process's
  lifetime, so re-reading them per-query would be wasted I/O.
"""

from functools import lru_cache
from pathlib import Path

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


@lru_cache(maxsize=None)
def _load_template(name: str) -> str:
    path = PROMPTS_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Prompt template not found: {path}")
    return path.read_text(encoding="utf-8").strip()


def build_system_prompt(zone_types: set[str]) -> str:
    """
    Assemble the system prompt for this query: system.txt always,
    plus regulations.txt and/or subjects.txt when the retrieved
    context actually includes chunks from that zone.

    zone_types: the set of zone_type values present among the chunks
    that passed the relevance gate (e.g. {"course"}, {"regulation",
    "table"}, {"course", "regulation"}).
    """
    sections = [_load_template("system.txt")]

    if "regulation" in zone_types or "table" in zone_types:
        sections.append(_load_template("regulations.txt"))
    if "course" in zone_types:
        sections.append(_load_template("subjects.txt"))

    return "\n\n".join(sections)
