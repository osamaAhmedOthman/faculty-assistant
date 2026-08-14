"""
generator.py — answer generation

Responsibility: given a user question, retrieve relevant SWE chunks
and generate a grounded answer via the LLM, staying strictly within
what the retrieved context actually supports.

Design notes:
- STRICT GROUNDING: the system prompt (assembled by rag/prompts.py
  from prompts/*.txt) instructs the model to answer only from the
  provided context, and to say so explicitly when the context doesn't
  cover the question, rather than falling back on general knowledge.
  This is a deliberate scope decision, not just phrasing — the
  assistant is a SWE-regulations lookup tool, not a general CS
  chatbot, so it declines out-of-scope questions the same way it
  declines under-evidenced ones.
- RELEVANCE GATE: a fixed similarity-score cutoff decides whether
  retrieval found anything usable, rather than asking the LLM to
  judge relevance itself. This is cheap, deterministic, and gives the
  evaluation harness a fixed number to test against, instead of a
  judgment call baked invisibly into a prompt.
- STRUCTURED OUTPUT: the model returns JSON ({answer, sources,
  confidence}) rather than freeform prose, so api/ and dashboard/
  don't need to parse natural language, and citations are a checkable
  field rather than text an eval script would have to regex out.
- retrieved_chunks is always attached to the result (even on a
  no-match response) so callers — especially evaluation code — can
  inspect what retrieval actually found without a second query.
"""

import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.config import GROQ_MODEL, RETRIEVAL_TOP_K, require_keys
from clients.groq_client import GroqClient
from rag.retriever import Retriever
from rag.prompts import build_system_prompt

# Cosine similarity floor below which a retrieved chunk is treated as
# not actually relevant. Chosen from observed real query scores: SWE
# regulation/course matches score ~0.55-0.75, unrelated queries score
# ~0.15-0.20 against this index — 0.3 sits clearly in the gap between
# the two, not tuned to any single query.
MIN_RELEVANCE_SCORE = 0.3

NO_MATCH_ANSWER = "I don't have information about that in the SWE program regulations."


def format_context(results: list[dict]) -> str:
    """
    Turn retrieved chunks into labeled context blocks for the prompt.
    Each block is tagged with the citation label the model is asked
    to reuse verbatim (course code or Article N), so citations in the
    model's answer are traceable back to a specific retrieved chunk
    rather than invented.
    """
    blocks = []
    for r in results:
        md = r["metadata"]
        zone = md.get("zone_type")
        if zone == "course":
            label = f"[Course {md.get('course_code')}]"
            prereq_names = md.get("prerequisite_names") or []
            extra = f"\nPrerequisite names: {', '.join(prereq_names)}" if prereq_names else ""
        elif zone == "regulation":
            label = f"[Article {md.get('article_num')}]"
            extra = ""
        else:
            label = f"[Table, page {md.get('page_number')}]"
            extra = ""
        blocks.append(f"{label}\n{r['text']}{extra}")
    return "\n\n---\n\n".join(blocks)


class Generator:
    def __init__(self, retriever: Retriever | None = None, groq_client: GroqClient | None = None):
        # Same injectable-client pattern as Retriever, for testing
        # without a real Groq call or Pinecone connection.
        self.retriever = retriever or Retriever()
        self.groq_client = groq_client or GroqClient()

    def answer(
        self,
        query: str,
        top_k: int = RETRIEVAL_TOP_K,
        zone_filter: str | None = None,
    ) -> dict:
        """
        Retrieve context for the query and generate a grounded answer.

        Returns a dict:
            {"answer": str, "sources": list[str], "confidence": str,
             "retrieved_chunks": list[dict]}
        On a parse failure from the model, an additional
        "parse_error": True key is included rather than silently
        discarding the model's raw text.
        """
        results = self.retriever.retrieve(query, top_k=top_k, zone_filter=zone_filter)
        relevant = [r for r in results if r["score"] >= MIN_RELEVANCE_SCORE]

        if not relevant:
            return {
                "answer": NO_MATCH_ANSWER,
                "sources": [],
                "confidence": "low",
                "retrieved_chunks": results,
            }

        context = format_context(relevant)
        user_message = f"CONTEXT:\n{context}\n\nQUESTION: {query}"

        zone_types = {r["metadata"].get("zone_type") for r in relevant}
        system_prompt = build_system_prompt(zone_types)

        raw_response = self.groq_client.chat(system=system_prompt, user=user_message)
        parsed = self._parse_response(raw_response)
        parsed["retrieved_chunks"] = relevant
        return parsed

    @staticmethod
    def _parse_response(raw: str) -> dict:
        """
        Parse the model's JSON reply. Models occasionally wrap JSON in
        prose or code fences despite instructions — fail toward
        surfacing the raw text with parse_error=True rather than
        silently dropping the answer, so this is visible in testing
        and evaluation rather than masked as an empty response.
        """
        try:
            data = json.loads(raw)
            return {
                "answer": str(data.get("answer", "")).strip(),
                "sources": data.get("sources", []),
                "confidence": data.get("confidence", "low"),
            }
        except (json.JSONDecodeError, TypeError):
            return {
                "answer": raw.strip() if isinstance(raw, str) else "",
                "sources": [],
                "confidence": "low",
                "parse_error": True,
            }


if __name__ == "__main__":
    # Manual smoke test — run this file directly with a query string
    # to sanity-check end-to-end generation against the real index and
    # a real Groq call before trusting it inside api/ or dashboard/.
    if len(sys.argv) < 2:
        print('Usage: python -m rag.generator "your question here"')
        sys.exit(1)

    require_keys("GROQ_API_KEY")

    query = sys.argv[1]
    generator = Generator()
    result = generator.answer(query)

    print(f"Query: {query!r}\n")
    print(f"Answer: {result['answer']}")
    print(f"Sources: {result['sources']}")
    print(f"Confidence: {result['confidence']}")
    if result.get("parse_error"):
        print("\nWARNING: model response was not valid JSON — showing raw text above.")
    if result["sources"] or result["answer"] != NO_MATCH_ANSWER:
        print(f"\nRetrieved {len(result['retrieved_chunks'])} chunk(s) above the relevance threshold ({MIN_RELEVANCE_SCORE}).")
    else:
        print(f"\nNo chunks reached the relevance threshold ({MIN_RELEVANCE_SCORE}) — "
              f"showing all {len(result['retrieved_chunks'])} raw retrieved chunk(s) for reference.")
