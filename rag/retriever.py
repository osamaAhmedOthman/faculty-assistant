"""
retriever.py — retrieval logic

Responsibility: Fetch top-k relevant chunks from Pinecone for a user query. 
This file handles query embedding, Pinecone retrieval, filtering, and 
formatting context for rag/generator.py.

Design notes:
- STATEFUL CLASS: Keeps persistent embedding and Pinecone client instances 
  in memory to prevent re-initialization overhead across queries.
- EMBEDDING CONSISTENCY: Uses the same model defined in core/config.py to 
  ensure query embeddings match index vector embeddings.
- SCOPED FILTERING: Accepts optional metadata filters (e.g., program="SWE") 
  without deciding query intent, leaving orchestration to pipeline.py.
- CLEAN OUTPUTS: Returns sanitized dictionaries (text, score, metadata) 
  instead of raw Pinecone response objects.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.config import RETRIEVAL_TOP_K
from clients.embeddings import EmbeddingClient
from clients.pinecone_client import PineconeClient


class Retriever:
    def __init__(self, embedding_client: EmbeddingClient | None = None, pinecone_client: PineconeClient | None = None):
        # Allow injecting stub/mock clients for testing (see
        # tests/test_retrieval.py) without needing a real model
        # download or a real Pinecone connection — same pattern used
        # to test embed.py and upload.py earlier in this project.
        self.embedding_client = embedding_client or EmbeddingClient()
        self.pinecone_client = pinecone_client or PineconeClient()

    def retrieve(
        self,
        query: str,
        top_k: int = RETRIEVAL_TOP_K,
        program_filter: str | None = None,
        zone_filter: str | None = None,
    ) -> list[dict]:
        """
        Embed the query and return the top_k most relevant chunks.

        program_filter: restrict to one program's chunks, e.g. "SWE".
        zone_filter: restrict to one zone type, e.g. "course" or
        "regulation" — useful for query types where you already know
        the answer lives in one zone (a GPA-calculation question
        should only ever search regulation chunks, not course
        descriptions).

        Returns a list of dicts:
            {"text": str, "score": float, "metadata": dict}
        ordered by relevance (highest score first — Pinecone already
        returns results in that order for cosine similarity).
        """
        if not query or not query.strip():
            raise ValueError("retrieve() called with an empty query")

        query_vector = self.embedding_client.embed_query(query)

        pinecone_filter = self._build_filter(program_filter, zone_filter)
        matches = self.pinecone_client.query(
            vector=query_vector,
            top_k=top_k,
            filter=pinecone_filter,
        )

        results = []
        for match in matches:
            metadata = match["metadata"]
            results.append(
                {
                    "text": metadata.get("text", ""),
                    "score": match["score"],
                    "metadata": metadata,
                }
            )
        return results

    @staticmethod
    def _build_filter(program_filter: str | None, zone_filter: str | None) -> dict | None:
        """
        Build a Pinecone metadata filter from the given constraints.
        Returns None (no filter) if neither is set, rather than an
        empty dict — Pinecone's client treats an empty filter dict
        differently in some SDK versions, so being explicit about
        "no filter at all" avoids a subtle version-dependent bug.
        """
        conditions = {}
        if program_filter:
            conditions["program"] = {"$eq": program_filter}
        if zone_filter:
            conditions["zone_type"] = {"$eq": zone_filter}

        if not conditions:
            return None
        return conditions


if __name__ == "__main__":
    # Manual smoke test — run this file directly with a query string
    # to sanity-check retrieval against the real Pinecone index before
    # trusting it inside the full pipeline.
    if len(sys.argv) < 2:
        print('Usage: python rag/retriever.py "your question here" [program_filter]')
        sys.exit(1)

    query = sys.argv[1]
    program = sys.argv[2] if len(sys.argv) > 2 else None

    retriever = Retriever()
    results = retriever.retrieve(query, program_filter=program)

    print(f"Query: {query!r}")
    print(f"Program filter: {program or 'none'}")
    print(f"\nTop {len(results)} results:\n")
    for i, r in enumerate(results, 1):
        print(f"--- Result {i} (score: {r['score']:.4f}) ---")
        print(f"program: {r['metadata'].get('program')}, zone: {r['metadata'].get('zone_type')}")
        print("METADATA:", r["metadata"])
        print(r["text"][:200])
        print()




