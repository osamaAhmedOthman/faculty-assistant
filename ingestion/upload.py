"""
upload.py — final ingestion stage: push embedded chunks to Pinecone

Responsibility: Take embedded chunks from data/processed/chunks.json 
and upsert them into Pinecone as queryable vectors.

Design notes:
- METADATA SANITIZATION: Ensures all metadata values match Pinecone's 
  supported types (string, number, boolean, or list of strings). This 
  flattens complex structures like table rows so upserts won't fail.
- TEXT IN METADATA: Copies the raw chunk text into the vector's metadata 
  so it can be retrieved alongside search results.
- IDEMPOTENT UPSERTS: Uses chunk_id as the Pinecone vector ID. Re-running 
  upload.py overwrites matching existing vectors instead of duplicating them.
"""

import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.config import DATA_PROCESSED_DIR
from clients.pinecone_client import PineconeClient

# Pinecone metadata size limit is 40KB per vector. Course descriptions
# are the longest text we have and are nowhere near this, but we guard
# for it anyway rather than assuming — a silently-truncated or rejected
# upsert would be a nasty thing to debug later.
MAX_METADATA_TEXT_CHARS = 35_000  # conservative margin under Pinecone's 40KB cap


def sanitize_metadata_value(value):
    """
    Coerce a single metadata value into something Pinecone accepts:
    str, int, float, bool, or list[str]. Anything else gets JSON-
    stringified as a fallback rather than dropped — preserves the data
    (recoverable via json.loads downstream) instead of silently losing
    it, while still satisfying Pinecone's type constraints.
    """
    if value is None:
        return ""  # Pinecone rejects None; empty string is the safe stand-in
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        if all(isinstance(v, str) for v in value):
            return value
        # list of non-strings (e.g. table rows: list[list]) — stringify
        return json.dumps(value, ensure_ascii=False)
    # dicts or anything else unexpected
    return json.dumps(value, ensure_ascii=False)


def build_pinecone_vector(chunk: dict) -> dict:
    """
    Convert one embedded chunk (chunk.py + embed.py's combined output
    shape) into the {id, values, metadata} shape Pinecone's upsert
    expects.
    """
    metadata = {k: sanitize_metadata_value(v) for k, v in chunk.get("metadata", {}).items()}

    # zone_type is promoted to a top-level metadata field (not just
    # nested under whatever chunk.py called it) since it's the primary
    # thing retriever.py will want to filter on later (e.g. "only
    # search course chunks for this query").
    metadata["zone_type"] = chunk["zone_type"]

    text = chunk["text"]
    if len(text) > MAX_METADATA_TEXT_CHARS:
        text = text[:MAX_METADATA_TEXT_CHARS]  # truncate defensively; log if this ever fires
    metadata["text"] = text

    return {
        "id": chunk["chunk_id"],
        "values": chunk["embedding"],
        "metadata": metadata,
    }


def main(chunks_path: Path, namespace: str = ""):
    if not chunks_path.exists():
        raise FileNotFoundError(
            f"{chunks_path} not found — run ingestion/embed.py first to produce it."
        )

    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    print(f"Loaded {len(chunks)} embedded chunks from {chunks_path}")

    missing_embeddings = [c["chunk_id"] for c in chunks if "embedding" not in c]
    if missing_embeddings:
        raise ValueError(
            f"{len(missing_embeddings)} chunk(s) have no embedding — "
            f"did embed.py run to completion? First few: {missing_embeddings[:5]}"
        )

    vectors = [build_pinecone_vector(c) for c in chunks]

    client = PineconeClient()
    client.create_index_if_not_exists()

    print(f"\nUpserting {len(vectors)} vectors to index '{client.index_name}'...")
    total = client.upsert_all(vectors, namespace=namespace)
    print(f"\nUpsert complete: {total} vectors upserted.")

    stats = client.get_stats()
    print(f"\nIndex stats after upload:\n{stats}")

    if total != len(vectors):
        print(
            f"\nWARNING: upserted count ({total}) does not match vectors sent "
            f"({len(vectors)}) — investigate before assuming the corpus is fully loaded."
        )


if __name__ == "__main__":
    chunks_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DATA_PROCESSED_DIR / "chunks.json"
    main(chunks_path)
