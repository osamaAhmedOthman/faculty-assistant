"""
embed.py — embedding generation stage

Responsibility: Combine per-program chunk files from chunk.py into a single 
data/processed/chunks.json file, attaching dense embedding vectors to every chunk.

Design notes:
- CORPUS MERGING: Unifies separate document-level chunk files into a single, 
  corpus-wide list optimized for vector database ingestion.
- TEXT-ONLY EMBEDDINGS: Encodes only chunk["text"] to prevent metadata 
  (e.g., course codes, program labels) from diluting the semantic vector signal.
- BATCH INFERENCE: Executes vector encoding across the full merged dataset 
  in a single batch operation to minimize computational overhead.
- ID PRESERVATION: Retains unique `chunk_id` attributes to maintain compatibility 
  with upload.py's idempotent Pinecone upserts.
"""

import sys
import json
from pathlib import Path

# Allow `python ingestion/embed.py` to be run directly (as this project
# has been run throughout) by putting the project root on sys.path, so
# `from core.config import ...` and `from clients.embeddings import ...`
# resolve correctly regardless of which directory the script is invoked
# from or how it's invoked.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.config import DATA_PROCESSED_DIR
from clients.embeddings import EmbeddingClient


def load_chunk_file(path: Path) -> list[dict]:
    """
    Flatten one *_chunks.json file (chunk.py's output format, which
    groups chunks by zone_type: regulation_chunks / table_chunks /
    course_chunks) into a single flat list of chunk dicts.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    flat = []
    flat.extend(data.get("regulation_chunks", []))
    flat.extend(data.get("table_chunks", []))
    flat.extend(data.get("course_chunks", []))
    return flat


def embed_chunks(chunks: list[dict], client: EmbeddingClient) -> list[dict]:
    """
    Attach an "embedding" field to each chunk dict, returning new dicts
    (does not mutate the input list) so callers can compare
    before/after if needed for debugging.
    """
    texts = [c["text"] for c in chunks]
    vectors = client.embed_texts(texts)

    if len(vectors) != len(chunks):
        # Fail loudly rather than silently misaligning embeddings to
        # chunks — a desync here would be a genuinely nasty bug to
        # trace later (wrong chunk retrieved for a query, with no
        # obvious error anywhere).
        raise RuntimeError(
            f"Embedding count mismatch: {len(vectors)} vectors for {len(chunks)} chunks"
        )

    embedded = []
    for chunk, vector in zip(chunks, vectors):
        embedded_chunk = dict(chunk)
        embedded_chunk["embedding"] = vector
        embedded.append(embedded_chunk)

    return embedded


def main(chunk_file_paths: list[Path], output_path: Path):
    client = EmbeddingClient()
    print(f"Loaded embedding model: {client.model_name}")

    all_chunks = []
    for path in chunk_file_paths:
        if not path.exists():
            print(f"  WARNING: {path} not found, skipping")
            continue
        chunks = load_chunk_file(path)
        print(f"  {path.name}: {len(chunks)} chunks")
        all_chunks.extend(chunks)

    print(f"\nTotal chunks to embed: {len(all_chunks)}")
    if not all_chunks:
        print("Nothing to embed — check that chunk files exist and contain data.")
        return

    embedded_chunks = embed_chunks(all_chunks, client)

    # Sanity check: confirm embedding dimension is consistent across
    # every chunk before writing to disk. A mismatch here would mean
    # something went wrong mid-batch (e.g. mixed model calls) — better
    # to catch it now than after upload.py has already pushed to Pinecone.
    dims = {len(c["embedding"]) for c in embedded_chunks}
    if len(dims) > 1:
        raise RuntimeError(f"Inconsistent embedding dimensions found: {dims}")
    print(f"Embedding dimension: {dims.pop()}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(embedded_chunks, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nSaved {len(embedded_chunks)} embedded chunks to {output_path}")

    # Zone-type breakdown for a quick sanity glance without re-opening
    # the file — mirrors chunk.py's own summary style.
    zone_counts: dict[str, int] = {}
    for c in embedded_chunks:
        zone_counts[c["zone_type"]] = zone_counts.get(c["zone_type"], 0) + 1
    print(f"Zone breakdown: {zone_counts}")


if __name__ == "__main__":
    # Default: pick up the three known program chunk files from
    # data/processed/. Can be overridden via CLI args if you want to
    # embed a subset (e.g. while iterating on one document's chunking).
    if len(sys.argv) > 1:
        chunk_files = [Path(p) for p in sys.argv[1:]]
    else:
        chunk_files = [
            DATA_PROCESSED_DIR / "ai_chunks.json",
            DATA_PROCESSED_DIR / "swe_chunks.json",
            DATA_PROCESSED_DIR / "bio__chunks.json",
        ]

    main(chunk_files, DATA_PROCESSED_DIR / "chunks.json")
