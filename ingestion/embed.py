"""
embed.py — embedding generation stage

Responsibility: take the per-program *_chunks.json files produced by
chunk.py (one file per document: AI, SWE, BIO) and produce a SINGLE
combined data/processed/chunks.json where every chunk has an embedding
vector attached — matching the architecture's target output file.

WHY combine all 3 programs into one file at this stage (not earlier):
chunk.py deliberately stays per-document, since zone detection and
catalog-boundary logic are document-specific concerns. But embedding
and upload are corpus-wide concerns — Pinecone doesn't care which
program a chunk came from at write time (that's what metadata.program
is for at query time), so this is the natural point to merge the 3
programs into one flat list of chunks ready for upload.py.

Design notes:
- We embed chunk["text"] only — not the metadata. Embedding metadata
  (course codes, program names) alongside description text would
  dilute the semantic signal the embedding is supposed to capture.
  Metadata stays as metadata, used for filtering/display, not for
  the vector itself.
- Table chunks are currently empty (tables: 0 across all 3 docs — see
  chunk.py's documented limitation). This script doesn't special-case
  that; it just embeds whatever chunk lists exist. When table chunking
  is eventually implemented, this script needs no changes.
- We batch across ALL chunks from ALL programs in one embed_texts()
  call rather than one call per document, since batched inference is
  faster and there's no reason to pay the per-call overhead 3 times.
- Each output chunk keeps a stable chunk_id (already unique per
  chunk.py's design: prefixed by source_file) — this is what upload.py
  will use as the Pinecone vector ID, so re-running embed.py on
  unchanged chunks produces the same IDs and upload.py can safely
  upsert (overwrite) rather than duplicate.
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
