"""
pinecone_client.py — Pinecone wrapper

Responsibility: talk to Pinecone. Nothing else — no knowledge of what
a "chunk" is conceptually, no embedding logic. ingestion/upload.py
decides WHAT to upload and prepares the payload shape; this module
only knows HOW to get vectors into (and later, out of) Pinecone.

Design notes:
- Index creation is idempotent (create_index_if_not_exists): safe to
  call every time upload.py runs, without accidentally erroring on a
  second run or silently doing nothing on the first. This matters
  because you'll re-run ingestion repeatedly while iterating.
- Upserts are batched (default 100 vectors/call) rather than sent one
  at a time — Pinecone's own docs recommend batching for throughput,
  and sending 208 individual requests would be needlessly slow.
- We fail loudly on a batch error rather than silently continuing to
  the next batch, since a partial upload (some chunks present, some
  missing, with no record of which) would be a much worse debugging
  problem later than the pipeline just stopping now.
"""

from pinecone import Pinecone, ServerlessSpec

from core.config import (
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    PINECONE_CLOUD,
    PINECONE_REGION,
    PINECONE_METRIC,
    EMBEDDING_DIMENSION,
)


class PineconeClient:
    def __init__(
        self,
        api_key: str = PINECONE_API_KEY,
        index_name: str = PINECONE_INDEX_NAME,
        dimension: int = EMBEDDING_DIMENSION,
    ):
        if not api_key:
            raise ValueError(
                "PINECONE_API_KEY is not set. Set it as an environment variable "
                "(see .env.example) before running upload.py."
            )

        self.index_name = index_name
        self.dimension = dimension
        self._pc = Pinecone(api_key=api_key)
        self._index = None  # lazily connected in .index property

    def create_index_if_not_exists(self):
        """
        Idempotent index creation. Safe to call on every upload.py run.

        Note: if an index already exists with a DIFFERENT dimension than
        EMBEDDING_DIMENSION (e.g. you switched embedding models without
        re-creating the index), this will NOT catch that mismatch —
        Pinecone will only surface it as an upsert-time error. Worth
        checking manually if you ever change EMBEDDING_MODEL_NAME.
        """
        existing = [idx["name"] for idx in self._pc.list_indexes()]
        if self.index_name in existing:
            print(f"Index '{self.index_name}' already exists, reusing it.")
            return

        print(f"Creating index '{self.index_name}' (dim={self.dimension}, metric={PINECONE_METRIC})...")
        self._pc.create_index(
            name=self.index_name,
            dimension=self.dimension,
            metric=PINECONE_METRIC,
            spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
        )

    @property
    def index(self):
        if self._index is None:
            self._index = self._pc.Index(self.index_name)
        return self._index

    def upsert_batch(self, vectors: list[dict], namespace: str = "") -> int:
        """
        Upsert one batch of vectors. Each vector dict must have the
        shape Pinecone expects: {"id": str, "values": list[float],
        "metadata": dict}. Returns the number of vectors upserted
        (from Pinecone's own response) so the caller can verify the
        count matches what was sent, rather than assuming success.
        """
        response = self.index.upsert(vectors=vectors, namespace=namespace)
        return response.get("upserted_count", 0)

    def upsert_all(self, vectors: list[dict], batch_size: int = 100, namespace: str = "") -> int:
        """
        Upsert a full list of vectors in batches. Returns total
        upserted count. Raises immediately on any batch failure rather
        than continuing — see module docstring for reasoning.
        """
        total_upserted = 0
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i : i + batch_size]
            count = self.upsert_batch(batch, namespace=namespace)
            total_upserted += count
            print(f"  Upserted batch {i // batch_size + 1} ({count} vectors, {total_upserted}/{len(vectors)} total)")

        return total_upserted

    def query(self, vector: list[float], top_k: int = 5, namespace: str = "", filter: dict | None = None) -> list[dict]:
        """
        Query the index for the top_k nearest vectors to `vector`.

        filter: optional Pinecone metadata filter, e.g.
            {"program": {"$eq": "AI"}} or {"zone_type": {"$eq": "course"}}
        This is how retriever.py will support "only search AI program
        courses" style scoping, without needing separate indexes per
        program.

        Returns a list of dicts with id/score/metadata — deliberately
        NOT Pinecone's raw response object, so callers (retriever.py)
        don't need to know anything about the Pinecone SDK's internal
        result shape. This keeps the Pinecone dependency contained to
        this one file, same principle as clients/embeddings.py isolating
        the embedding backend.
        """
        response = self.index.query(
            vector=vector,
            top_k=top_k,
            namespace=namespace,
            filter=filter,
            include_metadata=True,
        )
        return [
            {"id": match["id"], "score": match["score"], "metadata": match["metadata"]}
            for match in response.get("matches", [])
        ]

    def get_stats(self) -> dict:
        """Return index stats (total vector count, dimension, etc.) —
        useful as a post-upload sanity check."""
        return self.index.describe_index_stats()

    def query(
        self,
        vector: list[float],
        top_k: int = 5,
        namespace: str = "",
        filter: dict | None = None,
        include_metadata: bool = True,
    ) -> list[dict]:
        """
        Similarity search against the index. Returns Pinecone's raw
        match list (each match has id, score, metadata) — retriever.py
        is responsible for shaping this into whatever format the rest
        of the RAG pipeline expects, keeping this wrapper a thin,
        format-agnostic Pinecone interface rather than something that
        knows about "chunks" or RAG concepts.

        filter: optional Pinecone metadata filter, e.g.
            {"program": {"$eq": "AI"}} or {"zone_type": {"$eq": "course"}}
        to restrict search to a subset of the corpus. None means search
        the whole index.
        """
        response = self.index.query(
            vector=vector,
            top_k=top_k,
            namespace=namespace,
            filter=filter,
            include_metadata=include_metadata,
        )
        return response.get("matches", [])
