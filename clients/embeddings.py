"""
Embedding client wrapper for vector generation.

Architecture & Design Notes:
- Provider Abstraction: Decouples vector generation from vector storage and chunking, establishing 
  a clean seam for swapping local models with external embedding APIs without affecting callers.
- Persistent Model Instance: Loads the underlying model once upon class instantiation to avoid repeated, 
  costly model initialization across calls.
- Batched Vectorization: Vectorizes text in batches to optimize CPU/GPU inference throughput.
- Serialized Primitive Outputs: Converts raw NumPy outputs into plain Python list structures to simplify 
  JSON serialization and ensure compatibility with vector database clients.
"""

from core.config import EMBEDDING_MODEL_NAME, EMBEDDING_BATCH_SIZE


class EmbeddingClient:
    """Thin wrapper around a sentence-transformers model.

    Swap backends by changing model_name (any sentence-transformers
    model works via this same interface) or by writing an alternate
    class with the same embed_texts() signature (e.g. an OpenAIEmbeddingClient
    calling the API instead) — nothing else in the codebase needs to change
    as long as the interface (embed_texts: list[str] -> list[list[float]]) holds.
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        # Imported lazily inside __init__ (not at module top) so that
        # code which only needs config.py or other lightweight pieces
        # doesn't pay the cost of importing sentence-transformers
        # (and its torch dependency) unless it actually instantiates
        # this client.
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self._model = SentenceTransformer(model_name)

    def embed_texts(self, texts: list[str], batch_size: int = EMBEDDING_BATCH_SIZE) -> list[list[float]]:
        """
        Embed a list of texts, returning one vector per input text in
        the same order. Empty strings are embedded as-is (the model
        will produce some vector for them) rather than silently
        skipped — silently dropping an entry would desync the
        embeddings list from the caller's chunk list, which is exactly
        the kind of quiet bug this project has been actively hunting.
        """
        if not texts:
            return []

        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return embeddings.tolist()

    def embed_query(self, text: str) -> list[float]:
        """Convenience method for embedding a single query string at
        retrieval time (rag/retriever.py). Kept separate from
        embed_texts() so call sites reading the code can tell at a
        glance whether they're bulk-embedding a corpus or embedding
        one live user query."""
        return self.embed_texts([text])[0]
