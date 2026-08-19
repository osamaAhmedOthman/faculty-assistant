"""
Central configuration settings and environment loading.

Architecture & Design Notes:
- Single Source of Truth: Centralizes environment variables, paths, and constants to prevent 
  hardcoded magic values across the pipeline. All modules import keys/settings directly from this file.
- Eager Environment Initialization: Invokes `load_dotenv()` at import time to populate `os.environ` 
  before any downstream module attempts to access configuration values.
- Lightweight Implementation: Uses `os.getenv()` with sane defaults instead of heavy configuration 
  libraries (`pydantic-settings`), maintaining a zero-dependency footprint until complex validation is required.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")  # no-op (safely) if .env doesn't exist

# --- Paths -------------------------------------------------------------
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# --- Embedding model -----------------------------------------------------
# Multilingual model chosen deliberately: the corpus mixes Arabic
# (regulatory articles) and English (course catalog) in the same
# collection. An English-only model would badly under-represent the
# Arabic content in embedding space. This model is local (no API key,
# no per-call cost, fully reproducible) and specifically trained on
# parallel multilingual data including Arabic.
#
# IMPORTANT: whatever model is used here MUST also be used at query
# time in rag/retriever.py — embeddings from two different models are
# not comparable in the same vector space. If you ever change this,
# you must re-embed and re-upload the entire corpus, not just new docs.
EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL_NAME", "paraphrase-multilingual-MiniLM-L12-v2"
)
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "384"))  # matches the model above
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))

# --- Vector DB / LLM (used by later pipeline stages, not embed.py itself) --
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "faculty-assistant")
# Serverless index location. Pinecone's free tier supports serverless
# indexes on AWS us-east-1 — kept as explicit config (not hardcoded in
# pinecone_client.py) since it's an infra choice, not application logic.
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")
# Cosine similarity is the standard choice for sentence-transformers
# embeddings (they're trained/evaluated with cosine similarity), so it's
# the default here rather than euclidean/dotproduct.
PINECONE_METRIC = os.getenv("PINECONE_METRIC", "cosine")

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

# --- Retrieval defaults --------------------------------------------------
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "5"))


def require_keys(*names: str):
    """
    Call at the START of any script that needs specific env vars
    (e.g. rag/pipeline.py needs PINECONE_API_KEY and GROQ_API_KEY).
    Fails immediately with a clear message naming exactly which key is
    missing, rather than letting the failure surface later as an
    opaque error from deep inside a client library (e.g. Pinecone's
    "unauthenticated request" error, which doesn't tell you WHICH env
    var was supposed to be set).
    """
    missing = [n for n in names if not globals().get(n)]
    if missing:
        raise EnvironmentError(
            f"Missing required environment variable(s): {', '.join(missing)}. "
            f"Set them in your .env file (see .env.example)."
        )
