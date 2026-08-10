"""
config.py — central settings

Responsibility: one place for env vars and pipeline-wide constants, so
nothing downstream hardcodes a model name, a path, or a magic number
that has to be found and changed in five files later.

We deliberately keep this dependency-light (no pydantic-settings) since
the project doesn't need validation/typing machinery yet — just reads
env vars with sane defaults. Upgrade to pydantic BaseSettings later if
config grows complex enough to need it (e.g. nested settings, secrets
validation).

ENV LOADING: load_dotenv() reads a .env file in the project root (if
present) and populates os.environ BEFORE the os.getenv() calls below
run. This must happen here, at import time, and before any other
module reads an API key — every other file in this project (embed.py,
upload.py, retriever.py, etc.) gets its keys by importing FROM this
module, never by calling os.getenv() directly. That's a deliberate
single-source-of-truth rule: if API keys were read ad-hoc in multiple
files, a .env change could silently apply in one place and not
another, which is exactly the kind of quiet inconsistency this project
has been actively hunting down elsewhere.
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
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

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
