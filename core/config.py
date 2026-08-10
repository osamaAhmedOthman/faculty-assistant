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
"""

import os
from pathlib import Path

# --- Paths -------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
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
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
