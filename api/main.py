"""
FastAPI application entry point.

Architecture & Design Notes:
- Process Entry Point: Initializes the FastAPI app, configures global logging (via `core/logger.py`), 
  and mounts application routes (`routes.py`) without holding core business logic.
- CORS Configuration: Permits cross-origin HTTP calls from the separate Streamlit dashboard service. 
  Configured via `API_ALLOWED_ORIGINS` (defaults to local Streamlit ports, overridable via env var).
- Decoupled Lifecycle: Emits startup logs without running blocking dependency health checks (Pinecone/Groq). 
  Relies on import-time initialization in `routes.py` to catch missing keys or critical setup errors.
"""

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from core.logger import configure_logging, get_logger

configure_logging()
logger = get_logger(__name__)

from api.routes import router  # noqa: E402 — imported after configure_logging() so Pipeline's own construction logs (if any) are formatted correctly

API_ALLOWED_ORIGINS = os.getenv("API_ALLOWED_ORIGINS", "http://localhost:8501").split(",")

app = FastAPI(
    title="Faculty Assistant API",
    description="RAG API for Mansoura University SWE program regulations.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=API_ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.on_event("startup")
def on_startup():
    logger.info("Faculty Assistant API starting up — allowed origins: %s", API_ALLOWED_ORIGINS)
