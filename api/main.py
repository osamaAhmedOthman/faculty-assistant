"""
api/main.py — FastAPI application entry point

Responsibility: construct the FastAPI app, configure logging once at
process start, wire in CORS (needed because dashboard/app.py is a
SEPARATE service calling this one over HTTP, not an in-process import —
see the architecture decision this follows), and mount routes.py's
router. No business logic lives here.

Design notes:
- configure_logging() CALLED HERE, NOT IN routes.py: this is the
  actual process entry point (see core/logger.py's own docstring on
  why configuration belongs at the entry point, not inside library
  modules) — routes.py, pipeline.py, and everything downstream just
  call logging.getLogger(__name__) and inherit this configuration.
- CORS IS PERMISSIVE BY DESIGN, SCOPED BY ENV VAR: the dashboard runs
  as its own container/service (see the HTTP-over-import decision),
  so browser requests from Streamlit's origin need explicit CORS
  clearance — FastAPI blocks cross-origin requests by default.
  API_ALLOWED_ORIGINS defaults to Streamlit's typical local port so
  `docker-compose up` works out of the box, but stays overridable via
  env var rather than hardcoded, for when this is deployed somewhere
  other than localhost.
- STARTUP LOG, NOT A STARTUP CHECK: the startup event logs that the
  service is up; it deliberately does NOT eagerly ping Pinecone/Groq
  during startup, matching health_endpoint's same "don't couple
  liveness to external dependency reachability" reasoning in
  routes.py. Pipeline()'s own constructor already runs at import time
  (see routes.py's module-scoped singleton) — if the embedding model
  or Pinecone client fails to construct at all, that surfaces as an
  import-time crash on startup regardless, which is the correct
  failure mode for a genuinely broken config (e.g. missing API key).
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
