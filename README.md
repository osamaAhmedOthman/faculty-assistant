# Faculty Assistant

A Retrieval-Augmented Generation (RAG) system that answers student questions about Software Engineering program regulations at Mansoura University's Faculty of Computers and Information, grounded in the official bilingual (Arabic/English) program regulations document.

This project's purpose is **not** to demonstrate RAG fundamentals — it's built to showcase evaluation methodology, guardrail design, prompt engineering, and LLM reliability patterns on top of a working RAG pipeline.

## Architecture

```
                    ┌─────────────┐
                    │  Streamlit  │  (dashboard, port 8501)
                    │  dashboard  │
                    └──────┬──────┘
                           │ HTTP
                    ┌──────▼──────┐
                    │   FastAPI   │  (api, port 8000)
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │  Pipeline   │  input guardrail → retrieval →
                    │             │  generation → output guardrail →
                    │             │  reliability (retry/breaker/fallback)
                    └──────┬──────┘
              ┌────────────┼────────────┐
       ┌──────▼─────┐ ┌────▼────┐ ┌─────▼─────┐
       │  Retriever │ │Generator│ │ Guardrails │
       └──────┬─────┘ └────┬────┘ └───────────┘
       ┌──────▼─────┐ ┌────▼────┐
       │  Pinecone  │ │  Groq   │
       └────────────┘ └─────────┘
```

The dashboard and API are **separate services communicating over HTTP**, not a single process — a deliberate choice over an in-process import, matching how this system would actually be deployed (the dashboard has no business holding a Pinecone connection or an embedding model in memory).

### Pipeline stages

1. **Input guardrail** (`guardrails/validators.py::validate_input`) — blocks empty/oversized queries and known prompt-injection patterns *before* retrieval or generation ever run. Fails **closed**: a false-positive block costs the user one rephrase.
2. **Retrieval** (`rag/retriever.py`) — embeds the query (`paraphrase-multilingual-MiniLM-L12-v2`, local, multilingual) and queries Pinecone for the top-k most similar chunks, with optional program/zone metadata filters.
3. **Generation** (`rag/generator.py`) — chunks below a relevance-score floor are dropped; the LLM (Groq, `openai/gpt-oss-120b`) is instructed via `prompts/system.txt` to answer *only* from retrieved context and return structured JSON (`answer`, `sources`, `confidence`).
4. **Output guardrail** (`guardrails/validators.py::verify_citations`) — checks every cited source against what was *actually* retrieved, catching hallucinated citations. Fails **open**: a bad citation downgrades confidence and attaches a warning rather than discarding an otherwise-correct answer.
5. **Reliability layer** (`reliability/`) — `retry.py` (tenacity-based exponential backoff for transient Groq failures) wraps generation, itself wrapped by `circuit_breaker.py` (a three-state breaker that fails fast during a sustained outage instead of retrying against an already-broken dependency). Total failure degrades to `fallback.py`'s fixed low-confidence response — never an unhandled exception reaching the API layer.

### A known limitation, stated deliberately

The output guardrail verifies that a cited source (e.g. "Article 12") was actually *retrieved* — it does not verify that the model's claim about what Article 12 *says* is accurate. A citation can be real and still attached to a fabricated or misremembered detail ("citation washing"). This is exactly the failure class `RAGAS`'s `faithfulness` metric is designed to catch statistically, which is why the evaluation harness (`evaluation/ragas_eval.py`) exists as a second, independent layer rather than trusting the citation guardrail alone.

## Repository layout

```
ingestion/      extract.py → preprocess.py → chunk.py → embed.py → upload.py
rag/            retriever.py, prompts.py, generator.py, pipeline.py
guardrails/     schemas.py (structural), validators.py (semantic/citation)
reliability/    retry.py, circuit_breaker.py, fallback.py
evaluation/     dataset.py (golden set), ragas_eval.py (RAGAS scoring)
api/            models.py, routes.py, main.py (FastAPI)
dashboard/      app.py (Streamlit, HTTP client of api/)
core/           config.py (env vars, central constants), logger.py
tests/          pytest, injected fakes at every external-dependency boundary
```

## Running locally (no Docker)

Two terminals, both from the project root:

```bash
# Terminal 1 — API
uvicorn api.main:app --reload --port 8000

# Terminal 2 — Dashboard (PowerShell)
$env:API_BASE_URL="http://localhost:8000"
streamlit run dashboard/app.py

# Terminal 2 — Dashboard (bash/zsh)
API_BASE_URL=http://localhost:8000 streamlit run dashboard/app.py
```

Requires a `.env` file in the project root (see `.env.example`) with `GROQ_API_KEY` and `PINECONE_API_KEY` set, and the SWE corpus already ingested and upserted to Pinecone (`python -m ingestion.upload`).

## Running with Docker

```bash
docker-compose up --build
```

Dashboard: `http://localhost:8501` · API docs: `http://localhost:8000/docs`

The API and dashboard are separate images (`api/Dockerfile`, `dashboard/Dockerfile`) on a shared Compose network — the dashboard reaches the API at `http://api:8000` (Docker's service-name DNS resolution), not `localhost`. The dashboard image intentionally does **not** install `sentence-transformers`/`torch`/`pinecone`/`groq` — it only ever calls the API over HTTP (see `dashboard/requirements.txt`).

Docker Compose here builds fixed images, not a live-reload dev loop — after a code change, `docker-compose up --build` again, or run the bare `uvicorn --reload` / `streamlit run` commands above for iterative development.

## Testing

```bash
pytest
```

96+ tests, all external dependencies (Groq, Pinecone, the embedding model) replaced with injected fakes at the constructor boundary — no real API calls or model downloads happen during the suite, and it runs in under 3 seconds.

## Evaluation

```bash
python -m evaluation.ragas_eval --list-categories   # see available question categories
python -m evaluation.ragas_eval --limit 5           # run a subset against a free-tier rate limit
```

Runs the golden set (`evaluation/dataset.py`) through the real `Pipeline`, scores results with RAGAS (`faithfulness`, `answer_relevancy`, `context_precision`, `context_recall`) using a judge model kept separate from the generation model to isolate rate-limit quotas, and saves timestamped results to `evaluation/reports/`.

## Design principles this project holds itself to

- **Fail-open vs. fail-closed guardrail asymmetry** — input validation blocks on uncertainty (cheap to recover from: rephrase); output citation checking flags-and-continues rather than discarding a possibly-correct answer (expensive to recover from: the answer is just gone).
- **Real data validation over synthetic assumption** — every golden-set entry and every documented bug fix in this codebase was verified against actual pipeline output, not derived from what the regulations document is assumed to say.
- **No dead code, no speculative scaffolding** — a module with no real content is deleted, not stubbed, until something concrete justifies it existing.
