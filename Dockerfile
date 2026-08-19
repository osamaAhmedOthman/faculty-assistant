# Single Dockerfile for both API and Dashboard services.
#
# Architecture Note: Both services build from this shared image but run as 
# isolated containers with distinct startup commands (defined in docker-compose.yml).
# The dashboard communicates with the API strictly over HTTP (`requests`), keeping 
# process boundaries separated despite sharing a base runtime environment.
#
# - Environment: Requires all core dependencies (API, RAG engine, Streamlit).
# - Production Config: Reload flags disabled; local dev loops should run bare tools instead.
# - Default Entrypoint: Defaults to running the API if started without a compose command override.

FROM python:3.11-slim

WORKDIR /app

# poppler-utils: pdf2image (ingestion/extract.py's OCR fallback) needs it.
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

EXPOSE 8000 8501

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
