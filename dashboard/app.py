"""
dashboard/app.py — Streamlit dashboard

Responsibility: a thin UI over the FastAPI service. Calls the API over
HTTP via `requests` — deliberately NOT `from rag.pipeline import
Pipeline` — matching the two-services architecture decision: dashboard
and API are independent processes/containers, so this file has no
business importing rag/, guardrails/, or reliability/ at all. If it
did, this container would need every pipeline dependency installed
(sentence-transformers, torch, pinecone, groq) just to render a chat
box, and would carry its own separate Pipeline instance with its own
separate circuit-breaker state — the exact duplication running two
independent in-process pipelines would create.

Design notes:
- API_BASE_URL VIA ENV VAR: defaults to the FastAPI service's
  docker-compose service name (see docker-compose.yml — service
  hostnames are resolvable between containers on the same Docker
  network) so this works out of the box under `docker-compose up`,
  but stays overridable for local dev (e.g. running the dashboard
  bare with `streamlit run` against a locally-running `uvicorn`).
- NO top_k / zone_filter CONTROLS: both were originally exposed as
  sidebar widgets, but neither is a decision a student asking a
  regulations question should have to make — "restrict search to
  Course/Regulation/Table" requires already knowing which zone the
  answer lives in, which is exactly what the student is asking the
  assistant to figure out. DEFAULT_TOP_K and zone_filter=None (search
  everything) are fixed constants instead. If a future version wants
  to auto-select a zone_filter based on the question's shape, that
  belongs in Pipeline/Generator as a real classification step, not as
  a manual dropdown pushed onto the user.
- st.session_state FOR HISTORY: Streamlit reruns this whole script top
  to bottom on every interaction, so anything that needs to persist
  across reruns (chat history) must live in st.session_state, not a
  plain module-level list.
- NO RETRY/CIRCUIT-BREAKER LOGIC HERE: that reliability layer already
  lives inside Pipeline.run() on the API side (see reliability/) — a
  degraded-but-valid low-confidence fallback answer comes back as an
  ordinary 200 response, so this file only needs to handle genuine
  HTTP/network failures (the API process itself being unreachable),
  not "the LLM is down" (which the API already turned into a normal
  response for exactly this reason).
- CITATION WARNINGS SURFACED, NOT HIDDEN: citations_valid=False is a
  real signal from the guardrail layer (guardrails/validators.py) that
  a source may be hallucinated — the fail-open design deliberately
  keeps the answer visible rather than blocking it, so the dashboard's
  job is to show BOTH: the answer, and a visible warning the guardrail
  raised about it.
"""

import os

import requests
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://api:8000")
REQUEST_TIMEOUT_SECONDS = 60  # generous: a cold circuit-breaker retry sequence on the API side can itself take several seconds before falling back

# Fixed defaults — see module docstring's "NO top_k / zone_filter
# CONTROLS" note. DEFAULT_TOP_K matches core.config.RETRIEVAL_TOP_K's
# own default (5); kept as a plain constant here rather than importing
# core.config, since this file deliberately has zero imports from the
# rag/ codebase (see module docstring).
DEFAULT_TOP_K = 5

st.set_page_config(page_title="SWE Faculty Assistant", page_icon="🎓", layout="centered")
st.title("🎓 SWE Faculty Assistant")
st.caption("Ask about Software Engineering program regulations, courses, and prerequisites at Mansoura University's Faculty of Computers and Information.")

if "history" not in st.session_state:
    st.session_state.history = []  # list of {"question": str, "response": dict | None, "error": str | None}

with st.sidebar:
    if st.button("Clear conversation"):
        st.session_state.history = []
        st.rerun()
    st.divider()
    st.caption(f"API: `{API_BASE_URL}`")


def call_api(question: str) -> dict:
    """
    POST to the API's /query endpoint. Raises requests.RequestException
    on a network-level failure (API unreachable, timeout, connection
    refused) — the caller is responsible for catching that and showing
    a UI-appropriate error, since a raw exception here means the API
    PROCESS is unreachable, a fundamentally different situation from
    the API being reachable but reporting a low-confidence fallback
    answer (which arrives as a normal 200 response, not an exception).

    zone_filter is intentionally omitted from the payload — QueryRequest
    on the API side already defaults it to None (search every zone),
    so not sending it is equivalent to explicitly sending null.
    """
    payload = {"question": question, "top_k": DEFAULT_TOP_K}

    response = requests.post(f"{API_BASE_URL}/query", json=payload, timeout=REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()
    return response.json()


def render_response(response: dict) -> None:
    confidence = response.get("confidence", "low")
    confidence_icon = {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(confidence, "⚪")

    st.markdown(response["answer"])
    st.caption(f"{confidence_icon} Confidence: {confidence}")

    if response.get("citations_valid") is False:
        st.warning(
            "The guardrail flagged a possible citation issue in this answer:\n\n"
            + "\n".join(f"- {w}" for w in response.get("citation_warnings", []))
        )

    if response.get("parse_error"):
        st.info("The model's raw response wasn't valid JSON — showing raw text above.")


# Replay history
for turn in st.session_state.history:
    with st.chat_message("user"):
        st.markdown(turn["question"])
    with st.chat_message("assistant"):
        if turn["error"]:
            st.error(turn["error"])
        else:
            render_response(turn["response"])

# New input
question = st.chat_input("Ask a question about the SWE program regulations...")
if question:
    with st.chat_message("user"):
        st.markdown(question)

    turn = {"question": question, "response": None, "error": None}
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = call_api(question)
                turn["response"] = result
                render_response(result)
            except requests.RequestException as exc:
                error_message = f"Could not reach the Faculty Assistant API at `{API_BASE_URL}`: {exc}"
                turn["error"] = error_message
                st.error(error_message)

    st.session_state.history.append(turn)
