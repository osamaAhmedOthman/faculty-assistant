"""
Streamlit dashboard providing a lightweight UI over the FastAPI service.

Architecture & Design Notes:
- Process Isolation: Communicates with the API purely over HTTP (`requests`). Avoids importing
  internal pipeline modules (`rag/`, `guardrails/`), keeping dependencies lean and avoiding state duplication.
- Config & Networking: `API_BASE_URL` reads from environment variables (defaults to `http://api:8000` 
  for Docker network DNS, overridable for local dev).
- Fixed Retrieval Parameters: Hides `top_k` and `zone_filter` from the UI to avoid forcing 
  manual domain decisions onto end users.
- State & Error Handling: Uses `st.session_state` to retain chat history across Streamlit reruns. 
  Relies on API-side reliability layers for LLM fallback handling, catching only raw network failures.
- Guardrail Transparency: Displays citation verification warnings directly alongside 
  responses when `citations_valid=False`.
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
