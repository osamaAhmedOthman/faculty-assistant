"""
groq_client.py — Groq wrapper

Responsibility: talk to Groq's chat completion API. Nothing else — no
prompt design, no retrieval logic, no knowledge of "context" or
"citations". Mirrors pinecone_client.py and embeddings.py: this
project's convention is that clients/ isolates a single external
dependency behind a small, swappable interface, so rag/ modules never
import a provider SDK directly. Swapping providers later (e.g. moving
off Groq) means changing this one file, nothing upstream.
"""

from groq import Groq

from core.config import GROQ_API_KEY, GROQ_MODEL


class GroqClient:
    def __init__(self, api_key: str = GROQ_API_KEY, model: str = GROQ_MODEL):
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY is not set. Set it as an environment variable "
                "(see .env.example) before running generator.py."
            )
        self.model = model
        self._client = Groq(api_key=api_key)

    def chat(self, system: str, user: str, temperature: float = 0.0) -> str:
        """
        Single-turn chat completion. temperature=0.0 by default —
        grounded factual answers over a regulations document should be
        deterministic, not creative. Returns the raw text content of
        the model's reply; parsing/validating that text is the
        caller's responsibility (see generator.py), not this client's.
        """
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
        )
        return response.choices[0].message.content
