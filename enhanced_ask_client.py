"""
Enhanced Ask Across Documents - Reasoning capability.
Extends the original ask_documents feature with full document text context
so the AI can truly reason with the content, check feasibility, and provide
evidence-based answers (e.g., "Given the weather data in these documents,
is it possible to schedule an outdoor event?").
"""

from __future__ import annotations

import json
import os
from typing import Any

from dotenv import load_dotenv

load_dotenv(override=True)

# Patch httpx for OpenRouter compatibility
import openai._base_client as _obc
_orig_sync = _obc.SyncHttpxClientWrapper.__init__
def _patched_sync(self, *args, **kwargs):
    kwargs.pop('proxies', None)
    return _orig_sync(self, *args, **kwargs)
_obc.SyncHttpxClientWrapper.__init__ = _patched_sync

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise RuntimeError("OPENROUTER_API_KEY is not set.")

_HTTP_REFERER = os.getenv("OPENROUTER_HTTP_REFERER", "")
_APP_TITLE = os.getenv("OPENROUTER_APP_TITLE", "")

_default_headers = {}
if _HTTP_REFERER:
    _default_headers["HTTP-Referer"] = _HTTP_REFERER
if _APP_TITLE:
    _default_headers["X-OpenRouter-Title"] = _APP_TITLE

import openai as _openai
_client = _openai.OpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",
    default_headers=_default_headers if _default_headers else None,
)

import time
_last_call_timestamp: float = 0.0
_MIN_DELAY_SECONDS = 4.0

def _enforce_min_delay() -> None:
    global _last_call_timestamp
    now = time.time()
    elapsed = now - _last_call_timestamp
    if elapsed < _MIN_DELAY_SECONDS:
        time.sleep(_MIN_DELAY_SECONDS - elapsed)
    _last_call_timestamp = time.time()


def _call_openrouter_with_backoff(
    model: str,
    messages: list[dict[str, Any]],
    max_retries: int = 2,
) -> str:
    global _last_call_timestamp
    last_exception: Exception | None = None
    base_delay = 5.0

    for attempt in range(max_retries + 1):
        try:
            _enforce_min_delay()
            response = _client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=8192,
            )
            _last_call_timestamp = time.time()
            return response.choices[0].message.content or ""
        except Exception as exc:
            error_str = str(exc).lower()
            last_exception = exc
            is_rate_limit = "429" in error_str or "rate limit" in error_str or "quota" in error_str
            if is_rate_limit and attempt < max_retries:
                time.sleep(base_delay * (2 ** attempt))
                continue
            if attempt < max_retries:
                time.sleep(base_delay)
                continue
            raise exc
    raise last_exception  # type: ignore[misc]


def ask_documents_reasoned(
    documents: list[dict[str, Any]],
    question: str,
) -> dict[str, Any]:
    """
    Enhanced version of ask_documents with full reasoning capability.
    
    This version includes the FULL document text content (not just summaries),
    enabling the AI to:
    - Reason with specific data, figures, and conditions in the documents
    - Check feasibility (e.g., "Given the weather data, is it possible to...?")
    - Cross-reference information across documents
    - Provide evidence-backed reasoning with citations
    
    documents: list of dicts with keys:
        - filename: str
        - full_text: str (the complete extracted text from the document)
        - summary: str
        - key_points: str (JSON array)
        - entities: str (JSON array)
        - structured_data: str (JSON object)
        - risks: str (JSON array)
    
    question: natural language question (including reasoning/feasibility questions)
    
    Returns dict with:
        - answer: reasoned answer with evidence
        - reasoning: step-by-step reasoning process
        - sources: list of {"document": str, "evidence": str}
        - feasibility: "possible" | "not_possible" | "insufficient_data" (for feasibility questions)
    """
    model = "google/gemini-2.5-flash"

    doc_sections = []
    for doc in documents:
        filename = doc.get("filename", "Unknown")
        full_text = doc.get("full_text", "")
        summary = doc.get("summary", "")
        key_points = doc.get("key_points", "[]")
        structured_data = doc.get("structured_data", "{}")
        risks = doc.get("risks", "[]")

        section = f"Document: {filename}\n"
        if summary:
            section += f"Summary: {summary}\n"
        if key_points and key_points != "[]":
            section += f"Key Points: {key_points}\n"
        if structured_data and structured_data != "{}":
            section += f"Structured Data: {structured_data}\n"
        if risks and risks != "[]":
            section += f"Risks: {risks}\n"
        # Include full text (truncated to manage context, but much more than just summary)
        if full_text:
            # Use up to 8000 chars of full text per document for reasoning
            section += f"Full Document Content:\n{full_text[:8000]}\n"
        doc_sections.append(section)

    docs_context = "\n---\n".join(doc_sections)

    messages = [
        {
            "role": "user",
            "content": f"""You are an advanced document analysis assistant with full reasoning capability.

Your task: Using ONLY the provided document context below, answer the user's question.
You MUST reason step-by-step based on the actual content in the documents.

If the question involves feasibility (e.g., "Is it possible to...", "Can we...", "Given [conditions], would it work to..."), you should:
1. Extract the relevant conditions/constraints from the documents
2. Reason step-by-step whether the proposal is feasible
3. Cite specific evidence from the documents

If the answer cannot be found in the provided context, say "I cannot find this information in the provided documents."

Return ONLY a valid JSON object (no markdown fences) with these exact keys:
- "answer": your reasoned answer
- "reasoning": a brief step-by-step explanation of how you arrived at the answer
- "feasibility": "possible" if the documents support feasibility, "not_possible" if they contradict it, "insufficient_data" if there isn't enough information, or "not_applicable" if the question is not about feasibility
- "sources": a list of objects with:
  - "document": the filename of the source document
  - "evidence": the specific text from that document that supports this answer

Documents:
{docs_context}

Question: {question}

Example format:
{{"answer": "Based on the weather data in the documents, ...", "reasoning": "1. Document A shows rainfall of 50mm on that date... 2. Document B indicates... Therefore...", "feasibility": "not_possible", "sources": [{{"document": "weather_report.pdf", "evidence": "Rainfall: 50mm, wind speed: 40km/h"}}]}}""",
        }
    ]

    try:
        response_text = _call_openrouter_with_backoff(model, messages)
        # Parse JSON from response (strip markdown if needed)
        import re
        cleaned = response_text.strip()
        cleaned = re.sub(r"^```(?:json)?\s*\n?", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\n?```\s*$", "", cleaned.strip())
        parsed = json.loads(cleaned)

        return {
            "answer": parsed.get("answer", "Could not generate an answer."),
            "reasoning": parsed.get("reasoning", ""),
            "feasibility": parsed.get("feasibility", "not_applicable"),
            "sources": parsed.get("sources", []),
        }
    except Exception as exc:
        raise RuntimeError(f"Reasoned ask failed: {exc}") from exc