"""OpenRouter API client with rate limiting and retry logic for FRAS."""

from __future__ import annotations

import base64
import io
import json
import os
import re
import time
from typing import Any

import fitz  # PyMuPDF for OCR fallback on scanned PDFs
from dotenv import load_dotenv
from PIL import Image
from PyPDF2 import PdfReader

load_dotenv(override=True)

# Patch OpenAI's internal httpx wrapper to ignore unsupported 'proxies' argument
import openai._base_client as _obc
_orig_sync = _obc.SyncHttpxClientWrapper.__init__
def _patched_sync(self, *args, **kwargs):
    kwargs.pop('proxies', None)
    return _orig_sync(self, *args, **kwargs)
_obc.SyncHttpxClientWrapper.__init__ = _patched_sync

# Also patch httpx.Client directly as a fallback
import httpx as _httpx
_orig_httpx_init = _httpx.Client.__init__
def _patched_httpx(self, *args, **kwargs):
    kwargs.pop('proxies', None)
    return _orig_httpx_init(self, *args, **kwargs)
_httpx.Client.__init__ = _patched_httpx

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise RuntimeError("OPENROUTER_API_KEY is not set. Please add it to your .env file.")

# Optional app attribution headers (for OpenRouter leaderboards)
_HTTP_REFERER = os.getenv("OPENROUTER_HTTP_REFERER", "")
_APP_TITLE = os.getenv("OPENROUTER_APP_TITLE", "")

_default_headers = {}
if _HTTP_REFERER:
    _default_headers["HTTP-Referer"] = _HTTP_REFERER
if _APP_TITLE:
    _default_headers["X-OpenRouter-Title"] = _APP_TITLE

# Initialize client once
import openai as _openai
_client = _openai.OpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",
    default_headers=_default_headers if _default_headers else None,
)

# Rate limiting state
_last_call_timestamp: float = 0.0
_MIN_DELAY_SECONDS = 4.0


def _enforce_min_delay() -> None:
    """Ensure at least MIN_DELAY_SECONDS have passed since the last API call."""
    global _last_call_timestamp
    now = time.time()
    elapsed = now - _last_call_timestamp
    if elapsed < _MIN_DELAY_SECONDS:
        time.sleep(_MIN_DELAY_SECONDS - elapsed)
    _last_call_timestamp = time.time()


def _strip_markdown_fences(text: str) -> str:
    """Remove ```json ... ``` or ``` ... ``` wrappers from LLM output."""
    # Remove opening ```json or ```
    text = re.sub(r"^```(?:json)?\s*\n?", "", text.strip(), flags=re.IGNORECASE)
    # Remove closing ```
    text = re.sub(r"\n?```\s*$", "", text.strip())
    return text.strip()


def _parse_json(response_text: str) -> dict[str, Any]:
    """Defensively parse JSON from LLM response, stripping markdown fences."""
    cleaned = _strip_markdown_fences(response_text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to parse JSON response: {exc}\nRaw: {cleaned[:500]}") from exc


def _call_openrouter_with_backoff(
    model: str,
    messages: list[dict[str, Any]],
    max_retries: int = 2,
) -> str:
    """
    Call OpenRouter with:
    - Minimum delay between consecutive calls (rate limiting)
    - Exponential backoff on 429 errors
    - Generic retry on transient failures
    """
    global _last_call_timestamp

    last_exception: Exception | None = None
    base_delay = 5.0

    for attempt in range(max_retries + 1):
        try:
            _enforce_min_delay()
            response = _client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=6000,
            )
            _last_call_timestamp = time.time()
            return response.choices[0].message.content or ""

        except Exception as exc:
            error_str = str(exc).lower()
            last_exception = exc

            is_rate_limit = "429" in error_str or "rate limit" in error_str or "quota" in error_str

            if is_rate_limit and attempt < max_retries:
                wait_time = base_delay * (2 ** attempt)
                time.sleep(wait_time)
                continue

            if attempt < max_retries:
                time.sleep(base_delay)
                continue

            raise exc

    raise last_exception  # type: ignore[misc]


def _pdf_has_text_layer(file_path: str) -> bool:
    """Check if a PDF has extractable text (vs. being a scanned/image-only PDF)."""
    try:
        reader = PdfReader(file_path)
        for page in reader.pages:
            text = page.extract_text() or ""
            if len(text.strip()) > 20:
                return True
        return False
    except Exception:
        return False


def _extract_pdf_as_images_base64(file_path: str) -> str:
    """
    Render PDF pages as images using PyMuPDF and return base64-encoded JPEG data.
    Multiple pages are concatenated with a marker.
    Returns: base64 data (up to max 5 pages, using JPEG at 72 DPI to keep payload minimal)
    """
    doc = fitz.open(file_path)
    page_images = []
    # Limit to 5 pages max and 72 DPI to stay within free tier token limits
    max_pages = min(len(doc), 5)

    for page_num in range(max_pages):
        page = doc[page_num]
        # Render page at 72 DPI - minimal for OCR, saves token budget
        pix = page.get_pixmap(dpi=72)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        buf = io.BytesIO()
        # Use JPEG at quality 70 - aggressive compression for token economy
        img.save(buf, format="JPEG", quality=70, optimize=True)
        page_images.append(base64.b64encode(buf.getvalue()).decode("utf-8"))

    doc.close()
    # Join multiple pages with a marker for the vision model to process
    return "===MULTI_PAGE_PDF===" + "|||".join(page_images)


def _extract_text_from_file(file_path: str, file_type: str) -> str:
    """Extract text content from uploaded file for processing."""
    if file_type == ".pdf":
        reader = PdfReader(file_path)
        text = "\n\n".join(page.extract_text() or "" for page in reader.pages)
        # If no text layer detected (scanned PDF), fall back to image-based extraction
        if not text or len(text.strip()) < 20:
            return _extract_pdf_as_images_base64(file_path)
        return text
    elif file_type in {".txt"}:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    elif file_type in {".png", ".jpg", ".jpeg"}:
        # For images, return base64 encoded for vision models
        with open(file_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    else:
        # For docx and other types, try reading as text
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        except Exception:
            return ""


def _extract_text_content_for_text_only(file_path: str, file_type: str) -> str:
    """
    Extract text content for use in text-only contexts (risk detection, ask tab,
    report generation). For scanned PDFs without text layers, returns a
    descriptive placeholder since the content is only available via vision model.
    """
    raw = _extract_text_from_file(file_path, file_type)
    # If the result is base64 image data, return a placeholder instead
    if raw.startswith("===MULTI_PAGE_PDF===") or file_type in {".png", ".jpg", ".jpeg"} and len(raw) > 100:
        from pathlib import Path
        return f"[Scanned document: {Path(file_path).name}. Content is only available via the vision model during document processing.]"
    return raw


# ---------------------------------------------------------------------------
# Main Extraction Pipeline
# ---------------------------------------------------------------------------

def _build_vision_messages_for_pdf(text_content: str) -> list[dict[str, Any]]:
    """
    Build a vision API message payload from base64-encoded PDF page images.
    The text_content contains the marker '===MULTI_PAGE_PDF===' followed by
    '|||'-separated base64 PNG data for each page.
    """
    pages_base64 = text_content.split("===MULTI_PAGE_PDF===", 1)[1].split("|||")

    # Build content array: instruction text + each page as an image
    content_parts: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": """You are analyzing a scanned PDF document that has been converted to images. 
Examine ALL pages carefully and return ONLY a valid JSON object (no markdown fences, no extra text) with these exact keys:
- "summary": a 2-4 sentence plain-language summary of the document
- "key_points": a list of the most critical facts, figures, or action items
- "document_type": your best guess (report, invoice, memo, contract, brochure, etc.)
- "entities": a flat list of people, organizations, dates, and amounts mentioned

If the document is an invoice, also include:
- "structured_data": {"amount": number, "due_date": "YYYY-MM-DD", "vendor": "..."}
If the document is a contract, also include:
- "structured_data": {"parties": ["..."], "duration": "...", "obligations": "..."}
Otherwise, include "structured_data": {}

Example format:
{"summary": "...", "key_points": ["..."], "document_type": "...", "entities": ["..."], "structured_data": {}}""",
        },
    ]

    for i, b64_page in enumerate(pages_base64):
        if b64_page.strip():
            content_parts.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{b64_page}",
                    },
                }
            )

    return [{"role": "user", "content": content_parts}]


def extract_document_content(file_path: str, file_type: str) -> dict[str, Any]:
    """
    Extract structured content from a document using OpenRouter.
    Returns dict with: summary, key_points, document_type, entities, raw_response,
                       risks, structured_data
    """
    model = "google/gemini-2.5-flash"

    text_content = _extract_text_from_file(file_path, file_type)

    # Detect if this is a scanned PDF rendered as images
    is_scanned_pdf = text_content.startswith("===MULTI_PAGE_PDF===")

    if is_scanned_pdf:
        # Use vision model for scanned PDFs (rendered as page images)
        messages = _build_vision_messages_for_pdf(text_content)
    elif file_type in {".png", ".jpg", ".jpeg"}:
        # Use vision model for regular images
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": """Analyze this image and return ONLY a valid JSON object (no markdown fences, no extra text) with these exact keys:
- "summary": a 2-4 sentence plain-language summary
- "key_points": a list of the most critical facts, figures, or action items
- "document_type": your best guess (report, invoice, memo, contract, etc.)
- "entities": a flat list of people, organizations, dates, and amounts mentioned

If the document is an invoice, also include:
- "structured_data": {"amount": number, "due_date": "YYYY-MM-DD", "vendor": "..."}
If the document is a contract, also include:
- "structured_data": {"parties": ["..."], "duration": "...", "obligations": "..."}
Otherwise, include "structured_data": {}

Example format:
{"summary": "...", "key_points": ["..."], "document_type": "...", "entities": ["..."], "structured_data": {}}""",
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/{file_type[1:]};base64,{text_content}"
                        },
                    },
                ],
            }
        ]
    else:
        # Text-based processing
        messages = [
            {
                "role": "user",
                "content": f"""Analyze this document and return ONLY a valid JSON object (no markdown fences, no extra text) with these exact keys:
- "summary": a 2-4 sentence plain-language summary
- "key_points": a list of the most critical facts, figures, or action items
- "document_type": your best guess (report, invoice, memo, contract, etc.)
- "entities": a flat list of people, organizations, dates, and amounts mentioned

If the document is an invoice, also include:
- "structured_data": {{"amount": number, "due_date": "YYYY-MM-DD", "vendor": "..."}}
If the document is a contract, also include:
- "structured_data": {{"parties": ["..."], "duration": "...", "obligations": "..."}}
Otherwise, include "structured_data": {{}}

Document content:
{text_content[:4000]}  # Limit context size

Example format:
{{"summary": "...", "key_points": ["..."], "document_type": "...", "entities": ["..."], "structured_data": {{}}}}""",
            }
        ]

    try:
        response_text = _call_openrouter_with_backoff(model, messages)
        parsed = _parse_json(response_text)

        result = {
            "summary": parsed.get("summary", ""),
            "key_points": json.dumps(parsed.get("key_points", [])),
            "document_type": parsed.get("document_type", "unknown"),
            "entities": json.dumps(parsed.get("entities", [])),
            "raw_response": response_text,
            "structured_data": json.dumps(parsed.get("structured_data", {})),
        }

        # Second pass: Risk detection (non-blocking, failure won't break main extraction)
        try:
            risk_messages_content: str | list[dict[str, Any]]
            if is_scanned_pdf:
                # For scanned PDFs, reuse the vision-based risk detection
                risk_messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": """Analyze this scanned document and identify risks, inconsistencies, or missing critical information.
Return ONLY a valid JSON object with this exact key:
- "risks": a list of strings describing risks found (empty list if none)

Example format:
{"risks": ["Risk 1", "Risk 2"]}""",
                            },
                        ],
                    }
                ]
                # Add the page images to risk messages
                pages_base64 = text_content.split("===MULTI_PAGE_PDF===", 1)[1].split("|||")
                for b64_page in pages_base64:
                    if b64_page.strip():
                        risk_messages[0]["content"].append(
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{b64_page}"},
                            }
                        )
            else:
                risk_messages = [
                    {
                        "role": "user",
                        "content": f"""Analyze this document and identify risks, inconsistencies, or missing critical information.
Return ONLY a valid JSON object with this exact key:
- "risks": a list of strings describing risks found (empty list if none)

Document content:
{text_content[:4000]}

Example format:
{{"risks": ["Risk 1", "Risk 2"]}}""",
                    }
                ]
            risk_response = _call_openrouter_with_backoff(model, risk_messages)
            risk_parsed = _parse_json(risk_response)
            result["risks"] = json.dumps(risk_parsed.get("risks", []))
        except Exception:
            result["risks"] = json.dumps([])

        return result

    except Exception as exc:
        raise RuntimeError(f"OpenRouter extraction failed: {exc}") from exc


# ---------------------------------------------------------------------------
# Feature 1: Ask Across Documents
# ---------------------------------------------------------------------------

def ask_documents(documents: list[dict[str, Any]], question: str) -> dict[str, Any]:
    """
    Answer a question using ONLY the provided document context.
    documents: list of dicts with keys: filename, summary, key_points
    question: natural language question
    Returns dict with: answer, sources
    """
    model = "google/gemini-2.5-flash"

    docs_context = "\n\n".join(
        f"Document: {doc['filename']}\nSummary: {doc['summary']}\nKey Points: {doc.get('key_points', 'N/A')}"
        for doc in documents
    )

    messages = [
        {
            "role": "user",
            "content": f"""You are a document analysis assistant. Using ONLY the provided document context below, answer the user's question.

If the answer cannot be found in the provided context, say "I cannot find this information in the provided documents."

Return ONLY a valid JSON object (no markdown fences) with these exact keys:
- "answer": your answer based on the context
- "sources": a list of objects with:
  - "document": the filename of the source document
  - "evidence": the specific text from that document that supports this answer

Documents:
{docs_context}

Question: {question}

Example format:
{{"answer": "Based on the documents provided, ...", "sources": [{{"document": "file1.pdf", "evidence": "..."}}]}}""",
        }
    ]

    try:
        response_text = _call_openrouter_with_backoff(model, messages)
        parsed = _parse_json(response_text)

        return {
            "answer": parsed.get("answer", "Could not generate an answer."),
            "sources": parsed.get("sources", []),
        }
    except Exception as exc:
        raise RuntimeError(f"Ask documents failed: {exc}") from exc


# ---------------------------------------------------------------------------
# Feature 2: Insight Engine
# ---------------------------------------------------------------------------

def generate_insights(documents: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Analyze multiple documents together to extract cross-document insights.
    documents: list of dicts with keys: filename, summary, key_points, entities
    Returns dict with: issues, entities, trends, risks
    """
    model = "google/gemini-2.5-flash"

    docs_context = "\n\n".join(
        f"Document: {doc['filename']}\nSummary: {doc['summary']}\nKey Points: {doc.get('key_points', 'N/A')}\nEntities: {doc.get('entities', 'N/A')}"
        for doc in documents
    )

    messages = [
        {
            "role": "user",
            "content": f"""You are a document insight engine. Analyze the following documents together and extract cross-document insights.

Return ONLY a valid JSON object (no markdown fences) with these exact keys:
- "issues": a list of top recurring issues (max 5), each as a string
- "entities": a dict grouping entities by category (e.g. {{"people": [...], "organizations": [...], "dates": [...], "amounts": [...]}})
- "trends": a list of trends or patterns observed across documents (max 5), each as a string
- "risks": a list of risks observed (max 5), each as a string

Documents:
{docs_context}

Example format:
{{"issues": ["Issue 1", "Issue 2"], "entities": {{"people": [], "organizations": []}}, "trends": ["Trend 1"], "risks": ["Risk 1"]}}""",
        }
    ]

    try:
        response_text = _call_openrouter_with_backoff(model, messages)
        parsed = _parse_json(response_text)

        return {
            "issues": parsed.get("issues", []),
            "entities": parsed.get("entities", {}),
            "trends": parsed.get("trends", []),
            "risks": parsed.get("risks", []),
        }
    except Exception as exc:
        raise RuntimeError(f"Insight generation failed: {exc}") from exc


# ---------------------------------------------------------------------------
# Feature 3: Risk Detection (standalone)
# ---------------------------------------------------------------------------

def detect_risks(text_content: str) -> list[str]:
    """
    Standalone risk detection pass on a document text.
    Returns list of risk strings.
    """
    model = "google/gemini-2.5-flash"

    messages = [
        {
            "role": "user",
            "content": f"""Analyze this document and identify risks, inconsistencies, or missing critical information.
Return ONLY a valid JSON object (no markdown fences) with this exact key:
- "risks": a list of strings describing risks found (empty list if none)

Document content:
{text_content[:4000]}

Example format:
{{"risks": ["Risk 1", "Risk 2"]}}""",
        }
    ]

    try:
        response_text = _call_openrouter_with_backoff(model, messages)
        parsed = _parse_json(response_text)
        return parsed.get("risks", [])
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Feature 7: Semantic Search
# ---------------------------------------------------------------------------

def semantic_search(query: str, documents: list[dict[str, Any]]) -> list[str]:
    """
    Use Gemini to find the most relevant documents for a natural language query.
    documents: list of dicts with keys: filename, summary, key_points
    Returns list of matching filenames.
    """
    model = "google/gemini-2.5-flash"

    docs_context = "\n\n".join(
        f"Filename: {doc['filename']}\nSummary: {doc['summary']}\nKey Points: {doc.get('key_points', 'N/A')}"
        for doc in documents
    )

    messages = [
        {
            "role": "user",
            "content": f"""You are a semantic document search engine. Given a user query and a list of documents, return the most relevant documents.

Return ONLY a valid JSON object (no markdown fences) with this exact key:
- "matches": a list of filenames (strings) that are most relevant to the query (can be empty if no matches)

Documents:
{docs_context}

Query: {query}

Example format:
{{"matches": ["filename1.pdf", "filename2.pdf"]}}""",
        }
    ]

    try:
        response_text = _call_openrouter_with_backoff(model, messages)
        parsed = _parse_json(response_text)
        return parsed.get("matches", [])
    except Exception as exc:
        raise RuntimeError(f"Semantic search failed: {exc}") from exc


# ---------------------------------------------------------------------------
# Legacy: Report Generation
# ---------------------------------------------------------------------------

def generate_report_content(
    selected_summaries: list[dict[str, Any]],
    report_sections: list[str] | None = None,
) -> str:
    """
    Generate a structured report from multiple document summaries.
    selected_summaries: list of dicts with 'filename', 'summary', 'key_points'
    Returns the report as Markdown text.
    """
    model = "google/gemini-2.5-flash"

    sections = report_sections or [
        "Overview",
        "Key Findings",
        "Per-Document Breakdown",
        "Recommendations/Next Steps",
    ]

    docs_context = "\n\n".join(
        f"Document: {item['filename']}\nSummary: {item['summary']}\nKey Points: {item.get('key_points', 'N/A')}"
        for item in selected_summaries
    )

    messages = [
        {
            "role": "user",
            "content": f"""You are a professional report writer. Using ONLY the information provided below, generate a structured report with the following sections: {', '.join(sections)}.

For each section:
- Overview: 2-3 sentences summarizing the collective content
- Key Findings: bullet points of the most important facts across all documents
- Per-Document Breakdown: for each document, summarize its contribution
- Recommendations/Next Steps: actionable next steps based on the content

Documents:
{docs_context}

Return the report as clean Markdown. Do not include any JSON, code fences, or meta-commentary.""",
        }
    ]

    try:
        response_text = _call_openrouter_with_backoff(model, messages)
        return response_text

    except Exception as exc:
        raise RuntimeError(f"OpenRouter report generation failed: {exc}") from exc