"""
RAG node for retrieving evidence before final Sonnet reasoning.

Why this module exists
----------------------
This node calls the separate RAG service in retrieve-only mode. The RAG service
returns evidence; it does not write the final answer for the gateway.

The current RAG corpus is focused on skin cancer and related skin topics, and
the gateway is built so the corpus can grow later. For image routes, this node
can combine the user's question with HTAN segmentation context or Haiku vision
context before retrieval.

Pipeline trace
--------------
  rag_only:
    intent_router -> rag -> medical_llm -> report

  htan_rag:
    intent_router -> htan -> rag -> medical_llm -> report

  vision_rag:
    intent_router -> vision -> rag -> medical_llm -> report

State written here
------------------
  rag_query_used : exact query sent to the RAG service
  retrieved_docs : normalized evidence documents
  rag_context    : readable evidence/status context for Sonnet
"""

from __future__ import annotations

import logging
from typing import Any

from app import clients, config
from app.nodes.quality_gate import gate_rag
from app.state import MediLinkState

logger = logging.getLogger("medilink.gateway")

# Gateway intent -> RAG-service intent.
#
# Keep this conservative. The RAG service can store a broad skin-cancer corpus,
# but its API may still expect a smaller set of retrieval modes. Unknown intent
# values fall back to "general" instead of breaking retrieval.
_INTENT_MAP = {
    "symptoms": "symptoms",
    "symptom_description": "symptoms",
    "treatment": "treatment",
    "mechanism": "mechanism",
    "diagnosis": "diagnosis",
    "prognosis": "prognosis",
    "medication": "medication",
    "second_opinion": "general",
    "followup": "general",
    "skin_cancer": "general",
    "skin_lesion": "symptoms",
    "general": "general",
}


def _clean_string(value: Any, default: str = "") -> str:
    """Normalize arbitrary evidence fields into strings."""
    if value is None:
        return default
    text = str(value).strip()
    return text or default


def _normalize_doc(doc: dict[str, Any], index: int) -> dict[str, str]:
    """
    Normalize one evidence document from the RAG service.

    RAG service responses can evolve. Downstream modules should get predictable
    keys even if a document is missing PMID/title/text/source.
    """
    return {
        "pmid": _clean_string(doc.get("pmid")),
        "source": _clean_string(doc.get("source"), f"rag-{index}"),
        "title": _clean_string(doc.get("title"), "Untitled evidence"),
        "text": _clean_string(doc.get("text")),
    }


def _normalize_evidence(evidence: Any) -> list[dict[str, str]]:
    """Normalize a RAG evidence list and drop entries with no usable text."""
    if not isinstance(evidence, list):
        return []

    normalized = []
    for i, doc in enumerate(evidence, 1):
        if not isinstance(doc, dict):
            continue
        cleaned = _normalize_doc(doc, i)
        if cleaned["text"]:
            normalized.append(cleaned)
    return normalized


def _format_context(evidence: list[dict[str, str]]) -> str:
    """
    Render normalized evidence into the citation context Sonnet receives.

    PMID evidence keeps the [PMID:...] tag. Non-PMID sources get a stable
    [SOURCE:...] tag so Sonnet can still cite supplied context.
    """
    blocks = []
    for i, doc in enumerate(evidence, 1):
        pmid = doc.get("pmid", "")
        tag = f"[PMID:{pmid}]" if pmid else f"[SOURCE:{doc.get('source', '?')}-{i}]"
        blocks.append(f"{tag} {doc.get('title', '')}\n{doc.get('text', '')}".strip())
    return "\n\n---\n\n".join(blocks)


def _status_context(reason: str) -> str:
    """Return clear RAG status context instead of an unexplained empty string."""
    return f"RAG evidence was not available: {reason}"


def _build_query(state: MediLinkState) -> str:
    """
    Build the final retrieval query.

    Priority:
      1. Haiku router's rag_query, if present
      2. raw user message
      3. generic image-context query if the request is image-only

    HTAN and vision context are appended so skin-cancer/skin-topic retrieval can
    use both user text and image/document context when present.
    """
    question = (state.get("user_message") or "").strip()
    cv_text = (state.get("cv_text") or "").strip()
    vision_text = (state.get("vision_text") or "").strip()

    query = (state.get("rag_query") or question).strip()
    if not query and (cv_text or vision_text):
        query = "skin cancer or skin-related medical assessment based on uploaded image context"

    if cv_text:
        query = f"{query}\n\nHTAN context:\n{cv_text}".strip()
    if vision_text:
        query = f"{query}\n\nVision/document context:\n{vision_text}".strip()

    return query


def rag_node(state: MediLinkState) -> dict:
    """
    LangGraph node entry point.

    Reads:
      user_message, intent, rag_query, cv_text, vision_text,
      patient_mode, conversation_history

    Writes:
      rag_query_used, retrieved_docs, rag_context
    """
    question = (state.get("user_message") or "").strip()
    query = _build_query(state)
    intent = _INTENT_MAP.get(state.get("intent", "general"), "general")
    patient_mode = state.get("patient_mode")
    if patient_mode is None:
        patient_mode = config.DEFAULT_PATIENT_MODE

    if not query:
        reason = "no user question or image context was available for retrieval"
        return {
            "rag_query_used": "",
            "retrieved_docs": [],
            "rag_context": _status_context(reason),
        }

    try:
        result = clients.call_rag(
            query,
            intent=intent,
            patient_mode=patient_mode,
            retrieve_only=True,
            history=state.get("conversation_history") or [],
            top_k=config.RAG_TOP_K,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("RAG service call failed")
        return {
            "error": f"RAG service error: {exc}",
            "rag_query_used": query,
            "retrieved_docs": [],
            "rag_context": _status_context(f"service error: {exc}"),
        }

    evidence = _normalize_evidence(result.get("evidence", []))
    if not evidence:
        return {
            "rag_query_used": query,
            "retrieved_docs": [],
            "rag_context": _status_context("the RAG service returned no usable evidence"),
        }

    # Gate the same final query that was sent to retrieval, including HTAN or
    # vision context. This makes the gate judge relevance against the full task.
    if not gate_rag(evidence, query):
        logger.info("RAG output dropped by quality gate.")
        return {
            "rag_query_used": query,
            "retrieved_docs": [],
            "rag_context": _status_context("retrieved evidence was dropped by the quality gate"),
        }

    return {
        "rag_query_used": query,
        "retrieved_docs": evidence,
        "rag_context": _format_context(evidence),
    }
