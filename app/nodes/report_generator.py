"""
Report generator node.

Why this module exists
----------------------
Earlier nodes produce state:
  - intent_router.py writes route/intent/safety/router_reason
  - htan_node.py writes cv_result/cv_text
  - vision_node.py writes vision_result/vision_text
  - rag_node.py writes retrieved_docs/rag_context/rag_query_used
  - medical_llm.py writes draft_answer/citations

This node is the final formatter. It does not call any model or downstream
service. It returns:
  - final_answer: the user-facing answer
  - doctor_report: a structured trace/debug/clinician report

The doctor_report is useful because the API answer alone hides how the pipeline
arrived there. This report shows which route ran, what evidence was retrieved,
whether any service failed, and whether citations were valid.
"""

from __future__ import annotations

from typing import Any

from app.state import MediLinkState

_PREVIEW_CHARS = 1000


def _preview(value: Any, limit: int = _PREVIEW_CHARS) -> str:
    """Return a compact string preview for long trace fields."""
    if value is None:
        return ""
    text = str(value).strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."


def _image_summary(cv: dict | None) -> dict | None:
    """Return the clinician-facing subset of normalized HTAN output."""
    if not cv:
        return None
    return {
        "target_detected": cv.get("target_detected"),
        "modality": cv.get("modality"),
        "model": cv.get("model"),
        "segmented_area_percent": cv.get("segmented_area_percent"),
        "relative_size": cv.get("relative_size"),
        "location": cv.get("location"),
        "num_components": cv.get("num_components"),
        "largest_component_span_px": cv.get("largest_component_span_px"),
        "severity_estimate": cv.get("severity_estimate"),
    }


def _normalize_citations(citations: dict | None) -> dict:
    """
    Keep citation shape stable even when generation fails or a node omits fields.
    """
    citations = citations or {}
    fabricated_pmids = citations.get("fabricated_pmids") or []
    fabricated_sources = citations.get("fabricated_sources") or []
    return {
        "cited_pmids": citations.get("cited_pmids") or [],
        "fabricated_pmids": fabricated_pmids,
        "cited_sources": citations.get("cited_sources") or [],
        "fabricated_sources": fabricated_sources,
        "all_valid": citations.get("all_valid", not fabricated_pmids and not fabricated_sources),
        "num_citations": citations.get("num_citations", 0),
    }


def _evidence_summary(docs: list[dict]) -> list[dict]:
    """Return lightweight evidence metadata without copying full passage text."""
    summary = []
    for index, doc in enumerate(docs, 1):
        text = str(doc.get("text", "") or "")
        summary.append({
            "index": index,
            "pmid": doc.get("pmid", ""),
            "source": doc.get("source", ""),
            "title": doc.get("title", ""),
            "has_text": bool(text.strip()),
        })
    return summary


def _rag_context_status(rag_context: str) -> str:
    """
    Classify RAG context for quick report scanning.

    rag_node.py writes status text that starts with "RAG evidence was not
    available:" when retrieval failed, returned no docs, or was dropped.
    """
    if not rag_context:
        return "missing"
    if rag_context.startswith("RAG evidence was not available:"):
        return "unavailable"
    return "available"


def report_generator_node(state: MediLinkState) -> dict:
    """
    LangGraph node entry point.

    Reads the final state from router/HTAN/vision/RAG/medical_llm and writes:
      final_answer
      doctor_report
    """
    draft = (state.get("draft_answer") or "").strip()
    cv = state.get("cv_result")
    vision = state.get("vision_result")
    docs = state.get("retrieved_docs") or []
    citations = _normalize_citations(state.get("citations"))
    rag_context = state.get("rag_context") or ""

    final = draft or "I don't have enough information to answer that. Please consult a healthcare professional."

    fabricated_citations = citations["fabricated_pmids"] + citations["fabricated_sources"]

    doctor_report = {
        # Router trace.
        "intent": state.get("intent"),
        "route": state.get("route"),
        "safety_level": state.get("safety_level"),
        "router_reason": state.get("router_reason"),

        # Error trace from any recoverable node failure.
        "error": state.get("error"),

        # HTAN trace.
        "image_findings": _image_summary(cv),
        "cv_text_preview": _preview(state.get("cv_text")),

        # Vision trace.
        "vision_findings": vision,
        "vision_text_preview": _preview(state.get("vision_text")),

        # Triage trace.
        "router_triage_questions": state.get("router_triage_questions") or [],
        "triage_questions": state.get("triage_questions") or [],

        # RAG trace.
        "rag_query_used": state.get("rag_query_used"),
        "rag_context_status": _rag_context_status(rag_context),
        "rag_context_preview": _preview(rag_context),
        "evidence_count": len(docs),
        "evidence": _evidence_summary(docs),

        # Medical LLM trace.
        "answer": draft,
        "citations": citations,

        # Flags make the report easy to scan in clients/logs.
        "flags": {
            "fabricated_citations": fabricated_citations,
            "fabricated_pmids": citations["fabricated_pmids"],
            "fabricated_sources": citations["fabricated_sources"],
            "no_evidence": len(docs) == 0,
            "rag_unavailable": _rag_context_status(rag_context) != "available",
            "has_error": bool(state.get("error")),
        },
    }

    return {"final_answer": final, "doctor_report": doctor_report}
