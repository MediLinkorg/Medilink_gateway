"""
Medical LLM node - final Sonnet clinical reasoning.

Why this module exists
----------------------
Earlier nodes gather context:
  - htan_node.py writes cv_text for HTAN segmentation context.
  - vision_node.py writes vision_text for documents/radiology/general images.
  - rag_node.py writes rag_context and retrieved_docs for evidence.

This node combines the available context and asks Sonnet to write the draft
answer. It does not format the final API response; report_generator.py does
that after this node finishes.

Pipeline trace
--------------
  htan_only  : htan -> medical_llm -> report
  vision_only: vision -> medical_llm -> report
  rag_only   : rag -> medical_llm -> report
  htan_rag   : htan -> rag -> medical_llm -> report
  vision_rag : vision -> rag -> medical_llm -> report

State written here
------------------
  draft_answer : Sonnet's answer before final formatting
  citations    : citation validation report
  error        : recoverable generation error, if Sonnet fails
"""

from __future__ import annotations

import logging
import re

from app import config, llm
from app.state import MediLinkState

logger = logging.getLogger("medilink.gateway")

CITATION_SYSTEM = """
You are MediLink, a general medical clinical reasoning assistant.

Use supplied HTAN image findings, vision/document context, and retrieved
evidence when available. If retrieved evidence is unavailable or was dropped,
you may still answer from general medical knowledge, but clearly acknowledge
the limitation and do not fabricate citations.

Rules:
1. Do not invent statistics, source identifiers, test results, or image findings.
2. Cite claims with supplied [PMID:...] or [SOURCE:...] identifiers when retrieved
   evidence is provided.
3. If the available information is incomplete, ask for the missing details instead
   of over-concluding.
4. Provide focused clinical assessment and differential reasoning when enough
   information is available.
5. Treat HTAN segmentation as image-analysis context, not a standalone diagnosis.
6. Be clear, concise, and clinically careful.
""".strip()


def _empty_citations() -> dict:
    """Citation report used when no answer/evidence citation check is possible."""
    return {
        "cited_pmids": [],
        "fabricated_pmids": [],
        "cited_sources": [],
        "fabricated_sources": [],
        "all_valid": True,
        "num_citations": 0,
    }


def _source_id(doc: dict, index: int) -> str:
    """Return the SOURCE identifier used by rag_node._format_context()."""
    return f"{doc.get('source', '?')}-{index}"


def _verify_citations(answer: str, docs: list[dict]) -> dict:
    """
    Validate citations Sonnet used against retrieved_docs.

    PMID citations are valid when the PMID exists in retrieved_docs.
    SOURCE citations are valid when they match the generated source tag pattern:
      [SOURCE:{source}-{1-based index}]
    """
    valid_pmids = {str(d.get("pmid", "")).strip() for d in docs if d.get("pmid")}
    valid_sources = {_source_id(d, i) for i, d in enumerate(docs, 1) if not d.get("pmid")}

    cited_pmids = set(re.findall(r"\[PMID:(\d+)\]", answer))
    cited_sources = set(re.findall(r"\[SOURCE:([^\]]+)\]", answer))

    fabricated_pmids = sorted(cited_pmids - valid_pmids)
    fabricated_sources = sorted(cited_sources - valid_sources)

    return {
        "cited_pmids": sorted(cited_pmids),
        "fabricated_pmids": fabricated_pmids,
        "cited_sources": sorted(cited_sources),
        "fabricated_sources": fabricated_sources,
        "all_valid": not fabricated_pmids and not fabricated_sources,
        "num_citations": len(re.findall(r"\[(?:PMID|SOURCE):", answer)),
    }


def _rag_context_section(rag_context: str) -> str:
    """
    Label RAG context accurately.

    rag_node.py now returns clear status text when evidence is unavailable,
    empty, or dropped. That status should not be mislabeled as evidence.
    """
    if rag_context.startswith("RAG evidence was not available:"):
        return f"RAG STATUS:\n{rag_context}"
    return f"RETRIEVED EVIDENCE:\n{rag_context}"


def _build_prompt(state: MediLinkState) -> str:
    """
    Build the Sonnet user prompt from whatever context exists in state.

    This helper keeps prompt construction testable and traceable. It does not
    call the model.
    """
    question = (state.get("user_message") or "").strip()
    cv_text = (state.get("cv_text") or "").strip()
    vision_text = (state.get("vision_text") or "").strip()
    rag_context = (state.get("rag_context") or "").strip()

    sections = []
    if cv_text:
        sections.append(f"HTAN IMAGE FINDINGS:\n{cv_text}")
    if vision_text:
        sections.append(f"VISION / DOCUMENT CONTEXT:\n{vision_text}")
    if rag_context:
        sections.append(_rag_context_section(rag_context))

    if not question and (cv_text or vision_text):
        question = "Explain the uploaded medical context and what should be considered next."

    sections.append(
        f"QUESTION: {question}\n\n"
        "Write a clear, patient-friendly clinical assessment. Use supplied "
        "evidence, image findings, and document context where available. Cite "
        "medical claims when source identifiers are supplied. If information is "
        "missing, ask for the missing details instead of over-concluding."
    )
    return "\n\n".join(sections)


def medical_llm_node(state: MediLinkState) -> dict:
    """
    LangGraph node entry point.

    Reads:
      user_message, cv_text, vision_text, rag_context, retrieved_docs,
      conversation_history

    Writes:
      draft_answer, citations, and error on recoverable Sonnet failure
    """
    question = (state.get("user_message") or "").strip()
    cv_text = (state.get("cv_text") or "").strip()
    vision_text = (state.get("vision_text") or "").strip()
    rag_context = (state.get("rag_context") or "").strip()
    docs = state.get("retrieved_docs") or []

    # If the graph reaches this node with no text and no usable context, do not
    # spend a Sonnet call. Ask the user for usable input.
    if not question and not cv_text and not vision_text and not rag_context:
        return {
            "draft_answer": (
                "I couldn't get a clear read on the input, and there's no question "
                "to work from. Could you send a clearer image or describe what you'd "
                "like to know?"
            ),
            "citations": _empty_citations(),
        }

    prompt = _build_prompt(state)
    history = state.get("conversation_history") or []

    try:
        answer = llm.call_sonnet(
            prompt,
            system=CITATION_SYSTEM,
            max_tokens=config.SONNET_MAX_TOKENS,
            history=history,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Sonnet generation failed")
        return {
            "error": f"Generation error: {exc}",
            "draft_answer": "I wasn't able to generate an answer just now. Please try again.",
            "citations": _empty_citations(),
        }

    return {"draft_answer": answer, "citations": _verify_citations(answer, docs)}
