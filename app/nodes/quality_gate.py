"""
Reusable quality gates for HTAN and RAG outputs.

Trace:
  1. htan_node.py calls gate_htan() after HTAN segmentation returns.
  2. rag_node.py calls gate_rag() after evidence retrieval returns.
  3. The gate asks a fast LLM for one word: pass or drop.
  4. If all gate providers fail, llm.call_with_fallback(..., fail_open=True)
     returns "pass" so a provider outage does not block the whole pipeline.

The gate does not diagnose and does not judge medical truth. It only checks
whether the upstream output is coherent and useful enough to pass to Sonnet.
"""

from __future__ import annotations

import logging

from app.llm import call_with_fallback

logger = logging.getLogger("medilink.gateway")

_GATE_SYSTEM = (
    "You are a quality gate for a medical AI pipeline. "
    "Reply with one word only: pass or drop. Nothing else."
)

_HTAN_PROMPT = """A medical image segmentation model returned this result for a patient interaction.

Question/context: {question}
Modality: {modality}
Segmentation output:
  detected: {detected}
  area: {area}% of image
  relative size: {size}
  location: {location}
  components: {components}
  severity estimate: {severity}

Does this segmentation output make clinical sense and add useful information
given the patient's question and the imaging modality?
Reply: pass or drop"""

_RAG_PROMPT = """A biomedical retrieval pipeline returned these passages. This is
the same evidence context the answer model will receive.

Question: {question}

Evidence context ({n_docs} passages):
{context}

Is this context, taken together, sufficient and relevant to give a medically
accurate answer to the question? Judge the whole set, not any single passage.
Reply: pass or drop"""

# Gate prompt budget. The answer model receives all formatted evidence from
# rag_node.py; the gate sees a bounded preview so the pass/drop call stays cheap.
_GATE_CONTEXT_DOCS = 5
_GATE_DOC_CHARS = 800


def _parse(raw: str) -> bool:
    """Parse one-word gate output. True means pass; False means drop."""
    clean = raw.strip().lower().split()[0] if raw.strip() else "pass"
    return clean != "drop"


def gate_htan(cv_result: dict, question: str) -> bool:
    """
    Evaluate HTAN segmentation output for coherence.

    Returns:
      True  -> keep cv_text/cv_result and continue the graph
      False -> drop HTAN output and continue with clear failure context
    """
    if not cv_result:
        return False

    prompt = _HTAN_PROMPT.format(
        question=question or "image analysis",
        modality=cv_result.get("modality", "unknown"),
        detected=cv_result.get("target_detected", False),
        area=cv_result.get("segmented_area_percent", 0),
        size=cv_result.get("relative_size", "unknown"),
        location=cv_result.get("location", "unknown"),
        components=cv_result.get("num_components", 0),
        severity=cv_result.get("severity_estimate", "unknown"),
    )

    raw = call_with_fallback(prompt, system=_GATE_SYSTEM, max_tokens=5, fail_open=True)
    result = _parse(raw)
    logger.info("HTAN gate: %s (raw=%r)", "PASS" if result else "DROP", raw)
    return result


def gate_rag(docs: list[dict], question: str) -> bool:
    """
    Evaluate retrieved evidence for relevance/sufficiency before Sonnet.

    Returns:
      True  -> keep evidence and send it to medical_llm.py
      False -> drop evidence and send a clear RAG status instead
    """
    if not docs:
        return False

    blocks = []
    for i, doc in enumerate(docs[:_GATE_CONTEXT_DOCS], 1):
        pmid = doc.get("pmid", "")
        tag = f"[PMID:{pmid}]" if pmid else f"[SOURCE:{doc.get('source', '?')}-{i}]"
        text = str(doc.get("text", ""))[:_GATE_DOC_CHARS]
        blocks.append(f"{tag} {doc.get('title', '')}\n{text}".strip())
    context = "\n\n---\n\n".join(blocks)

    prompt = _RAG_PROMPT.format(
        question=question or "medical question",
        n_docs=len(docs),
        context=context,
    )

    raw = call_with_fallback(prompt, system=_GATE_SYSTEM, max_tokens=5, fail_open=True)
    result = _parse(raw)
    logger.info("RAG gate: %s on %d docs (raw=%r)", "PASS" if result else "DROP", len(docs), raw)
    return result
