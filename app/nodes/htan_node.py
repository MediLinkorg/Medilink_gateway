"""
HTAN node for segmentation-capable medical images.

Why this module exists
----------------------
HTAN is the gateway's segmentation branch. It should only receive image
modalities that the HTAN service is validated to process:
  - dermoscopy
  - histology
  - microscopy

Medical documents, radiology, and other general medical images belong to
vision_node.py instead.

Pipeline trace
--------------
  1. intent_router.py chooses route htan_only or htan_rag and writes modality.

  2. graph.py sends the shared MediLinkState to this node.

  3. this node validates image_bytes and modality.

  4. this node calls the HTAN service through clients.call_htan().

  5. this node normalizes the HTAN JSON into a stable cv_result contract.

  6. quality_gate.py checks whether the segmentation output is useful.

  7. this node writes:
       cv_result = normalized HTAN JSON, or None if dropped/failed
       cv_text   = readable segmentation context for RAG/Sonnet

  8. graph.py decides the next step:
       htan_only -> medical_llm
       htan_rag  -> rag -> medical_llm

Important boundary
------------------
This node does not diagnose. It only converts HTAN segmentation output into
context that later modules can use.
"""

from __future__ import annotations

import logging
from typing import Any

from app import clients, config
from app.nodes.quality_gate import gate_htan
from app.state import MediLinkState

logger = logging.getLogger("medilink.gateway")


def _clean_string(value: Any, default: str = "") -> str:
    """Normalize arbitrary values into readable strings."""
    if value is None:
        return default
    text = str(value).strip()
    return text or default


def _clean_float(value: Any, default: float = 0.0) -> float:
    """Normalize numeric HTAN fields without crashing on malformed JSON."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clean_int(value: Any, default: int = 0) -> int:
    """Normalize integer HTAN fields without crashing on malformed JSON."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _normalize_cv_result(raw: dict[str, Any], *, modality: str) -> dict[str, Any]:
    """
    Convert the HTAN service response into a stable contract.

    Downstream modules should not depend on every optional key being present in
    the HTAN service response. This function keeps expected fields predictable.
    """
    target_detected = bool(raw.get("target_detected"))
    return {
        "target_detected": target_detected,
        "modality": _clean_string(raw.get("modality"), modality),
        "model": _clean_string(raw.get("model"), "htan"),
        "segmented_area_percent": _clean_float(raw.get("segmented_area_percent"), 0.0),
        "relative_size": _clean_string(raw.get("relative_size"), "unknown"),
        "location": _clean_string(raw.get("location"), "unknown"),
        "num_components": _clean_int(raw.get("num_components"), 0 if not target_detected else 1),
        "largest_component_span_px": _clean_float(raw.get("largest_component_span_px"), 0.0),
        "severity_estimate": _clean_string(raw.get("severity_estimate"), "unknown"),
    }


def cv_result_to_text(cv: dict[str, Any] | None) -> str:
    """Render normalized HTAN JSON into context text for RAG and Sonnet."""
    if not cv:
        return "HTAN image analysis: no segmentation result is available."
    if not cv.get("target_detected"):
        return "HTAN image analysis: no distinct region was segmented in the image."
    return (
        f"HTAN image analysis ({cv.get('modality', 'image')}, model {cv.get('model', 'htan')}): "
        f"a {cv.get('relative_size', 'unknown')} region was segmented covering "
        f"{cv.get('segmented_area_percent', 0.0)}% of the image, "
        f"located {cv.get('location', 'unknown')}, "
        f"with {cv.get('num_components', 0)} distinct component(s), "
        f"largest span {cv.get('largest_component_span_px', 0.0)} px, "
        f"and relative screening priority '{cv.get('severity_estimate', 'unknown')}'."
    )


def _failure_text(reason: str) -> str:
    """Create a clear downstream context string for recoverable HTAN failures."""
    return f"HTAN image analysis was not available: {reason}"


def htan_node(state: MediLinkState) -> dict:
    """
    LangGraph node entry point.

    Reads:
      image_bytes, modality, user_message

    Writes:
      cv_result, cv_text, and error on recoverable failures
    """
    image_bytes = state.get("image_bytes")
    modality = state.get("modality")
    question = (state.get("user_message") or "").strip()

    # The graph should only route here with an image. Keep this defensive guard
    # so future graph changes do not crash the service.
    if not image_bytes:
        reason = "no image bytes in state"
        return {
            "error": f"htan_node: {reason}.",
            "cv_result": None,
            "cv_text": _failure_text(reason),
        }

    # Do not silently default to dermoscopy. The router must provide a modality
    # that HTAN supports; otherwise this request belongs to vision_node.py or
    # should be rejected before reaching this node.
    if modality not in config.SUPPORTED_MODALITIES:
        reason = f"unsupported or missing modality: {modality!r}"
        return {
            "error": f"htan_node: {reason}.",
            "cv_result": None,
            "cv_text": _failure_text(reason),
        }

    try:
        raw_cv = clients.call_htan(
            image_bytes,
            modality=modality,
            tta=config.HTAN_TTA,
            image_media_type=state.get("image_media_type"),
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("HTAN service call failed")
        reason = f"service error: {exc}"
        return {
            "error": f"HTAN service error: {exc}",
            "cv_result": None,
            "cv_text": _failure_text(reason),
        }

    cv = _normalize_cv_result(raw_cv, modality=modality)

    # Haiku evaluates whether this segmentation is coherent and useful for the
    # user's question. The gate fails open if all configured LLM gate providers
    # are unavailable, as implemented in quality_gate.py.
    if not gate_htan(cv, question):
        logger.info("HTAN output dropped by quality gate.")
        return {
            "cv_result": None,
            "cv_text": _failure_text("segmentation output was dropped by the quality gate"),
        }

    return {"cv_result": cv, "cv_text": cv_result_to_text(cv)}
