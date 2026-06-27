"""
Intent router node.

Haiku owns medical routing decisions: safety classification, triage need,
follow-up questions, and broad route selection. Python only normalizes the
structured decision into routes the LangGraph can execute safely.
"""

from __future__ import annotations

import base64
import json
import logging
import re
from typing import Any

import app.llm as _llm

from app import config
from app.state import MediLinkState

logger = logging.getLogger("medilink.gateway")

ROUTES = {
    "direct",
    "triage_question",
    "rag_only",
    "htan_only",
    "htan_rag",
    "vision_only",
    "vision_rag",
}

VISION_IMAGE_TYPES = {
    "medical_document",
    "xray",
    "ct",
    "mri",
    "ultrasound",
    "radiology",
    "general_medical_image",
}

IMAGE_TYPES = {
    *config.SUPPORTED_MODALITIES,
    *VISION_IMAGE_TYPES,
    "blurry",
    "non_medical",
    "unknown",
}

ROUTER_SYSTEM = """
You are MediLink's medical intake router for a general medical assistant.

Return valid JSON only. Do not wrap it in markdown.

Decide:
- whether this is emergency/crisis content,
- whether symptom triage/follow-up questions are needed before answering,
- whether evidence retrieval is useful,
- whether HTAN segmentation is appropriate,
- whether vision extraction/description is needed for a document, radiology,
  or other general medical image.

HTAN is allowed only for the configured HTAN-supported modalities.
Medical documents, X-ray/CT/MRI/ultrasound/radiology, and other medical images
must use vision routes, not HTAN.

Allowed routes:
- direct
- triage_question
- rag_only
- htan_only
- htan_rag
- vision_only
- vision_rag

Use triage_question when the user describes symptoms but key clinical details
are missing and a focused assessment would be premature. Ask concise follow-up
questions that collect the missing symptom information.

JSON schema:
{
  "intent": "string",
  "safety_level": "none|urgent|emergency|crisis",
  "route": "one allowed route",
  "needs_triage": true,
  "missing_questions": ["question"],
  "needs_rag": true,
  "needs_htan": false,
  "needs_vision": false,
  "image_type": null,
  "modality": null,
  "rag_query": null,
  "direct_answer": null,
  "reason": "short reason"
}
""".strip()


def _json_object(raw: str) -> dict[str, Any]:
    """Parse a JSON object, tolerating accidental prose around it."""
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if not match:
            return {}
        try:
            parsed = json.loads(match.group(0))
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}


def _as_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return default


def _as_questions(value: Any) -> list[str]:
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    if not isinstance(value, list):
        return []
    questions = []
    for item in value:
        if isinstance(item, str) and item.strip():
            questions.append(item.strip())
    return questions[:5]


def _direct_answer_for(decision: dict[str, Any]) -> str:
    safety = str(decision.get("safety_level") or "none").lower()
    answer = str(decision.get("direct_answer") or "").strip()
    if safety == "crisis":
        return answer or config.CRISIS_RESPONSE
    if safety == "emergency":
        return answer or config.EMERGENCY_RESPONSE
    return answer or "How can I help you today?"


def _classify_image_type(image_bytes: bytes | None, media_type: str = "image/jpeg") -> str | None:
    """
    Classify an uploaded image with Haiku vision.

    Returns None when no image was supplied. This avoids treating text-only
    requests as dermoscopy by default.
    """
    if not image_bytes:
        return None
    b64 = base64.standard_b64encode(image_bytes).decode("ascii")
    try:
        if _llm._anthropic_client is None:
            return "unknown"
        response = _llm._anthropic_client.messages.create(
            model=config.HAIKU_MODEL,
            max_tokens=20,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": b64}},
                    {"type": "text", "text": """Classify this image into ONE category:
- dermoscopy: skin lesion / mole / dermatology photo
- histology: stained tissue slide / pathology
- microscopy: fluorescence or cell microscopy
- xray: X-ray image
- ct: CT scan
- mri: MRI image
- ultrasound: ultrasound image
- radiology: other radiology image
- medical_document: report, prescription, lab result, discharge summary, or text document
- general_medical_image: other medical image
- blurry: too blurry/dark to use
- non_medical: not a medical image
Reply with ONE word only."""},
                ],
            }],
        )
        label = response.content[0].text.strip().lower()
        label = re.sub(r"[^a-z_]", "", label)
        return label if label in IMAGE_TYPES else "unknown"
    except Exception as exc:  # noqa: BLE001
        logger.warning("Image classification failed: %s", exc)
        return "unknown"


def _haiku_router_decision(
    *,
    question: str,
    image_type: str | None,
    history: list,
    patient_mode: bool | None,
) -> dict[str, Any]:
    raw = _llm.call_haiku(
        prompt=json.dumps(
            {
                "user_message": question,
                "has_image": image_type is not None,
                "image_type": image_type,
                "history": history[-6:],
                "patient_mode": patient_mode,
            },
            ensure_ascii=True,
        ),
        system=ROUTER_SYSTEM,
        max_tokens=700,
    )
    decision = _json_object(raw)
    if not decision:
        logger.warning("Router returned invalid JSON: %r", raw)
    return decision


def _fallback_decision(question: str, image_type: str | None) -> dict[str, Any]:
    """
    Conservative fallback for outages or malformed router JSON. This fallback
    does not make safety calls from hardcoded medical keywords; it only keeps
    the graph usable when the model fails.
    """
    if not question and image_type is None:
        return {
            "intent": "empty",
            "route": "direct",
            "direct_answer": "Hello! How can I help you today?",
            "needs_rag": False,
            "needs_htan": False,
            "needs_vision": False,
            "needs_triage": False,
        }
    if image_type in config.SUPPORTED_MODALITIES:
        return {
            "intent": "image_analysis" if not question else "image_question",
            "route": "htan_rag" if question else "htan_only",
            "needs_rag": bool(question),
            "needs_htan": True,
            "needs_vision": False,
            "needs_triage": False,
            "modality": image_type,
        }
    if image_type in VISION_IMAGE_TYPES or image_type == "unknown":
        return {
            "intent": "medical_image" if not question else "image_question",
            "route": "vision_rag" if question else "vision_only",
            "needs_rag": bool(question),
            "needs_htan": False,
            "needs_vision": True,
            "needs_triage": False,
        }
    if image_type == "blurry":
        return {
            "intent": "image_blurry",
            "route": "rag_only" if question else "direct",
            "direct_answer": "The image is unclear. Please send a clearer image or describe what you need help with.",
            "needs_rag": bool(question),
            "needs_htan": False,
            "needs_vision": False,
            "needs_triage": False,
        }
    return {
        "intent": "general",
        "route": "rag_only",
        "needs_rag": True,
        "needs_htan": False,
        "needs_vision": False,
        "needs_triage": False,
    }


def _normalize_decision(
    decision: dict[str, Any],
    *,
    question: str,
    image_type: str | None,
) -> dict[str, Any]:
    if not decision:
        decision = _fallback_decision(question, image_type)

    route = str(decision.get("route") or "").strip().lower()
    if route not in ROUTES:
        route = _fallback_decision(question, image_type)["route"]

    safety = str(decision.get("safety_level") or "none").strip().lower()
    if safety in {"emergency", "crisis"}:
        route = "direct"

    questions = _as_questions(decision.get("missing_questions"))
    needs_triage = _as_bool(decision.get("needs_triage"), False)
    if needs_triage and questions:
        route = "triage_question"

    raw_modality = decision.get("modality")
    modality = raw_modality if isinstance(raw_modality, str) and raw_modality in config.SUPPORTED_MODALITIES else None

    needs_htan = _as_bool(decision.get("needs_htan"), False)
    needs_vision = _as_bool(decision.get("needs_vision"), False)
    needs_rag = _as_bool(decision.get("needs_rag"), False)

    # Enforce service boundaries while leaving the medical choice to Haiku.
    if (
        route in {"htan_only", "htan_rag"}
        and raw_modality is not None
        and modality not in config.SUPPORTED_MODALITIES
    ):
        route = "vision_rag" if question else "vision_only"
        needs_htan = False
        needs_vision = True

    if route in {"vision_only", "vision_rag"}:
        needs_vision = True
        needs_htan = False
    if route in {"htan_only", "htan_rag"}:
        needs_htan = True
    if route in {"rag_only", "htan_rag", "vision_rag"}:
        needs_rag = True

    normalized = {
        "intent": str(decision.get("intent") or "general").strip().lower(),
        "route": route,
        "safety_level": safety,
        "needs_rag": needs_rag,
        "needs_htan": needs_htan,
        "needs_vision": needs_vision,
        "needs_triage": route == "triage_question",
        "router_triage_questions": questions,
        "image_type": image_type,
        "modality": modality,
        "router_reason": str(decision.get("reason") or "").strip(),
    }

    rag_query = decision.get("rag_query")
    if isinstance(rag_query, str) and rag_query.strip():
        normalized["rag_query"] = rag_query.strip()

    if route == "direct":
        normalized["final_answer"] = _direct_answer_for(decision)

    return normalized


def intent_router_node(state: MediLinkState) -> dict:
    question = (state.get("user_message") or "").strip()
    image_bytes = state.get("image_bytes")
    image_media_type = state.get("image_media_type") or "image/jpeg"
    image_type = _classify_image_type(image_bytes, media_type=image_media_type) if image_bytes is not None else None

    decision = _haiku_router_decision(
        question=question,
        image_type=image_type,
        history=state.get("conversation_history") or [],
        patient_mode=state.get("patient_mode"),
    )

    return _normalize_decision(decision, question=question, image_type=image_type)
