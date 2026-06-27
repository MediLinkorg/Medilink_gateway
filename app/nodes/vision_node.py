"""
Vision node for medical documents, radiology, and general medical images.

Why this module exists
----------------------
HTAN is a segmentation service. It is only appropriate for the image modalities
it was built for: dermoscopy, histology, and microscopy.

This node handles the other supported image cases:
  - medical documents, such as reports, prescriptions, labs, discharge notes
  - X-ray / CT / MRI / ultrasound / radiology-style images
  - other general medical images that should be described, not segmented

Pipeline trace
--------------
  1. intent_router.py classifies the image and chooses:
       route = "vision_only" or route = "vision_rag"

  2. graph.py sends the shared MediLinkState to this node.

  3. this node sends the uploaded image to Haiku vision.

  4. Haiku returns structured JSON:
       image_type, extracted_text, summary, limitations

  5. this node normalizes that JSON so downstream modules get predictable types.

  6. this node writes:
       vision_result = normalized structured JSON
       vision_text   = readable context string for RAG/Sonnet

  7. graph.py decides the next step:
       vision_only -> medical_llm
       vision_rag  -> rag -> medical_llm

Important boundary
------------------
This node does not produce the final answer. It only extracts useful image or
document context. medical_llm.py performs the final clinical reasoning.
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

# System prompt used for the Haiku vision call.
#
# The output is intentionally JSON because this node is part of a pipeline. A
# stable structure makes it easier for RAG, Sonnet, and the doctor_report to use
# the extracted image/document context without guessing what the model meant.
VISION_SYSTEM = """
You extract useful medical context from uploaded images for a general medical
assistant. Return valid JSON only.

For medical documents, extract the visible text and summarize key values,
dates, medications, diagnoses, or recommendations.

For radiology or other medical images, describe visible findings cautiously.
Do not invent details. If the image quality or view limits interpretation,
state that in limitations.

JSON schema:
{
  "image_type": "string",
  "extracted_text": "string",
  "summary": "string",
  "limitations": ["string"]
}
""".strip()


def _json_object(raw: str) -> dict[str, Any]:
    """
    Parse a model response into a JSON object.

    Haiku is instructed to return JSON only, but this helper tolerates accidental
    prose around the JSON block so the graph can continue gracefully.
    """
    # Preferred path: the model obeyed the instruction and returned pure JSON.
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        # Defensive path: sometimes a model may add short prose around JSON.
        # We recover the first {...} block instead of failing the whole request.
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if not match:
            return {}
        try:
            parsed = json.loads(match.group(0))
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}


def _clean_string(value: Any) -> str:
    """Normalize any scalar-ish value into a clean string."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _clean_string_list(value: Any) -> list[str]:
    """Normalize limitations into a list of clean strings."""
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    if not isinstance(value, list):
        return []
    cleaned = []
    for item in value:
        text = _clean_string(item)
        if text:
            cleaned.append(text)
    return cleaned[:8]


def _normalize_result(raw: dict[str, Any], *, image_type: str | None) -> dict[str, Any]:
    """
    Convert Haiku's JSON into predictable state.

    Downstream nodes should never need to guess whether a value is missing,
    a list, a number, or another object. This function makes the contract stable.
    """
    # Keep only the fields this pipeline understands. If Haiku returns extra
    # fields, they are ignored here so downstream modules have a stable contract.
    normalized = {
        "image_type": _clean_string(raw.get("image_type")) or image_type or "unknown",
        "extracted_text": _clean_string(raw.get("extracted_text")),
        "summary": _clean_string(raw.get("summary")),
        "limitations": _clean_string_list(raw.get("limitations")),
    }

    # Medical documents may contain useful extracted text even when the model
    # omits a summary. Add a neutral summary so medical_llm.py has context.
    if not normalized["summary"] and normalized["extracted_text"]:
        normalized["summary"] = "Visible medical document text was extracted."

    # If neither text nor summary is available, preserve that limitation in the
    # state instead of returning a blank context. This helps Sonnet explain that
    # the image was not usable.
    if not normalized["summary"] and not normalized["extracted_text"]:
        normalized["limitations"].append("No reliable visual details could be extracted.")

    return normalized


def _call_vision(
    image_bytes: bytes,
    *,
    image_type: str | None,
    question: str,
    media_type: str,
) -> dict[str, Any]:
    """
    Call Haiku vision using the configured model and original upload MIME type.
    """
    # Vision requires the Anthropic multimodal client. The text-only fallback in
    # llm.py cannot inspect image bytes, so there is no equivalent fallback here.
    if _llm._anthropic_client is None:
        raise RuntimeError("Anthropic vision client is not configured.")

    # Anthropic vision expects the image as base64 plus the original MIME type.
    # main.py stores the upload MIME type in state["image_media_type"].
    b64 = base64.standard_b64encode(image_bytes).decode("ascii")

    response = _llm._anthropic_client.messages.create(
        # Use config so model upgrades happen in one place.
        model=config.HAIKU_MODEL,
        max_tokens=1200,
        system=VISION_SYSTEM,
        messages=[{
            "role": "user",
            "content": [
                # Image part: the actual uploaded image.
                {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": b64}},
                # Text part: routing context from intent_router.py plus the
                # user's question. This helps Haiku decide whether to extract
                # document text, summarize radiology content, or note limits.
                {"type": "text", "text": json.dumps({
                    "image_type": image_type,
                    "user_question": question,
                    "task": "Extract document text or describe the medical image for downstream clinical reasoning.",
                })},
            ],
        }],
    )
    return _json_object(response.content[0].text.strip())


def _result_to_text(result: dict[str, Any]) -> str:
    """
    Render normalized vision JSON into context text for RAG and Sonnet.
    """
    # medical_llm.py and rag_node.py consume text context, not raw JSON.
    # This rendering keeps the normalized structured fields readable.
    parts = []
    if result.get("image_type"):
        parts.append(f"Image type: {result['image_type']}")
    if result.get("summary"):
        parts.append(f"Vision summary: {result['summary']}")
    if result.get("extracted_text"):
        parts.append(f"Extracted document text:\n{result['extracted_text']}")
    if result.get("limitations"):
        parts.append("Limitations: " + "; ".join(result["limitations"]))
    return "\n\n".join(parts).strip()


def vision_node(state: MediLinkState) -> dict:
    """
    LangGraph node entry point.

    Reads:
      image_bytes, image_media_type, image_type, user_message

    Writes:
      vision_result, vision_text, and error on recoverable failures
    """
    # Read input/context created by earlier modules.
    #
    # image_bytes      -> uploaded image from main.py
    # image_media_type -> upload MIME type from main.py
    # image_type       -> Haiku router classification from intent_router.py
    # user_message     -> user's text question from main.py
    image_bytes = state.get("image_bytes")
    image_type = state.get("image_type")
    media_type = state.get("image_media_type") or "image/jpeg"
    question = (state.get("user_message") or "").strip()

    # The graph should only route here when an image exists, but this guard
    # keeps the node safe if a future route is misconfigured.
    if not image_bytes:
        return {"vision_result": None, "vision_text": ""}

    try:
        # Call Haiku vision and parse the raw JSON response.
        raw_result = _call_vision(
            image_bytes,
            image_type=image_type,
            question=question,
            media_type=media_type,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Vision extraction failed")
        # Do not crash the graph. Return a clear context string so the next node
        # can still answer from the user's text if possible.
        return {
            "error": f"Vision extraction error: {exc}",
            "vision_result": None,
            "vision_text": "Vision extraction failed. Use the user's text only if available.",
        }

    # Normalize before writing state. This is the contract used by rag_node.py,
    # medical_llm.py, and report_generator.py.
    result = _normalize_result(raw_result, image_type=image_type)
    return {"vision_result": result, "vision_text": _result_to_text(result)}
