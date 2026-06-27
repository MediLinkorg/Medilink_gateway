"""
HTTP clients for downstream MediLink services.

This module is intentionally small and non-medical:
  1. graph nodes decide what work is needed.
  2. this module formats HTTP requests for HTAN and RAG.
  3. downstream services return JSON.
  4. graph nodes normalize that JSON into state fields.

Keeping the transport code here gives every node the same retry, timeout, and
error behavior.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from app import config

logger = logging.getLogger("medilink.gateway")


def _response_error_detail(response: httpx.Response) -> str:
    """Return a compact response body preview for logs/errors."""
    body = response.text.strip()
    if len(body) > 500:
        body = f"{body[:500]}..."
    return f"HTTP {response.status_code}: {body or response.reason_phrase}"


def _post(url: str, *, timeout: float, **kwargs: Any) -> dict:
    """
    Shared POST helper used by RAG and HTAN.

    Trace:
      - opens a short-lived httpx client with the caller's timeout.
      - sends the prepared JSON or multipart payload.
      - retries according to SERVICE_RETRIES.
      - returns parsed JSON when successful.
      - raises a detailed RuntimeError when all attempts fail.
    """
    last_error: Exception | None = None
    for attempt in range(config.SERVICE_RETRIES + 1):
        try:
            with httpx.Client(timeout=timeout) as http:
                resp = http.post(url, **kwargs)
                if resp.status_code >= 400:
                    raise RuntimeError(_response_error_detail(resp))
                try:
                    return resp.json()
                except ValueError as error:
                    raise RuntimeError(f"Invalid JSON response: {_response_error_detail(resp)}") from error
        except Exception as error:  # noqa: BLE001
            last_error = error
            logger.warning("POST %s failed (attempt %d): %s", url, attempt + 1, error)
    raise RuntimeError(f"Service call failed: {url}: {last_error}")


def call_htan(
    image_bytes: bytes,
    modality: str,
    *,
    tta: str | None = None,
    image_media_type: str | None = None,
) -> dict:
    """
    Send an uploaded medical image to the HTAN segmentation service.

    The caller must provide a validated HTAN modality. This client does not
    default missing modality to dermoscopy; htan_node.py owns that validation.
    """
    url = f"{config.HTAN_SERVICE_URL}{config.API_PREFIX}/segment"
    files = {"image": ("upload.jpg", image_bytes, image_media_type or "application/octet-stream")}
    data = {"modality": modality, "tta": tta or config.HTAN_TTA}
    return _post(url, timeout=config.HTAN_TIMEOUT, files=files, data=data)


def call_rag(
    question: str,
    *,
    intent: str | None = None,
    patient_mode: bool = True,
    retrieve_only: bool = True,
    history: list | None = None,
    top_k: int | None = None,
) -> dict:
    """
    Send a retrieval query to the RAG service.

    rag_node.py already builds the final query by combining the user question,
    HTAN text, and/or vision document text. This function only preserves that
    query and the retrieval controls in the payload shape expected by medilink-rag.
    """
    url = f"{config.RAG_SERVICE_URL}{config.API_PREFIX}/query"
    payload = {
        "question": question,
        "intent": intent,
        "patient_mode": patient_mode,
        "retrieve_only": retrieve_only,
        "history": history or [],
    }
    if top_k is not None:
        payload["top_k"] = top_k
    return _post(url, timeout=config.RAG_TIMEOUT, json=payload)


def htan_healthy() -> bool:
    """Return True only when the HTAN service answers its health endpoint."""
    try:
        with httpx.Client(timeout=config.SERVICE_HEALTH_TIMEOUT) as http:
            return http.get(f"{config.HTAN_SERVICE_URL}/health").status_code == 200
    except Exception:  # noqa: BLE001
        return False


def rag_healthy() -> bool:
    """Return True only when the RAG service answers its health endpoint."""
    try:
        with httpx.Client(timeout=config.SERVICE_HEALTH_TIMEOUT) as http:
            return http.get(f"{config.RAG_SERVICE_URL}/health").status_code == 200
    except Exception:  # noqa: BLE001
        return False
