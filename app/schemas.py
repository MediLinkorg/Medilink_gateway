"""
Public API schemas for the MediLink gateway.

These models are the contract between the gateway and any frontend/mobile/API
client. The LangGraph state can contain many internal fields, but main.py maps
only the fields below into the HTTP response.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class AgentResponse(BaseModel):
    """
    Response returned by POST /api/v1/agent.

    Field trace:
      - answer: patient-facing answer or triage prompt.
      - intent/route/safety_level: router decision metadata.
      - image_type/modality: image classification and HTAN modality metadata.
      - router_reason: short Haiku explanation of why this route was selected.
      - router_triage_questions: Haiku's suggested missing-information hints.
      - triage_questions: final Sonnet-generated questions shown to the user.
      - rag_query_used: exact query sent to the RAG service after augmentation.
      - doctor_report: structured trace/report for debugging or clinician review.
      - error: node/graph error surfaced without hiding the response body.
    """

    answer: str
    intent: str | None = None
    route: str | None = None
    safety_level: str | None = None
    image_type: str | None = None
    modality: str | None = None
    router_reason: str | None = None
    router_triage_questions: list[str] | None = None
    triage_questions: list[str] | None = None
    rag_query_used: str | None = None
    doctor_report: dict[str, Any] | None = None
    error: str | None = None


class HealthResponse(BaseModel):
    """Response returned by GET /health for service readiness checks."""

    status: str
    htan_service: bool
    rag_service: bool
    llm_configured: bool
    warnings: list[str] = Field(default_factory=list)
