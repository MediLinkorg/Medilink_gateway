"""
MediLink Gateway public FastAPI entrypoint.

Trace:
  1. A frontend/mobile client calls POST /api/v1/agent with text and/or image.
  2. main.py reads multipart form data and normalizes conversation history.
  3. main.py builds the initial MediLinkState for graph.py.
  4. LangGraph runs router -> selected nodes -> report formatter.
  5. main.py maps final graph state into the public AgentResponse schema.

This module should stay thin. Medical decisions belong in graph nodes; HTTP
service calls belong in clients.py; model calls belong in llm.py.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
import json
import logging
from typing import Any

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app import clients, config, llm
from app.graph import medilink_graph
from app.schemas import AgentResponse, HealthResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
logger = logging.getLogger("medilink.gateway")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan hook.

    Startup initializes LLM clients once, logs configuration warnings, and leaves
    request-time work to the endpoint functions below.
    """
    llm.init_clients()
    for warning in config.validate():
        logger.warning("CONFIG: %s", warning)
    logger.info("Gateway ready. HTAN=%s RAG=%s", config.HTAN_SERVICE_URL, config.RAG_SERVICE_URL)
    yield


app = FastAPI(
    title="MediLink Gateway",
    version="1.0.0",
    description="Orchestrates HTAN image analysis + biomedical RAG into a single answer.",
    lifespan=lifespan,
)

# Mobile/web clients call this directly. Use CORS_ALLOW_ORIGINS in production to
# replace the development wildcard with explicit frontend origins.
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ALLOW_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _parse_history(history: str | list | None) -> list[dict[str, str]]:
    """
    Parse and normalize client conversation history.

    Accepted input is either a JSON string or a Python list of turns. Only clean
    {"role": "user"|"assistant", "content": "..."} entries are kept so malformed
    client data does not reach the router, RAG service, or Sonnet prompts.
    """
    if isinstance(history, str):
        try:
            raw_history = json.loads(history) if history else []
        except json.JSONDecodeError:
            raw_history = []
    elif isinstance(history, list):
        raw_history = history
    else:
        raw_history = []

    normalized = []
    for turn in raw_history:
        if not isinstance(turn, dict):
            continue
        role = turn.get("role")
        content = turn.get("content")
        if role not in {"user", "assistant"}:
            continue
        if not isinstance(content, str) or not content.strip():
            continue
        normalized.append({"role": role, "content": content.strip()})
    return normalized


def _build_initial_state(
    *,
    message: str,
    image_bytes: bytes | None,
    image_media_type: str | None,
    user_id: str,
    session_id: str,
    history: str | list | None,
    patient_mode: bool,
) -> dict[str, Any]:
    """
    Build the first graph state from HTTP input.

    Keeping this in one helper makes it easy to trace which request fields enter
    LangGraph and which fields are intentionally left out.
    """
    return {
        "user_message": message or "",
        "image_bytes": image_bytes,
        "image_media_type": image_media_type,
        "user_id": user_id,
        "session_id": session_id,
        "conversation_history": _parse_history(history),
        "patient_mode": patient_mode,
    }


def _response_from_state(final_state: dict[str, Any]) -> AgentResponse:
    """
    Convert graph final_state into the public API response schema.

    The graph may include many internal fields. This helper is the single place
    that decides what leaves the gateway API.
    """
    answer = final_state.get("final_answer") or final_state.get("draft_answer") or (
        "I wasn't able to produce an answer. Please rephrase or try again."
    )
    return AgentResponse(
        answer=answer,
        intent=final_state.get("intent"),
        route=final_state.get("route"),
        safety_level=final_state.get("safety_level"),
        image_type=final_state.get("image_type"),
        modality=final_state.get("modality"),
        router_reason=final_state.get("router_reason"),
        router_triage_questions=final_state.get("router_triage_questions"),
        triage_questions=final_state.get("triage_questions"),
        rag_query_used=final_state.get("rag_query_used"),
        doctor_report=final_state.get("doctor_report"),
        error=final_state.get("error"),
    )


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health() -> HealthResponse:
    """Return gateway dependency readiness for load balancers and operators."""
    return HealthResponse(
        status="ok",
        htan_service=clients.htan_healthy(),
        rag_service=clients.rag_healthy(),
        llm_configured=bool(config.ANTHROPIC_API_KEY),
        warnings=config.validate(),
    )


@app.post(config.API_PREFIX + "/agent", response_model=AgentResponse, tags=["agent"])
async def agent(
    message: str = Form(default=""),
    image: UploadFile | None = File(default=None),
    user_id: str = Form(default="anon"),
    session_id: str = Form(default="default"),
    history: str = Form(default="[]"),
    patient_mode: bool = Form(default=config.DEFAULT_PATIENT_MODE),
) -> AgentResponse:
    """
    Main chat endpoint.

    Accepts:
      - message: optional user text
      - image: optional uploaded image/document
      - user/session ids: trace identifiers
      - history: JSON conversation history
      - patient_mode: passed to RAG for patient-friendly retrieval behavior
    """
    image_bytes = await image.read() if image is not None else None
    image_media_type = image.content_type if image is not None else None

    initial = _build_initial_state(
        message=message,
        image_bytes=image_bytes,
        image_media_type=image_media_type,
        user_id=user_id,
        session_id=session_id,
        history=history,
        patient_mode=patient_mode,
    )

    try:
        final_state = await medilink_graph.ainvoke(initial)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Agent graph failed")
        return AgentResponse(
            answer="Something went wrong while processing your request. Please try again.",
            error=str(exc),
        )

    return _response_from_state(final_state)
