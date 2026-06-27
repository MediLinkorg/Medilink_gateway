"""Tests for public API schemas and response mapping."""

import asyncio

from app import main
from app.schemas import AgentResponse, HealthResponse


def test_health_response_warnings_use_independent_lists():
    first = HealthResponse(status="ok", htan_service=True, rag_service=True, llm_configured=True)
    second = HealthResponse(status="ok", htan_service=True, rag_service=True, llm_configured=True)

    first.warnings.append("first warning")

    assert first.warnings == ["first warning"]
    assert second.warnings == []


def test_agent_response_accepts_trace_fields():
    response = AgentResponse(
        answer="Answer",
        intent="medical_question",
        route="rag_only",
        safety_level="none",
        image_type="medical_document",
        modality=None,
        router_reason="Need evidence.",
        router_triage_questions=["How long?"],
        triage_questions=["How long has this been present?"],
        rag_query_used="skin cancer warning signs",
        doctor_report={"evidence_count": 2},
        error=None,
    )

    data = response.model_dump()
    assert data["router_reason"] == "Need evidence."
    assert data["router_triage_questions"] == ["How long?"]
    assert data["triage_questions"] == ["How long has this been present?"]
    assert data["rag_query_used"] == "skin cancer warning signs"
    assert data["doctor_report"]["evidence_count"] == 2


def test_parse_history_keeps_only_clean_turns():
    history = main._parse_history(
        """[
            {"role":"user","content":"  first question  ","extra":"ignored"},
            {"role":"system","content":"bad role"},
            {"role":"assistant","content":"answer"},
            {"role":"user","content":""},
            "not a turn"
        ]"""
    )

    assert history == [
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "answer"},
    ]


def test_build_initial_state_preserves_gateway_inputs():
    state = main._build_initial_state(
        message="Question",
        image_bytes=b"image",
        image_media_type="image/png",
        user_id="user-1",
        session_id="session-1",
        history=[{"role": "user", "content": "Previous"}],
        patient_mode=False,
    )

    assert state == {
        "user_message": "Question",
        "image_bytes": b"image",
        "image_media_type": "image/png",
        "user_id": "user-1",
        "session_id": "session-1",
        "conversation_history": [{"role": "user", "content": "Previous"}],
        "patient_mode": False,
    }


def test_response_from_state_uses_draft_answer_fallback():
    response = main._response_from_state({
        "draft_answer": "Draft answer",
        "route": "rag_only",
        "rag_query_used": "query",
    })

    assert response.answer == "Draft answer"
    assert response.route == "rag_only"
    assert response.rag_query_used == "query"


def test_agent_endpoint_maps_new_trace_fields(monkeypatch):
    async def _fake_ainvoke(initial):
        return {
            "final_answer": "Final answer",
            "intent": "medical_question",
            "route": "rag_only",
            "safety_level": "none",
            "image_type": "medical_document",
            "modality": None,
            "router_reason": "Router chose RAG.",
            "router_triage_questions": ["Router hint"],
            "triage_questions": ["Final question"],
            "rag_query_used": "query used",
            "doctor_report": {"route": "rag_only"},
            "error": None,
        }

    monkeypatch.setattr(main.medilink_graph, "ainvoke", _fake_ainvoke)

    response = asyncio.run(main.agent(
        message="Question",
        image=None,
        user_id="anon",
        session_id="default",
        history="[]",
        patient_mode=True,
    ))

    assert response.answer == "Final answer"
    assert response.router_reason == "Router chose RAG."
    assert response.router_triage_questions == ["Router hint"]
    assert response.triage_questions == ["Final question"]
    assert response.rag_query_used == "query used"
    assert response.doctor_report == {"route": "rag_only"}
