"""Triage node - Sonnet generates focused follow-up questions."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from app import llm
from app.state import MediLinkState

logger = logging.getLogger("medilink.gateway")

TRIAGE_SYSTEM = """
You are MediLink's clinical triage question generator.

The router has already decided that more information is needed before a focused
medical assessment. Your job is to ask only the most important missing questions.

Return valid JSON only:
{
  "questions": ["question"],
  "message": "patient-facing message"
}

Rules:
1. Ask concise, clinically useful questions.
2. Prefer 3 to 5 questions.
3. Include urgent red-flag questions when relevant.
4. Do not provide a diagnosis or treatment plan in this step.
5. Do not add generic disclaimer text.
""".strip()


def _json_object(raw: str) -> dict[str, Any]:
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


def _clean_questions(value: Any) -> list[str]:
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    if not isinstance(value, list):
        return []
    questions = []
    for item in value:
        if isinstance(item, str) and item.strip():
            questions.append(item.strip())
    return questions[:6]


def _fallback_questions(state: MediLinkState) -> list[str]:
    suggested = _clean_questions(state.get("router_triage_questions"))
    if not suggested:
        suggested = _clean_questions(state.get("triage_questions"))
    if suggested:
        return suggested
    return [
        "When did this start?",
        "How severe is it, and is it getting better, worse, or staying the same?",
        "Do you have fever, shortness of breath, severe pain, weakness, bleeding, or fainting?",
        "Do you have any known medical conditions or take any regular medications?",
    ]


def _format_message(questions: list[str], message: str | None = None) -> str:
    intro = (message or "I need a little more information before I can give a focused assessment:").strip()
    lines = [intro]
    lines.extend(f"{i}. {question}" for i, question in enumerate(questions, 1))
    return "\n".join(lines)


def triage_node(state: MediLinkState) -> dict:
    question = (state.get("user_message") or "").strip()
    history = state.get("conversation_history") or []
    suggested_questions = _clean_questions(state.get("router_triage_questions"))

    prompt = json.dumps(
        {
            "user_message": question,
            "conversation_history": history[-8:],
            "router_suggested_questions": suggested_questions,
            "router_reason": state.get("router_reason"),
            "intent": state.get("intent"),
            "image_type": state.get("image_type"),
        },
        ensure_ascii=True,
    )

    try:
        raw = llm.call_sonnet(
            prompt,
            system=TRIAGE_SYSTEM,
            max_tokens=700,
            history=history,
        )
        parsed = _json_object(raw)
        questions = _clean_questions(parsed.get("questions"))
        if not questions:
            questions = _fallback_questions(state)
        message = parsed.get("message") if isinstance(parsed.get("message"), str) else None
    except Exception as exc:  # noqa: BLE001
        logger.exception("Sonnet triage question generation failed")
        questions = _fallback_questions(state)
        message = None

    return {
        "triage_questions": questions,
        "final_answer": _format_message(questions, message),
    }
