"""
Shared LLM provider clients for the MediLink gateway.

This module is the model boundary for the gateway:
  1. main.py calls init_clients() once during FastAPI startup.
  2. router/gate nodes call Haiku-style helpers for fast decisions.
  3. triage_node.py and medical_llm.py call Sonnet for patient-facing reasoning.
  4. vision_node.py and image classification code may reuse _anthropic_client
     directly when they need Anthropic's multimodal message format.

The module does not decide routes, triage questions, RAG queries, or diagnoses.
It only owns provider initialization, message formatting, fallback behavior, and
safe text extraction from provider responses.
"""

from __future__ import annotations

import logging
from typing import Any

from app import config

logger = logging.getLogger("medilink.gateway")

# These globals are initialized once at application startup. Tests monkeypatch
# them directly, and vision/image-router modules reuse _anthropic_client for
# image messages that do not fit the plain text helper functions below.
_anthropic_client = None
_gemini_client = None


def init_clients() -> None:
    """
    Initialize provider clients from config.

    Trace:
      - Anthropic is required for Sonnet and direct multimodal calls.
      - Gemini is optional and only used as a text fallback for Haiku/gates.
      - Missing keys do not crash startup; config.validate() reports warnings.
    """
    global _anthropic_client, _gemini_client

    if config.ANTHROPIC_API_KEY:
        from anthropic import Anthropic

        _anthropic_client = Anthropic(api_key=config.ANTHROPIC_API_KEY)
    else:
        logger.info("Anthropic client not initialized: ANTHROPIC_API_KEY is missing.")

    if config.GEMINI_API_KEY:
        try:
            import google.generativeai as genai

            genai.configure(api_key=config.GEMINI_API_KEY)
            _gemini_client = genai.GenerativeModel(config.GEMINI_MODEL)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Gemini client init failed: %s", exc)
    else:
        logger.info("Gemini fallback not initialized: GEMINI_API_KEY is missing.")


def _extract_text(response: Any) -> str:
    """
    Extract text from provider responses in a defensive way.

    Anthropic usually returns response.content[0].text. Gemini usually returns
    response.text. Tests and provider SDK changes can vary these shapes, so this
    helper keeps parsing errors clear instead of failing with an attribute error.
    """
    text = getattr(response, "text", None)
    if isinstance(text, str) and text.strip():
        return text.strip()

    content = getattr(response, "content", None)
    if isinstance(content, list):
        parts = []
        for item in content:
            item_text = getattr(item, "text", None)
            if item_text is None and isinstance(item, dict):
                item_text = item.get("text")
            if isinstance(item_text, str) and item_text.strip():
                parts.append(item_text.strip())
        if parts:
            return "\n".join(parts).strip()

    raise RuntimeError("Provider returned no extractable text.")


def _messages_from_history(prompt: str, history: list | None = None) -> list[dict]:
    """
    Build Anthropic-compatible messages.

    The gateway stores conversation turns as plain dictionaries. We pass through
    only role/content pairs so accidental extra fields from clients do not leak
    into the provider request.
    """
    messages = []
    for turn in history or []:
        role = turn.get("role") if isinstance(turn, dict) else None
        content = turn.get("content") if isinstance(turn, dict) else None
        if role in {"user", "assistant"} and isinstance(content, str) and content.strip():
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": prompt})
    return messages


def _call_anthropic(
    prompt: str,
    system: str,
    max_tokens: int,
    *,
    model: str,
    history: list | None = None,
) -> str:
    """Call Anthropic with a text prompt and return clean text."""
    if _anthropic_client is None:
        raise RuntimeError("Anthropic client not initialized.")

    response = _anthropic_client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=_messages_from_history(prompt, history),
    )
    return _extract_text(response)


def _call_gemini(prompt: str, system: str, max_tokens: int, history: list | None = None) -> str:
    """
    Call Gemini fallback with a text prompt and return clean text.

    Gemini fallback is used only for lightweight Haiku-style text tasks. We fold
    history into the prompt because this SDK path accepts one prompt string here.
    """
    if _gemini_client is None:
        raise RuntimeError("Gemini client not initialized.")

    history_text = ""
    for turn in history or []:
        if isinstance(turn, dict) and turn.get("role") and turn.get("content"):
            history_text += f"{turn['role']}: {turn['content']}\n"

    full_prompt = f"{system}\n\n{history_text}{prompt}" if system else f"{history_text}{prompt}"
    response = _gemini_client.generate_content(
        full_prompt,
        generation_config={"max_output_tokens": max_tokens},
    )
    return _extract_text(response)


def call_with_fallback(
    prompt: str,
    system: str = "",
    max_tokens: int = 20,
    fail_open: bool = True,
    history: list | None = None,
) -> str:
    """
    Try Anthropic Haiku, then Gemini, then deterministic fail-open/fail-closed.

    Quality gates use fail_open=True so a provider outage does not silently block
    the pipeline. Router-style calls can still receive a raw provider answer when
    Anthropic or Gemini is available.
    """
    try:
        return _call_anthropic(
            prompt,
            system,
            max_tokens,
            model=config.HAIKU_MODEL,
            history=history,
        )
    except Exception as exc:  # noqa: BLE001
        if _gemini_client is None:
            logger.warning("Anthropic call failed and Gemini is not configured: %s", exc)
        else:
            logger.warning("Anthropic call failed; trying Gemini fallback: %s", exc)

    try:
        return _call_gemini(prompt, system, max_tokens, history=history)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Gemini fallback failed; returning %s: %s", "pass" if fail_open else "drop", exc)

    return "pass" if fail_open else "drop"


def call_haiku(
    prompt: str,
    system: str = "",
    max_tokens: int = 200,
    history: list | None = None,
) -> str:
    """
    Call the fast reasoning model for routing/classification style tasks.

    Unlike the previous version, history is now forwarded into provider calls,
    which matters for follow-up turns where the router needs conversation memory.
    """
    return call_with_fallback(
        prompt,
        system=system,
        max_tokens=max_tokens,
        fail_open=True,
        history=history,
    )


def call_sonnet(
    prompt: str,
    system: str = "",
    max_tokens: int | None = None,
    history: list | None = None,
) -> str:
    """
    Call Sonnet for final reasoning or triage question generation.

    Sonnet intentionally has no Gemini fallback. If it fails, the calling node
    writes a clear error/fallback into graph state so report_generator.py can
    expose what happened.
    """
    token_limit = max_tokens if max_tokens is not None else config.SONNET_MAX_TOKENS
    return _call_anthropic(
        prompt,
        system,
        token_limit,
        model=config.SONNET_MODEL,
        history=history,
    )
