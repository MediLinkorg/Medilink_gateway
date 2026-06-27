"""Tests for shared LLM provider helpers."""

import pytest

from app import config, llm


class _TextPart:
    def __init__(self, text):
        self.text = text


class _AnthropicResponse:
    def __init__(self, text):
        self.content = [_TextPart(text)]


class _Messages:
    def __init__(self, calls, text="anthropic answer", error=None):
        self.calls = calls
        self.text = text
        self.error = error

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return _AnthropicResponse(self.text)


class _AnthropicClient:
    def __init__(self, calls, text="anthropic answer", error=None):
        self.messages = _Messages(calls, text=text, error=error)


class _GeminiResponse:
    def __init__(self, text):
        self.text = text


class _GeminiClient:
    def __init__(self, calls, text="gemini answer", error=None):
        self.calls = calls
        self.text = text
        self.error = error

    def generate_content(self, prompt, generation_config):
        self.calls.append((prompt, generation_config))
        if self.error:
            raise self.error
        return _GeminiResponse(self.text)


def test_call_haiku_uses_config_model_and_passes_history(monkeypatch):
    calls = []
    monkeypatch.setattr(config, "HAIKU_MODEL", "test-haiku")
    monkeypatch.setattr(llm, "_anthropic_client", _AnthropicClient(calls, text="route json"))
    monkeypatch.setattr(llm, "_gemini_client", None)

    out = llm.call_haiku(
        "current question",
        system="router system",
        history=[{"role": "user", "content": "previous question"}],
    )

    assert out == "route json"
    assert calls[0]["model"] == "test-haiku"
    assert calls[0]["system"] == "router system"
    assert calls[0]["messages"] == [
        {"role": "user", "content": "previous question"},
        {"role": "user", "content": "current question"},
    ]


def test_call_with_fallback_uses_gemini_when_anthropic_fails(monkeypatch):
    anthropic_calls = []
    gemini_calls = []
    monkeypatch.setattr(llm, "_anthropic_client", _AnthropicClient(anthropic_calls, error=RuntimeError("down")))
    monkeypatch.setattr(llm, "_gemini_client", _GeminiClient(gemini_calls, text="gemini pass"))

    out = llm.call_with_fallback("gate prompt", system="gate", max_tokens=5)

    assert out == "gemini pass"
    assert anthropic_calls[0]["model"] == config.HAIKU_MODEL
    assert gemini_calls[0][1] == {"max_output_tokens": 5}
    assert "gate prompt" in gemini_calls[0][0]


def test_call_with_fallback_returns_drop_when_fail_closed(monkeypatch):
    monkeypatch.setattr(llm, "_anthropic_client", _AnthropicClient([], error=RuntimeError("anthropic down")))
    monkeypatch.setattr(llm, "_gemini_client", _GeminiClient([], error=RuntimeError("gemini down")))

    assert llm.call_with_fallback("prompt", fail_open=False) == "drop"


def test_call_sonnet_uses_config_model_and_default_tokens(monkeypatch):
    calls = []
    monkeypatch.setattr(config, "SONNET_MODEL", "test-sonnet")
    monkeypatch.setattr(config, "SONNET_MAX_TOKENS", 777)
    monkeypatch.setattr(llm, "_anthropic_client", _AnthropicClient(calls, text="final answer"))

    out = llm.call_sonnet("final prompt", system="medical system")

    assert out == "final answer"
    assert calls[0]["model"] == "test-sonnet"
    assert calls[0]["max_tokens"] == 777
    assert calls[0]["messages"] == [{"role": "user", "content": "final prompt"}]


def test_extract_text_raises_clear_error_for_empty_response():
    with pytest.raises(RuntimeError, match="no extractable text"):
        llm._extract_text(object())
