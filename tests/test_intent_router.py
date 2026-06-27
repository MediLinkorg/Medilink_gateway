"""Tests for intent router helper behavior."""

from app import config
import app.nodes.intent_router as ir


class _TextPart:
    def __init__(self, text):
        self.text = text


class _VisionResponse:
    def __init__(self, text):
        self.content = [_TextPart(text)]


class _Messages:
    def __init__(self, calls):
        self.calls = calls

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return _VisionResponse("medical_document")


class _AnthropicClient:
    def __init__(self, calls):
        self.messages = _Messages(calls)


def test_image_classifier_uses_configured_haiku_model(monkeypatch):
    calls = []
    monkeypatch.setattr(config, "HAIKU_MODEL", "test-haiku-vision")
    monkeypatch.setattr(ir._llm, "_anthropic_client", _AnthropicClient(calls))

    label = ir._classify_image_type(b"fake-image", media_type="image/png")

    assert label == "medical_document"
    assert calls[0]["model"] == "test-haiku-vision"
    assert calls[0]["messages"][0]["content"][0]["source"]["media_type"] == "image/png"
