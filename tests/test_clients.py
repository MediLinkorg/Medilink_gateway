"""Tests for the gateway HTTP client boundary."""

import pytest

from app import clients, config


class _FakeResponse:
    def __init__(self, status_code=200, payload=None, text="OK", reason_phrase="OK"):
        self.status_code = status_code
        self._payload = payload if payload is not None else {}
        self.text = text
        self.reason_phrase = reason_phrase

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


def test_call_htan_uses_config_tta_and_preserves_media_type(monkeypatch):
    calls = []

    def _fake_post(url, *, timeout, **kwargs):
        calls.append((url, timeout, kwargs))
        return {"ok": True}

    monkeypatch.setattr(clients, "_post", _fake_post)

    out = clients.call_htan(
        b"image",
        modality="dermoscopy",
        image_media_type="image/png",
    )

    assert out == {"ok": True}
    assert calls[0][0] == f"{config.HTAN_SERVICE_URL}{config.API_PREFIX}/segment"
    assert calls[0][1] == config.HTAN_TIMEOUT
    assert calls[0][2]["data"] == {"modality": "dermoscopy", "tta": config.HTAN_TTA}
    assert calls[0][2]["files"]["image"] == ("upload.jpg", b"image", "image/png")


def test_call_rag_omits_top_k_when_not_provided(monkeypatch):
    calls = []
    monkeypatch.setattr(clients, "_post", lambda url, *, timeout, **kwargs: calls.append(kwargs) or {"ok": True})

    clients.call_rag("question", intent="diagnosis", top_k=None)

    payload = calls[0]["json"]
    assert payload["question"] == "question"
    assert payload["intent"] == "diagnosis"
    assert payload["history"] == []
    assert "top_k" not in payload


def test_post_reports_http_body_after_retries(monkeypatch):
    class _FakeClient:
        def __init__(self, timeout):
            self.timeout = timeout

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def post(self, url, **kwargs):
            return _FakeResponse(status_code=503, text="RAG unavailable", reason_phrase="Service Unavailable")

    monkeypatch.setattr(clients.httpx, "Client", _FakeClient)
    monkeypatch.setattr(config, "SERVICE_RETRIES", 0)

    with pytest.raises(RuntimeError) as exc:
        clients._post("http://service/api", timeout=1, json={})

    assert "HTTP 503: RAG unavailable" in str(exc.value)


def test_health_uses_config_timeout(monkeypatch):
    timeouts = []

    class _FakeClient:
        def __init__(self, timeout):
            timeouts.append(timeout)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def get(self, url):
            return _FakeResponse(status_code=200)

    monkeypatch.setattr(clients.httpx, "Client", _FakeClient)

    assert clients.htan_healthy() is True
    assert clients.rag_healthy() is True
    assert timeouts == [config.SERVICE_HEALTH_TIMEOUT, config.SERVICE_HEALTH_TIMEOUT]
