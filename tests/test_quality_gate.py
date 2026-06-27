"""Tests for HTAN/RAG quality gate prompt behavior."""

import app.nodes.quality_gate as qg


def test_rag_gate_uses_source_citation_tags(monkeypatch):
    calls = []

    def _fake_gate(prompt, *, system, max_tokens, fail_open):
        calls.append(prompt)
        return "pass"

    monkeypatch.setattr(qg, "call_with_fallback", _fake_gate)

    assert qg.gate_rag([{"source": "cdc", "title": "Title", "text": "Evidence text"}], "Question") is True
    assert "[SOURCE:cdc-1]" in calls[0]


def test_gate_parse_defaults_to_pass_on_unclear_output():
    assert qg._parse("") is True
    assert qg._parse("maybe") is True
    assert qg._parse("drop because irrelevant") is False
