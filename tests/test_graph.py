"""Gateway graph tests with Haiku routing and downstream services stubbed."""

import importlib

import app.nodes.intent_router as ir
import app.nodes.medical_llm as mll
import app.nodes.report_generator as rg
from app import clients
from app.graph import build_graph


def _route(monkeypatch, decision, image_type=None):
    monkeypatch.setattr(ir, "_haiku_router_decision", lambda **kwargs: decision)
    monkeypatch.setattr(ir, "_classify_image_type", lambda image_bytes, media_type="image/jpeg": image_type)


def _sonnet(monkeypatch, answer):
    responder = answer if callable(answer) else (lambda *a, **k: answer)
    monkeypatch.setattr(
        mll,
        "llm",
        type("M", (), {"call_sonnet": staticmethod(responder)})(),
    )


def test_direct_path_greeting(monkeypatch):
    _route(monkeypatch, {
        "intent": "greeting",
        "safety_level": "none",
        "route": "direct",
        "direct_answer": "Hi! How can I help?",
    })

    out = build_graph().invoke({"user_message": "hello", "image_bytes": None})

    assert out["route"] == "direct"
    assert out["final_answer"] == "Hi! How can I help?"


def test_triage_question_stops_before_rag(monkeypatch):
    _route(monkeypatch, {
        "intent": "symptom_description",
        "safety_level": "none",
        "route": "triage_question",
        "needs_triage": True,
        "missing_questions": [
            "How long has this been happening?",
            "How severe is the pain?",
        ],
    })
    tn = importlib.import_module("app.nodes.triage_node")
    sonnet_called = []
    monkeypatch.setattr(
        tn,
        "llm",
        type("M", (), {"call_sonnet": staticmethod(
            lambda *a, **k: sonnet_called.append(1) or (
                '{"questions":["When did the stomach pain start?",'
                '"Do you have vomiting, fever, fainting, or blood in stool?"],'
                '"message":"I need two details before I can assess this:"}'
            )
        )})(),
    )

    out = build_graph().invoke({"user_message": "I have stomach pain", "image_bytes": None})

    assert out["route"] == "triage_question"
    assert len(sonnet_called) == 1
    assert out["router_triage_questions"] == [
        "How long has this been happening?",
        "How severe is the pain?",
    ]
    assert out["triage_questions"] == [
        "When did the stomach pain start?",
        "Do you have vomiting, fever, fainting, or blood in stool?",
    ]
    assert "I need two details" in out["final_answer"]
    assert "When did the stomach pain start?" in out["final_answer"]
    assert "retrieved_docs" not in out


def test_rag_only_gate_passes(monkeypatch):
    _route(monkeypatch, {
        "intent": "medical_question",
        "safety_level": "none",
        "route": "rag_only",
        "needs_rag": True,
    })
    rag_calls = []

    def _fake_rag(*args, **kwargs):
        rag_calls.append((args, kwargs))
        return {
            "evidence": [
                {"pmid": 123, "source": "pubmed", "title": "T", "text": "Melanoma can spread."},
                {"source": "empty", "title": "No text"},
            ],
        }

    monkeypatch.setattr(clients, "call_rag", _fake_rag)
    rn = importlib.import_module("app.nodes.rag_node")
    gate_inputs = []
    monkeypatch.setattr(rn, "gate_rag", lambda docs, q: gate_inputs.append((docs, q)) or True)
    _sonnet(monkeypatch, "Yes, melanoma can spread [PMID:123].")

    out = build_graph().invoke({"user_message": "Can melanoma spread?", "image_bytes": None})

    assert out["route"] == "rag_only"
    assert rag_calls[0][0][0] == "Can melanoma spread?"
    assert rag_calls[0][1]["top_k"] == 5
    assert out["rag_query_used"] == "Can melanoma spread?"
    assert out["retrieved_docs"] == [
        {"pmid": "123", "source": "pubmed", "title": "T", "text": "Melanoma can spread."},
    ]
    assert gate_inputs[0][1] == "Can melanoma spread?"
    assert "[PMID:123]" in out["final_answer"]


def test_rag_service_failure_returns_clear_context(monkeypatch):
    _route(monkeypatch, {
        "intent": "medical_question",
        "safety_level": "none",
        "route": "rag_only",
        "needs_rag": True,
    })
    monkeypatch.setattr(clients, "call_rag", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("down")))
    _sonnet(monkeypatch, "I can answer, but RAG was unavailable.")

    out = build_graph().invoke({"user_message": "Can skin cancer spread?", "image_bytes": None})

    assert out["retrieved_docs"] == []
    assert out["rag_query_used"] == "Can skin cancer spread?"
    assert "service error" in out["rag_context"]
    assert out["error"] == "RAG service error: down"


def test_rag_gate_drop_returns_clear_context(monkeypatch):
    _route(monkeypatch, {
        "intent": "medical_question",
        "safety_level": "none",
        "route": "rag_only",
        "needs_rag": True,
        "rag_query": "skin cancer warning signs",
    })
    monkeypatch.setattr(
        clients,
        "call_rag",
        lambda *a, **k: {
            "evidence": [{"source": "pubmed", "title": "T", "text": "Evidence text."}],
        },
    )
    rn = importlib.import_module("app.nodes.rag_node")
    gate_queries = []
    monkeypatch.setattr(rn, "gate_rag", lambda docs, q: gate_queries.append(q) or False)
    _sonnet(monkeypatch, "The retrieved evidence was not usable.")

    out = build_graph().invoke({"user_message": "What are warning signs?", "image_bytes": None})

    assert out["rag_query_used"] == "skin cancer warning signs"
    assert gate_queries == ["skin cancer warning signs"]
    assert out["retrieved_docs"] == []
    assert "quality gate" in out["rag_context"]


def test_medical_llm_labels_rag_status_and_uses_config_tokens(monkeypatch):
    calls = []

    def _fake_sonnet(*args, **kwargs):
        calls.append((args, kwargs))
        return "RAG was unavailable, so this is a cautious answer."

    _sonnet(monkeypatch, _fake_sonnet)

    out = mll.medical_llm_node({
        "user_message": "Can skin cancer spread?",
        "rag_context": "RAG evidence was not available: service error: down",
        "retrieved_docs": [],
    })

    prompt = calls[0][0][0]
    assert "RAG STATUS:" in prompt
    assert "RETRIEVED EVIDENCE:" not in prompt
    assert calls[0][1]["max_tokens"] == 1500
    assert out["citations"]["all_valid"] is True


def test_medical_llm_validates_source_citations():
    report = mll._verify_citations(
        "Use this source [SOURCE:cdc-1] but not this one [SOURCE:fake-9].",
        [{"pmid": "", "source": "cdc", "title": "T", "text": "Evidence"}],
    )

    assert report["cited_sources"] == ["cdc-1", "fake-9"]
    assert report["fabricated_sources"] == ["fake-9"]
    assert report["all_valid"] is False


def test_medical_llm_failure_returns_empty_citations(monkeypatch):
    _sonnet(monkeypatch, lambda *a, **k: (_ for _ in ()).throw(RuntimeError("sonnet down")))

    out = mll.medical_llm_node({
        "user_message": "What is melanoma?",
        "rag_context": "[PMID:123] T\nEvidence",
        "retrieved_docs": [{"pmid": "123", "source": "pubmed", "title": "T", "text": "Evidence"}],
    })

    assert out["error"] == "Generation error: sonnet down"
    assert out["citations"]["all_valid"] is True
    assert out["citations"]["num_citations"] == 0


def test_htan_only_dropped_no_text_is_graceful(monkeypatch):
    _route(monkeypatch, {
        "intent": "image_analysis",
        "safety_level": "none",
        "route": "htan_only",
        "needs_htan": True,
        "modality": "dermoscopy",
    }, image_type="dermoscopy")
    monkeypatch.setattr(
        clients,
        "call_htan",
        lambda *a, **k: {"target_detected": True, "modality": "dermoscopy", "segmented_area_percent": 0.2},
    )
    hn = importlib.import_module("app.nodes.htan_node")
    monkeypatch.setattr(hn, "gate_htan", lambda cv, q: False)

    sonnet_called = []
    _sonnet(monkeypatch, lambda *a, **k: sonnet_called.append(1) or "I could not use the segmentation result.")

    out = build_graph().invoke({"user_message": "", "image_bytes": b"fakeimagebytes"})

    assert out["route"] == "htan_only"
    assert len(sonnet_called) == 1
    assert "quality gate" in out["cv_text"]
    assert "could not use" in out["final_answer"]


def test_htan_missing_modality_does_not_default_to_dermoscopy(monkeypatch):
    _route(monkeypatch, {
        "intent": "image_analysis",
        "safety_level": "none",
        "route": "htan_only",
        "needs_htan": True,
    }, image_type="dermoscopy")

    htan_calls = []
    monkeypatch.setattr(clients, "call_htan", lambda *a, **k: htan_calls.append((a, k)) or {})
    _sonnet(monkeypatch, "HTAN was unavailable because modality was missing.")

    out = build_graph().invoke({"user_message": "", "image_bytes": b"fakeimagebytes"})

    assert out["route"] == "htan_only"
    assert htan_calls == []
    assert "unsupported or missing modality" in out["cv_text"]
    assert "HTAN was unavailable" in out["final_answer"]


def test_htan_normalizes_output_and_uses_config_tta(monkeypatch):
    _route(monkeypatch, {
        "intent": "image_analysis",
        "safety_level": "none",
        "route": "htan_only",
        "needs_htan": True,
        "modality": "histology",
    }, image_type="histology")

    calls = []

    def _fake_htan(*args, **kwargs):
        calls.append((args, kwargs))
        return {
            "target_detected": True,
            "segmented_area_percent": "7.5",
            "num_components": "2",
            "largest_component_span_px": "18",
            "severity_estimate": "medium",
        }

    monkeypatch.setattr(clients, "call_htan", _fake_htan)
    hn = importlib.import_module("app.nodes.htan_node")
    monkeypatch.setattr(hn, "gate_htan", lambda cv, q: True)
    _sonnet(monkeypatch, "HTAN context was used.")

    out = build_graph().invoke({"user_message": "", "image_bytes": b"fakeimagebytes"})

    assert calls[0][1]["modality"] == "histology"
    assert calls[0][1]["tta"] == "basic"
    assert out["cv_result"]["modality"] == "histology"
    assert out["cv_result"]["segmented_area_percent"] == 7.5
    assert out["cv_result"]["num_components"] == 2
    assert out["cv_result"]["largest_component_span_px"] == 18.0
    assert "largest span 18.0 px" in out["cv_text"]


def test_medical_document_uses_vision_then_rag(monkeypatch):
    _route(monkeypatch, {
        "intent": "document_question",
        "safety_level": "none",
        "route": "vision_rag",
        "needs_vision": True,
        "needs_rag": True,
    }, image_type="medical_document")

    vn = importlib.import_module("app.nodes.vision_node")
    vision_calls = []

    def _fake_vision(*args, **kwargs):
        vision_calls.append(kwargs)
        return {
            "image_type": "medical_document",
            "extracted_text": "HbA1c 9.2%",
            "summary": 123,
            "limitations": ["Some text is cropped.", 456],
        }

    monkeypatch.setattr(vn, "_call_vision", _fake_vision)
    rag_calls = []

    def _fake_rag(*args, **kwargs):
        rag_calls.append((args, kwargs))
        return {
            "evidence": [{"pmid": "456", "source": "pubmed", "title": "Diabetes", "text": "HbA1c reflects glycemia."}],
        }

    monkeypatch.setattr(clients, "call_rag", _fake_rag)
    rn = importlib.import_module("app.nodes.rag_node")
    monkeypatch.setattr(rn, "gate_rag", lambda docs, q: True)
    _sonnet(monkeypatch, "The report suggests elevated average blood sugar [PMID:456].")

    out = build_graph().invoke({
        "user_message": "What does this mean?",
        "image_bytes": b"document",
        "image_media_type": "image/png",
    })

    assert out["route"] == "vision_rag"
    assert out["image_type"] == "medical_document"
    assert vision_calls[0]["media_type"] == "image/png"
    assert out["vision_result"]["summary"] == "123"
    assert out["vision_result"]["limitations"] == ["Some text is cropped.", "456"]
    assert "HbA1c" in out["vision_text"]
    assert "Vision/document context" in out["rag_query_used"]
    assert "HbA1c 9.2%" in out["rag_query_used"]
    assert "HbA1c 9.2%" in rag_calls[0][0][0]
    assert "[PMID:456]" in out["final_answer"]


def test_report_generator_includes_trace_fields_and_normalized_flags():
    out = rg.report_generator_node({
        "intent": "medical_question",
        "route": "rag_only",
        "safety_level": "none",
        "router_reason": "test route",
        "error": "RAG service error: down",
        "cv_result": {
            "target_detected": True,
            "modality": "dermoscopy",
            "model": "htan",
            "segmented_area_percent": 3.2,
            "largest_component_span_px": 20.0,
        },
        "cv_text": "HTAN image analysis text",
        "vision_result": {"image_type": "medical_document", "summary": "report"},
        "vision_text": "Vision context text",
        "router_triage_questions": ["router question"],
        "triage_questions": ["sonnet question"],
        "rag_query_used": "skin cancer warning signs",
        "rag_context": "RAG evidence was not available: service error: down",
        "retrieved_docs": [{"pmid": "", "source": "cdc", "title": "T", "text": "Evidence"}],
        "draft_answer": "Draft answer",
        "citations": {
            "cited_sources": ["cdc-1", "fake-9"],
            "fabricated_sources": ["fake-9"],
            "num_citations": 2,
        },
    })

    report = out["doctor_report"]

    assert out["final_answer"] == "Draft answer"
    assert report["error"] == "RAG service error: down"
    assert report["image_findings"]["largest_component_span_px"] == 20.0
    assert report["cv_text_preview"] == "HTAN image analysis text"
    assert report["vision_text_preview"] == "Vision context text"
    assert report["rag_context_status"] == "unavailable"
    assert report["rag_context_preview"].startswith("RAG evidence was not available")
    assert report["evidence_count"] == 1
    assert report["evidence"][0]["has_text"] is True
    assert report["citations"]["fabricated_pmids"] == []
    assert report["citations"]["fabricated_sources"] == ["fake-9"]
    assert report["flags"]["fabricated_citations"] == ["fake-9"]
    assert report["flags"]["rag_unavailable"] is True
    assert report["flags"]["has_error"] is True
