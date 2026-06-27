"""Tests for gateway configuration helpers and validation."""

import importlib

from app import config


def test_safe_numeric_parsers_fall_back_on_bad_env(monkeypatch):
    monkeypatch.setenv("BAD_INT", "not-int")
    monkeypatch.setenv("BAD_FLOAT", "not-float")

    assert config._i("BAD_INT", 7) == 7
    assert config._f("BAD_FLOAT", 2.5) == 2.5


def test_url_and_api_prefix_helpers_normalize_values(monkeypatch):
    monkeypatch.setenv("SERVICE_URL", "http://example.test///")
    monkeypatch.setenv("CUSTOM_PREFIX", "api/custom/")

    assert config._url("SERVICE_URL", "http://fallback") == "http://example.test"
    assert config._api_prefix("CUSTOM_PREFIX", "/api/v1") == "/api/custom"


def test_api_prefix_and_supported_modalities_are_env_configurable(monkeypatch):
    monkeypatch.setenv("API_PREFIX", "gateway/v2")
    monkeypatch.setenv("SUPPORTED_MODALITIES", "dermoscopy, histology, cytology")
    try:
        reloaded = importlib.reload(config)
        assert reloaded.API_PREFIX == "/gateway/v2"
        assert reloaded.SUPPORTED_MODALITIES == ("dermoscopy", "histology", "cytology")
    finally:
        monkeypatch.delenv("API_PREFIX", raising=False)
        monkeypatch.delenv("SUPPORTED_MODALITIES", raising=False)
        importlib.reload(config)


def test_validate_reports_risky_settings(monkeypatch):
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", "")
    monkeypatch.setattr(config, "CORS_ALLOW_ORIGINS", ["*"])
    monkeypatch.setattr(config, "SERVICE_RETRIES", -1)
    monkeypatch.setattr(config, "RAG_TOP_K", 0)
    monkeypatch.setattr(config, "SONNET_MAX_TOKENS", 0)
    monkeypatch.setattr(config, "SUPPORTED_MODALITIES", ())
    monkeypatch.setattr(config, "HTAN_TTA", "experimental")
    monkeypatch.setattr(config, "EMERGENCY_RESPONSE", "Call 911.")
    monkeypatch.setattr(config, "CRISIS_RESPONSE", "Call 988.")

    warnings = config.validate()

    assert any("ANTHROPIC_API_KEY" in warning for warning in warnings)
    assert any("CORS_ALLOW_ORIGINS" in warning for warning in warnings)
    assert any("SERVICE_RETRIES" in warning for warning in warnings)
    assert any("RAG_TOP_K" in warning for warning in warnings)
    assert any("SONNET_MAX_TOKENS" in warning for warning in warnings)
    assert any("SUPPORTED_MODALITIES" in warning for warning in warnings)
    assert any("HTAN_TTA" in warning for warning in warnings)
    assert any("US-specific" in warning for warning in warnings)
