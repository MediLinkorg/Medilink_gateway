"""
Gateway configuration.

Trace:
  1. load_dotenv() reads local .env values during app startup/import.
  2. small parser helpers convert environment strings into typed values.
  3. modules import constants from this file instead of reading os.environ.
  4. validate() reports risky or incomplete settings at startup and /health.

The gateway orchestrates external services over HTTP, so this module owns URLs,
timeouts, model names, safety text, and gateway behavior flags. Retrieval/index
internals stay inside the RAG service.
"""

from __future__ import annotations

import logging
import os

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger("medilink.gateway")


def _s(key: str, default: str = "") -> str:
    """Read a string env var and trim surrounding whitespace."""
    return os.environ.get(key, default).strip()


def _i(key: str, default: int) -> int:
    """Read an integer env var, falling back safely when malformed."""
    raw = os.environ.get(key)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw.strip())
    except ValueError:
        logger.warning("Invalid integer for %s=%r; using default %s.", key, raw, default)
        return default


def _f(key: str, default: float) -> float:
    """Read a float env var, falling back safely when malformed."""
    raw = os.environ.get(key)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw.strip())
    except ValueError:
        logger.warning("Invalid float for %s=%r; using default %s.", key, raw, default)
        return default


def _b(key: str, default: bool) -> bool:
    """Read a boolean env var using common truthy strings."""
    return os.environ.get(key, str(default)).strip().lower() in ("1", "true", "yes", "on")


def _csv(key: str, default: str) -> list[str]:
    """Read a comma-separated env var into a clean list."""
    return [item.strip() for item in os.environ.get(key, default).split(",") if item.strip()]


def _url(key: str, default: str) -> str:
    """Read a service URL and remove trailing slashes to avoid double-slash paths."""
    return _s(key, default).rstrip("/")


def _api_prefix(key: str = "API_PREFIX", default: str = "/api/v1") -> str:
    """Read the API prefix and normalize it to exactly one leading slash."""
    raw = _s(key, default).strip("/")
    return f"/{raw}" if raw else ""


# --- Gateway service ---
GATEWAY_HOST = _s("GATEWAY_HOST", "0.0.0.0")
GATEWAY_PORT = _i("GATEWAY_PORT", 8000)
API_PREFIX = _api_prefix()
CORS_ALLOW_ORIGINS = _csv("CORS_ALLOW_ORIGINS", "*")

# --- Claude / Anthropic ---
HAIKU_MODEL = _s("HAIKU_MODEL", "claude-haiku-4-5-20251001")
SONNET_MODEL = _s("SONNET_MODEL", "claude-sonnet-4-6")
SONNET_MAX_TOKENS = _i("SONNET_MAX_TOKENS", 1500)
ANTHROPIC_API_KEY = _s("ANTHROPIC_API_KEY", "")

# --- Gemini fallback ---
GEMINI_API_KEY = _s("GEMINI_API_KEY", "")
GEMINI_MODEL = _s("GEMINI_MODEL", "gemini-2.0-flash")

# --- Downstream HTTP services ---
HTAN_SERVICE_URL = _url("HTAN_SERVICE_URL", "http://htan:8001")
RAG_SERVICE_URL = _url("RAG_SERVICE_URL", "http://rag:8002")
HTAN_TIMEOUT = _f("HTAN_TIMEOUT", 60.0)
RAG_TIMEOUT = _f("RAG_TIMEOUT", 60.0)
SERVICE_HEALTH_TIMEOUT = _f("SERVICE_HEALTH_TIMEOUT", 5.0)
SERVICE_RETRIES = _i("SERVICE_RETRIES", 1)

# --- Pipeline behavior ---
HTAN_TTA = _s("HTAN_TTA", "basic")
RAG_TOP_K = _i("RAG_TOP_K", 5)
DEFAULT_PATIENT_MODE = _b("DEFAULT_PATIENT_MODE", True)

# Kept for compatibility with older deployments; current gateway code does not
# persist uploaded images to disk.
IMAGE_TEMP_DIR = _s("IMAGE_TEMP_DIR", "/tmp/medilink_images")

# HTAN is currently validated for these modalities. Future HTAN deployments can
# opt into more modalities with SUPPORTED_MODALITIES=dermoscopy,histology,...
SUPPORTED_MODALITIES = tuple(_csv("SUPPORTED_MODALITIES", "dermoscopy,histology,microscopy"))

# --- LangSmith observability (optional) ---
if _s("LANGCHAIN_API_KEY"):
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = _s("LANGCHAIN_API_KEY")
    os.environ["LANGCHAIN_PROJECT"] = _s("LANGCHAIN_PROJECT", "medilink")

# --- Safety responses ---
# These are routing safety responses, not diagnosis disclaimers. Override them
# with region-specific emergency/crisis wording before production deployment.
EMERGENCY_RESPONSE = _s(
    "EMERGENCY_RESPONSE",
    "This may be a medical emergency. Call your local emergency number or go "
    "to the nearest emergency department immediately.",
)
CRISIS_RESPONSE = _s(
    "CRISIS_RESPONSE",
    "I'm concerned about your safety. Please reach out right now to a local crisis "
    "or suicide-prevention helpline, or contact emergency services. You are not alone, "
    "and trained people are available to help.",
)


def validate() -> list[str]:
    """
    Return human-readable config warnings.

    This does not block startup. It surfaces issues through logs and /health so
    deployment can decide whether to continue.
    """
    warnings = []
    if not ANTHROPIC_API_KEY:
        warnings.append("ANTHROPIC_API_KEY is not set; routing/generation will fail.")
    if CORS_ALLOW_ORIGINS == ["*"]:
        warnings.append("CORS_ALLOW_ORIGINS allows all origins; restrict this before production.")
    if SERVICE_RETRIES < 0:
        warnings.append("SERVICE_RETRIES is negative; use 0 or higher.")
    if RAG_TOP_K <= 0:
        warnings.append("RAG_TOP_K should be greater than 0.")
    if SONNET_MAX_TOKENS <= 0:
        warnings.append("SONNET_MAX_TOKENS should be greater than 0.")
    if not SUPPORTED_MODALITIES:
        warnings.append("SUPPORTED_MODALITIES is empty; HTAN routes will not be usable.")
    if HTAN_TTA not in {"none", "basic", "advanced"}:
        warnings.append("HTAN_TTA is not one of: none, basic, advanced.")
    if "911" in EMERGENCY_RESPONSE or "988" in CRISIS_RESPONSE:
        warnings.append("Emergency/crisis numbers look US-specific; confirm they match your users' region.")
    return warnings
