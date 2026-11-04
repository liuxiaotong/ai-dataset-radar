"""The Codex fallback model matches the weekly workflow default."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import analyze_insights  # noqa: E402


def test_codex_default_model_is_gpt_6_1_sol(monkeypatch):
    for name in (
        "INSIGHTS_OPENAI_API_KEY",
        "SENTINEL_CODEX_API_KEY",
        "INSIGHTS_OPENAI_BASE_URL",
        "OPENAI_BASE_URL",
        "LLM_BASE_URL",
        "INSIGHTS_OPENAI_MODEL",
        "OPENAI_MODEL",
        "LLM_MODEL",
        "INSIGHTS_MODEL",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("SENTINEL_CODEX_API_KEY", "kly-proxy-test")
    monkeypatch.setattr(analyze_insights, "HAS_CREW", False)

    config = analyze_insights.detect_api_config()

    assert analyze_insights.DEFAULT_CODEX_MODEL == "gpt-6.1-sol"
    assert config["model"] == "gpt-6.1-sol"
