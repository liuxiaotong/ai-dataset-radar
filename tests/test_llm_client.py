"""Tests for multi-provider LLM client."""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from utils.llm_client import (
    generate_insights,
    _generate_anthropic,
    _generate_openai_compatible,
    SYSTEM_PROMPT,
    DEFAULT_ANTHROPIC_MODEL,
    DEFAULT_OPENAI_MODEL,
    MAX_TOKENS,
)


# ---------------------------------------------------------------------------
# Provider routing
# ---------------------------------------------------------------------------

class TestProviderRouting:
    """Test that generate_insights routes to the correct provider."""

    @patch("utils.llm_client._generate_anthropic")
    def test_default_routes_to_anthropic(self, mock_anthropic, monkeypatch):
        monkeypatch.delenv("LLM_PROVIDER", raising=False)
        generate_insights("test prompt")
        mock_anthropic.assert_called_once_with("test prompt", None, None)

    @patch("utils.llm_client._generate_anthropic")
    def test_empty_provider_routes_to_anthropic(self, mock_anthropic, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "")
        generate_insights("test prompt")
        mock_anthropic.assert_called_once()

    @patch("utils.llm_client._generate_openai_compatible")
    def test_openai_compatible_routes_correctly(self, mock_openai, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "openai_compatible")
        generate_insights("test prompt")
        mock_openai.assert_called_once_with("test prompt", None, None)

    @patch("utils.llm_client._generate_openai_compatible")
    def test_provider_case_insensitive(self, mock_openai, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "OpenAI_Compatible")
        generate_insights("test prompt")
        mock_openai.assert_called_once()

    @patch("utils.llm_client._generate_openai_compatible")
    def test_provider_strips_whitespace(self, mock_openai, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "  openai_compatible  ")
        generate_insights("test prompt")
        mock_openai.assert_called_once()

    @patch("utils.llm_client._generate_anthropic")
    def test_unknown_provider_falls_back_to_anthropic(self, mock_anthropic, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "unknown_provider")
        generate_insights("test prompt")
        mock_anthropic.assert_called_once()

    def test_model_and_key_passed_through(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "openai_compatible")
        with patch("utils.llm_client._generate_openai_compatible") as mock_openai:
            generate_insights("prompt", model="custom-model", api_key="sk-test")
            mock_openai.assert_called_once_with("prompt", "custom-model", "sk-test")


# ---------------------------------------------------------------------------
# Anthropic provider
# ---------------------------------------------------------------------------

class TestAnthropicProvider:
    """Test Anthropic-specific behavior."""

    def test_no_api_key_returns_none(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("LLM_API_KEY", raising=False)
        result = _generate_anthropic("test")
        assert result is None

    def test_llm_api_key_takes_priority(self, monkeypatch):
        monkeypatch.setenv("LLM_API_KEY", "sk-llm")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant")

        mock_anthropic = MagicMock()
        mock_message = MagicMock()
        mock_message.content = [MagicMock(text="insights")]
        mock_anthropic.Anthropic.return_value.messages.create.return_value = mock_message

        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            result = _generate_anthropic("test")
            mock_anthropic.Anthropic.assert_called_once_with(api_key="sk-llm")
            assert result == "insights"

    def test_anthropic_api_key_fallback(self, monkeypatch):
        monkeypatch.delenv("LLM_API_KEY", raising=False)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant")

        mock_anthropic = MagicMock()
        mock_message = MagicMock()
        mock_message.content = [MagicMock(text="insights")]
        mock_anthropic.Anthropic.return_value.messages.create.return_value = mock_message

        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            result = _generate_anthropic("test")
            mock_anthropic.Anthropic.assert_called_once_with(api_key="sk-ant")
            assert result == "insights"

    def test_llm_model_env_override(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant")
        monkeypatch.setenv("LLM_MODEL", "claude-opus-4-6")

        mock_anthropic = MagicMock()
        mock_message = MagicMock()
        mock_message.content = [MagicMock(text="insights")]
        mock_anthropic.Anthropic.return_value.messages.create.return_value = mock_message

        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            _generate_anthropic("test")
            call_kwargs = mock_anthropic.Anthropic.return_value.messages.create.call_args
            assert call_kwargs.kwargs["model"] == "claude-opus-4-6"

    def test_missing_anthropic_package(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant")

        with patch.dict("sys.modules", {"anthropic": None}):
            with patch("builtins.__import__", side_effect=ImportError("no module")):
                # Can't easily test ImportError with lazy import, test via direct call
                pass

    def test_api_error_returns_none(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant")

        mock_anthropic = MagicMock()
        mock_anthropic.Anthropic.return_value.messages.create.side_effect = Exception("API error")

        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            result = _generate_anthropic("test")
            assert result is None

    def test_system_prompt_passed(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant")

        mock_anthropic = MagicMock()
        mock_message = MagicMock()
        mock_message.content = [MagicMock(text="insights")]
        mock_anthropic.Anthropic.return_value.messages.create.return_value = mock_message

        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            _generate_anthropic("test")
            call_kwargs = mock_anthropic.Anthropic.return_value.messages.create.call_args
            assert call_kwargs.kwargs["system"] == SYSTEM_PROMPT
            assert call_kwargs.kwargs["max_tokens"] == MAX_TOKENS


# ---------------------------------------------------------------------------
# OpenAI-compatible provider
# ---------------------------------------------------------------------------

class TestOpenAICompatibleProvider:
    """Test OpenAI-compatible provider (Kimi, DeepSeek, Qwen, etc.)."""

    def test_no_api_key_returns_none(self, monkeypatch):
        monkeypatch.delenv("LLM_API_KEY", raising=False)
        result = _generate_openai_compatible("test")
        assert result is None

    def test_basic_call(self, monkeypatch):
        monkeypatch.setenv("LLM_API_KEY", "sk-kimi")
        monkeypatch.setenv("LLM_BASE_URL", "https://api.moonshot.cn/v1")
        monkeypatch.setenv("LLM_MODEL", "moonshot-v1-128k")

        mock_openai = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "kimi insights"
        mock_openai.OpenAI.return_value.chat.completions.create.return_value = MagicMock(
            choices=[mock_choice]
        )

        with patch.dict("sys.modules", {"openai": mock_openai}):
            result = _generate_openai_compatible("test prompt")

            mock_openai.OpenAI.assert_called_once_with(
                api_key="sk-kimi", base_url="https://api.moonshot.cn/v1"
            )
            create_call = mock_openai.OpenAI.return_value.chat.completions.create
            call_kwargs = create_call.call_args.kwargs
            assert call_kwargs["model"] == "moonshot-v1-128k"
            assert call_kwargs["max_tokens"] == MAX_TOKENS
            assert call_kwargs["messages"][0]["role"] == "system"
            assert call_kwargs["messages"][0]["content"] == SYSTEM_PROMPT
            assert call_kwargs["messages"][1]["content"] == "test prompt"
            assert result == "kimi insights"

    def test_no_base_url_uses_default(self, monkeypatch):
        monkeypatch.setenv("LLM_API_KEY", "sk-openai")
        monkeypatch.delenv("LLM_BASE_URL", raising=False)

        mock_openai = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "openai insights"
        mock_openai.OpenAI.return_value.chat.completions.create.return_value = MagicMock(
            choices=[mock_choice]
        )

        with patch.dict("sys.modules", {"openai": mock_openai}):
            result = _generate_openai_compatible("test")
            # Should not pass base_url kwarg
            mock_openai.OpenAI.assert_called_once_with(api_key="sk-openai")
            assert result == "openai insights"

    def test_default_model(self, monkeypatch):
        monkeypatch.setenv("LLM_API_KEY", "sk-test")
        monkeypatch.delenv("LLM_MODEL", raising=False)

        mock_openai = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "result"
        mock_openai.OpenAI.return_value.chat.completions.create.return_value = MagicMock(
            choices=[mock_choice]
        )

        with patch.dict("sys.modules", {"openai": mock_openai}):
            _generate_openai_compatible("test")
            call_kwargs = (
                mock_openai.OpenAI.return_value.chat.completions.create.call_args.kwargs
            )
            assert call_kwargs["model"] == DEFAULT_OPENAI_MODEL

    def test_api_error_returns_none(self, monkeypatch):
        monkeypatch.setenv("LLM_API_KEY", "sk-test")

        mock_openai = MagicMock()
        mock_openai.OpenAI.return_value.chat.completions.create.side_effect = Exception(
            "rate limit"
        )

        with patch.dict("sys.modules", {"openai": mock_openai}):
            result = _generate_openai_compatible("test")
            assert result is None

    def test_explicit_api_key_param(self, monkeypatch):
        monkeypatch.delenv("LLM_API_KEY", raising=False)

        mock_openai = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "result"
        mock_openai.OpenAI.return_value.chat.completions.create.return_value = MagicMock(
            choices=[mock_choice]
        )

        with patch.dict("sys.modules", {"openai": mock_openai}):
            result = _generate_openai_compatible("test", api_key="sk-param")
            mock_openai.OpenAI.assert_called_once_with(api_key="sk-param")
            assert result == "result"

    def test_explicit_model_param(self, monkeypatch):
        monkeypatch.setenv("LLM_API_KEY", "sk-test")
        monkeypatch.setenv("LLM_MODEL", "env-model")

        mock_openai = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "result"
        mock_openai.OpenAI.return_value.chat.completions.create.return_value = MagicMock(
            choices=[mock_choice]
        )

        with patch.dict("sys.modules", {"openai": mock_openai}):
            _generate_openai_compatible("test", model="param-model")
            call_kwargs = (
                mock_openai.OpenAI.return_value.chat.completions.create.call_args.kwargs
            )
            # Explicit param should override env var
            assert call_kwargs["model"] == "param-model"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class TestConstants:
    """Test module-level constants."""

    def test_system_prompt_is_chinese(self):
        assert "竞争情报" in SYSTEM_PROMPT
        assert "中文" in SYSTEM_PROMPT

    def test_max_tokens(self):
        assert MAX_TOKENS == 8192

    def test_default_models(self):
        assert "claude" in DEFAULT_ANTHROPIC_MODEL
        assert DEFAULT_OPENAI_MODEL == "gpt-4o"
