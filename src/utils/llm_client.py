"""Lightweight LLM client for generating insights reports.

Supports multiple providers via environment variables:
  - Anthropic (default): ANTHROPIC_API_KEY
  - OpenAI-compatible (Kimi/DeepSeek/Qwen/Zhipu/OpenAI):
      LLM_PROVIDER=openai_compatible
      LLM_API_KEY=sk-xxx
      LLM_BASE_URL=https://api.moonshot.cn/v1
      LLM_MODEL=moonshot-v1-128k
"""

import os
import logging

logger = logging.getLogger("llm_client")

DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-5-20250929"
DEFAULT_OPENAI_MODEL = "gpt-4o"
MAX_TOKENS = 8192

SYSTEM_PROMPT = (
    "你是 AI 训练数据行业的资深竞争情报分析师。"
    "请根据提供的数据生成结构化的竞争情报分析报告。"
    "报告应具体、可执行，引用具体的数据集名称、组织名称和论文标题。"
    "用中文回答。"
)


def generate_insights(prompt: str, model: str = None, api_key: str = None) -> str | None:
    """Generate insights from scan data using the configured LLM provider.

    Provider selection (via LLM_PROVIDER env var):
      - "openai_compatible": Uses OpenAI SDK (Kimi, DeepSeek, Qwen, Zhipu, OpenAI)
      - "" or unset: Uses Anthropic SDK (default, backward compatible)

    Args:
        prompt: The full insights prompt (data + analysis instructions).
        model: Model ID override.
        api_key: API key override.

    Returns:
        Generated insights markdown string, or None if unavailable.
    """
    provider = os.environ.get("LLM_PROVIDER", "").lower().strip()

    if provider == "openai_compatible":
        return _generate_openai_compatible(prompt, model, api_key)
    return _generate_anthropic(prompt, model, api_key)


def _generate_anthropic(prompt: str, model: str = None, api_key: str = None) -> str | None:
    """Generate insights via Anthropic API (Claude)."""
    api_key = (
        api_key
        or os.environ.get("LLM_API_KEY")
        or os.environ.get("ANTHROPIC_API_KEY")
    )
    if not api_key:
        logger.info("No API key found — falling back to environment LLM")
        return None

    model = model or os.environ.get("LLM_MODEL") or DEFAULT_ANTHROPIC_MODEL

    try:
        import anthropic
    except ImportError:
        logger.warning("anthropic package not installed — run: pip install anthropic")
        return None

    logger.info("Generating insights via Anthropic %s ...", model)

    try:
        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model=model,
            max_tokens=MAX_TOKENS,
            messages=[{"role": "user", "content": prompt}],
            system=SYSTEM_PROMPT,
        )

        result = message.content[0].text
        logger.info("Insights generated successfully (%d chars)", len(result))
        return result

    except Exception as e:
        logger.error("Failed to generate insights via Anthropic: %s", e)
        return None


def _generate_openai_compatible(
    prompt: str, model: str = None, api_key: str = None
) -> str | None:
    """Generate insights via OpenAI-compatible API.

    Supports: Kimi (Moonshot), DeepSeek, Qwen, Zhipu, OpenAI, and any
    provider that implements the OpenAI chat completions API.

    Configure via environment variables:
      LLM_API_KEY: API key for the provider
      LLM_BASE_URL: API base URL (e.g. https://api.moonshot.cn/v1)
      LLM_MODEL: Model ID (e.g. moonshot-v1-128k)
    """
    api_key = api_key or os.environ.get("LLM_API_KEY")
    if not api_key:
        logger.info("No LLM_API_KEY found — falling back to environment LLM")
        return None

    model = model or os.environ.get("LLM_MODEL") or DEFAULT_OPENAI_MODEL
    base_url = os.environ.get("LLM_BASE_URL")

    try:
        from openai import OpenAI
    except ImportError:
        logger.warning("openai package not installed — run: pip install openai")
        return None

    provider_name = base_url or "OpenAI"
    logger.info("Generating insights via %s %s ...", provider_name, model)

    try:
        kwargs = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url

        client = OpenAI(**kwargs)
        response = client.chat.completions.create(
            model=model,
            max_tokens=MAX_TOKENS,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )

        result = response.choices[0].message.content
        logger.info("Insights generated successfully (%d chars)", len(result))
        return result

    except Exception as e:
        logger.error("Failed to generate insights via %s: %s", provider_name, e)
        return None
