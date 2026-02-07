"""Lightweight LLM client for generating insights reports."""

import os
import logging

logger = logging.getLogger("llm_client")

DEFAULT_MODEL = "claude-sonnet-4-5-20250929"
MAX_TOKENS = 8192


def generate_insights(prompt: str, model: str = None, api_key: str = None) -> str | None:
    """Call Anthropic API to generate insights from the scan prompt.

    Args:
        prompt: The full insights prompt (data + analysis instructions).
        model: Model ID to use. Defaults to Claude Sonnet.
        api_key: Anthropic API key. Falls back to ANTHROPIC_API_KEY env var.

    Returns:
        Generated insights markdown string, or None if no API key available.
    """
    api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        logger.info("No ANTHROPIC_API_KEY found — falling back to environment LLM")
        return None

    model = model or DEFAULT_MODEL

    try:
        import anthropic
    except ImportError:
        logger.warning("anthropic package not installed — run: pip install anthropic")
        return None

    logger.info("Generating insights via %s ...", model)

    try:
        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model=model,
            max_tokens=MAX_TOKENS,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            system="你是 AI 训练数据行业的资深竞争情报分析师。请根据提供的数据生成结构化的竞争情报分析报告。报告应具体、可执行，引用具体的数据集名称、组织名称和论文标题。不要输出第5节（异常与待排查）。用中文回答。",
        )

        result = message.content[0].text
        logger.info("Insights generated successfully (%d chars)", len(result))
        return result

    except Exception as e:
        logger.error("Failed to generate insights: %s", e)
        return None
