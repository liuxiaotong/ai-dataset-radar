"""Lightweight LLM client for generating insights reports.

Supports multiple providers via environment variables:
  - Anthropic (default): ANTHROPIC_API_KEY
  - OpenAI-compatible (Kimi/DeepSeek/Qwen/Zhipu/OpenAI):
      LLM_PROVIDER=openai_compatible
      LLM_API_KEY=sk-xxx
      LLM_BASE_URL=https://api.moonshot.cn/v1
      LLM_MODEL=moonshot-v1-128k
"""

import logging
import json
import os

try:
    from crew.organization import resolve_model_config, get_default_model
    HAS_CREW = True
except ImportError:
    HAS_CREW = False

logger = logging.getLogger("llm_client")

_FALLBACK_ANTHROPIC_MODEL = "claude-sonnet-4-5-20250929"
DEFAULT_ANTHROPIC_MODEL = get_default_model("default") if HAS_CREW else _FALLBACK_ANTHROPIC_MODEL
DEFAULT_OPENAI_MODEL = "gpt-4o"
MAX_TOKENS = 8192
DEFAULT_SENTINEL_BASE_URL = "https://sentinel.knowlyr.com"
DEFAULT_SENTINEL_OPENAI_BASE_URL = f"{DEFAULT_SENTINEL_BASE_URL}/v1"

SYSTEM_PROMPT = (
    "你是 AI 训练数据行业的资深竞争情报分析师。"
    "请根据提供的数据生成结构化的竞争情报分析报告。"
    "报告应具体、可执行，引用具体的数据集名称、组织名称和论文标题。"
    "用中文回答。"
)


def _is_sentinel_key(api_key: str | None) -> bool:
    return str(api_key or "").startswith("kly-proxy-")


def _normalize_openai_content(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") in {"text", "output_text"}:
                text = item.get("text")
                if text:
                    parts.append(text)
        return "".join(parts)
    return str(content or "")


def _post_openai_chat_completion(
    *,
    base_url: str,
    api_key: str,
    model: str,
    system_prompt: str,
    prompt: str,
    max_tokens: int,
) -> str:
    import httpx

    response = httpx.post(
        f"{base_url.rstrip('/')}/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "max_completion_tokens": max_tokens,
        },
        timeout=300.0,
    )
    response.raise_for_status()
    data = response.json()
    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError(f"OpenAI-compatible response missing choices: {json.dumps(data)[:500]}")
    message = choices[0].get("message") or {}
    return _normalize_openai_content(message.get("content"))


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
    if not provider and os.environ.get("SENTINEL_CODEX_API_KEY"):
        provider = "openai_compatible"

    if provider == "openai_compatible":
        return _generate_openai_compatible(prompt, model, api_key)
    return _generate_anthropic(prompt, model, api_key)


def _generate_anthropic(prompt: str, model: str = None, api_key: str = None) -> str | None:
    """Generate insights via Anthropic API (Claude)."""
    # 优先 SSOT 配置
    ssot_cfg = None
    if HAS_CREW and not api_key:
        try:
            ssot_cfg = resolve_model_config("default")
        except Exception:
            pass

    api_key = (
        api_key
        or (ssot_cfg.api_key if ssot_cfg and ssot_cfg.api_key else None)
        or os.environ.get("LLM_API_KEY")
        or os.environ.get("ANTHROPIC_API_KEY")
    )
    if not api_key:
        logger.info("No API key found — falling back to environment LLM")
        return None

    model = model or (ssot_cfg.model if ssot_cfg and ssot_cfg.model else None) or os.environ.get("LLM_MODEL") or DEFAULT_ANTHROPIC_MODEL
    base_url = (
        (ssot_cfg.base_url if ssot_cfg and ssot_cfg.base_url else None)
        or os.environ.get("ANTHROPIC_BASE_URL")
        or (DEFAULT_SENTINEL_BASE_URL if _is_sentinel_key(api_key) else None)
    )

    try:
        import anthropic
    except ImportError:
        logger.warning("anthropic package not installed — run: pip install anthropic")
        return None

    logger.info("Generating insights via Anthropic %s ...", model)

    try:
        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        client = anthropic.Anthropic(**client_kwargs)
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
    api_key = api_key or os.environ.get("LLM_API_KEY") or os.environ.get("SENTINEL_CODEX_API_KEY")
    if not api_key:
        logger.info("No LLM_API_KEY found — falling back to environment LLM")
        return None

    model = model or os.environ.get("LLM_MODEL") or DEFAULT_OPENAI_MODEL
    base_url = os.environ.get("LLM_BASE_URL") or (
        DEFAULT_SENTINEL_OPENAI_BASE_URL if _is_sentinel_key(api_key) else None
    )

    provider_name = base_url or "OpenAI"
    logger.info("Generating insights via %s %s ...", provider_name, model)

    try:
        if base_url:
            result = _post_openai_chat_completion(
                base_url=base_url,
                api_key=api_key,
                model=model,
                system_prompt=SYSTEM_PROMPT,
                prompt=prompt,
                max_tokens=MAX_TOKENS,
            )
        else:
            from openai import OpenAI

            client = OpenAI(api_key=api_key)
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
