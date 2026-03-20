#!/usr/bin/env python3
"""
前沿洞察分析脚本 — 通过 Anthropic API 生成竞争情报分析
替代 `claude -p` 调用，避免与 Claude Code CLI 的 OAuth session 冲突。

用法:
    python3 analyze_insights.py --date 2026-03-04

模型配置优先走 SSOT（organization.yaml），回退到显式配置的 Anthropic 兼容路由或 ANTHROPIC_API_KEY。
"""

import argparse
import json
import os
import sys
from pathlib import Path

# 加载 .env
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass

# SSOT 模型配置（knowlyr-crew）
try:
    from crew.organization import resolve_model_config, get_default_model
    HAS_CREW = True
except ImportError:
    HAS_CREW = False


DEFAULT_SENTINEL_BASE_URL = "https://sentinel.knowlyr.com"
DEFAULT_SENTINEL_OPENAI_BASE_URL = f"{DEFAULT_SENTINEL_BASE_URL}/v1"
DEFAULT_CODEX_MODEL = "gpt-5.4"


def _first_env(*names: str) -> str:
    for name in names:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return ""


def _is_sentinel_key(api_key: str) -> bool:
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


def _openai_chat_completion(base_url: str, api_key: str, model: str, system_prompt: str, user_message: str, max_tokens: int) -> str:
    import httpx

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "max_completion_tokens": max_tokens,
    }

    response = httpx.post(
        f"{base_url.rstrip('/')}/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=300.0,
    )
    response.raise_for_status()
    data = response.json()
    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError(f"OpenAI-compatible response missing choices: {json.dumps(data)[:500]}")
    message = choices[0].get("message") or {}
    return _normalize_openai_content(message.get("content"))


def detect_api_config(tier: str = "strong"):
    """检测 API 配置：优先 SSOT → 显式 Anthropic 兼容路由 → 环境变量直连。"""
    # 1. 优先走 SSOT
    if HAS_CREW:
        try:
            cfg = resolve_model_config(tier)
            if cfg and cfg.model and cfg.api_key:
                return {
                    "base_url": cfg.base_url or "https://api.anthropic.com",
                    "api_key": cfg.api_key,
                    "model": cfg.model,
                    "source": f"SSOT ({tier})",
                }
        except Exception as e:
            print(f"  ⚠ SSOT 配置读取失败（{e}），回退到备用方案", file=sys.stderr)

    # 2. Sentinel Codex / OpenAI-compatible
    codex_api_key = _first_env(
        "INSIGHTS_OPENAI_API_KEY",
        "SENTINEL_CODEX_API_KEY",
    )
    openai_base_url = _first_env(
        "INSIGHTS_OPENAI_BASE_URL",
        "OPENAI_BASE_URL",
        "LLM_BASE_URL",
    ).rstrip("/")
    openai_model = _first_env(
        "INSIGHTS_OPENAI_MODEL",
        "OPENAI_MODEL",
        "LLM_MODEL",
        "INSIGHTS_MODEL",
    )
    if codex_api_key:
        base_url = openai_base_url or DEFAULT_SENTINEL_OPENAI_BASE_URL
        return {
            "provider": "openai_compatible",
            "base_url": base_url,
            "api_key": codex_api_key,
            "model": openai_model or DEFAULT_CODEX_MODEL,
            "source": f"Codex route ({base_url})",
        }

    # 3. Fallback: 显式 Anthropic 兼容路由
    base_url = _first_env(
        "INSIGHTS_BASE_URL",
        "SENTINEL_ANTHROPIC_BASE_URL",
        "ANTHROPIC_BASE_URL",
        "SENTINEL_BASE_URL",
    ).rstrip("/")
    api_key = _first_env(
        "INSIGHTS_API_KEY",
        "SENTINEL_ANTHROPIC_KEY",
        "SENTINEL_CODEX_API_KEY",
        "ANTHROPIC_API_KEY",
        "LLM_API_KEY",
    )
    model = _first_env("INSIGHTS_MODEL")

    if base_url and api_key:
        return {
            "provider": "anthropic",
            "base_url": base_url,
            "api_key": api_key,
            "model": model or None,
            "source": f"Configured route ({base_url})",
        }

    if _is_sentinel_key(api_key):
        return {
            "provider": "anthropic",
            "base_url": DEFAULT_SENTINEL_BASE_URL,
            "api_key": api_key,
            "model": model or None,
            "source": f"Sentinel tunnel ({DEFAULT_SENTINEL_BASE_URL})",
        }

    # 4. Fallback: 环境变量直连
    if api_key:
        return {
            "provider": "anthropic",
            "base_url": "https://api.anthropic.com",
            "api_key": api_key,
            "model": model or None,
            "source": "Anthropic API (direct)",
        }

    return None


def main():
    parser = argparse.ArgumentParser(description="前沿洞察分析")
    parser.add_argument("--date", required=True, help="报告日期 YYYY-MM-DD")
    parser.add_argument("--radar-dir", default=str(Path(__file__).parent),
                        help="ai-dataset-radar 目录")
    default_model = get_default_model("strong") if HAS_CREW else "claude-opus-4-20250514"
    parser.add_argument("--model", default=default_model)
    parser.add_argument("--max-tokens", type=int, default=8192)
    args = parser.parse_args()

    radar_dir = Path(args.radar_dir)
    reports_dir = radar_dir / "data" / "reports" / args.date

    # ── 1. 检查文件 ──
    prompt_file = reports_dir / f"intel_report_{args.date}_insights_prompt.md"
    changes_file = reports_dir / f"intel_report_{args.date}_changes.md"
    recipe_file = reports_dir / "recipe" / "aggregate_summary.json"
    output_file = reports_dir / f"intel_report_{args.date}_insights.md"
    pipeline_md = radar_dir / "pipeline-CLAUDE.md"

    if not prompt_file.exists():
        print(f"✗ 缺少扫描数据: {prompt_file}", file=sys.stderr)
        sys.exit(1)

    # ── 2. 构建 system prompt ──
    system_prompt = "你是 AI 训练数据行业的资深竞争情报分析师。"
    if pipeline_md.exists():
        system_prompt = pipeline_md.read_text()

    # ── 3. 构建 user message ──
    user_parts = []
    user_parts.append(f"## 扫描数据（{args.date}）\n\n")
    user_parts.append(prompt_file.read_text())

    if changes_file.exists():
        user_parts.append(f"\n\n## 变化数据\n\n{changes_file.read_text()}")

    if recipe_file.exists():
        user_parts.append(f"\n\n## DataRecipe 汇总\n\n{recipe_file.read_text()}")

    user_parts.append("\n\n---\n请基于以上数据撰写竞争情报分析报告，严格按照你的系统指令中的格式要求输出。")

    user_message = "".join(user_parts)

    # ── 4. 调用 API ──
    config = detect_api_config()
    if not config:
        print("✗ 无可用 API：未配置可用的 Anthropic/Sentinel 路由或 key", file=sys.stderr)
        sys.exit(1)

    # SSOT 配置的模型优先（用户未手动指定时）
    if config.get("model") and args.model == default_model:
        args.model = config["model"]

    print(f"  → API: {config['source']}")
    print(f"  → Model: {args.model}")
    print(f"  → Input: {len(user_message):,} chars")

    print(f"  → 正在分析（{args.model}，预计 1-3 分钟）...")
    for attempt in range(2):
        try:
            if config.get("provider") == "openai_compatible":
                result = _openai_chat_completion(
                    base_url=config["base_url"],
                    api_key=config["api_key"],
                    model=args.model,
                    system_prompt=system_prompt,
                    user_message=user_message,
                    max_tokens=args.max_tokens,
                )
            else:
                try:
                    import anthropic
                except ImportError:
                    print("✗ 缺少 anthropic 包: pip install anthropic", file=sys.stderr)
                    sys.exit(1)

                client = anthropic.Anthropic(
                    base_url=config["base_url"],
                    api_key=config["api_key"],
                    timeout=300.0,
                )
                message = client.messages.create(
                    model=args.model,
                    max_tokens=args.max_tokens,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_message}],
                )
                result = message.content[0].text
            break
        except Exception as e:
            if attempt == 0:
                print(f"  ⚠ 第一次调用失败（{e}），重试中...")
                continue
            raise

    # ── 5. 写入文件 ──
    output_file.write_text(result)
    print(f"  ✓ 洞察分析完成 → {output_file} ({len(result):,} chars)")


if __name__ == "__main__":
    main()
