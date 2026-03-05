#!/usr/bin/env python3
"""
前沿洞察分析脚本 — 通过 Anthropic API 生成竞争情报分析
替代 `claude -p` 调用，避免与 Claude Code CLI 的 OAuth session 冲突。

用法:
    python3 analyze_insights.py --date 2026-03-04

在 SG 服务器上自动使用 claude-proxy (127.0.0.1:9100)。
本地可通过 ANTHROPIC_API_KEY 环境变量直接调 API。
"""

import argparse
import os
import sys
from pathlib import Path

# 加载 .env
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass


def detect_api_config():
    """检测 API 配置：优先 SG proxy，回退到直接 API。"""
    # SG 服务器上有 claude-proxy（TCP 探测，不依赖 HTTP 根路径响应）
    try:
        import socket
        s = socket.create_connection(("127.0.0.1", 9100), timeout=2)
        s.close()
        return {
            "base_url": "http://127.0.0.1:9100",
            "api_key": "proxy",  # proxy 不需要真实 key
            "source": "claude-proxy (SG)"
        }
    except Exception:
        pass

    # 直接 API
    api_key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("LLM_API_KEY")
    if api_key:
        return {
            "base_url": "https://api.anthropic.com",
            "api_key": api_key,
            "source": "Anthropic API (direct)"
        }

    return None


def main():
    parser = argparse.ArgumentParser(description="前沿洞察分析")
    parser.add_argument("--date", required=True, help="报告日期 YYYY-MM-DD")
    parser.add_argument("--radar-dir", default=str(Path(__file__).parent),
                        help="ai-dataset-radar 目录")
    parser.add_argument("--model", default="claude-opus-4-0")
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
        print("✗ 无可用 API：SG 上无 claude-proxy，也无 ANTHROPIC_API_KEY", file=sys.stderr)
        sys.exit(1)

    print(f"  → API: {config['source']}")
    print(f"  → Model: {args.model}")
    print(f"  → Input: {len(user_message):,} chars")

    try:
        import anthropic
    except ImportError:
        print("✗ 缺少 anthropic 包: pip install anthropic", file=sys.stderr)
        sys.exit(1)

    client = anthropic.Anthropic(
        base_url=config["base_url"],
        api_key=config["api_key"],
        timeout=300.0,  # Opus 生成长报告可能需 2-3 分钟
    )

    print("  → 正在分析（Opus 4.6，预计 1-3 分钟）...")
    for attempt in range(2):
        try:
            message = client.messages.create(
                model=args.model,
                max_tokens=args.max_tokens,
                system=system_prompt,
                messages=[{"role": "user", "content": user_message}],
            )
            break
        except Exception as e:
            if attempt == 0:
                print(f"  ⚠ 第一次调用失败（{e}），重试中...")
                continue
            raise

    result = message.content[0].text

    # ── 5. 写入文件 ──
    output_file.write_text(result)
    print(f"  ✓ 洞察分析完成 → {output_file} ({len(result):,} chars)")


if __name__ == "__main__":
    main()
