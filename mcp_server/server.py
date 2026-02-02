#!/usr/bin/env python3
"""MCP Server for AI Dataset Radar.

Exposes tools for Claude Desktop to analyze AI training datasets
and competitive intelligence.
"""

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import Tool, TextContent
except ImportError:
    print("Error: mcp package not installed. Run: pip install mcp", file=sys.stderr)
    sys.exit(1)


# Initialize MCP server
server = Server("ai-dataset-radar")


@server.list_tools()
async def list_tools():
    """List available tools."""
    return [
        Tool(
            name="radar_scan",
            description="运行 AI 数据集竞争情报扫描，监控 HuggingFace、GitHub、arXiv 和博客上的最新动态",
            inputSchema={
                "type": "object",
                "properties": {
                    "days": {
                        "type": "integer",
                        "description": "扫描天数 (默认 7 天)",
                        "default": 7
                    }
                }
            }
        ),
        Tool(
            name="radar_summary",
            description="获取最新扫描报告的摘要统计",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="radar_datasets",
            description="获取最新发现的数据集列表",
            inputSchema={
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "description": "按类型过滤: synthetic, sft, preference, evaluation, multimodal, multilingual, code, agent",
                        "default": ""
                    },
                    "limit": {
                        "type": "integer",
                        "description": "返回数量限制",
                        "default": 10
                    }
                }
            }
        ),
        Tool(
            name="radar_github",
            description="获取 GitHub 组织的最新活动",
            inputSchema={
                "type": "object",
                "properties": {
                    "relevance": {
                        "type": "string",
                        "description": "过滤相关性: high, medium, low",
                        "default": "high"
                    }
                }
            }
        ),
        Tool(
            name="radar_papers",
            description="获取最新相关论文",
            inputSchema={
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "返回数量限制",
                        "default": 10
                    }
                }
            }
        ),
        Tool(
            name="radar_config",
            description="查看当前监控配置（监控的组织、关键词等）",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
    ]


def get_latest_report() -> dict | None:
    """Get the latest JSON report."""
    reports_dir = PROJECT_ROOT / "data" / "reports"
    if not reports_dir.exists():
        return None

    json_files = sorted(reports_dir.glob("intel_report_*.json"), reverse=True)
    if not json_files:
        return None

    with open(json_files[0], "r", encoding="utf-8") as f:
        return json.load(f)


def get_latest_report_path() -> Path | None:
    """Get path to latest report."""
    reports_dir = PROJECT_ROOT / "data" / "reports"
    if not reports_dir.exists():
        return None

    json_files = sorted(reports_dir.glob("intel_report_*.json"), reverse=True)
    return json_files[0] if json_files else None


@server.call_tool()
async def call_tool(name: str, arguments: dict):
    """Handle tool calls."""

    if name == "radar_scan":
        days = arguments.get("days", 7)

        # Run the scan
        venv_python = PROJECT_ROOT / ".venv" / "bin" / "python"
        if not venv_python.exists():
            venv_python = "python3"
        else:
            venv_python = str(venv_python)

        cmd = [venv_python, str(PROJECT_ROOT / "src" / "main_intel.py"), "--days", str(days)]

        try:
            result = subprocess.run(
                cmd,
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True,
                timeout=300
            )

            # Get the generated report
            report = get_latest_report()
            if report:
                summary = report.get("summary", {})
                return [TextContent(
                    type="text",
                    text=f"""扫描完成！

**统计摘要:**
- 数据集: {summary.get('total_datasets', 0)} 个
- GitHub 组织: {summary.get('total_github_orgs', 0)} 个
- GitHub 仓库: {summary.get('total_github_repos', 0)} 个 ({summary.get('total_github_repos_high_relevance', 0)} 个高相关)
- 论文: {summary.get('total_papers', 0)} 篇
- 博客文章: {summary.get('total_blog_posts', 0)} 篇

报告已保存到: {get_latest_report_path()}

使用 `radar_datasets`、`radar_github`、`radar_papers` 查看详细内容。"""
                )]
            else:
                return [TextContent(type="text", text=f"扫描完成，但未找到报告。\n\n输出:\n{result.stdout}\n\n错误:\n{result.stderr}")]

        except subprocess.TimeoutExpired:
            return [TextContent(type="text", text="扫描超时 (5分钟)，请稍后重试。")]
        except Exception as e:
            return [TextContent(type="text", text=f"扫描失败: {e}")]

    elif name == "radar_summary":
        report = get_latest_report()
        if not report:
            return [TextContent(type="text", text="没有找到报告，请先运行 `radar_scan`。")]

        summary = report.get("summary", {})
        period = report.get("period", {})
        generated_at = report.get("generated_at", "未知")

        return [TextContent(
            type="text",
            text=f"""**AI Dataset Radar 报告摘要**

生成时间: {generated_at}
扫描周期: {period.get('days', 7)} 天 ({period.get('start', '')[:10]} ~ {period.get('end', '')[:10]})

**统计:**
| 类别 | 数量 |
|------|------|
| 数据集 | {summary.get('total_datasets', 0)} |
| GitHub 组织 | {summary.get('total_github_orgs', 0)} |
| GitHub 仓库 | {summary.get('total_github_repos', 0)} |
| 高相关仓库 | {summary.get('total_github_repos_high_relevance', 0)} |
| 论文 | {summary.get('total_papers', 0)} |
| 博客文章 | {summary.get('total_blog_posts', 0)} |

**数据集类型分布:**
{json.dumps(report.get('datasets_by_type', {}), indent=2, ensure_ascii=False)}
"""
        )]

    elif name == "radar_datasets":
        report = get_latest_report()
        if not report:
            return [TextContent(type="text", text="没有找到报告，请先运行 `radar_scan`。")]

        datasets = report.get("datasets", [])
        category = arguments.get("category", "")
        limit = arguments.get("limit", 10)

        if category:
            datasets = [d for d in datasets if d.get("category") == category or category in d.get("all_categories", [])]

        datasets = datasets[:limit]

        if not datasets:
            return [TextContent(type="text", text=f"没有找到{'类型为 ' + category + ' 的' if category else ''}数据集。")]

        lines = ["**最新数据集:**\n"]
        for ds in datasets:
            cat = ds.get("category", "unknown")
            lines.append(f"- **{ds.get('id')}** [{cat}]")
            lines.append(f"  - Downloads: {ds.get('downloads', 0):,} | Likes: {ds.get('likes', 0)}")
            if ds.get("description"):
                desc = ds.get("description", "")[:100]
                lines.append(f"  - {desc}...")
            lines.append("")

        return [TextContent(type="text", text="\n".join(lines))]

    elif name == "radar_github":
        report = get_latest_report()
        if not report:
            return [TextContent(type="text", text="没有找到报告，请先运行 `radar_scan`。")]

        github_activity = report.get("github_activity", [])
        relevance_filter = arguments.get("relevance", "high")

        lines = [f"**GitHub 活动 (相关性: {relevance_filter}):**\n"]

        for org in github_activity:
            org_name = org.get("org", "unknown")
            repos = org.get("repos_updated", [])

            filtered_repos = [r for r in repos if r.get("relevance") == relevance_filter] if relevance_filter else repos

            if filtered_repos:
                lines.append(f"### {org_name}")
                for repo in filtered_repos[:5]:
                    signals = repo.get("relevance_signals", [])
                    signals_str = ", ".join(signals) if signals else "无"
                    lines.append(f"- **{repo.get('name')}** ⭐ {repo.get('stars', 0)}")
                    lines.append(f"  - 信号: {signals_str}")
                    if repo.get("description"):
                        lines.append(f"  - {repo.get('description', '')[:80]}")
                lines.append("")

        if len(lines) == 1:
            return [TextContent(type="text", text=f"没有找到相关性为 {relevance_filter} 的仓库。")]

        return [TextContent(type="text", text="\n".join(lines))]

    elif name == "radar_papers":
        report = get_latest_report()
        if not report:
            return [TextContent(type="text", text="没有找到报告，请先运行 `radar_scan`。")]

        papers = report.get("papers", [])
        limit = arguments.get("limit", 10)
        papers = papers[:limit]

        if not papers:
            return [TextContent(type="text", text="没有找到论文。")]

        lines = ["**最新论文:**\n"]
        for paper in papers:
            title = paper.get("title", "未知标题")
            url = paper.get("url", "")
            abstract = paper.get("abstract", "")[:150]
            lines.append(f"- **{title}**")
            if url:
                lines.append(f"  - URL: {url}")
            if abstract:
                lines.append(f"  - {abstract}...")
            lines.append("")

        return [TextContent(type="text", text="\n".join(lines))]

    elif name == "radar_config":
        config_path = PROJECT_ROOT / "config.yaml"
        if not config_path.exists():
            return [TextContent(type="text", text="未找到配置文件。")]

        import yaml
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        # Extract key info
        watched_orgs = config.get("watched_orgs", {})
        watched_vendors = config.get("watched_vendors", {})
        priority_types = config.get("priority_data_types", {})

        lines = ["**AI Dataset Radar 配置:**\n"]

        lines.append("### 监控的 AI Labs")
        for category, orgs in watched_orgs.items():
            org_names = list(orgs.keys()) if isinstance(orgs, dict) else orgs
            lines.append(f"- **{category}**: {', '.join(org_names[:5])}{'...' if len(org_names) > 5 else ''}")

        lines.append("\n### 监控的数据供应商")
        for category, vendors in watched_vendors.items():
            if category == "blogs":
                continue
            vendor_names = list(vendors.keys()) if isinstance(vendors, dict) else vendors
            lines.append(f"- **{category}**: {', '.join(vendor_names[:5]) if vendor_names else '无'}")

        lines.append("\n### 关注的数据类型")
        for dtype in list(priority_types.keys())[:8]:
            lines.append(f"- {dtype}")

        return [TextContent(type="text", text="\n".join(lines))]

    else:
        return [TextContent(type="text", text=f"未知工具: {name}")]


async def main():
    """Run the MCP server."""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
