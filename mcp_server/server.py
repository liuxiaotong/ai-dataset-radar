#!/usr/bin/env python3
"""MCP Server for AI Dataset Radar.

Exposes tools for Claude Desktop to analyze AI training datasets
and competitive intelligence.
"""

import json
import subprocess
import sys
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
                        "default": 7,
                    },
                    "sources": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": ["github", "blogs", "papers", "labs", "vendors", "x"],
                        },
                        "description": "只扫描指定的数据源（默认全部扫描）",
                        "default": [],
                    },
                },
            },
        ),
        Tool(
            name="radar_summary",
            description="获取最新扫描报告的摘要统计",
            inputSchema={"type": "object", "properties": {}},
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
                        "default": "",
                    },
                    "org": {
                        "type": "string",
                        "description": "按组织/作者过滤 (如 'openai', 'meta-llama')",
                        "default": "",
                    },
                    "limit": {"type": "integer", "description": "返回数量限制", "default": 10},
                },
            },
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
                        "default": "high",
                    },
                    "org": {
                        "type": "string",
                        "description": "按组织名过滤 (如 'argilla-io', 'scaleapi')",
                        "default": "",
                    },
                },
            },
        ),
        Tool(
            name="radar_papers",
            description="获取最新相关论文",
            inputSchema={
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "description": "返回数量限制", "default": 10}
                },
            },
        ),
        Tool(
            name="radar_config",
            description="查看当前监控配置（监控的组织、关键词等）",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="radar_blogs",
            description="获取最新博客文章（来自 OpenAI、Anthropic、Mistral、Scale AI、Stanford HAI 等 17 个博客源）",
            inputSchema={
                "type": "object",
                "properties": {
                    "source": {
                        "type": "string",
                        "description": "按博客源过滤，如 'OpenAI Blog', 'Mistral AI', 'Stanford HAI'",
                        "default": "",
                    },
                    "limit": {"type": "integer", "description": "返回数量限制", "default": 20},
                },
            },
        ),
        Tool(
            name="radar_search",
            description="跨所有数据源全文搜索（数据集、GitHub 仓库、论文、博客、X/Twitter），支持关键词和正则",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "搜索关键词（支持正则表达式）"},
                    "sources": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": ["datasets", "github", "papers", "blogs", "x"],
                        },
                        "description": "限制搜索的数据源（默认搜索全部）",
                        "default": [],
                    },
                    "limit": {
                        "type": "integer",
                        "description": "每个数据源返回的最大结果数",
                        "default": 10,
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="radar_trend",
            description="查询数据集增长趋势：上升最快、突破性增长、指定数据集的历史曲线",
            inputSchema={
                "type": "object",
                "properties": {
                    "mode": {
                        "type": "string",
                        "description": "查询模式: top_growing (增长最快), rising (上升中), breakthroughs (突破), dataset (指定数据集)",
                        "enum": ["top_growing", "rising", "breakthroughs", "dataset"],
                        "default": "top_growing",
                    },
                    "dataset_id": {
                        "type": "string",
                        "description": "指定数据集 ID (mode=dataset 时使用，如 'openai/gsm8k')",
                    },
                    "days": {
                        "type": "integer",
                        "description": "趋势周期: 7 或 30 天",
                        "default": 7,
                    },
                    "limit": {"type": "integer", "description": "返回数量限制", "default": 10},
                },
            },
        ),
        Tool(
            name="radar_history",
            description="查看历史扫描报告时间线，展示各期报告的统计摘要和变化趋势",
            inputSchema={
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "description": "返回最近几期报告", "default": 10}
                },
            },
        ),
        Tool(
            name="radar_diff",
            description="对比两期报告，自动识别新增/消失的数据集、仓库、论文等变化",
            inputSchema={
                "type": "object",
                "properties": {
                    "date_a": {
                        "type": "string",
                        "description": "较早的报告日期 (YYYY-MM-DD)，留空则使用倒数第二份报告",
                    },
                    "date_b": {
                        "type": "string",
                        "description": "较新的报告日期 (YYYY-MM-DD)，留空则使用最新报告",
                    },
                },
            },
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


def get_report_by_date(date_str: str) -> dict | None:
    """Get a report by date string (YYYY-MM-DD)."""
    reports_dir = PROJECT_ROOT / "data" / "reports"
    if not reports_dir.exists():
        return None

    target = reports_dir / f"intel_report_{date_str}.json"
    if target.exists():
        with open(target, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def get_all_reports_sorted() -> list[Path]:
    """Get all report paths sorted by date descending."""
    reports_dir = PROJECT_ROOT / "data" / "reports"
    if not reports_dir.exists():
        return []
    return sorted(reports_dir.glob("intel_report_*.json"), reverse=True)


def _fmt_growth(value) -> str:
    """Format a growth rate value for display.

    Args:
        value: Growth rate (float, e.g. 0.5 = 50%) or None.

    Returns:
        Formatted string like "+50.0%" or "N/A".
    """
    if value is None:
        return "N/A"
    if value == float("inf"):
        return "New ∞"
    sign = "+" if value > 0 else ""
    return f"{sign}{value * 100:.1f}%"


def search_in_report(report: dict, query: str, sources: list[str], limit: int) -> dict:
    """Full-text search across all data sources in a report.

    Args:
        report: Loaded report dict.
        query: Search keyword or regex pattern.
        sources: List of source types to search (empty = all).
        limit: Max results per source.

    Returns:
        Dict of source -> list of matching items.
    """
    import re as _re

    try:
        pattern = _re.compile(query, _re.IGNORECASE)
    except _re.error:
        # Fall back to literal match if regex is invalid
        pattern = _re.compile(_re.escape(query), _re.IGNORECASE)

    search_all = not sources
    results = {}

    # Search datasets
    if search_all or "datasets" in sources:
        matched = []
        for ds in report.get("datasets", []):
            text = " ".join(
                [
                    ds.get("id", ""),
                    ds.get("description", ""),
                    ds.get("category", ""),
                    " ".join(ds.get("all_categories", [])),
                ]
            )
            if pattern.search(text):
                matched.append(
                    {
                        "id": ds.get("id"),
                        "category": ds.get("category"),
                        "downloads": ds.get("downloads", 0),
                        "description": (ds.get("description") or "")[:120],
                    }
                )
            if len(matched) >= limit:
                break
        if matched:
            results["datasets"] = matched

    # Search GitHub repos
    if search_all or "github" in sources:
        matched = []
        for org in report.get("github_activity", []):
            for repo in org.get("repos_updated", []):
                text = " ".join(
                    [
                        repo.get("name", ""),
                        repo.get("full_name", ""),
                        repo.get("description", ""),
                        " ".join(repo.get("topics", [])),
                        " ".join(repo.get("signals", [])),
                    ]
                )
                if pattern.search(text):
                    matched.append(
                        {
                            "name": repo.get("full_name"),
                            "description": (repo.get("description") or "")[:120],
                            "stars": repo.get("stars", 0),
                            "relevance": repo.get("relevance"),
                            "url": repo.get("url"),
                        }
                    )
                if len(matched) >= limit:
                    break
            if len(matched) >= limit:
                break
        if matched:
            results["github"] = matched

    # Search papers
    if search_all or "papers" in sources:
        matched = []
        for paper in report.get("papers", []):
            text = " ".join(
                [
                    paper.get("title", ""),
                    paper.get("abstract", ""),
                    " ".join(paper.get("categories", [])),
                ]
            )
            if pattern.search(text):
                matched.append(
                    {
                        "title": paper.get("title"),
                        "url": paper.get("url"),
                        "source": paper.get("source"),
                    }
                )
            if len(matched) >= limit:
                break
        if matched:
            results["papers"] = matched

    # Search blog posts
    if search_all or "blogs" in sources:
        matched = []
        for blog in report.get("blog_posts", []):
            source_name = blog.get("source", "")
            for article in blog.get("articles", []):
                text = " ".join(
                    [
                        source_name,
                        article.get("title", ""),
                        article.get("snippet", ""),
                        " ".join(article.get("signals", [])),
                    ]
                )
                if pattern.search(text):
                    matched.append(
                        {
                            "source": source_name,
                            "title": article.get("title"),
                            "url": article.get("url"),
                            "date": article.get("date", ""),
                        }
                    )
                if len(matched) >= limit:
                    break
            if len(matched) >= limit:
                break
        if matched:
            results["blogs"] = matched

    # Search X/Twitter
    if search_all or "x" in sources:
        matched = []
        x_data = report.get("x_activity", {})
        for acct in x_data.get("accounts", []):
            for tweet in acct.get("relevant_tweets", []):
                text = " ".join(
                    [
                        tweet.get("username", ""),
                        tweet.get("text", ""),
                    ]
                )
                if pattern.search(text):
                    matched.append(
                        {
                            "username": tweet.get("username"),
                            "text": tweet.get("text", "")[:200],
                            "url": tweet.get("url"),
                            "date": tweet.get("date", ""),
                        }
                    )
                if len(matched) >= limit:
                    break
            if len(matched) >= limit:
                break
        if matched:
            results["x"] = matched

    return results


def diff_reports(report_a: dict, report_b: dict) -> dict:
    """Compare two reports and identify changes.

    Args:
        report_a: Older report.
        report_b: Newer report.

    Returns:
        Dict with summary changes and new/removed items.
    """
    diff = {
        "period_a": report_a.get("generated_at", "")[:10],
        "period_b": report_b.get("generated_at", "")[:10],
        "summary_changes": {},
        "new_items": {},
        "removed_items": {},
    }

    # Compare summary counts
    sum_a = report_a.get("summary", {})
    sum_b = report_b.get("summary", {})
    for key in [
        "total_datasets",
        "total_github_repos",
        "total_github_repos_high_relevance",
        "total_papers",
        "total_blog_posts",
    ]:
        val_a = sum_a.get(key, 0)
        val_b = sum_b.get(key, 0)
        delta = val_b - val_a
        if delta != 0:
            diff["summary_changes"][key] = {"before": val_a, "after": val_b, "delta": delta}

    # Compare datasets
    ds_ids_a = {d.get("id") for d in report_a.get("datasets", [])}
    ds_ids_b = {d.get("id") for d in report_b.get("datasets", [])}
    new_ds = ds_ids_b - ds_ids_a
    removed_ds = ds_ids_a - ds_ids_b
    if new_ds:
        diff["new_items"]["datasets"] = sorted(new_ds)
    if removed_ds:
        diff["removed_items"]["datasets"] = sorted(removed_ds)

    # Compare GitHub repos
    def _get_repo_names(report):
        names = set()
        for org in report.get("github_activity", []):
            for repo in org.get("repos_updated", []):
                names.add(repo.get("full_name", ""))
        return names

    repos_a = _get_repo_names(report_a)
    repos_b = _get_repo_names(report_b)
    new_repos = repos_b - repos_a
    removed_repos = repos_a - repos_b
    if new_repos:
        diff["new_items"]["github_repos"] = sorted(new_repos)
    if removed_repos:
        diff["removed_items"]["github_repos"] = sorted(removed_repos)

    # Compare papers
    papers_a = {p.get("title", "") for p in report_a.get("papers", [])}
    papers_b = {p.get("title", "") for p in report_b.get("papers", [])}
    new_papers = papers_b - papers_a
    if new_papers:
        diff["new_items"]["papers"] = sorted(new_papers)

    # Compare blog articles
    def _get_blog_urls(report):
        urls = set()
        for blog in report.get("blog_posts", []):
            for article in blog.get("articles", []):
                urls.add(article.get("url", ""))
        return urls

    blogs_a = _get_blog_urls(report_a)
    blogs_b = _get_blog_urls(report_b)
    new_blogs = blogs_b - blogs_a
    if new_blogs:
        diff["new_items"]["blog_articles"] = sorted(new_blogs)

    return diff


@server.call_tool()
async def call_tool(name: str, arguments: dict):
    """Handle tool calls."""

    if name == "radar_scan":
        days = arguments.get("days", 7)
        sources = arguments.get("sources", [])

        # Run the scan
        venv_python = PROJECT_ROOT / ".venv" / "bin" / "python"
        if not venv_python.exists():
            venv_python = "python3"
        else:
            venv_python = str(venv_python)

        cmd = [venv_python, str(PROJECT_ROOT / "src" / "main_intel.py"), "--days", str(days)]

        # Map source filters to CLI flags
        if sources:
            all_sources = {"github", "blogs", "papers", "labs", "vendors", "x"}
            skip = all_sources - set(sources)
            flag_map = {
                "github": "--no-github",
                "blogs": "--no-blogs",
                "papers": "--no-papers",
                "labs": "--no-labs",
                "vendors": "--no-vendors",
                "x": "--no-x",
            }
            for src in skip:
                if src in flag_map:
                    cmd.append(flag_map[src])

        try:
            result = subprocess.run(
                cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True, timeout=300
            )

            # Get the generated report
            report = get_latest_report()
            if report:
                summary = report.get("summary", {})
                return [
                    TextContent(
                        type="text",
                        text=f"""扫描完成！

**统计摘要:**
- 数据集: {summary.get("total_datasets", 0)} 个
- GitHub 组织: {summary.get("total_github_orgs", 0)} 个
- GitHub 仓库: {summary.get("total_github_repos", 0)} 个 ({summary.get("total_github_repos_high_relevance", 0)} 个高相关)
- 论文: {summary.get("total_papers", 0)} 篇
- 博客文章: {summary.get("total_blog_posts", 0)} 篇

报告已保存到: {get_latest_report_path()}

使用 `radar_datasets`、`radar_github`、`radar_papers` 查看详细内容。""",
                    )
                ]
            else:
                return [
                    TextContent(
                        type="text",
                        text=f"扫描完成，但未找到报告。\n\n输出:\n{result.stdout}\n\n错误:\n{result.stderr}",
                    )
                ]

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

        return [
            TextContent(
                type="text",
                text=f"""**AI Dataset Radar 报告摘要**

生成时间: {generated_at}
扫描周期: {period.get("days", 7)} 天 ({period.get("start", "")[:10]} ~ {period.get("end", "")[:10]})

**统计:**
| 类别 | 数量 |
|------|------|
| 数据集 | {summary.get("total_datasets", 0)} |
| GitHub 组织 | {summary.get("total_github_orgs", 0)} |
| GitHub 仓库 | {summary.get("total_github_repos", 0)} |
| 高相关仓库 | {summary.get("total_github_repos_high_relevance", 0)} |
| 论文 | {summary.get("total_papers", 0)} |
| 博客文章 | {summary.get("total_blog_posts", 0)} |

**数据集类型分布:**
{json.dumps(report.get("datasets_by_type", {}), indent=2, ensure_ascii=False)}
""",
            )
        ]

    elif name == "radar_datasets":
        report = get_latest_report()
        if not report:
            return [TextContent(type="text", text="没有找到报告，请先运行 `radar_scan`。")]

        datasets = report.get("datasets", [])
        category = arguments.get("category", "")
        org_filter = arguments.get("org", "")
        limit = arguments.get("limit", 10)

        if category:
            datasets = [
                d
                for d in datasets
                if d.get("category") == category or category in d.get("all_categories", [])
            ]

        if org_filter:
            org_lower = org_filter.lower()
            datasets = [
                d for d in datasets if org_lower in (d.get("id") or "").lower().split("/")[0]
            ]

        datasets = datasets[:limit]

        if not datasets:
            filters = []
            if category:
                filters.append(f"类型={category}")
            if org_filter:
                filters.append(f"组织={org_filter}")
            filter_desc = f" ({', '.join(filters)})" if filters else ""
            return [TextContent(type="text", text=f"没有找到{filter_desc}数据集。")]

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
        org_filter = arguments.get("org", "")

        if org_filter:
            org_lower = org_filter.lower()
            github_activity = [o for o in github_activity if org_lower in o.get("org", "").lower()]

        lines = [f"**GitHub 活动 (相关性: {relevance_filter}):**\n"]

        for org in github_activity:
            org_name = org.get("org", "unknown")
            repos = org.get("repos_updated", [])

            filtered_repos = (
                [r for r in repos if r.get("relevance") == relevance_filter]
                if relevance_filter
                else repos
            )

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
            lines.append(
                f"- **{category}**: {', '.join(org_names[:5])}{'...' if len(org_names) > 5 else ''}"
            )

        lines.append("\n### 监控的数据供应商")
        for category, vendors in watched_vendors.items():
            if category == "blogs":
                continue
            vendor_names = list(vendors.keys()) if isinstance(vendors, dict) else vendors
            lines.append(
                f"- **{category}**: {', '.join(vendor_names[:5]) if vendor_names else '无'}"
            )

        lines.append("\n### 关注的数据类型")
        for dtype in list(priority_types.keys())[:8]:
            lines.append(f"- {dtype}")

        return [TextContent(type="text", text="\n".join(lines))]

    elif name == "radar_blogs":
        report = get_latest_report()
        if not report:
            return [TextContent(type="text", text="没有找到报告，请先运行 `radar_scan`。")]

        blog_posts = report.get("blog_posts", [])
        source_filter = arguments.get("source", "")
        limit = arguments.get("limit", 20)

        lines = ["**博客文章动态:**\n"]
        total_articles = 0
        shown_articles = 0

        for blog in blog_posts:
            source = blog.get("source", "unknown")
            articles = blog.get("articles", [])

            if not articles:
                continue

            # Filter by source if specified
            if source_filter and source_filter.lower() not in source.lower():
                continue

            total_articles += len(articles)

            lines.append(f"### {source} ({len(articles)} 篇)")
            for article in articles[:5]:  # Max 5 per source
                if shown_articles >= limit:
                    break
                title = article.get("title", "无标题")[:60]
                url = article.get("url", "")
                date = article.get("date", "")
                signals = article.get("signals", [])

                lines.append(f"- [{title}]({url})")
                if date or signals:
                    meta = []
                    if date:
                        meta.append(date)
                    if signals:
                        meta.append(f"信号: {', '.join(signals[:3])}")
                    lines.append(f"  - {' | '.join(meta)}")
                shown_articles += 1

            lines.append("")

            if shown_articles >= limit:
                break

        if total_articles == 0:
            return [
                TextContent(
                    type="text",
                    text=f"没有找到{'来自 ' + source_filter + ' 的' if source_filter else ''}博客文章。",
                )
            ]

        # Add summary
        active_sources = len([b for b in blog_posts if b.get("articles")])
        lines.insert(1, f"共 {active_sources} 个活跃博客源，{total_articles} 篇文章\n")

        return [TextContent(type="text", text="\n".join(lines))]

    elif name == "radar_trend":
        db_path = PROJECT_ROOT / "data" / "radar.db"
        if not db_path.exists():
            return [TextContent(type="text", text="数据库不存在。请先运行 `radar_scan` 积累数据。")]

        try:
            from db import RadarDatabase
            from analyzers.trend import TrendAnalyzer

            db = RadarDatabase(str(db_path))
            analyzer = TrendAnalyzer(db)

            mode = arguments.get("mode", "top_growing")
            days = arguments.get("days", 7)
            limit = arguments.get("limit", 10)
            dataset_id = arguments.get("dataset_id", "")

            if mode == "dataset" and dataset_id:
                result = analyzer.get_dataset_trend(dataset_id)
                if not result:
                    return [
                        TextContent(type="text", text=f"未找到数据集 '{dataset_id}' 的趋势数据。")
                    ]

                ds = result["dataset"]
                history = result["history"]
                lines = [f"**数据集趋势: {ds.get('dataset_id', dataset_id)}**\n"]
                lines.append(f"- 7 天增长: {_fmt_growth(result.get('growth_7d'))}")
                lines.append(f"- 30 天增长: {_fmt_growth(result.get('growth_30d'))}")
                if history:
                    lines.append(f"\n**最近 {len(history)} 天下载量:**")
                    for h in history[-10:]:
                        lines.append(f"- {h.get('date', '')}: {h.get('downloads', 0):,}")
                db.close()
                return [TextContent(type="text", text="\n".join(lines))]

            elif mode == "top_growing":
                datasets = analyzer.get_top_growing_datasets(days=days, limit=limit)
                lines = [f"**增长最快数据集 ({days} 天):**\n"]
                if not datasets:
                    lines.append("暂无趋势数据。需要多天扫描数据积累后才能计算。")
                for i, ds in enumerate(datasets, 1):
                    growth = _fmt_growth(ds.get("growth"))
                    downloads = ds.get("current_downloads", 0)
                    lines.append(f"{i}. **{ds.get('name', ds.get('dataset_id', '?'))}**")
                    lines.append(f"   增长: {growth} | 下载: {downloads:,}")
                db.close()
                return [TextContent(type="text", text="\n".join(lines))]

            elif mode == "rising":
                datasets = analyzer.get_rising_datasets(days=days, limit=limit)
                lines = [f"**上升中数据集 ({days} 天, 增长 ≥50%):**\n"]
                if not datasets:
                    lines.append("暂无上升数据集。")
                for ds in datasets:
                    growth = _fmt_growth(ds.get("growth"))
                    lines.append(f"- **{ds.get('name', ds.get('dataset_id', '?'))}** — {growth}")
                db.close()
                return [TextContent(type="text", text="\n".join(lines))]

            elif mode == "breakthroughs":
                datasets = analyzer.get_breakthrough_datasets(days=days, limit=limit)
                lines = [f"**突破性增长数据集 ({days} 天):**\n"]
                if not datasets:
                    lines.append("暂无突破性增长。")
                for ds in datasets:
                    old = ds.get("old_downloads", 0) or 0
                    current = ds.get("current_downloads", 0)
                    lines.append(
                        f"- **{ds.get('name', ds.get('dataset_id', '?'))}**: {old:,} → {current:,}"
                    )
                db.close()
                return [TextContent(type="text", text="\n".join(lines))]

            db.close()
            return [TextContent(type="text", text=f"未知模式: {mode}")]

        except Exception as e:
            return [TextContent(type="text", text=f"趋势查询失败: {e}")]

    elif name == "radar_history":
        all_reports = get_all_reports_sorted()
        limit = arguments.get("limit", 10)
        reports_to_show = all_reports[:limit]

        if not reports_to_show:
            return [TextContent(type="text", text="没有找到历史报告。请先运行 `radar_scan`。")]

        lines = [f"**历史扫描报告 (最近 {len(reports_to_show)} 期):**\n"]
        lines.append("| 日期 | 数据集 | GitHub 仓库 | 高相关 | 论文 | 博客 |")
        lines.append("|------|--------|-------------|--------|------|------|")

        for rp in reports_to_show:
            try:
                with open(rp, "r", encoding="utf-8") as f:
                    r = json.load(f)
                s = r.get("summary", {})
                date = r.get("generated_at", "")[:10]
                lines.append(
                    f"| {date} "
                    f"| {s.get('total_datasets', 0)} "
                    f"| {s.get('total_github_repos', 0)} "
                    f"| {s.get('total_github_repos_high_relevance', 0)} "
                    f"| {s.get('total_papers', 0)} "
                    f"| {s.get('total_blog_posts', 0)} |"
                )
            except (json.JSONDecodeError, OSError):
                continue

        # Add trend line if multiple reports
        if len(reports_to_show) >= 2:
            try:
                with open(reports_to_show[0], "r") as f:
                    latest = json.load(f)
                with open(reports_to_show[-1], "r") as f:
                    oldest = json.load(f)
                s_new = latest.get("summary", {})
                s_old = oldest.get("summary", {})
                lines.append("")
                lines.append(
                    f"**趋势 ({oldest.get('generated_at', '')[:10]} → {latest.get('generated_at', '')[:10]}):**"
                )
                for key, label in [
                    ("total_datasets", "数据集"),
                    ("total_github_repos", "GitHub 仓库"),
                    ("total_papers", "论文"),
                    ("total_blog_posts", "博客"),
                ]:
                    delta = s_new.get(key, 0) - s_old.get(key, 0)
                    sign = "+" if delta > 0 else ""
                    lines.append(f"- {label}: {sign}{delta}")
            except (json.JSONDecodeError, OSError):
                pass

        return [TextContent(type="text", text="\n".join(lines))]

    elif name == "radar_search":
        report = get_latest_report()
        if not report:
            return [TextContent(type="text", text="没有找到报告，请先运行 `radar_scan`。")]

        query = arguments.get("query", "")
        if not query:
            return [TextContent(type="text", text="请提供搜索关键词 (query 参数)。")]

        sources = arguments.get("sources", [])
        limit = arguments.get("limit", 10)

        results = search_in_report(report, query, sources, limit)

        if not results:
            source_desc = f" (数据源: {', '.join(sources)})" if sources else ""
            return [TextContent(type="text", text=f"未找到匹配 '{query}' 的结果{source_desc}。")]

        lines = [f"**搜索结果: '{query}'**\n"]
        total = 0

        if "datasets" in results:
            lines.append(f"### 数据集 ({len(results['datasets'])} 条)")
            for item in results["datasets"]:
                lines.append(
                    f"- **{item['id']}** [{item.get('category', '')}] — downloads: {item.get('downloads', 0):,}"
                )
                if item.get("description"):
                    lines.append(f"  {item['description']}")
            lines.append("")
            total += len(results["datasets"])

        if "github" in results:
            lines.append(f"### GitHub 仓库 ({len(results['github'])} 条)")
            for item in results["github"]:
                lines.append(
                    f"- **{item['name']}** ⭐ {item.get('stars', 0)} [{item.get('relevance', '')}]"
                )
                if item.get("description"):
                    lines.append(f"  {item['description']}")
                if item.get("url"):
                    lines.append(f"  {item['url']}")
            lines.append("")
            total += len(results["github"])

        if "papers" in results:
            lines.append(f"### 论文 ({len(results['papers'])} 条)")
            for item in results["papers"]:
                lines.append(f"- **{item['title']}**")
                if item.get("url"):
                    lines.append(f"  {item['url']}")
            lines.append("")
            total += len(results["papers"])

        if "blogs" in results:
            lines.append(f"### 博客 ({len(results['blogs'])} 条)")
            for item in results["blogs"]:
                lines.append(
                    f"- [{item.get('title', '')}]({item.get('url', '')}) — {item.get('source', '')}"
                )
            lines.append("")
            total += len(results["blogs"])

        if "x" in results:
            lines.append(f"### X/Twitter ({len(results['x'])} 条)")
            for item in results["x"]:
                lines.append(f"- @{item.get('username', '')}: {item.get('text', '')[:120]}...")
                if item.get("url"):
                    lines.append(f"  {item['url']}")
            lines.append("")
            total += len(results["x"])

        lines.insert(1, f"共 {total} 条匹配结果\n")

        return [TextContent(type="text", text="\n".join(lines))]

    elif name == "radar_diff":
        all_reports = get_all_reports_sorted()
        if len(all_reports) < 2:
            return [
                TextContent(
                    type="text",
                    text="需要至少两份报告才能对比。当前报告数: " + str(len(all_reports)),
                )
            ]

        date_a = arguments.get("date_a", "")
        date_b = arguments.get("date_b", "")

        if date_b:
            report_b = get_report_by_date(date_b)
            if not report_b:
                return [TextContent(type="text", text=f"未找到日期 {date_b} 的报告。")]
        else:
            with open(all_reports[0], "r", encoding="utf-8") as f:
                report_b = json.load(f)

        if date_a:
            report_a = get_report_by_date(date_a)
            if not report_a:
                return [TextContent(type="text", text=f"未找到日期 {date_a} 的报告。")]
        else:
            with open(all_reports[1], "r", encoding="utf-8") as f:
                report_a = json.load(f)

        diff = diff_reports(report_a, report_b)

        lines = [f"**报告对比: {diff['period_a']} → {diff['period_b']}**\n"]

        # Summary changes
        if diff["summary_changes"]:
            lines.append("### 统计变化")
            label_map = {
                "total_datasets": "数据集",
                "total_github_repos": "GitHub 仓库",
                "total_github_repos_high_relevance": "高相关仓库",
                "total_papers": "论文",
                "total_blog_posts": "博客文章",
            }
            for key, change in diff["summary_changes"].items():
                label = label_map.get(key, key)
                sign = "+" if change["delta"] > 0 else ""
                lines.append(
                    f"- {label}: {change['before']} → {change['after']} ({sign}{change['delta']})"
                )
            lines.append("")

        # New items
        if diff["new_items"]:
            lines.append("### 新增项目")
            for category, items in diff["new_items"].items():
                label_map = {
                    "datasets": "数据集",
                    "github_repos": "GitHub 仓库",
                    "papers": "论文",
                    "blog_articles": "博客文章",
                }
                label = label_map.get(category, category)
                lines.append(f"**{label}** (+{len(items)})")
                for item in items[:10]:
                    lines.append(f"- {item}")
                if len(items) > 10:
                    lines.append(f"  ...及其他 {len(items) - 10} 项")
                lines.append("")

        # Removed items
        if diff["removed_items"]:
            lines.append("### 消失项目")
            for category, items in diff["removed_items"].items():
                label_map = {"datasets": "数据集", "github_repos": "GitHub 仓库"}
                label = label_map.get(category, category)
                lines.append(f"**{label}** (-{len(items)})")
                for item in items[:10]:
                    lines.append(f"- {item}")
                if len(items) > 10:
                    lines.append(f"  ...及其他 {len(items) - 10} 项")
                lines.append("")

        if not diff["summary_changes"] and not diff["new_items"] and not diff["removed_items"]:
            lines.append("两份报告内容完全一致，无变化。")

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
