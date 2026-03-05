#!/usr/bin/env python3
"""AI Dataset Radar - Competitive Intelligence System.

Main entry point for the competitive intelligence workflow.
Integrates HuggingFace, GitHub, and Blog monitoring.
"""

import argparse
import asyncio
import json
import math
import os
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

import yaml
from dotenv import load_dotenv

# Load .env file (for ANTHROPIC_API_KEY etc.)
load_dotenv()

# Add src to path
src_dir = Path(__file__).parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from utils.async_http import AsyncHTTPClient
from utils.logging_config import get_logger, setup_logging

logger = get_logger("main_intel")

from trackers.org_tracker import OrgTracker
from trackers.github_tracker import GitHubTracker
from trackers.blog_tracker import BlogTracker
from trackers.x_tracker import XTracker
from trackers.reddit_tracker import RedditTracker
from trackers.hn_tracker import HNTracker
from trackers.github_trending_tracker import GitHubTrendingTracker
from trackers.producthunt_tracker import ProductHuntTracker
from analyzers.data_type_classifier import DataTypeClassifier, DataType
from analyzers.paper_filter import PaperFilter
from analyzers.competitor_matrix import CompetitorMatrix
from analyzers.dataset_lineage import DatasetLineageTracker
from analyzers.org_graph import OrgRelationshipGraph
from intel_report import IntelReportGenerator
from scrapers.arxiv import ArxivScraper
from scrapers.hf_papers import HFPapersScraper
from scrapers.huggingface import HuggingFaceScraper
from scrapers.paperswithcode import PapersWithCodeScraper
from output_formatter import DualOutputFormatter
from db import RadarDatabase
from analyzers.trend import TrendAnalyzer
from _version import __version__


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        Configuration dictionary, or empty dict if file is invalid.
    """
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
            return config if config else {}
    except FileNotFoundError:
        logger.warning("Config file not found: %s, using defaults", config_path)
        return {}
    except yaml.YAMLError as e:
        logger.warning("Invalid YAML in %s: %s, using defaults", config_path, e)
        return {}


def validate_config(config: dict) -> list[str]:
    """Validate config structure and warn about missing sections.

    Args:
        config: Loaded configuration dictionary.

    Returns:
        List of warning messages (empty if all OK).
    """
    warnings = []

    expected_sections = {
        "watched_orgs": dict,
        "sources": dict,
        "github": dict,
    }
    for section, expected_type in expected_sections.items():
        val = config.get(section)
        if val is None:
            warnings.append(f"Missing config section: '{section}'")
        elif not isinstance(val, expected_type):
            warnings.append(f"Config '{section}' should be {expected_type.__name__}, got {type(val).__name__}")

    # Check blog sources exist
    blogs = (
        config.get("data_vendors", {}).get("blogs", [])
        or config.get("watched_vendors", {}).get("blogs", [])
        or config.get("blogs", [])
    )
    if not blogs:
        warnings.append("No blog sources configured (checked data_vendors.blogs, watched_vendors.blogs, blogs)")

    # Check GitHub orgs
    github = config.get("github", {})
    orgs = github.get("orgs", {})
    if not orgs.get("ai_labs") and not orgs.get("data_vendors"):
        warnings.append("No GitHub orgs configured in github.orgs.ai_labs or github.orgs.data_vendors")

    for msg in warnings:
        logger.warning("Config: %s", msg)

    return warnings


def format_insights_prompt(
    all_datasets: list,
    blog_activity: list,
    github_activity: list,
    papers: list,
    datasets_by_type: dict,
    lab_activity: dict = None,
    vendor_activity: dict = None,
    x_activity: dict = None,
    reddit_activity: dict = None,
    pwc_datasets: list = None,
    hn_activity: dict = None,
    kaggle_datasets: list = None,
    semantic_scholar_papers: list = None,
    gh_trending: dict = None,
    producthunt: dict = None,
) -> str:
    """Format data with analysis prompt for LLM consumption.

    This output is designed to be read by Claude Code / Claude App,
    which will then perform the analysis using its native LLM capabilities.
    Surfaces all available intelligence data with full context.
    """
    lines = []
    lines.append("\n" + "=" * 60)
    lines.append("  AI Dataset Radar - 竞争情报分析材料")
    lines.append("=" * 60 + "\n")

    # ── Section 1: Lab Activity (org-by-org with datasets AND models) ──
    lines.append("## 一、AI Labs 动态（按组织）\n")
    labs = (lab_activity or {}).get("labs", {})
    has_lab_activity = False

    category_names = {
        "frontier_labs": "Frontier Labs（一线实验室）",
        "emerging_labs": "Emerging Labs（新兴实验室）",
        "research_labs": "Research Labs（研究机构）",
        "china_opensource": "中国开源大模型",
        "china_closedsource": "中国闭源大模型",
    }

    for cat_key, cat_display in category_names.items():
        cat_data = labs.get(cat_key, {})
        # Filter to orgs with actual activity
        active_orgs = {k: v for k, v in cat_data.items() if v.get("datasets") or v.get("models")}
        if not active_orgs:
            continue

        has_lab_activity = True
        lines.append(f"### {cat_display}\n")

        for org_name, org_data in active_orgs.items():
            org_display = org_name.replace("_", " ").title()
            ds_list = org_data.get("datasets", [])
            model_list = org_data.get("models", [])
            lines.append(f"**{org_display}** — {len(ds_list)} 数据集, {len(model_list)} 模型")

            # Datasets with full info
            for ds in ds_list:
                ds_id = ds.get("id", "")
                downloads = ds.get("downloads", 0)
                likes = ds.get("likes", 0)
                created = ds.get("created_at") or ds.get("createdAt") or ""
                if created:
                    created = created[:10]
                desc = ds.get("description", "")
                # Clean up description whitespace
                if desc:
                    desc = " ".join(desc.split())[:300]
                date_str = f" ({created})" if created else ""
                lines.append(f"- 📦 **{ds_id}**{date_str} downloads: {downloads:,}, likes: {likes}")
                if desc:
                    lines.append(f"  {desc}")
                # Show meaningful tags (filter out noise)
                tags = ds.get("tags", [])
                meaningful_tags = [
                    t
                    for t in tags
                    if not t.startswith(
                        (
                            "region:",
                            "library:",
                            "size_categories:",
                            "format:",
                            "arxiv:",
                            "language:",
                        )
                    )
                    and t not in ("region:us",)
                ][:8]
                if meaningful_tags:
                    lines.append(f"  标签: {', '.join(meaningful_tags)}")

            # Models with context - show top models by downloads+likes, limit noise
            notable_models = [
                m for m in model_list if m.get("downloads", 0) > 0 or m.get("likes", 0) > 0
            ]
            if not notable_models:
                # All models are zero-activity, just summarize
                if model_list:
                    sample = model_list[0].get("id", "").split("/")[-1] if model_list else ""
                    lines.append(f"- 🤖 {len(model_list)} 个模型（均无下载/点赞，如 {sample} 等）")
                model_list_to_show = []
            else:
                top_models = sorted(
                    notable_models, key=lambda m: -(m.get("downloads", 0) + m.get("likes", 0) * 100)
                )
                model_list_to_show = top_models[:5]
            for model in model_list_to_show:
                model_id = model.get("id", "")
                downloads = model.get("downloads", 0)
                likes = model.get("likes", 0)
                pipeline = model.get("pipeline_tag", "")
                created = model.get("created_at") or model.get("createdAt") or ""
                if created:
                    created = created[:10]
                model_tags = model.get("tags", [])
                # Extract meaningful tags for models
                meaningful = [
                    t
                    for t in model_tags
                    if not t.startswith(
                        ("region:", "base_model:", "endpoints_", "license:", "arxiv:")
                    )
                    and t
                    not in (
                        "safetensors",
                        "transformers",
                        "pytorch",
                        "en",
                        "model_hub_mixin",
                        "pytorch_model_hub_mixin",
                    )
                ][:6]
                model_date_str = f" ({created})" if created else ""
                lines.append(
                    f"- 🤖 **{model_id}**{model_date_str} downloads: {downloads:,}, likes: {likes}, pipeline: {pipeline}"
                )
                if meaningful:
                    lines.append(f"  标签: {', '.join(meaningful)}")
            if len(notable_models) > 5:
                lines.append(f"  （另有 {len(notable_models) - 5} 个模型省略）")

            lines.append("")

    if not has_lab_activity:
        lines.append("本周无 AI Labs 新活动\n")

    # ── Section 2: Vendor Activity ──
    lines.append("## 二、数据供应商动态（竞品）\n")
    vendors = (vendor_activity or {}).get("vendors", {})
    has_vendor_activity = False

    for tier_name, tier_data in vendors.items():
        active_vendors = {
            k: v for k, v in tier_data.items() if v.get("datasets") or v.get("models")
        }
        if not active_vendors:
            continue

        has_vendor_activity = True
        lines.append(f"### {tier_name.replace('_', ' ').title()}\n")

        for vendor_name, vendor_data in active_vendors.items():
            vendor_display = vendor_name.replace("_", " ").title()
            ds_list = vendor_data.get("datasets", [])
            model_list = vendor_data.get("models", [])
            lines.append(f"**{vendor_display}** — {len(ds_list)} 数据集, {len(model_list)} 模型")

            for ds in ds_list:
                ds_id = ds.get("id", "")
                downloads = ds.get("downloads", 0)
                created = ds.get("created_at") or ds.get("createdAt") or ""
                if created:
                    created = created[:10]
                desc = ds.get("description", "")
                if desc:
                    desc = " ".join(desc.split())[:300]
                date_str = f" ({created})" if created else ""
                lines.append(f"- 📦 **{ds_id}**{date_str} downloads: {downloads:,}")
                if desc:
                    lines.append(f"  {desc}")
            lines.append("")

    if not has_vendor_activity:
        lines.append("本周无供应商 HuggingFace 新活动\n")

    # ── Section 3: Dataset Classification Results ──
    lines.append("## 三、数据集分类分析\n")
    if datasets_by_type:
        # Show classified types first, "other" last
        classified = {
            k: v
            for k, v in datasets_by_type.items()
            if (k.value if hasattr(k, "value") else str(k)) != "other" and v
        }
        other = {
            k: v
            for k, v in datasets_by_type.items()
            if (k.value if hasattr(k, "value") else str(k)) == "other" and v
        }

        total = sum(len(v) for v in datasets_by_type.values())
        classified_count = sum(len(v) for v in classified.values())
        lines.append(f"共 {total} 个数据集，已分类 {classified_count} 个：\n")

        for dtype, ds_list in classified.items():
            type_name = dtype.value if hasattr(dtype, "value") else str(dtype)
            lines.append(
                f"- **{type_name}**: {len(ds_list)} 个 — {', '.join((ds.get('id', '') if isinstance(ds, dict) else str(ds)) for ds in ds_list[:5])}"
            )

        if other:
            other_list = list(other.values())[0]
            lines.append(
                f"- **未分类**: {len(other_list)} 个 — {', '.join((ds.get('id', '') if isinstance(ds, dict) else str(ds)) for ds in other_list[:5])}"
            )
        lines.append("")
    else:
        lines.append("无分类数据\n")

    # ── Section 4: Blog Activity (full titles, more articles) ──
    lines.append("## 四、博客要闻\n")
    if blog_activity:
        active_blogs = [b for b in blog_activity if b.get("articles")]
        if active_blogs:
            for blog in active_blogs:
                source = blog.get("source", "未知")
                articles = blog.get("articles", [])[:5]
                if articles:
                    lines.append(f"### {source}")
                    for art in articles:
                        title = art.get("title", "无标题")
                        url = art.get("url", "")
                        summary = art.get("summary", "")
                        if summary:
                            summary = " ".join(summary.split())[:200]
                        lines.append(f"- [{title}]({url})")
                        if summary:
                            lines.append(f"  {summary}")
                    lines.append("")
        else:
            lines.append("无博客更新\n")
    else:
        lines.append("无博客更新\n")

    # ── Section 5: GitHub Activity (high + medium relevance) ──
    lines.append("## 五、GitHub 活动\n")
    if github_activity:
        # Collect all repos with relevance info
        all_repos = []
        for org in github_activity:
            org_name = org.get("org", "")
            for repo in org.get("repos_updated", []):
                repo_copy = dict(repo)
                repo_copy["org"] = org_name
                all_repos.append(repo_copy)

        # High relevance
        high = [r for r in all_repos if r.get("relevance") == "high"]
        high = sorted(high, key=lambda x: -x.get("stars", 0))
        # Medium relevance
        medium = [r for r in all_repos if r.get("relevance") == "medium"]
        medium = sorted(medium, key=lambda x: -x.get("stars", 0))[:10]

        if high:
            lines.append(f"### 高相关 ({len(high)} 个)")
            for repo in high:
                lines.append(
                    f"- **{repo.get('org')}/{repo.get('name')}** ⭐ {repo.get('stars', 0)}"
                )
                if repo.get("description"):
                    lines.append(f"  {repo.get('description', '')[:120]}")
                signals = repo.get("signals", [])
                if signals:
                    lines.append(f"  信号: {', '.join(str(s) for s in signals[:5])}")
            lines.append("")

        if medium:
            lines.append(f"### 中相关 (Top {len(medium)})")
            for repo in medium:
                lines.append(
                    f"- **{repo.get('org')}/{repo.get('name')}** ⭐ {repo.get('stars', 0)}"
                )
                if repo.get("description"):
                    lines.append(f"  {repo.get('description', '')[:120]}")
            lines.append("")

        # Summary stats
        total_repos = len(all_repos)
        active_orgs = len([o for o in github_activity if o.get("repos_updated")])
        lines.append(f"共监控 {active_orgs} 个组织，{total_repos} 个活跃仓库\n")
    else:
        lines.append("无 GitHub 活动\n")

    # ── Section 5.5: X/Twitter Activity ──
    lines.append("## 5.5、X/Twitter 动态\n")
    x_data = x_activity or {}
    x_accounts = x_data.get("accounts", [])
    x_search = x_data.get("search_results", [])
    if x_accounts:
        for acct in x_accounts:
            username = acct.get("username", "")
            tweets = acct.get("relevant_tweets", [])
            if tweets:
                lines.append(f"### @{username} ({len(tweets)} 条相关推文)")
                for tweet in tweets[:5]:
                    text = tweet.get("text", "")[:200]
                    url = tweet.get("url", "")
                    date = tweet.get("date", "")
                    signals = tweet.get("signals", [])
                    lines.append(f"- [{date}] {text}")
                    if url:
                        lines.append(f"  链接: {url}")
                    if signals:
                        lines.append(f"  信号: {', '.join(signals)}")
                lines.append("")
    if x_search:
        lines.append("### 关键词搜索结果")
        for tweet in x_search[:10]:
            text = tweet.get("text", "")[:200]
            query = tweet.get("query", "")
            lines.append(f"- [{query}] {text}")
        lines.append("")
    if not x_accounts and not x_search:
        lines.append("无 X/Twitter 动态\n")

    # ── Section 5.6: Reddit ──
    lines.append("## 5.6、Reddit 社区动态\n")
    reddit_data = reddit_activity or {}
    reddit_posts = reddit_data.get("posts", [])
    if reddit_posts:
        for post in reddit_posts[:20]:
            title = post.get("title", "")[:150]
            url = post.get("url", "")
            date = post.get("date", "")
            score = post.get("score", 0)
            sub = post.get("subreddit", "")
            signals = post.get("signals", [])
            lines.append(f"- [{date}] r/{sub} (↑{score}) {title}")
            if url:
                lines.append(f"  链接: {url}")
            if signals:
                lines.append(f"  信号: {', '.join(signals)}")
        lines.append("")
    else:
        lines.append("无 Reddit 动态\n")

    # ── Section 5.7: Papers with Code ──
    lines.append("## 5.7、Papers with Code 数据集/榜单\n")
    pwc_items = (pwc_datasets or [])[:15]
    if pwc_items:
        for ds in pwc_items:
            name = ds.get("full_name") or ds.get("name", "")
            url = ds.get("url") or ds.get("homepage", "")
            paper_count = ds.get("paper_count", 0)
            desc = (ds.get("description") or "").strip()
            created_at = ds.get("created_at", "")
            meta = []
            if created_at:
                meta.append(created_at[:10])
            meta.append(f"论文: {paper_count}")
            modalities = ds.get("modalities") or ds.get("languages") or []
            if modalities:
                meta.append(f"模态: {', '.join(modalities[:3])}")
            title = name or "未命名"
            if url:
                lines.append(f"- **[{title}]({url})** ({'; '.join(meta)})")
            else:
                lines.append(f"- **{title}** ({'; '.join(meta)})")
            if desc:
                lines.append(f"  {desc[:400]}" + ("…" if len(desc) > 400 else ""))
        lines.append("")
    else:
        lines.append("本周无 Papers with Code 更新\n")

    # ── Section 5.8: Hacker News ──
    lines.append("## 5.8、Hacker News 社区动态\n")
    hn_data = hn_activity or {}
    hn_stories = hn_data.get("stories", [])
    if hn_stories:
        for story in hn_stories[:20]:
            title = story.get("title", "")[:150]
            url = story.get("url") or story.get("hn_url", "")
            date = story.get("date", "")
            points = story.get("points", 0)
            comments = story.get("num_comments", 0)
            signals = story.get("signals", [])
            lines.append(f"- [{date}] (↑{points}, {comments}评) {title}")
            if url:
                lines.append(f"  链接: {url}")
            if signals:
                lines.append(f"  信号: {', '.join(signals)}")
        lines.append("")
    else:
        lines.append("无 Hacker News 动态\n")

    # ── Section 5.9: Kaggle ──
    lines.append("## 5.9、Kaggle 数据集动态\n")
    kaggle_items = (kaggle_datasets or [])[:15]
    if kaggle_items:
        for ds in kaggle_items:
            name = ds.get("name", "")
            url = ds.get("url", "")
            author = ds.get("author", "")
            downloads = ds.get("downloads", 0)
            votes = ds.get("votes", 0)
            desc = (ds.get("description") or "").strip()[:200]
            tags = ds.get("tags", [])
            meta = []
            if author:
                meta.append(author)
            meta.append(f"↓{downloads}")
            if votes:
                meta.append(f"♥{votes}")
            if tags:
                meta.append(", ".join(tags[:3]))
            title_str = f"[{name}]({url})" if url else name
            lines.append(f"- **{title_str}** ({'; '.join(meta)})")
            if desc:
                lines.append(f"  {desc}")
        lines.append("")
    else:
        lines.append("无 Kaggle 数据集动态\n")

    # ── Section 5.10: Semantic Scholar ──
    lines.append("## 5.10、Semantic Scholar 高影响力论文\n")
    ss_items = (semantic_scholar_papers or [])[:20]
    if ss_items:
        for p in ss_items:
            title = p.get("title", "")[:150]
            url = p.get("url", "")
            citations = p.get("citation_count", 0)
            venue = p.get("venue", "")
            year = p.get("year", "")
            authors = ", ".join(p.get("authors", [])[:3])
            meta = []
            if citations:
                meta.append(f"引用 {citations}")
            if venue:
                meta.append(venue)
            if year:
                meta.append(str(year))
            title_str = f"[{title}]({url})" if url else title
            lines.append(f"- **{title_str}** ({'; '.join(meta)})")
            if authors:
                lines.append(f"  作者: {authors}")
        lines.append("")
    else:
        lines.append("无 Semantic Scholar 数据\n")

    # ── Section 5.11: GitHub Trending ──
    lines.append("## 5.11、GitHub Trending\n")
    gh_repos = (gh_trending or {}).get("repos", [])
    if gh_repos:
        for repo in gh_repos[:20]:
            name = repo.get("name", "")
            author = repo.get("author", "")
            url = repo.get("url", "")
            stars = repo.get("stars", 0)
            period_stars = repo.get("currentPeriodStars", 0)
            desc = (repo.get("description") or "")[:200]
            signals = repo.get("signals", [])
            title_str = f"[{author}/{name}]({url})" if url else f"{author}/{name}"
            lines.append(f"- **{title_str}** (★{stars}, +{period_stars}/周)")
            if desc:
                lines.append(f"  {desc}")
            if signals:
                lines.append(f"  信号: {', '.join(signals[:5])}")
        lines.append("")
    else:
        lines.append("无 GitHub Trending 数据\n")

    # ── Section 5.12: Product Hunt ──
    lines.append("## 5.12、Product Hunt AI 产品\n")
    ph_products = (producthunt or {}).get("products", [])
    if ph_products:
        for prod in ph_products[:15]:
            title = prod.get("title", "")
            url = prod.get("url", "")
            date = prod.get("date", "")
            tagline = prod.get("tagline", "")[:200]
            signals = prod.get("signals", [])
            title_str = f"[{title}]({url})" if url else title
            lines.append(f"- [{date}] **{title_str}**")
            if tagline:
                lines.append(f"  {tagline}")
            if signals:
                lines.append(f"  信号: {', '.join(signals[:5])}")
        lines.append("")
    else:
        lines.append("无 Product Hunt 数据\n")

    # ── Section 6: Papers (full titles, longer abstracts) ──
    lines.append("## 六、相关论文\n")
    if papers:
        # Group by category if available
        by_cat = {}
        for paper in papers:
            cat = paper.get("category", "其他")
            if cat not in by_cat:
                by_cat[cat] = []
            by_cat[cat].append(paper)

        for cat, paper_list in by_cat.items():
            if len(by_cat) > 1:
                lines.append(f"### {cat}\n")
            for paper in paper_list[:8]:
                title = paper.get("title", "无标题")
                source = paper.get("source", "")
                url = paper.get("url", "")
                abstract = paper.get("abstract", "")
                if abstract:
                    abstract = " ".join(abstract.split())[:400]
                matched_kw = paper.get("_matched_keywords", [])
                pub_date = paper.get("published_at") or paper.get("created_at") or ""
                if pub_date and len(pub_date) >= 10:
                    pub_date = pub_date[:10]

                link_str = f"[{source}]({url})" if url else f"[{source}]"
                paper_date_str = f"({pub_date}) " if pub_date else ""
                lines.append(f"- **{title}** {paper_date_str}{link_str}")
                if matched_kw:
                    lines.append(f"  关键词命中: {', '.join(matched_kw[:5])}")
                if abstract:
                    lines.append(f"  摘要: {abstract}")
            lines.append("")
    else:
        lines.append("无相关论文\n")

    # ── Analysis Prompt ──
    lines.append("=" * 60)
    lines.append("  分析要求")
    lines.append("=" * 60 + "\n")
    lines.append("""背景：你是 AI 训练数据行业的竞争情报分析师。读者是一家数据服务公司的管理层，需要从以上数据中获取可执行的商业洞察。

注意：上方数据中，数据集和模型名称后的括号内日期为发布/更新时间（YYYY-MM-DD），论文标题后的括号内日期为发表时间。请在分析中引用具体日期。

请提供以下分析：

### 1. 关键发现（Key Findings）
- 本周最值得关注的 3-5 个事件（数据集发布、模型动态、工具更新），标注具体日期，逐条说明原因和商业意义
- 按时间顺序排列，突出最新动态
- 特别关注：新发布的高价值训练数据集、RLHF/对齐相关动态、合成数据方向

### 2. 组织动态图谱
- 各 AI Lab 本周的数据策略动向（发了什么数据集？训练了什么模型？模型需要什么类型的数据？）
- 数据供应商竞品的最新动作（产品发布、开源工具、技术博客传递的信号）
- 中国 vs 海外 AI Labs 的数据布局差异

### 3. 数据需求信号
- 从模型发布反推：哪些类型的训练数据需求在上升？（如 RLHF、多模态、代码、Agent 等）
- 从论文方向看：学术界在探索什么新的数据方法论？（如新的标注范式、合成数据技术、数据质量评估）
- 从博客和 GitHub 看：数据工具链有什么新趋势？

### 4. 行动建议
- 针对数据服务公司，本周有哪些值得跟进的机会？
- 有哪些值得警惕的竞争威胁？
- 建议优先关注的数据类型或技术方向

### 5. 时间线（Timeline）
- 按日期列出本周重要事件时间线（数据集发布、模型发布、论文发表）
- 标注日期和关联组织

请用中文回答。分析应该具体、可执行，避免泛泛而谈。引用具体的数据集名称、组织名称、论文标题和日期。
""")

    return "\n".join(lines)


def format_anomalies_report(
    anomalies: list[str],
    all_datasets: list[dict],
    datasets_by_type: dict,
    x_activity: dict,
    blog_activity: list[dict],
    github_activity: list[dict],
    papers: list[dict],
) -> str:
    """Generate anomalies report for engineering/program optimization.

    This is separate from the insights report (which is for management).
    The anomalies report contains data quality warnings and actionable
    items for improving the scanning program.

    Args:
        anomalies: List of data quality warning strings.
        all_datasets: All collected datasets.
        datasets_by_type: Datasets grouped by classification type.
        x_activity: X/Twitter activity data.
        blog_activity: Blog activity data.
        github_activity: GitHub activity data.
        papers: Filtered papers list.

    Returns:
        Markdown string with anomalies report.
    """
    date_str = datetime.now().strftime("%Y-%m-%d")
    lines = [
        f"# 数据采集异常与待排查 — {date_str}",
        "",
        "> 本文件用于工程优化，不属于竞争情报分析报告。",
        "",
    ]

    # Section 1: Data quality warnings
    lines.append("## 1. 数据质量告警")
    lines.append("")
    if anomalies:
        for a in anomalies:
            lines.append(f"- :warning: {a}")
    else:
        lines.append("- 无告警，所有数据源正常返回数据")
    lines.append("")

    # Section 2: Source coverage stats
    lines.append("## 2. 数据源覆盖统计")
    lines.append("")

    # Datasets
    total_ds = len(all_datasets)
    classified = 0
    other_count = 0
    for dtype, ds_list in datasets_by_type.items():
        key = dtype.value if hasattr(dtype, "value") else str(dtype)
        if key == "other":
            other_count = len(ds_list)
        classified += len(ds_list)
    non_other = classified - other_count
    lines.append("| 数据源 | 数量 | 备注 |")
    lines.append("|--------|------|------|")
    lines.append(
        f"| 数据集 | {total_ds} | "
        f"已分类 {non_other}, 未分类(other) {other_count} |"
    )

    # X/Twitter
    x_data = x_activity or {}
    x_accounts = x_data.get("accounts", [])
    x_meta = x_data.get("metadata", {})
    x_active = len(x_accounts)
    x_tweets = sum(len(a.get("relevant_tweets", [])) for a in x_accounts)
    rsshub_ok = x_meta.get("rsshub_success", 0)
    rsshub_fail = x_meta.get("rsshub_fail", 0)
    rsshub_total = rsshub_ok + rsshub_fail
    x_note = f"活跃 {x_active}, 推文 {x_tweets}"
    if rsshub_total > 0:
        x_note += f", RSSHub {rsshub_ok}/{rsshub_total} ({rsshub_ok*100//rsshub_total}%)"
    lines.append(f"| X/Twitter | {x_active} 账号 | {x_note} |")

    # GitHub
    active_github = sum(1 for a in github_activity if a.get("repos_updated"))
    total_repos = sum(len(a.get("repos_updated", [])) for a in github_activity)
    lines.append(f"| GitHub | {active_github} 组织 | {total_repos} 个活跃仓库 |")

    # Blogs
    active_blogs = sum(1 for a in blog_activity if a.get("articles"))
    total_articles = sum(len(a.get("articles", [])) for a in blog_activity)
    lines.append(f"| 博客 | {active_blogs} 源 | {total_articles} 篇文章 |")

    # Papers
    lines.append(f"| 论文 | {len(papers)} 篇 | arXiv + HF Papers |")
    lines.append("")

    # Section 3: X/Twitter failed accounts
    failed_accounts = x_meta.get("failed_accounts", [])
    if failed_accounts:
        lines.append("## 3. X/Twitter 失败账号")
        lines.append("")
        lines.append("以下账号 RSSHub 获取失败，可能已改名、注销或被限制：")
        lines.append("")
        for acct in failed_accounts:
            lines.append(f"- `@{acct}`")
        lines.append("")
        lines.append("**建议**：人工核查后从 `config.yaml` 中清理无效账号。")
        lines.append("")

    # Section 4: Classification coverage
    if other_count > 0 and total_ds > 0:
        other_ratio = other_count * 100 // total_ds
        section_num = "4" if failed_accounts else "3"
        lines.append(f"## {section_num}. 数据集分类覆盖率")
        lines.append("")
        lines.append(
            f"- 未分类(other)占比: {other_ratio}% ({other_count}/{total_ds})"
        )
        lines.append("- **建议**：检查未分类数据集，考虑在 `DataTypeClassifier` 中增加关键词覆盖")
        lines.append("")
        # List unclassified datasets
        other_ds = datasets_by_type.get("other", [])
        if not other_ds:
            # Try enum key
            for dtype, ds_list in datasets_by_type.items():
                key = dtype.value if hasattr(dtype, "value") else str(dtype)
                if key == "other":
                    other_ds = ds_list
                    break
        if other_ds:
            lines.append("未分类数据集：")
            lines.append("")
            for ds in other_ds[:20]:
                name = ds.get("id") or ds.get("name", "unknown")
                desc = (ds.get("description") or "")[:80]
                lines.append(f"- `{name}` — {desc}")
            lines.append("")

    lines.append("---")
    lines.append(f"*自动生成于 {date_str} | 用于工程优化，非竞争情报报告*")
    lines.append("")

    return "\n".join(lines)


async def fetch_dataset_readmes(datasets: list[dict], hf_scraper: HuggingFaceScraper) -> list[dict]:
    """Fetch README content for datasets to improve classification.

    Uses concurrent fetching with semaphore for speed.

    Args:
        datasets: List of datasets.
        hf_scraper: HuggingFace scraper instance.

    Returns:
        Datasets with card_data populated.
    """
    logger.info("Fetching dataset READMEs for better classification...")
    to_fetch = [
        (i, ds) for i, ds in enumerate(datasets[:30]) if ds.get("id") and not ds.get("card_data")
    ]

    if not to_fetch:
        return datasets

    sem = asyncio.Semaphore(10)

    async def _fetch_one(idx: int, ds: dict):
        ds_id = ds.get("id", "")
        async with sem:
            try:
                card_data = await hf_scraper.fetch_dataset_readme(ds_id)
                return idx, card_data
            except Exception as e:
                logger.warning("Could not fetch README for %s: %s", ds_id, e)
                return idx, None

    results = await asyncio.gather(
        *[_fetch_one(idx, ds) for idx, ds in to_fetch], return_exceptions=True
    )

    count = 0
    for result in results:
        if isinstance(result, Exception):
            continue
        idx, card_data = result
        if card_data:
            datasets[idx]["card_data"] = card_data[:5000]
            count += 1

    logger.info("Fetched %d READMEs", count)
    return datasets


# ---------------------------------------------------------------------------
# DataRecipe integration
# ---------------------------------------------------------------------------

RECIPE_CATEGORY_PRIORITY = {
    "rlhf_preference": 10,
    "reward_model": 9,
    "sft_instruction": 8,
    "code": 7,
    "agent_tool": 7,
    "robotics": 6,
    "synthetic": 6,
    "rl_environment": 5,
    "multimodal": 4,
    "multilingual": 3,
    "evaluation": 2,
    "other": 1,
}


def rank_datasets_for_recipe(all_datasets, datasets_by_type, limit=5):
    """Rank datasets by value for DataRecipe deep analysis.

    Scoring (0-100):
      - Community traction (max 35): downloads (max 25) + likes (max 10)
      - Category priority (max 20): weighted by training-data relevance
      - Signals (max 18): meaningful metadata tags
      - Recency (max 12): gradual decay over 7/14/30 days
      - Min-downloads gate: <50 downloads → score halved
    """
    # Build category lookup from datasets_by_type
    category_map = {}
    for dtype, ds_list in datasets_by_type.items():
        cat_key = dtype.value if hasattr(dtype, "value") else str(dtype)
        for ds in ds_list:
            ds_id = ds.get("id", "")
            if ds_id:
                category_map[ds_id] = cat_key

    scored = []
    now = datetime.now(timezone.utc)

    for ds in all_datasets:
        ds_id = ds.get("id", "")
        if not ds_id:
            continue

        downloads = ds.get("downloads", 0) or 0
        likes = ds.get("likes", 0) or 0
        signals = ds.get("signals", []) or []
        category = category_map.get(ds_id, ds.get("category", "other"))

        # 1. Download score (log scale, max 25)
        download_score = min(25, math.log10(downloads + 1) * 8)

        # 2. Likes score (sqrt scale, max 10) — community endorsement
        likes_score = min(10, math.sqrt(likes) * 1.5)

        # 3. Signal score (max 18)
        meaningful = [s for s in signals if s and s != "-"]
        signal_score = min(18, len(meaningful) * 6)

        # 4. Category score (max 20)
        category_score = RECIPE_CATEGORY_PRIORITY.get(category, 1) * 2

        # 5. Recency bonus (gradual decay, max 12)
        recency_bonus = 0
        created_at = ds.get("created_at") or ds.get("createdAt") or ""
        if created_at:
            try:
                created_dt = datetime.fromisoformat(
                    created_at.replace("Z", "+00:00")
                )
                age_days = (now - created_dt).days
                if age_days <= 7:
                    recency_bonus = 12
                elif age_days <= 14:
                    recency_bonus = 8
                elif age_days <= 30:
                    recency_bonus = 4
            except (ValueError, TypeError):
                pass

        total = (
            download_score + likes_score + signal_score
            + category_score + recency_bonus
        )

        # Gate: penalize datasets with very few downloads
        if downloads < 50:
            total *= 0.5

        ds_copy = dict(ds)
        ds_copy["_recipe_score"] = round(total, 1)
        ds_copy["_recipe_category"] = category
        scored.append(ds_copy)

    scored.sort(key=lambda x: x["_recipe_score"], reverse=True)

    # Prefer non-"other" datasets when enough are available
    non_other = [d for d in scored if d["_recipe_category"] != "other"]
    if len(non_other) >= limit:
        return non_other[:limit]
    return scored[:limit]


async def run_recipe_analysis(selected_datasets, reports_dir, config):
    """Run DataRecipe deep analysis on selected datasets.

    Soft dependency — returns gracefully if datarecipe is not installed.
    Uses asyncio.to_thread() to run sync DeepAnalyzerCore without blocking.

    Args:
        selected_datasets: Ranked list from rank_datasets_for_recipe().
        reports_dir: Date-specific reports directory (e.g. data/reports/2026-02-08/).
        config: Radar config dict.
    """
    try:
        from datarecipe.core.deep_analyzer import DeepAnalyzerCore
        from datarecipe.integrations.radar import RadarIntegration
    except ImportError:
        logger.warning(
            "DataRecipe not installed. Install with: "
            "pip install -e /path/to/data-recipe"
        )
        return {"error": "datarecipe not installed", "results": []}

    date_str = datetime.now().strftime("%Y-%m-%d")
    recipe_dir = reports_dir / "recipe"
    recipe_dir.mkdir(parents=True, exist_ok=True)

    region = config.get("recipe", {}).get("region", "china") if config else "china"
    analyzer = DeepAnalyzerCore(
        output_dir=str(recipe_dir),
        region=region,
        use_llm=False,
    )

    results = []
    summaries = []

    for i, ds in enumerate(selected_datasets, 1):
        ds_id = ds.get("id", "")
        score = ds.get("_recipe_score", 0)
        category = ds.get("_recipe_category", "")

        logger.info(
            "Recipe [%d/%d]: %s (score=%.1f, category=%s)",
            i, len(selected_datasets), ds_id, score, category,
        )

        try:
            result = await asyncio.to_thread(
                analyzer.analyze,
                dataset_id=ds_id,
                sample_size=300,
            )

            if result.success:
                logger.info(
                    "  OK: type=%s, cost=$%.0f",
                    result.dataset_type,
                    result.reproduction_cost.get("total", 0),
                )
                summary_path = Path(result.output_dir) / "recipe_summary.json"
                if summary_path.exists():
                    summary = RadarIntegration.load_summary(str(summary_path))
                    summaries.append(summary)
            else:
                logger.warning("  FAIL: %s — %s", ds_id, result.error)

            results.append(result.to_dict())

        except Exception as e:
            logger.warning("  Exception analyzing %s: %s", ds_id, e)
            results.append({
                "dataset_id": ds_id,
                "success": False,
                "error": str(e),
            })

    # Aggregate
    aggregate = {}
    if summaries:
        aggregate = RadarIntegration.aggregate_summaries(summaries)

    # Save JSON summary
    with open(recipe_dir / "aggregate_summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "date": date_str,
                "datasets_selected": len(selected_datasets),
                "datasets_analyzed": sum(1 for r in results if r.get("success")),
                "datasets_failed": sum(1 for r in results if not r.get("success")),
                "aggregate": aggregate,
                "results": results,
            },
            f, ensure_ascii=False, indent=2, default=str,
        )

    # Save Markdown summary
    ok_count = sum(1 for r in results if r.get("success"))
    fail_count = sum(1 for r in results if not r.get("success"))
    md = [
        f"# DataRecipe Analysis — {date_str}\n",
        f"Selected: {len(selected_datasets)} | Analyzed: {ok_count} | Failed: {fail_count}\n",
    ]
    if aggregate:
        cost = aggregate.get("total_reproduction_cost", {})
        md.append("## Aggregate\n")
        md.append(f"- Total reproduction cost: ${cost.get('total', 0):,.0f}")
        md.append(f"  - Human: ${cost.get('human', 0):,.0f}")
        md.append(f"  - API: ${cost.get('api', 0):,.0f}")
        md.append(f"- Avg human %: {aggregate.get('avg_human_percentage', 0):.1f}%")
        md.append(f"- Types: {aggregate.get('type_distribution', {})}")
        md.append(f"- Difficulty: {aggregate.get('difficulty_distribution', {})}\n")

    md.append("## Results\n")
    for r in results:
        ds_id = r.get("dataset_id", "?")
        if r.get("success"):
            md.append(
                f"- **{ds_id}** — type={r.get('dataset_type', '?')}, "
                f"samples={r.get('sample_count', 0)}, "
                f"files={len(r.get('files_generated', []))}"
            )
        else:
            md.append(f"- **{ds_id}** — FAIL: {r.get('error', 'unknown')}")

    with open(recipe_dir / "recipe_analysis_summary.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")

    logger.info("Recipe summary saved to: %s", recipe_dir)

    return {
        "output_dir": str(recipe_dir),
        "datasets_selected": len(selected_datasets),
        "datasets_analyzed": ok_count,
        "aggregate": aggregate,
    }


def _effective_days(watermarks, source: str, default_days: int) -> int:
    """Convert watermark timestamp to equivalent days for a source.

    If no watermark exists, returns default_days.
    Adds +1 day safety margin to avoid missing data at boundaries.
    """
    wm = watermarks.get(source)
    if not wm:
        return default_days
    try:
        wm_dt = datetime.fromisoformat(wm)
        delta = datetime.now() - wm_dt
        return max(1, int(delta.total_seconds() / 86400) + 1)
    except (ValueError, TypeError):
        return default_days


def _filter_pwc_datasets(datasets: list[dict], days: int) -> list[dict]:
    """Filter Papers with Code datasets by introduced date."""
    if not datasets or not days:
        return datasets or []

    cutoff = datetime.now() - timedelta(days=days)
    filtered = []
    for ds in datasets:
        created_at = ds.get("created_at") or ds.get("introduced_date")
        if created_at:
            try:
                dt = datetime.fromisoformat(str(created_at))
            except ValueError:
                try:
                    dt = datetime.strptime(str(created_at)[:10], "%Y-%m-%d")
                except ValueError:
                    dt = None
        else:
            dt = None

        if dt and dt < cutoff:
            continue
        filtered.append(ds)
    return filtered


def _load_org_watermarks(raw_value) -> dict[str, str]:
    if not raw_value:
        return {}
    if isinstance(raw_value, dict):
        return raw_value
    if isinstance(raw_value, str):
        try:
            data = json.loads(raw_value)
            return data if isinstance(data, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def _update_watermarks(watermarks, all_data: dict) -> None:
    """Extract latest timestamps from scan results and update watermarks."""

    def _max_ts(timestamps: list[str]) -> str | None:
        valid = []
        for ts in timestamps:
            if ts:
                try:
                    datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
                    valid.append(str(ts))
                except (ValueError, TypeError):
                    pass
        return max(valid) if valid else None

    def _normalize_ts(value: str | None) -> str | None:
        if not value:
            return None
        value = str(value)
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            try:
                dt = datetime.strptime(value[:10], "%Y-%m-%d")
            except ValueError:
                return None
        return dt.isoformat()

    # Labs / Vendors: max(dataset["last_modified"])
    for source_key in ("labs", "vendors"):
        activity_key = f"{source_key}_activity"
        activity = all_data.get(activity_key, {})
        timestamps = []
        data_dict = activity.get(source_key, {})
        if isinstance(data_dict, dict):
            for category in data_dict.values():
                if isinstance(category, dict):
                    for org_data in category.values():
                        if isinstance(org_data, dict):
                            for ds in org_data.get("datasets", []):
                                ts = ds.get("last_modified") or ds.get("created_at")
                                norm = _normalize_ts(ts)
                                if norm:
                                    timestamps.append(norm)
        ts = _max_ts(timestamps)
        if ts:
            watermarks.set(source_key, ts)

    def _merge_org_map(existing_key: str, new_map: dict[str, str]) -> None:
        if not new_map:
            return
        existing = _load_org_watermarks(watermarks.get(existing_key))
        merged = dict(existing)
        changed = False
        for org_name, ts in new_map.items():
            prev = merged.get(org_name)
            best = _max_ts([prev, ts]) if prev else ts
            if best and best != prev:
                merged[org_name] = best
                changed = True
        if changed:
            watermarks.set(existing_key, merged)

    labs_org_map = {}
    labs_activity = all_data.get("labs_activity", {}).get("labs", {})
    if isinstance(labs_activity, dict):
        for category in labs_activity.values():
            if not isinstance(category, dict):
                continue
            for org_name, org_data in category.items():
                timestamps = []
                for ds in org_data.get("datasets", []):
                    ts = ds.get("last_modified") or ds.get("created_at")
                    norm = _normalize_ts(ts)
                    if norm:
                        timestamps.append(norm)
                for model in org_data.get("models", []):
                    ts = model.get("last_modified") or model.get("created_at")
                    norm = _normalize_ts(ts)
                    if norm:
                        timestamps.append(norm)
                org_ts = _max_ts(timestamps)
                if org_ts:
                    labs_org_map[org_name] = org_ts

    _merge_org_map("labs_orgs", labs_org_map)

    vendor_org_map = {}
    vendor_activity = all_data.get("vendor_activity", {}).get("vendors", {})
    if isinstance(vendor_activity, dict):
        for tier in vendor_activity.values():
            if not isinstance(tier, dict):
                continue
            for vendor_name, vendor_data in tier.items():
                timestamps = []
                for ds in vendor_data.get("datasets", []):
                    ts = ds.get("last_modified") or ds.get("created_at")
                    norm = _normalize_ts(ts)
                    if norm:
                        timestamps.append(norm)
                org_ts = _max_ts(timestamps)
                if org_ts:
                    vendor_org_map[vendor_name] = org_ts

    _merge_org_map("vendors_orgs", vendor_org_map)

    blog_map = {}
    for activity in all_data.get("blog_posts", []) or []:
        source = activity.get("source")
        if not source:
            continue
        article_dates = []
        for article in activity.get("articles", []) or []:
            norm = _normalize_ts(article.get("date"))
            if norm:
                article_dates.append(norm)
        src_ts = _max_ts(article_dates)
        if src_ts:
            blog_map[source] = src_ts

    _merge_org_map("blog_sources", blog_map)

    x_map = {}
    for account in all_data.get("x_activity", {}).get("accounts", []) or []:
        username = account.get("username")
        if not username:
            continue
        tweet_dates = []
        for tweet in account.get("relevant_tweets", []) or []:
            norm = _normalize_ts(tweet.get("date"))
            if norm:
                tweet_dates.append(norm)
        user_ts = _max_ts(tweet_dates)
        if user_ts:
            x_map[username] = user_ts

    _merge_org_map("x_accounts", x_map)

    reddit_map = {}
    for post in all_data.get("reddit_activity", {}).get("posts", []) or []:
        sub = post.get("subreddit")
        if not sub:
            continue
        norm = _normalize_ts(post.get("date")) or _normalize_ts(post.get("created_utc"))
        if not norm:
            continue
        reddit_map.setdefault(sub, []).append(norm)

    reddit_timestamps = []
    for timestamps in reddit_map.values():
        reddit_timestamps.extend(timestamps)

    ts = _max_ts(reddit_timestamps)
    if ts:
        watermarks.set("reddit", ts)

    reddit_agg = {sub: _max_ts(ts_list) for sub, ts_list in reddit_map.items()}
    _merge_org_map("reddit_sources", {k: v for k, v in reddit_agg.items() if v})

    # HN: max(story["date"])
    hn_timestamps = []
    for story in all_data.get("hn_activity", {}).get("stories", []) or []:
        norm = _normalize_ts(story.get("date"))
        if norm:
            hn_timestamps.append(norm)
    ts = _max_ts(hn_timestamps)
    if ts:
        watermarks.set("hn", ts)

    # Kaggle: max(dataset["last_modified"])
    kaggle_timestamps = []
    for ds in all_data.get("kaggle_datasets", []) or []:
        norm = _normalize_ts(ds.get("last_modified"))
        if norm:
            kaggle_timestamps.append(norm)
    ts = _max_ts(kaggle_timestamps)
    if ts:
        watermarks.set("kaggle", ts)

    # Semantic Scholar: max(paper["publication_date"])
    ss_timestamps = []
    for p in all_data.get("semantic_scholar_papers", []) or []:
        norm = _normalize_ts(p.get("publication_date"))
        if norm:
            ss_timestamps.append(norm)
    ts = _max_ts(ss_timestamps)
    if ts:
        watermarks.set("semantic_scholar", ts)

    # GitHub Trending: max(repo["date"])
    ght_timestamps = []
    for repo in (all_data.get("gh_trending") or {}).get("repos", []) or []:
        norm = _normalize_ts(repo.get("date"))
        if norm:
            ght_timestamps.append(norm)
    ts = _max_ts(ght_timestamps)
    if ts:
        watermarks.set("gh_trending", ts)

    # Product Hunt: max(product["date"])
    ph_timestamps = []
    for prod in (all_data.get("producthunt") or {}).get("products", []) or []:
        norm = _normalize_ts(prod.get("date"))
        if norm:
            ph_timestamps.append(norm)
    ts = _max_ts(ph_timestamps)
    if ts:
        watermarks.set("producthunt", ts)

    # update cooldown metadata
    cooldowns = _load_org_watermarks(watermarks.get("_cooldowns"))
    cooldowns.update(all_data.get("cooldowns", {}))
    if cooldowns:
        watermarks.set("_cooldowns", cooldowns)

    # GitHub: max(repo["updated_at"])
    gh_timestamps = []
    for org_entry in all_data.get("github_activity", []):
        for repo in org_entry.get("repos_updated", []):
            ts = repo.get("updated_at") or repo.get("pushed_at")
            norm = _normalize_ts(ts)
            if norm:
                gh_timestamps.append(norm)
    ts = _max_ts(gh_timestamps)
    if ts:
        watermarks.set("github", ts)

    github_org_map = {}
    for org_entry in all_data.get("github_activity", []) or []:
        org_name = org_entry.get("org")
        if not org_name:
            continue
        repo_timestamps = []
        for repo in org_entry.get("repos_updated", []) or []:
            ts = repo.get("updated_at") or repo.get("pushed_at")
            if ts:
                repo_timestamps.append(str(ts))
        org_ts = _max_ts(repo_timestamps)
        if org_ts:
            github_org_map[org_name] = org_ts

    if github_org_map:
        existing_orgs = _load_org_watermarks(watermarks.get("github_orgs"))
        merged = dict(existing_orgs)
        changed = False
        for org_name, ts in github_org_map.items():
            prev = merged.get(org_name)
            if prev:
                best = _max_ts([prev, ts])
            else:
                best = ts
            if best and best != prev:
                merged[org_name] = best
                changed = True
        if changed:
            watermarks.set("github_orgs", merged)

    # Blogs: max(article["date"])
    blog_timestamps = []
    for blog in all_data.get("blog_posts", []):
        for article in blog.get("articles", []):
            ts = article.get("date")
            if ts:
                blog_timestamps.append(str(ts))
    ts = _max_ts(blog_timestamps)
    if ts:
        watermarks.set("blogs", ts)

    # Papers: max(paper["created_at"])
    paper_timestamps = []
    for paper in all_data.get("papers", []):
        ts = paper.get("created_at") or paper.get("published")
        if ts:
            paper_timestamps.append(str(ts))
    ts = _max_ts(paper_timestamps)
    if ts:
        watermarks.set("papers", ts)

    # X/Twitter: max(tweet["date"])
    x_timestamps = []
    for account in all_data.get("x_activity", {}).get("accounts", []):
        for tweet in account.get("relevant_tweets", []):
            ts = tweet.get("date")
            if ts:
                x_timestamps.append(str(ts))
    ts = _max_ts(x_timestamps)
    if ts:
        watermarks.set("x", ts)

    # Reddit: max(post["date"])
    reddit_timestamps = []
    for post in all_data.get("reddit_activity", {}).get("posts", []):
        ts = post.get("date") or post.get("created_utc")
        if ts:
            reddit_timestamps.append(str(ts))
    ts = _max_ts(reddit_timestamps)
    if ts:
        watermarks.set("reddit", ts)

    # Papers with Code: max(dataset["created_at"])
    pwc_timestamps = []
    for ds in all_data.get("paperswithcode", []) or []:
        ts = ds.get("created_at") or ds.get("introduced_date")
        norm = _normalize_ts(ts)
        if norm:
            pwc_timestamps.append(norm)
    ts = _max_ts(pwc_timestamps)
    if ts:
        watermarks.set("pwc", ts)

    hf_general_ts = []
    for ds in all_data.get("huggingface_general", []) or []:
        ts = ds.get("last_modified") or ds.get("created_at")
        norm = _normalize_ts(ts)
        if norm:
            hf_general_ts.append(norm)
    ts = _max_ts(hf_general_ts)
    if ts:
        watermarks.set("huggingface_general", ts)


async def async_main(args):
    """Async main logic for the intelligence scan.

    Args:
        args: Parsed command-line arguments.
    """
    # Set up logging based on verbosity
    setup_logging(level="INFO")

    import time as _time
    _scan_start_mono = _time.monotonic()
    _scan_started_at = datetime.now()

    # Load config
    logger.info("=" * 60)
    logger.info("  AI Dataset Radar v%s", __version__)
    logger.info("  Competitive Intelligence System")
    logger.info("=" * 60)

    # Preflight checks — warn about missing optional deps
    _preflight_warnings = []

    config_path = Path(args.config)
    if not config_path.exists():
        logger.error("Config file not found: %s", config_path.absolute())
        logger.error("Run this command from the project root directory:")
        logger.error("  cd /path/to/ai-dataset-radar && python src/main_intel.py")
        return

    if not args.no_blogs:
        try:
            from playwright.async_api import async_playwright  # noqa: F401
        except ImportError:
            _preflight_warnings.append(
                "Playwright not installed — blog tracking will be limited. "
                "Fix: pip install playwright && playwright install chromium"
            )

    if not os.environ.get("GITHUB_TOKEN"):
        _preflight_warnings.append(
            "No GITHUB_TOKEN — GitHub API limited to 60 req/hr (may miss data). "
            "Fix: add GITHUB_TOKEN to .env"
        )

    # Only warn if user explicitly requested API insights but has no key
    if getattr(args, "api_insights", False):
        has_llm = (
            os.environ.get("LLM_PROVIDER")
            or os.environ.get("LLM_API_KEY")
            or os.environ.get("ANTHROPIC_API_KEY")
        )
        if not has_llm:
            _preflight_warnings.append(
                "--api-insights requested but no LLM API key found. "
                "Set ANTHROPIC_API_KEY or LLM_PROVIDER+LLM_API_KEY"
            )

    if _preflight_warnings:
        logger.warning("— Environment checks —")
        for w in _preflight_warnings:
            logger.warning("  ⚠ %s", w)
        logger.warning("—" * 40)

    # Progress indicator — total is set after config is loaded
    _step = 0
    _total = 0

    def _progress(msg: str) -> str:
        nonlocal _step
        _step += 1
        return "[%d/%d] %s" % (_step, _total, msg)

    config = load_config(args.config)
    validate_config(config)

    # Create shared async HTTP client
    http_client = AsyncHTTPClient()
    try:
        # Initialize components with shared HTTP client
        org_tracker = OrgTracker(config, http_client=http_client)
        github_tracker = GitHubTracker(config, http_client=http_client)
        blog_tracker = BlogTracker(config, http_client=http_client)
        x_tracker = (
            XTracker(config, http_client=http_client)
            if not args.no_x and config.get("x_tracker", {}).get("enabled", False)
            else None
        )
        reddit_tracker = (
            RedditTracker(config, http_client=http_client)
            if not args.no_reddit and config.get("reddit_tracker", {}).get("enabled", False)
            else None
        )
        hn_tracker = (
            HNTracker(config, http_client=http_client)
            if not args.no_hn and config.get("hn_tracker", {}).get("enabled", False)
            else None
        )
        from scrapers.kaggle_scraper import KaggleScraper
        kaggle_scraper = (
            KaggleScraper(config=config, http_client=http_client)
            if not args.no_kaggle and config.get("sources", {}).get("kaggle", {}).get("enabled", False)
            else None
        )
        from scrapers.semantic_scholar import SemanticScholarScraper
        ss_scraper = (
            SemanticScholarScraper(config=config, http_client=http_client)
            if not args.no_semantic_scholar and config.get("sources", {}).get("semantic_scholar", {}).get("enabled", False)
            else None
        )
        gh_trending_tracker = (
            GitHubTrendingTracker(config, http_client=http_client)
            if not args.no_gh_trending and config.get("github_trending", {}).get("enabled", False)
            else None
        )
        ph_tracker = (
            ProductHuntTracker(config, http_client=http_client)
            if not args.no_producthunt and config.get("producthunt_tracker", {}).get("enabled", False)
            else None
        )
        data_classifier = DataTypeClassifier(config)
        paper_filter = PaperFilter(config)
        report_generator = IntelReportGenerator(config)
        hf_scraper = HuggingFaceScraper(config, http_client=http_client)
        hf_cfg = config.get("sources", {}).get("huggingface", {})
        hf_mode = str(hf_cfg.get("mode", "targeted")).lower()
        hf_general_enabled = hf_mode in {"general", "hybrid"}
        hf_general_datasets = []

        # Incremental scan: watermark-based days calculation
        from utils.watermark import WatermarkStore

        output_dir = Path(config.get("report", {}).get("output_dir", "data"))
        watermarks = WatermarkStore(output_dir / "watermarks.json")
        full_scan = getattr(args, "full_scan", False)
        incremental = (
            not full_scan
            and watermarks.get("labs") is not None
        )

        def _days(source: str) -> int:
            if not incremental:
                return args.days
            return _effective_days(watermarks, source, args.days)

        if incremental:
            logger.info("增量扫描模式（基于上次水位线）")
        else:
            logger.info("全量扫描模式（%d 天窗口）", args.days)

        pwc_config = config.get("sources", {}).get("paperswithcode", {})
        pwc_scraper = None
        pwc_days = args.days
        if (
            not args.no_pwc
            and pwc_config.get("enabled", True)
        ):
            pwc_scraper = PapersWithCodeScraper(
                config=config,
                limit=pwc_config.get("limit", 50),
            )
            base_days = _days("pwc") if incremental else args.days
            config_days = pwc_config.get("days")
            if config_days is not None:
                pwc_days = min(config_days, base_days) if incremental else config_days
            else:
                pwc_days = base_days

        # 1-3. Fetch all data sources concurrently
        lab_activity = {"labs": {}}
        vendor_activity = {"vendors": {}}
        github_activity = []
        blog_activity = []
        x_activity = {"accounts": [], "search_results": []}
        reddit_activity = {"posts": [], "metadata": {}}
        hn_activity = {"stories": [], "metadata": {}}
        kaggle_datasets = []
        semantic_scholar_papers = []
        gh_trending = {"repos": [], "metadata": {}}
        producthunt = {"products": [], "metadata": {}}
        papers = []
        pwc_datasets = []

        # Pre-build paper scrapers
        arxiv_scraper = None
        hf_papers_scraper = None
        if not args.no_papers:
            arxiv_config = config.get("sources", {}).get("arxiv", {})
            if arxiv_config.get("enabled", True):
                arxiv_scraper = ArxivScraper(limit=50, config=config, http_client=http_client)
            hf_config = config.get("sources", {}).get("hf_papers", {})
            if hf_config.get("enabled", True):
                hf_papers_scraper = HFPapersScraper(
                    limit=50,
                    days=hf_config.get("days", 7),
                    http_client=http_client,
                )

        github_org_watermarks = _load_org_watermarks(watermarks.get("github_orgs"))
        labs_org_watermarks = _load_org_watermarks(watermarks.get("labs_orgs"))
        vendor_org_watermarks = _load_org_watermarks(watermarks.get("vendors_orgs"))
        blog_source_watermarks = _load_org_watermarks(watermarks.get("blog_sources"))
        x_account_watermarks = _load_org_watermarks(watermarks.get("x_accounts"))
        reddit_source_watermarks = _load_org_watermarks(watermarks.get("reddit_sources"))
        hn_source_watermarks = _load_org_watermarks(watermarks.get("hn_sources"))
        gh_trending_watermarks = _load_org_watermarks(watermarks.get("gh_trending_sources"))
        ph_source_watermarks = _load_org_watermarks(watermarks.get("producthunt_sources"))

        # Calculate total progress steps
        _n = 0
        if not args.no_labs:
            _n += 1
        if not args.no_vendors:
            _n += 1
        if not args.no_github:
            _n += 1
        if not args.no_blogs:
            _n += 1
        if x_tracker:
            _n += 1
        if reddit_tracker:
            _n += 1
        if hn_tracker:
            _n += 1
        if kaggle_scraper:
            _n += 1
        if ss_scraper:
            _n += 1
        if gh_trending_tracker:
            _n += 1
        if ph_tracker:
            _n += 1
        if arxiv_scraper or hf_papers_scraper:
            _n += 1
        if pwc_scraper:
            _n += 1
        _total = _n + 3  # + classify + report + finalize

        # Build async tasks
        tasks = {}
        if not args.no_labs:
            logger.info(_progress("HuggingFace Labs 追踪..."))
            tasks["labs"] = org_tracker.fetch_lab_activity(
                days=_days("labs"), org_watermarks=labs_org_watermarks
            )

        if not args.no_vendors:
            logger.info(_progress("HuggingFace Vendors 追踪..."))
            tasks["vendors"] = org_tracker.fetch_vendor_activity(
                days=_days("vendors"), org_watermarks=vendor_org_watermarks
            )

        if not args.no_github:
            logger.info(_progress("GitHub 组织扫描..."))
            tasks["github"] = github_tracker.fetch_all_orgs(
                days=_days("github"), org_watermarks=github_org_watermarks
            )

        if not args.no_blogs:
            logger.info(_progress("博客源抓取..."))
            tasks["blogs"] = blog_tracker.fetch_all_blogs(
                days=_days("blogs"), source_watermarks=blog_source_watermarks
            )

        if x_tracker:
            logger.info(_progress("X/Twitter 账号抓取..."))
            tasks["x"] = x_tracker.fetch_all(
                days=_days("x"), account_watermarks=x_account_watermarks
            )

        if reddit_tracker:
            logger.info(_progress("Reddit 社区追踪..."))
            tasks["reddit"] = reddit_tracker.fetch_all(
                days=_days("reddit"), source_watermarks=reddit_source_watermarks
            )

        if hn_tracker:
            logger.info(_progress("Hacker News 追踪..."))
            tasks["hn"] = hn_tracker.fetch_all(
                days=_days("hn"), source_watermarks=hn_source_watermarks
            )

        if kaggle_scraper:
            logger.info(_progress("Kaggle 数据集..."))
            tasks["kaggle"] = kaggle_scraper.fetch(days=_days("kaggle"))

        if ss_scraper:
            logger.info(_progress("Semantic Scholar 论文..."))
            tasks["semantic_scholar"] = ss_scraper.fetch(days=_days("semantic_scholar"))

        if gh_trending_tracker:
            logger.info(_progress("GitHub Trending..."))
            tasks["gh_trending"] = gh_trending_tracker.fetch_all(
                days=_days("gh_trending"), source_watermarks=gh_trending_watermarks
            )

        if ph_tracker:
            logger.info(_progress("Product Hunt..."))
            tasks["producthunt"] = ph_tracker.fetch_all(
                days=_days("producthunt"), source_watermarks=ph_source_watermarks
            )

        if arxiv_scraper:
            if not hf_papers_scraper:
                logger.info(_progress("论文抓取 (arXiv)..."))
            else:
                logger.info(_progress("论文抓取 (arXiv + HF Papers)..."))
            tasks["arxiv"] = arxiv_scraper.fetch()

        if hf_papers_scraper:
            if not arxiv_scraper:
                logger.info(_progress("论文抓取 (HF Papers)..."))
            tasks["hf_papers"] = hf_papers_scraper.fetch()

        if hf_general_enabled:
            hf_general_watermark = watermarks.get("huggingface_general")
            general_limit = hf_cfg.get("general_limit", hf_cfg.get("limit", 100))
            general_max_pages = hf_cfg.get("general_max_pages", 5)

            async def _fetch_hf_general():
                scraper = HuggingFaceScraper(
                    config,
                    limit=general_limit,
                    http_client=http_client,
                )
                return await scraper.fetch(
                    min_timestamp=hf_general_watermark,
                    max_pages=general_max_pages,
                )

            tasks["hf_general"] = _fetch_hf_general()

        if pwc_scraper:
            logger.info(_progress("Papers with Code 数据集..."))

            async def _fetch_pwc():
                result = await asyncio.to_thread(pwc_scraper.fetch)
                return _filter_pwc_datasets(result, pwc_days)

            tasks["pwc"] = _fetch_pwc()

        # Run all tasks concurrently
        if tasks:
            logger.info("  ↳ 等待数据采集完成...")
            keys = list(tasks.keys())
            results = await asyncio.gather(*tasks.values(), return_exceptions=True)

            for key, result in zip(keys, results):
                try:
                    if isinstance(result, Exception):
                        raise result
                    if key == "labs":
                        lab_activity = {"labs": result}
                    elif key == "vendors":
                        vendor_activity = {"vendors": result}
                    elif key == "github":
                        github_activity = result.get("vendors", []) + result.get("labs", [])
                        active_count = sum(1 for a in github_activity if a.get("repos_updated"))
                        repo_count = sum(len(a.get("repos_updated", [])) for a in github_activity)
                        logger.info(
                            "  ✓ GitHub: %d 组织, %d 仓库", active_count, repo_count
                        )
                    elif key == "blogs":
                        blog_activity = result
                        active_count = sum(1 for a in blog_activity if a.get("articles"))
                        article_count = sum(len(a.get("articles", [])) for a in blog_activity)
                        logger.info(
                            "  ✓ 博客: %d 源, %d 篇",
                            active_count,
                            article_count,
                        )
                    elif key == "x":
                        x_activity = result
                        x_acct_count = len(result.get("accounts", []))
                        x_tweets = sum(
                            len(a.get("relevant_tweets", [])) for a in result.get("accounts", [])
                        )
                        logger.info(
                            "  ✓ X: %d 账号, %d 推文",
                            x_acct_count,
                            x_tweets,
                        )
                    elif key == "reddit":
                        reddit_activity = result
                        reddit_posts = len(result.get("posts", []))
                        logger.info("  ✓ Reddit: %d 相关帖子", reddit_posts)
                    elif key == "hn":
                        hn_activity = result
                        hn_stories = len(result.get("stories", []))
                        logger.info("  ✓ Hacker News: %d 相关讨论", hn_stories)
                    elif key == "kaggle":
                        kaggle_datasets = result or []
                        logger.info("  ✓ Kaggle: %d 数据集", len(kaggle_datasets))
                    elif key == "semantic_scholar":
                        semantic_scholar_papers = result or []
                        logger.info("  ✓ Semantic Scholar: %d 论文", len(semantic_scholar_papers))
                    elif key == "gh_trending":
                        gh_trending = result
                        logger.info("  ✓ GitHub Trending: %d 仓库", len(result.get("repos", [])))
                    elif key == "producthunt":
                        producthunt = result
                        logger.info("  ✓ Product Hunt: %d 产品", len(result.get("products", [])))
                    elif key == "arxiv":
                        papers.extend(paper_filter.filter_papers(result))
                    elif key == "hf_papers":
                        filtered = paper_filter.filter_papers(result)
                        papers.extend(filtered)
                    elif key == "pwc":
                        pwc_datasets = result or []
                        logger.info("  ✓ Papers with Code: %d 数据集", len(pwc_datasets))
                except Exception as e:
                    logger.warning("  ✗ %s: %s", key, e)

            if papers:
                logger.info("  ✓ 论文: %d 篇", len(papers))

        # 4. Collect all datasets for classification
        all_datasets = []

        # From labs
        for category in lab_activity.get("labs", {}).values():
            for org_data in category.values():
                all_datasets.extend(org_data.get("datasets", []))

        # From vendors
        for tier in vendor_activity.get("vendors", {}).values():
            for vendor_data in tier.values():
                all_datasets.extend(vendor_data.get("datasets", []))

        if hf_general_datasets:
            all_datasets.extend(hf_general_datasets)

        logger.info("  ✓ 数据集: %d 个", len(all_datasets))

        # 5. Fetch dataset READMEs for better classification
        if not args.no_readme and all_datasets:
            all_datasets = await fetch_dataset_readmes(all_datasets, hf_scraper)

        # 6. Classify datasets
        logger.info(_progress("数据集分类..."))
        datasets_by_type = data_classifier.group_by_type(all_datasets)

        summary = data_classifier.summarize(all_datasets)
        logger.info("Classified datasets: %d/%d relevant", summary["relevant"], summary["total"])
        logger.info("Other ratio: %.1f%%", summary["other_ratio"] * 100)
        for dtype, count in summary["by_type"].items():
            if count > 0:
                logger.info("  %s: %d", dtype, count)

        # 7. Papers already fetched in parallel above (arXiv + HF Papers)

        # 7.1 Advanced analysis
        logger.info(_progress("高级分析..."))
        competitor_matrix = CompetitorMatrix(config).build(
            datasets=all_datasets,
            github_activity=github_activity,
            papers=papers,
            blog_posts=blog_activity,
        )
        dataset_lineage = DatasetLineageTracker(config).analyze(all_datasets)
        org_graph = OrgRelationshipGraph(config).build(
            datasets=all_datasets,
            github_activity=github_activity,
            papers=papers,
            blog_posts=blog_activity,
        )

        # 7.5 Data quality validation
        anomalies = []
        active_github = sum(1 for a in github_activity if a.get("repos_updated"))
        active_blogs = sum(1 for a in blog_activity if a.get("articles"))
        x_accounts = len(x_activity.get("accounts", []))

        if not args.no_github and active_github == 0:
            anomalies.append("GitHub: 0 active orgs (expected >0)")
        if not args.no_blogs and active_blogs == 0:
            anomalies.append("Blogs: 0 active blogs (expected >0)")
        if x_tracker and x_accounts == 0:
            anomalies.append("X/Twitter: 0 active accounts (check RSSHub/API connectivity)")
        if reddit_tracker and len(reddit_activity.get("posts", [])) == 0:
            anomalies.append("Reddit: 0 relevant posts (check subreddit config or Reddit API)")
        if not args.no_papers and len(papers) == 0:
            anomalies.append("Papers: 0 results from arXiv + HF Papers (check network)")
        if len(all_datasets) == 0 and not args.no_labs and not args.no_vendors:
            anomalies.append("Datasets: 0 from all tracked orgs (check HuggingFace API)")

        if anomalies:
            logger.warning("=" * 60)
            logger.warning("  DATA QUALITY WARNINGS")
            logger.warning("=" * 60)
            for a in anomalies:
                logger.warning("  ⚠ %s", a)
            logger.warning("=" * 60)

        # 8. Trend analysis (before report so trends appear in output)
        trend_data = {}
        try:
            db_path = output_dir / "radar.db"
            db = RadarDatabase(str(db_path))
            trend_analyzer = TrendAnalyzer(db, config)

            if all_datasets:
                trend_analyzer.record_daily_stats(all_datasets)
                trend_analyzer.calculate_trends(
                    dataset_ids=trend_analyzer.last_recorded_ids
                )

                # Inject growth rates into each dataset
                for ds in all_datasets:
                    ds_id = ds.get("id", "")
                    if ds_id:
                        ds_trend = trend_analyzer.get_dataset_trend(ds_id)
                        if ds_trend:
                            ds["growth_7d"] = ds_trend.get("growth_7d")
                            ds["growth_30d"] = ds_trend.get("growth_30d")

                trend_data = {
                    "top_growing_7d": trend_analyzer.get_top_growing_datasets(days=7, limit=10),
                    "rising_7d": trend_analyzer.get_rising_datasets(days=7, limit=10),
                }

            db.close()
        except Exception as e:
            logger.warning("Trend analysis skipped: %s", e)

        # 9. Generate report
        logger.info(_progress("生成报告..."))

        report = report_generator.generate(
            lab_activity=lab_activity,
            vendor_activity=vendor_activity,
            datasets_by_type=datasets_by_type,
            papers=papers,
            github_activity=github_activity,
            blog_activity=blog_activity,
            x_activity=x_activity,
            reddit_activity=reddit_activity,
            hn_activity=hn_activity,
            kaggle_datasets=kaggle_datasets,
            trend_data=trend_data,
            competitor_matrix=competitor_matrix,
            dataset_lineage=dataset_lineage,
            org_graph=org_graph,
            pwc_datasets=pwc_datasets,
        )

        # Prepare structured data for JSON output
        datasets_json = {}
        for dtype, ds_list in datasets_by_type.items():
            key = dtype.value if isinstance(dtype, DataType) else str(dtype)
            datasets_json[key] = [
                {k: v for k, v in ds.items() if not k.startswith("_")} for ds in ds_list
            ]

        _scan_elapsed = _time.monotonic() - _scan_start_mono
        all_data = {
            "data_quality_warnings": anomalies,
            "scan_mode": "incremental" if incremental else "full",
            "scan_started_at": _scan_started_at.isoformat(),
            "scan_duration_seconds": round(_scan_elapsed, 1),
            "period": {
                "days": args.days,
                "start": None,
                "end": datetime.now().isoformat(),
            },
            "labs_activity": lab_activity,
            "vendor_activity": vendor_activity,
            "github_activity": github_activity,
            "blog_posts": blog_activity,
            "x_activity": x_activity,
            "reddit_activity": reddit_activity,
            "hn_activity": hn_activity,
            "kaggle_datasets": kaggle_datasets,
            "semantic_scholar_papers": semantic_scholar_papers,
            "gh_trending": gh_trending,
            "producthunt": producthunt,
            "huggingface_general": hf_general_datasets,
            "paperswithcode": pwc_datasets,
            "datasets": all_datasets,
            "datasets_by_type": datasets_json,
            "papers": papers,
            "trend_data": trend_data,
            "competitor_matrix": competitor_matrix,
            "dataset_lineage": dataset_lineage,
            "org_graph": org_graph,
        }

        # Determine output directory — reports grouped by date
        date_str = datetime.now().strftime("%Y-%m-%d")
        reports_dir = output_dir / "reports" / date_str
        reports_dir.mkdir(parents=True, exist_ok=True)

        # Initialize dual formatter
        formatter = DualOutputFormatter(output_dir=str(reports_dir))

        # Use custom output path if specified
        if args.output:
            output_path = Path(args.output)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(report)
            logger.info("Report saved to: %s", output_path)

            if args.json:
                json_path = output_path.with_suffix(".json")
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(
                        formatter._format_json_output(all_data),
                        f,
                        ensure_ascii=False,
                        indent=2,
                        default=str,
                    )
                logger.info("JSON data saved to: %s", json_path)
        else:
            # Use DualOutputFormatter for default path
            md_path, json_path = formatter.save_reports(
                markdown_content=report, data=all_data, filename_prefix="intel_report"
            )
            logger.info("Report saved to: %s", md_path)
            logger.info("JSON data saved to: %s", json_path)

        # 9.5 Daily change summary & finalize
        logger.info(_progress("变化追踪 & 收尾..."))
        try:
            from analyzers.change_tracker import generate_change_summary

            changes_path = generate_change_summary(output_dir / "reports", date_str)
            if changes_path:
                logger.info("  ✓ 变化报告: %s", changes_path)
        except Exception as e:
            logger.warning("Change summary skipped: %s", e)

        # 9.6 Alert evaluation
        alert_cfg = config.get("alerting", {})
        if alert_cfg.get("enabled", False):
            try:
                from alerting import AlertManager
                from analyzers.change_tracker import find_previous_report

                alert_mgr = AlertManager(config)
                prev_path = find_previous_report(output_dir / "reports", date_str)
                prev_report = None
                if prev_path:
                    with open(prev_path, "r", encoding="utf-8") as f:
                        prev_report = json.load(f)
                alerts = alert_mgr.evaluate(all_data, prev_report)
                if alerts:
                    logger.info("  ✓ Triggered %d alert(s)", len(alerts))
            except Exception as e:
                logger.warning("Alert evaluation skipped: %s", e)

        # Update watermarks for next incremental scan
        if not full_scan:
            try:
                _update_watermarks(watermarks, all_data)
                logger.info("  ✓ 水位线已更新")
            except Exception as e:
                logger.warning("Watermark update skipped: %s", e)

        # Print console summary
        logger.info(
            report_generator.generate_console_summary(
                lab_activity,
                vendor_activity,
                datasets_by_type,
                github_activity,
                blog_activity,
                pwc_datasets,
            )
        )

        _total_elapsed = _time.monotonic() - _scan_start_mono
        logger.info("Done! (%.1fs)", _total_elapsed)

        # Output insights prompt for LLM analysis (Claude Code / Claude App)
        if not args.no_insights:
            insights_content = format_insights_prompt(
                all_datasets=all_datasets,
                blog_activity=blog_activity,
                github_activity=github_activity,
                papers=papers,
                datasets_by_type=datasets_by_type,
                lab_activity=lab_activity,
                vendor_activity=vendor_activity,
                x_activity=x_activity,
                reddit_activity=reddit_activity,
                hn_activity=hn_activity,
                kaggle_datasets=kaggle_datasets,
                semantic_scholar_papers=semantic_scholar_papers,
                gh_trending=gh_trending,
                producthunt=producthunt,
                pwc_datasets=pwc_datasets,
            )

            # Save insights prompt to file (always — for Claude Code environment)
            insights_prompt_path = reports_dir / f"intel_report_{date_str}_insights_prompt.md"
            with open(insights_prompt_path, "w", encoding="utf-8") as f:
                f.write(insights_content)
            logger.info("Insights prompt saved to: %s", insights_prompt_path)

            insights_path = reports_dir / f"intel_report_{date_str}_insights.md"

            if args.api_insights:
                # Explicitly requested: generate insights via LLM API
                from utils.llm_client import generate_insights

                insights_result = await asyncio.to_thread(generate_insights, insights_content)

                if insights_result:
                    with open(insights_path, "w", encoding="utf-8") as f:
                        f.write(insights_result)
                    logger.info("Insights report saved to: %s", insights_path)
                else:
                    logger.warning(
                        "--api-insights requested but no LLM API key configured. "
                        "Set ANTHROPIC_API_KEY or LLM_PROVIDER+LLM_API_KEY"
                    )
                    logger.info("INSIGHTS_OUTPUT_PATH=%s", insights_path)
            else:
                # Default: let Claude Code / environment LLM handle analysis
                logger.info(
                    "Insights prompt ready for environment LLM: %s",
                    insights_prompt_path,
                )
                logger.info("INSIGHTS_OUTPUT_PATH=%s", insights_path)

            # Generate anomalies report (separate from insights — for engineering use)
            anomalies_content = format_anomalies_report(
                anomalies=anomalies,
                all_datasets=all_datasets,
                datasets_by_type=datasets_by_type,
                x_activity=x_activity,
                blog_activity=blog_activity,
                github_activity=github_activity,
                papers=papers,
            )
            anomalies_path = reports_dir / f"intel_report_{date_str}_anomalies.md"
            with open(anomalies_path, "w", encoding="utf-8") as f:
                f.write(anomalies_content)
            logger.info("Anomalies report saved to: %s", anomalies_path)

        # 10. DataRecipe deep analysis (if --recipe)
        if getattr(args, "recipe", False) and all_datasets:
            logger.info("=" * 60)
            logger.info("  DataRecipe Deep Analysis")
            logger.info("=" * 60)

            selected = rank_datasets_for_recipe(
                all_datasets=all_datasets,
                datasets_by_type=datasets_by_type,
                limit=getattr(args, "recipe_limit", 5),
            )

            if selected:
                logger.info("Selected %d datasets:", len(selected))
                for ds in selected:
                    logger.info(
                        "  %.1f  %s (%s)",
                        ds["_recipe_score"], ds["id"], ds["_recipe_category"],
                    )

                recipe_result = await run_recipe_analysis(
                    selected_datasets=selected,
                    reports_dir=reports_dir,
                    config=config,
                )

                if recipe_result.get("error"):
                    logger.warning("Recipe skipped: %s", recipe_result["error"])
                else:
                    logger.info(
                        "Recipe complete: %d/%d analyzed → %s",
                        recipe_result["datasets_analyzed"],
                        recipe_result["datasets_selected"],
                        recipe_result["output_dir"],
                    )
            else:
                logger.info("No datasets eligible for recipe analysis")
        elif getattr(args, "recipe", False):
            logger.warning("--recipe requested but no datasets found")
    finally:
        await http_client.close()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description=f"AI Dataset Radar v{__version__} - Competitive Intelligence System"
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=7,
        help="Look back period in days (default: 7)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output file path (default: data/intel_report_DATE.md)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Also save raw data as JSON",
    )
    parser.add_argument(
        "--no-labs",
        action="store_true",
        help="Skip AI labs tracking",
    )
    parser.add_argument(
        "--no-vendors",
        action="store_true",
        help="Skip vendor tracking",
    )
    parser.add_argument(
        "--no-github",
        action="store_true",
        help="Skip GitHub tracking",
    )
    parser.add_argument(
        "--no-blogs",
        action="store_true",
        help="Skip blog tracking",
    )
    parser.add_argument(
        "--no-papers",
        action="store_true",
        help="Skip paper fetching",
    )
    parser.add_argument(
        "--no-readme",
        action="store_true",
        help="Skip fetching dataset READMEs",
    )
    parser.add_argument(
        "--no-pwc",
        action="store_true",
        help="Skip Papers with Code dataset scraping",
    )
    parser.add_argument(
        "--no-x",
        action="store_true",
        help="Skip X/Twitter tracking",
    )
    parser.add_argument(
        "--no-reddit",
        action="store_true",
        help="Skip Reddit tracking",
    )
    parser.add_argument(
        "--no-hn",
        action="store_true",
        help="Skip Hacker News tracking",
    )
    parser.add_argument(
        "--no-kaggle",
        action="store_true",
        help="Skip Kaggle dataset scraping",
    )
    parser.add_argument(
        "--no-semantic-scholar",
        action="store_true",
        help="Skip Semantic Scholar paper scraping",
    )
    parser.add_argument(
        "--no-gh-trending",
        action="store_true",
        help="Skip GitHub Trending tracking",
    )
    parser.add_argument(
        "--no-producthunt",
        action="store_true",
        help="Skip Product Hunt tracking",
    )
    parser.add_argument(
        "--no-insights",
        action="store_true",
        help="Skip insights prompt generation entirely",
    )
    parser.add_argument(
        "--api-insights",
        action="store_true",
        help="Use LLM API (Anthropic/Kimi/DeepSeek) to auto-generate insights. "
        "Default: save prompt for Claude Code environment to analyze",
    )
    parser.add_argument(
        "--recipe",
        action="store_true",
        help="Run DataRecipe deep analysis on top-ranked datasets after scan",
    )
    parser.add_argument(
        "--recipe-limit",
        type=int,
        default=5,
        help="Max datasets to analyze with DataRecipe (default: 5)",
    )
    parser.add_argument(
        "--full-scan",
        action="store_true",
        help="Force full scan ignoring watermarks",
    )

    args = parser.parse_args()
    asyncio.run(async_main(args))


async def run_intel_scan(
    days: int = 7, api_insights: bool = False, full_scan: bool = False,
) -> dict:
    """Run an intelligence scan programmatically (used by the API).

    Args:
        days: Look back period in days.
        api_insights: If True, use LLM API to generate insights. Default: False.
        full_scan: If True, ignore watermarks and do a full scan.

    Returns:
        Summary dict with scan results.
    """
    import time as _time
    _scan_start_mono = _time.monotonic()
    _scan_started_at = datetime.now()

    setup_logging(level="INFO")
    config = load_config()
    validate_config(config)

    from utils.watermark import WatermarkStore

    output_dir = Path(config.get("report", {}).get("output_dir", "data"))
    watermarks = WatermarkStore(output_dir / "watermarks.json")
    incremental = (
        not full_scan
        and watermarks.get("labs") is not None
    )

    if incremental:
        logger.info("增量扫描模式（基于上次水位线）")
    else:
        logger.info("全量扫描模式（%d 天窗口）", days)

    def _days(source: str) -> int:
        if not incremental:
            return days
        return _effective_days(watermarks, source, days)

    github_org_watermarks = _load_org_watermarks(watermarks.get("github_orgs"))
    labs_org_watermarks = _load_org_watermarks(watermarks.get("labs_orgs"))
    vendor_org_watermarks = _load_org_watermarks(watermarks.get("vendors_orgs"))
    blog_source_watermarks = _load_org_watermarks(watermarks.get("blog_sources"))
    x_account_watermarks = _load_org_watermarks(watermarks.get("x_accounts"))
    reddit_source_watermarks = _load_org_watermarks(watermarks.get("reddit_sources"))
    blog_source_watermarks = _load_org_watermarks(watermarks.get("blog_sources"))
    x_account_watermarks = _load_org_watermarks(watermarks.get("x_accounts"))
    reddit_source_watermarks = _load_org_watermarks(watermarks.get("reddit_sources"))
    hn_source_watermarks = _load_org_watermarks(watermarks.get("hn_sources"))
    gh_trending_watermarks = _load_org_watermarks(watermarks.get("gh_trending_sources"))
    ph_source_watermarks = _load_org_watermarks(watermarks.get("producthunt_sources"))

    http_client = AsyncHTTPClient()
    try:
        org_tracker = OrgTracker(config, http_client=http_client)
        github_tracker = GitHubTracker(config, http_client=http_client)
        blog_tracker = BlogTracker(config, http_client=http_client)
        x_tracker = (
            XTracker(config, http_client=http_client)
            if config.get("x_tracker", {}).get("enabled", False)
            else None
        )
        reddit_tracker = (
            RedditTracker(config, http_client=http_client)
            if config.get("reddit_tracker", {}).get("enabled", False)
            else None
        )
        hn_tracker = (
            HNTracker(config, http_client=http_client)
            if config.get("hn_tracker", {}).get("enabled", False)
            else None
        )
        from scrapers.kaggle_scraper import KaggleScraper
        kaggle_scraper = (
            KaggleScraper(config=config, http_client=http_client)
            if config.get("sources", {}).get("kaggle", {}).get("enabled", False)
            else None
        )
        from scrapers.semantic_scholar import SemanticScholarScraper
        ss_scraper = (
            SemanticScholarScraper(config=config, http_client=http_client)
            if config.get("sources", {}).get("semantic_scholar", {}).get("enabled", False)
            else None
        )
        gh_trending_tracker = (
            GitHubTrendingTracker(config, http_client=http_client)
            if config.get("github_trending", {}).get("enabled", False)
            else None
        )
        ph_tracker = (
            ProductHuntTracker(config, http_client=http_client)
            if config.get("producthunt_tracker", {}).get("enabled", False)
            else None
        )
        data_classifier = DataTypeClassifier(config)
        paper_filter = PaperFilter(config)
        report_generator = IntelReportGenerator(config)
        hf_scraper = HuggingFaceScraper(config, http_client=http_client)
        hf_cfg = config.get("sources", {}).get("huggingface", {})
        hf_mode = str(hf_cfg.get("mode", "targeted")).lower()
        hf_general_enabled = hf_mode in {"general", "hybrid"}
        hf_general_datasets = []
        pwc_config = config.get("sources", {}).get("paperswithcode", {})
        pwc_scraper = None
        pwc_days = days
        if pwc_config.get("enabled", True):
            pwc_scraper = PapersWithCodeScraper(
                config=config,
                limit=pwc_config.get("limit", 50),
            )
            base_days = _days("pwc") if incremental else days
            config_days = pwc_config.get("days")
            if config_days is not None:
                pwc_days = min(config_days, base_days) if incremental else config_days
            else:
                pwc_days = base_days

        # Build async tasks
        tasks = {
            "labs": org_tracker.fetch_lab_activity(
                days=_days("labs"), org_watermarks=labs_org_watermarks
            ),
            "vendors": org_tracker.fetch_vendor_activity(
                days=_days("vendors"), org_watermarks=vendor_org_watermarks
            ),
            "github": github_tracker.fetch_all_orgs(
                days=_days("github"), org_watermarks=github_org_watermarks
            ),
            "blogs": blog_tracker.fetch_all_blogs(
                days=_days("blogs"), source_watermarks=blog_source_watermarks
            ),
        }
        if x_tracker:
            tasks["x"] = x_tracker.fetch_all(
                days=_days("x"), account_watermarks=x_account_watermarks
            )
        if reddit_tracker:
            tasks["reddit"] = reddit_tracker.fetch_all(
                days=_days("reddit"), source_watermarks=reddit_source_watermarks
            )
        if hn_tracker:
            tasks["hn"] = hn_tracker.fetch_all(
                days=_days("hn"), source_watermarks=hn_source_watermarks
            )
        if kaggle_scraper:
            tasks["kaggle"] = kaggle_scraper.fetch(days=_days("kaggle"))
        if ss_scraper:
            tasks["semantic_scholar"] = ss_scraper.fetch(days=_days("semantic_scholar"))
        if gh_trending_tracker:
            tasks["gh_trending"] = gh_trending_tracker.fetch_all(
                days=_days("gh_trending"), source_watermarks=gh_trending_watermarks
            )
        if ph_tracker:
            tasks["producthunt"] = ph_tracker.fetch_all(
                days=_days("producthunt"), source_watermarks=ph_source_watermarks
            )

        arxiv_config = config.get("sources", {}).get("arxiv", {})
        if arxiv_config.get("enabled", True):
            arxiv_scraper = ArxivScraper(limit=50, config=config, http_client=http_client)
            tasks["arxiv"] = arxiv_scraper.fetch()

        hf_config = config.get("sources", {}).get("hf_papers", {})
        if hf_config.get("enabled", True):
            hf_papers_scraper = HFPapersScraper(
                limit=50, days=hf_config.get("days", 7), http_client=http_client,
            )
            tasks["hf_papers"] = hf_papers_scraper.fetch()

        if hf_general_enabled:
            hf_general_watermark = watermarks.get("huggingface_general")
            general_limit = hf_cfg.get("general_limit", hf_cfg.get("limit", 100))
            general_max_pages = hf_cfg.get("general_max_pages", 5)

            async def _fetch_hf_general():
                scraper = HuggingFaceScraper(
                    config,
                    limit=general_limit,
                    http_client=http_client,
                )
                return await scraper.fetch(
                    min_timestamp=hf_general_watermark,
                    max_pages=general_max_pages,
                )

            tasks["hf_general"] = _fetch_hf_general()

        if pwc_scraper:
            async def _fetch_pwc():
                result = await asyncio.to_thread(pwc_scraper.fetch)
                return _filter_pwc_datasets(result, pwc_days)

            tasks["pwc"] = _fetch_pwc()

        # Run all tasks concurrently
        lab_activity = {"labs": {}}
        vendor_activity = {"vendors": {}}
        github_activity = []
        blog_activity = []
        x_activity = {"accounts": [], "search_results": []}
        reddit_activity = {"posts": [], "metadata": {}}
        hn_activity = {"stories": [], "metadata": {}}
        kaggle_datasets = []
        semantic_scholar_papers = []
        gh_trending = {"repos": [], "metadata": {}}
        producthunt = {"products": [], "metadata": {}}
        papers = []
        pwc_datasets = []

        keys = list(tasks.keys())
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)

        for key, result in zip(keys, results):
            try:
                if isinstance(result, Exception):
                    raise result
                if key == "labs":
                    lab_activity = {"labs": result}
                elif key == "vendors":
                    vendor_activity = {"vendors": result}
                elif key == "github":
                    github_activity = result.get("vendors", []) + result.get("labs", [])
                elif key == "blogs":
                    blog_activity = result
                elif key == "x":
                    x_activity = result
                elif key == "reddit":
                    reddit_activity = result
                elif key == "hn":
                    hn_activity = result
                elif key == "kaggle":
                    kaggle_datasets = result or []
                elif key == "semantic_scholar":
                    semantic_scholar_papers = result or []
                elif key == "gh_trending":
                    gh_trending = result
                elif key == "producthunt":
                    producthunt = result
                elif key == "arxiv":
                    papers.extend(paper_filter.filter_papers(result))
                elif key == "hf_papers":
                    papers.extend(paper_filter.filter_papers(result))
                elif key == "hf_general":
                    hf_general_datasets = result or []
                    logger.info(
                        "  ✓ HuggingFace 通用: %d 数据集", len(hf_general_datasets)
                    )
                elif key == "pwc":
                    pwc_datasets = result or []
            except Exception as e:
                logger.warning("Error fetching %s: %s", key, e)

        # Collect and classify datasets
        all_datasets = []
        for category in lab_activity.get("labs", {}).values():
            for org_data in category.values():
                all_datasets.extend(org_data.get("datasets", []))
        for tier in vendor_activity.get("vendors", {}).values():
            for vendor_data in tier.values():
                all_datasets.extend(vendor_data.get("datasets", []))

        if hf_general_datasets:
            all_datasets.extend(hf_general_datasets)

        if all_datasets:
            all_datasets = await fetch_dataset_readmes(all_datasets, hf_scraper)

        datasets_by_type = data_classifier.group_by_type(all_datasets)
        summary = data_classifier.summarize(all_datasets)

        # Advanced analysis
        competitor_matrix = CompetitorMatrix(config).build(
            datasets=all_datasets,
            github_activity=github_activity,
            papers=papers,
            blog_posts=blog_activity,
        )
        dataset_lineage = DatasetLineageTracker(config).analyze(all_datasets)
        org_graph = OrgRelationshipGraph(config).build(
            datasets=all_datasets,
            github_activity=github_activity,
            papers=papers,
            blog_posts=blog_activity,
        )

        # Data quality validation
        anomalies = []
        active_github = sum(1 for a in github_activity if a.get("repos_updated"))
        active_blogs = sum(1 for a in blog_activity if a.get("articles"))
        x_accounts = len(x_activity.get("accounts", []))

        if active_github == 0:
            anomalies.append("GitHub: 0 active orgs (expected >0)")
        if active_blogs == 0:
            anomalies.append("Blogs: 0 active blogs (expected >0)")
        if x_tracker and x_accounts == 0:
            anomalies.append("X/Twitter: 0 active accounts (check RSSHub/API connectivity)")
        if len(papers) == 0:
            anomalies.append("Papers: 0 results from arXiv + HF Papers (check network)")
        if len(all_datasets) == 0:
            anomalies.append("Datasets: 0 from all tracked orgs (check HuggingFace API)")

        if anomalies:
            logger.warning("Data quality warnings: %s", "; ".join(anomalies))

        # Trend analysis (before report so trends appear in output)
        trend_data = {}
        try:
            db_path = output_dir / "radar.db"
            db = RadarDatabase(str(db_path))
            trend_analyzer = TrendAnalyzer(db, config)
            if all_datasets:
                trend_analyzer.record_daily_stats(all_datasets)
                trend_analyzer.calculate_trends(
                    dataset_ids=trend_analyzer.last_recorded_ids
                )
                for ds in all_datasets:
                    ds_id = ds.get("id", "")
                    if ds_id:
                        ds_trend = trend_analyzer.get_dataset_trend(ds_id)
                        if ds_trend:
                            ds["growth_7d"] = ds_trend.get("growth_7d")
                            ds["growth_30d"] = ds_trend.get("growth_30d")
                trend_data = {
                    "top_growing_7d": trend_analyzer.get_top_growing_datasets(days=7, limit=10),
                    "rising_7d": trend_analyzer.get_rising_datasets(days=7, limit=10),
                }
            db.close()
        except Exception as e:
            logger.warning("Trend analysis skipped: %s", e)

        # Generate and save report
        report = report_generator.generate(
            lab_activity=lab_activity,
            vendor_activity=vendor_activity,
            datasets_by_type=datasets_by_type,
            papers=papers,
            github_activity=github_activity,
            blog_activity=blog_activity,
            x_activity=x_activity,
            reddit_activity=reddit_activity,
            hn_activity=hn_activity,
            kaggle_datasets=kaggle_datasets,
            trend_data=trend_data,
            competitor_matrix=competitor_matrix,
            dataset_lineage=dataset_lineage,
            org_graph=org_graph,
            pwc_datasets=pwc_datasets,
        )

        date_str = datetime.now().strftime("%Y-%m-%d")
        reports_dir = output_dir / "reports" / date_str
        reports_dir.mkdir(parents=True, exist_ok=True)
        formatter = DualOutputFormatter(output_dir=str(reports_dir))

        datasets_json = {}
        for dtype, ds_list in datasets_by_type.items():
            key = dtype.value if isinstance(dtype, DataType) else str(dtype)
            datasets_json[key] = [
                {k: v for k, v in ds.items() if not k.startswith("_")} for ds in ds_list
            ]

        _scan_elapsed = _time.monotonic() - _scan_start_mono
        all_data = {
            "data_quality_warnings": anomalies,
            "scan_mode": "incremental" if incremental else "full",
            "scan_started_at": _scan_started_at.isoformat(),
            "scan_duration_seconds": round(_scan_elapsed, 1),
            "period": {"days": days, "end": datetime.now().isoformat()},
            "labs_activity": lab_activity,
            "vendor_activity": vendor_activity,
            "github_activity": github_activity,
            "blog_posts": blog_activity,
            "x_activity": x_activity,
            "reddit_activity": reddit_activity,
            "hn_activity": hn_activity,
            "kaggle_datasets": kaggle_datasets,
            "semantic_scholar_papers": semantic_scholar_papers,
            "gh_trending": gh_trending,
            "producthunt": producthunt,
            "huggingface_general": hf_general_datasets,
            "paperswithcode": pwc_datasets,
            "datasets": all_datasets,
            "datasets_by_type": datasets_json,
            "papers": papers,
            "trend_data": trend_data,
            "competitor_matrix": competitor_matrix,
            "dataset_lineage": dataset_lineage,
            "org_graph": org_graph,
        }

        formatter.save_reports(markdown_content=report, data=all_data, filename_prefix="intel_report")

        # Daily change summary
        try:
            from analyzers.change_tracker import generate_change_summary

            changes_path = generate_change_summary(output_dir / "reports", date_str)
            if changes_path:
                logger.info("Change summary saved to: %s", changes_path)
        except Exception as e:
            logger.warning("Change summary skipped: %s", e)

        # Alert evaluation
        alert_cfg = config.get("alerting", {})
        if alert_cfg.get("enabled", False):
            try:
                from alerting import AlertManager
                from analyzers.change_tracker import find_previous_report

                alert_mgr = AlertManager(config)
                prev_path = find_previous_report(output_dir / "reports", date_str)
                prev_report = None
                if prev_path:
                    with open(prev_path, "r", encoding="utf-8") as f:
                        prev_report = json.load(f)
                alerts = alert_mgr.evaluate(all_data, prev_report)
                if alerts:
                    logger.info("Triggered %d alert(s)", len(alerts))
            except Exception as e:
                logger.warning("Alert evaluation skipped: %s", e)

        # Update watermarks for next incremental scan
        if not full_scan:
            try:
                _update_watermarks(watermarks, all_data)
                logger.info("水位线已更新")
            except Exception as e:
                logger.warning("Watermark update skipped: %s", e)

        # Insights prompt (always saved for environment LLM)
        insights_text = None
        insights_content = None
        insights_prompt_path = None
        try:
            insights_content = format_insights_prompt(
                all_datasets=all_datasets,
                blog_activity=blog_activity,
                github_activity=github_activity,
                papers=papers,
                datasets_by_type=datasets_by_type,
                lab_activity=lab_activity,
                vendor_activity=vendor_activity,
                x_activity=x_activity,
                reddit_activity=reddit_activity,
                hn_activity=hn_activity,
                kaggle_datasets=kaggle_datasets,
                semantic_scholar_papers=semantic_scholar_papers,
                gh_trending=gh_trending,
                producthunt=producthunt,
                pwc_datasets=pwc_datasets,
            )

            insights_prompt_path = reports_dir / f"intel_report_{date_str}_insights_prompt.md"
            with open(insights_prompt_path, "w", encoding="utf-8") as f:
                f.write(insights_content)
            logger.info("Insights prompt saved to: %s", insights_prompt_path)

            if api_insights:
                from utils.llm_client import generate_insights

                insights_text = await asyncio.to_thread(generate_insights, insights_content)
                insights_path = reports_dir / f"intel_report_{date_str}_insights.md"
                if insights_text:
                    with open(insights_path, "w", encoding="utf-8") as f:
                        f.write(insights_text)
                    logger.info("Insights report saved to: %s", insights_path)
                else:
                    logger.info("--api-insights but no LLM key configured")
            else:
                logger.info("Insights prompt ready for environment LLM")
        except Exception as e:
            logger.warning("Insights generation error: %s", e)

        return {
            "datasets": len(all_datasets),
            "papers": len(papers),
            "blogs": sum(len(b.get("articles", [])) for b in blog_activity),
            "github_repos": sum(len(a.get("repos_updated", [])) for a in github_activity),
            "classification": summary,
            "insights": insights_text,
            "insights_prompt_path": str(insights_prompt_path) if insights_content else None,
        }
    finally:
        await http_client.close()


if __name__ == "__main__":
    main()
