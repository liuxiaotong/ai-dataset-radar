#!/usr/bin/env python3
"""AI Dataset Radar v5 - Competitive Intelligence System.

Main entry point for the competitive intelligence workflow.
Integrates HuggingFace, GitHub, and Blog monitoring.
"""

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import yaml

# Add src to path
src_dir = Path(__file__).parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from utils.logging_config import get_logger, setup_logging

logger = get_logger("main_intel")

from trackers.org_tracker import OrgTracker
from trackers.github_tracker import GitHubTracker
from trackers.blog_tracker import BlogTracker
from analyzers.data_type_classifier import DataTypeClassifier, DataType
from analyzers.paper_filter import PaperFilter
from intel_report import IntelReportGenerator
from scrapers.arxiv import ArxivScraper
from scrapers.hf_papers import HFPapersScraper
from scrapers.huggingface import HuggingFaceScraper
from output_formatter import DualOutputFormatter


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



def format_insights_prompt(
    all_datasets: list,
    blog_activity: list,
    github_activity: list,
    papers: list,
    datasets_by_type: dict,
    lab_activity: dict = None,
    vendor_activity: dict = None,
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
        active_orgs = {
            k: v for k, v in cat_data.items()
            if v.get("datasets") or v.get("models")
        }
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
                desc = ds.get("description", "")
                # Clean up description whitespace
                if desc:
                    desc = " ".join(desc.split())[:300]
                lines.append(f"- 📦 **{ds_id}** (downloads: {downloads:,}, likes: {likes})")
                if desc:
                    lines.append(f"  {desc}")
                # Show meaningful tags (filter out noise)
                tags = ds.get("tags", [])
                meaningful_tags = [
                    t for t in tags
                    if not t.startswith(("region:", "library:", "size_categories:",
                                        "format:", "arxiv:", "language:"))
                    and t not in ("region:us",)
                ][:8]
                if meaningful_tags:
                    lines.append(f"  标签: {', '.join(meaningful_tags)}")

            # Models with context - show top models by downloads+likes, limit noise
            notable_models = [m for m in model_list if m.get("downloads", 0) > 0 or m.get("likes", 0) > 0]
            if not notable_models:
                # All models are zero-activity, just summarize
                if model_list:
                    sample = model_list[0].get("id", "").split("/")[-1] if model_list else ""
                    lines.append(f"- 🤖 *{len(model_list)} 个模型（均无下载/点赞，如 {sample} 等）*")
                model_list_to_show = []
            else:
                top_models = sorted(notable_models, key=lambda m: -(m.get("downloads", 0) + m.get("likes", 0) * 100))
                model_list_to_show = top_models[:5]
            for model in model_list_to_show:
                model_id = model.get("id", "")
                downloads = model.get("downloads", 0)
                likes = model.get("likes", 0)
                pipeline = model.get("pipeline_tag", "")
                model_tags = model.get("tags", [])
                # Extract meaningful tags for models
                meaningful = [
                    t for t in model_tags
                    if not t.startswith(("region:", "base_model:", "endpoints_",
                                        "license:", "arxiv:"))
                    and t not in ("safetensors", "transformers", "pytorch", "en",
                                  "model_hub_mixin", "pytorch_model_hub_mixin")
                ][:6]
                lines.append(f"- 🤖 **{model_id}** (downloads: {downloads:,}, likes: {likes}, pipeline: {pipeline})")
                if meaningful:
                    lines.append(f"  标签: {', '.join(meaningful)}")
            if len(notable_models) > 5:
                lines.append(f"  *(另有 {len(notable_models) - 5} 个模型省略)*")

            lines.append("")

    if not has_lab_activity:
        lines.append("*本周无 AI Labs 新活动*\n")

    # ── Section 2: Vendor Activity ──
    lines.append("## 二、数据供应商动态（竞品）\n")
    vendors = (vendor_activity or {}).get("vendors", {})
    has_vendor_activity = False

    for tier_name, tier_data in vendors.items():
        active_vendors = {
            k: v for k, v in tier_data.items()
            if v.get("datasets") or v.get("models")
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
                desc = ds.get("description", "")
                if desc:
                    desc = " ".join(desc.split())[:300]
                lines.append(f"- 📦 **{ds_id}** (downloads: {downloads:,})")
                if desc:
                    lines.append(f"  {desc}")
            lines.append("")

    if not has_vendor_activity:
        lines.append("*本周无供应商 HuggingFace 新活动*\n")

    # ── Section 3: Dataset Classification Results ──
    lines.append("## 三、数据集分类分析\n")
    if datasets_by_type:
        # Show classified types first, "other" last
        classified = {k: v for k, v in datasets_by_type.items()
                      if (k.value if hasattr(k, 'value') else str(k)) != "other" and v}
        other = {k: v for k, v in datasets_by_type.items()
                 if (k.value if hasattr(k, 'value') else str(k)) == "other" and v}

        total = sum(len(v) for v in datasets_by_type.values())
        classified_count = sum(len(v) for v in classified.values())
        lines.append(f"共 {total} 个数据集，已分类 {classified_count} 个：\n")

        for dtype, ds_list in classified.items():
            type_name = dtype.value if hasattr(dtype, 'value') else str(dtype)
            lines.append(f"- **{type_name}**: {len(ds_list)} 个 — {', '.join(ds.get('id', '') for ds in ds_list[:5])}")

        if other:
            other_list = list(other.values())[0]
            lines.append(f"- **未分类**: {len(other_list)} 个 — {', '.join(ds.get('id', '') for ds in other_list[:5])}")
        lines.append("")
    else:
        lines.append("*无分类数据*\n")

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
            lines.append("*无博客更新*\n")
    else:
        lines.append("*无博客更新*\n")

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
                lines.append(f"- **{repo.get('org')}/{repo.get('name')}** ⭐ {repo.get('stars', 0)}")
                if repo.get("description"):
                    lines.append(f"  {repo.get('description', '')[:120]}")
                signals = repo.get("signals", [])
                if signals:
                    lines.append(f"  信号: {', '.join(str(s) for s in signals[:5])}")
            lines.append("")

        if medium:
            lines.append(f"### 中相关 (Top {len(medium)})")
            for repo in medium:
                lines.append(f"- **{repo.get('org')}/{repo.get('name')}** ⭐ {repo.get('stars', 0)}")
                if repo.get("description"):
                    lines.append(f"  {repo.get('description', '')[:120]}")
            lines.append("")

        # Summary stats
        total_repos = len(all_repos)
        active_orgs = len([o for o in github_activity if o.get("repos_updated")])
        lines.append(f"*共监控 {active_orgs} 个组织，{total_repos} 个活跃仓库*\n")
    else:
        lines.append("*无 GitHub 活动*\n")

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

                link_str = f"[{source}]({url})" if url else f"[{source}]"
                lines.append(f"- **{title}** {link_str}")
                if matched_kw:
                    lines.append(f"  关键词命中: {', '.join(matched_kw[:5])}")
                if abstract:
                    lines.append(f"  摘要: {abstract}")
            lines.append("")
    else:
        lines.append("*无相关论文*\n")

    # ── Analysis Prompt ──
    lines.append("=" * 60)
    lines.append("  分析要求")
    lines.append("=" * 60 + "\n")
    lines.append("""背景：你是 AI 训练数据行业的竞争情报分析师。读者是一家数据服务公司的管理层，需要从以上数据中获取可执行的商业洞察。

请提供以下分析：

### 1. 关键发现（Key Findings）
- 本周最值得关注的 3-5 个事件（数据集发布、模型动态、工具更新），逐条说明原因和商业意义
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

### 5. 异常与待排查
- 数据采集中是否有异常（如某数据源返回 0 结果、分类覆盖率过低等）
- 值得人工复查的条目

请用中文回答。分析应该具体、可执行，避免泛泛而谈。引用具体的数据集名称、组织名称和论文标题。
""")

    return "\n".join(lines)


def validate_config(config: dict) -> list[str]:
    """Validate configuration has required sections.

    Args:
        config: Configuration dictionary.

    Returns:
        List of warning messages.
    """
    warnings = []

    if not config:
        warnings.append("Configuration is empty, using defaults")
        return warnings

    # Check for watched orgs
    watched_orgs = config.get("watched_orgs", {})
    if not watched_orgs:
        warnings.append("No watched_orgs configured - no HuggingFace orgs will be tracked")

    # Check for watched vendors
    watched_vendors = config.get("watched_vendors", {})
    if not watched_vendors:
        warnings.append("No watched_vendors configured - no vendors will be tracked")

    # Check for blogs
    blogs = watched_vendors.get("blogs", [])
    if not blogs:
        warnings.append("No blogs configured - blog tracking disabled")

    return warnings


def fetch_dataset_readmes(datasets: list[dict], hf_scraper: HuggingFaceScraper) -> list[dict]:
    """Fetch README content for datasets to improve classification.

    Uses parallel fetching with rate-limited workers for speed.

    Args:
        datasets: List of datasets.
        hf_scraper: HuggingFace scraper instance.

    Returns:
        Datasets with card_data populated.
    """
    logger.info("Fetching dataset READMEs for better classification...")
    to_fetch = [
        (i, ds) for i, ds in enumerate(datasets[:30])
        if ds.get("id") and not ds.get("card_data")
    ]

    if not to_fetch:
        return datasets

    def _fetch_one(idx_ds):
        idx, ds = idx_ds
        ds_id = ds.get("id", "")
        try:
            card_data = hf_scraper.fetch_dataset_readme(ds_id)
            return idx, card_data
        except Exception as e:
            logger.warning("Could not fetch README for %s: %s", ds_id, e)
            return idx, None

    count = 0
    with ThreadPoolExecutor(max_workers=5) as executor:
        for idx, card_data in executor.map(_fetch_one, to_fetch):
            if card_data:
                datasets[idx]["card_data"] = card_data[:5000]
                count += 1

    logger.info("Fetched %d READMEs", count)
    return datasets


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="AI Dataset Radar v5 - Competitive Intelligence System"
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
        "--no-insights",
        action="store_true",
        help="Skip LLM analysis prompt output (enabled by default)",
    )

    args = parser.parse_args()

    # Set up logging based on verbosity
    setup_logging(level="INFO")

    # Load config
    logger.info("=" * 60)
    logger.info("  AI Dataset Radar v5")
    logger.info("  Competitive Intelligence System")
    logger.info("=" * 60)

    config = load_config(args.config)

    # Initialize components
    org_tracker = OrgTracker(config)
    github_tracker = GitHubTracker(config)
    blog_tracker = BlogTracker(config)
    data_classifier = DataTypeClassifier(config)
    paper_filter = PaperFilter(config)
    report_generator = IntelReportGenerator(config)
    hf_scraper = HuggingFaceScraper(config)

    # 1-3. Fetch all data sources in parallel for maximum speed
    lab_activity = {"labs": {}}
    vendor_activity = {"vendors": {}}
    github_activity = []
    blog_activity = []
    papers = []

    # Pre-build paper scrapers so they're ready for parallel submission
    arxiv_scraper = None
    hf_papers_scraper = None
    if not args.no_papers:
        arxiv_config = config.get("sources", {}).get("arxiv", {})
        if arxiv_config.get("enabled", True):
            arxiv_scraper = ArxivScraper(limit=50, config=config)
        hf_config = config.get("sources", {}).get("hf_papers", {})
        if hf_config.get("enabled", True):
            hf_papers_scraper = HFPapersScraper(
                limit=50,
                days=hf_config.get("days", 7),
            )

    futures = {}
    with ThreadPoolExecutor(max_workers=6, thread_name_prefix="radar") as executor:
        if not args.no_labs:
            logger.info("Tracking AI labs on HuggingFace...")
            futures["labs"] = executor.submit(org_tracker.fetch_lab_activity, days=args.days)

        if not args.no_vendors:
            logger.info("Tracking data vendors on HuggingFace...")
            futures["vendors"] = executor.submit(org_tracker.fetch_vendor_activity, days=args.days)

        if not args.no_github:
            logger.info("Tracking GitHub organizations...")
            futures["github"] = executor.submit(github_tracker.fetch_all_orgs, days=args.days)

        if not args.no_blogs:
            logger.info("Tracking company blogs...")
            futures["blogs"] = executor.submit(blog_tracker.fetch_all_blogs, days=args.days)

        if arxiv_scraper:
            logger.info("Fetching from arXiv...")
            futures["arxiv"] = executor.submit(arxiv_scraper.fetch)

        if hf_papers_scraper:
            logger.info("Fetching from HuggingFace Papers...")
            futures["hf_papers"] = executor.submit(hf_papers_scraper.fetch)

        # Collect results as they complete
        for key, future in futures.items():
            try:
                result = future.result()
                if key == "labs":
                    lab_activity = {"labs": result}
                elif key == "vendors":
                    vendor_activity = {"vendors": result}
                elif key == "github":
                    github_activity = result.get("vendors", []) + result.get("labs", [])
                    active_count = sum(1 for a in github_activity if a.get("repos_updated"))
                    repo_count = sum(len(a.get("repos_updated", [])) for a in github_activity)
                    logger.info("Found %d active orgs with %d updated repos", active_count, repo_count)
                elif key == "blogs":
                    blog_activity = result
                    active_count = sum(1 for a in blog_activity if a.get("articles"))
                    article_count = sum(len(a.get("articles", [])) for a in blog_activity)
                    logger.info("Found %d active blogs with %d relevant articles", active_count, article_count)
                elif key == "arxiv":
                    logger.info("Found %d arXiv papers", len(result))
                    papers.extend(paper_filter.filter_papers(result))
                    logger.info("Relevant arXiv: %d", len(papers))
                elif key == "hf_papers":
                    logger.info("Found %d HF papers", len(result))
                    filtered = paper_filter.filter_papers(result)
                    papers.extend(filtered)
                    logger.info("Relevant HF papers: %d", len(filtered))
            except Exception as e:
                logger.warning("Error fetching %s: %s", key, e)

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

    logger.info("Collected %d datasets from tracked organizations", len(all_datasets))

    # 5. Fetch dataset READMEs for better classification
    if not args.no_readme and all_datasets:
        all_datasets = fetch_dataset_readmes(all_datasets, hf_scraper)

    # 6. Classify datasets
    logger.info("Classifying datasets by training type...")
    datasets_by_type = data_classifier.group_by_type(all_datasets)

    summary = data_classifier.summarize(all_datasets)
    logger.info("Classified datasets: %d/%d relevant", summary['relevant'], summary['total'])
    logger.info("Other ratio: %.1f%%", summary['other_ratio'] * 100)
    for dtype, count in summary["by_type"].items():
        if count > 0:
            logger.info("  %s: %d", dtype, count)

    # 7. Papers already fetched in parallel above (arXiv + HF Papers)

    # 8. Generate report
    logger.info("Generating intelligence report...")

    report = report_generator.generate(
        lab_activity=lab_activity,
        vendor_activity=vendor_activity,
        datasets_by_type=datasets_by_type,
        papers=papers,
        github_activity=github_activity,
        blog_activity=blog_activity,
    )

    # Prepare structured data for JSON output
    datasets_json = {}
    for dtype, ds_list in datasets_by_type.items():
        key = dtype.value if isinstance(dtype, DataType) else str(dtype)
        datasets_json[key] = [
            {k: v for k, v in ds.items() if not k.startswith("_")}
            for ds in ds_list
        ]

    all_data = {
        "period": {
            "days": args.days,
            "start": None,
            "end": datetime.now().isoformat(),
        },
        "labs_activity": lab_activity,
        "vendor_activity": vendor_activity,
        "github_activity": github_activity,
        "blog_posts": blog_activity,
        "datasets": all_datasets,
        "datasets_by_type": datasets_json,
        "papers": papers,
    }

    # Determine output directory and save reports
    output_dir = Path(config.get("report", {}).get("output_dir", "data"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize dual formatter
    formatter = DualOutputFormatter(output_dir=str(output_dir / "reports"))

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
                    f, ensure_ascii=False, indent=2, default=str
                )
            logger.info("JSON data saved to: %s", json_path)
    else:
        # Use DualOutputFormatter for default path
        md_path, json_path = formatter.save_reports(
            markdown_content=report,
            data=all_data,
            filename_prefix="intel_report"
        )
        logger.info("Report saved to: %s", md_path)
        logger.info("JSON data saved to: %s", json_path)

    # Print console summary
    logger.info(report_generator.generate_console_summary(
        lab_activity, vendor_activity, datasets_by_type,
        github_activity, blog_activity
    ))

    logger.info("Done!")

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
        )
        print(insights_content)

        # Save insights prompt to file for reference
        insights_prompt_path = output_dir / "reports" / f"intel_report_{datetime.now().strftime('%Y-%m-%d')}_insights_prompt.md"
        with open(insights_prompt_path, "w", encoding="utf-8") as f:
            f.write(insights_content)
        logger.info("Insights prompt saved to: %s", insights_prompt_path)
        logger.info("")
        logger.info(">>> AI 分析完成后，请将分析结果保存到:")
        logger.info(">>> %s", str(insights_prompt_path).replace("_prompt.md", ".md"))


if __name__ == "__main__":
    main()
