#!/usr/bin/env python3
"""Main entry point for AI Dataset Radar v3 - High-Value Dataset Discovery System."""

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import yaml

from utils.logging_config import get_logger

logger = get_logger("main")

from scrapers import (
    HuggingFaceScraper,
    PapersWithCodeScraper,
    ArxivScraper,
    GitHubScraper,
    HFPapersScraper,
    SemanticScholarScraper,
    PwCSOTAScraper,
)
from filters import filter_datasets, DomainFilter, OrganizationFilter
from notifiers import create_notifiers, expand_env_vars, BusinessIntelNotifier
from db import get_database
from analyzers import (
    ModelDatasetAnalyzer,
    TrendAnalyzer,
    OpportunityAnalyzer,
    ModelCardAnalyzer,
    ValueScorer,
    ValueAggregator,
)
from report import generate_value_report


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file.

    Args:
        config_path: Path to configuration file.

    Returns:
        Configuration dictionary.
    """
    # Try multiple paths for config
    paths_to_try = [
        config_path,
        Path(__file__).parent.parent / "config.yaml",
        Path.cwd() / "config.yaml",
    ]

    for path in paths_to_try:
        if Path(path).exists():
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)

    print(f"Warning: Config file not found, using defaults")
    return {}


def fetch_all_data(config: dict) -> dict:
    """Fetch data from all enabled sources in parallel.

    Uses ThreadPoolExecutor to fetch from multiple sources concurrently,
    reducing total fetch time from 3-4 minutes to under 1 minute.

    Args:
        config: Configuration dictionary.

    Returns:
        Dictionary with data from each source.
    """
    data = {}
    sources_cfg = config.get("sources", {})
    tasks = []

    def fetch_huggingface():
        hf_cfg = sources_cfg.get("huggingface", {})
        if not hf_cfg.get("enabled", True):
            return "huggingface", []
        logger.info("Fetching Hugging Face datasets...")
        scraper = HuggingFaceScraper(limit=hf_cfg.get("limit", 50))
        result = scraper.fetch()
        logger.info("Found %d datasets from HuggingFace", len(result))
        return "huggingface", result

    def fetch_paperswithcode():
        pwc_cfg = sources_cfg.get("paperswithcode", {})
        if not pwc_cfg.get("enabled", True):
            return "paperswithcode", []
        logger.info("Fetching Papers with Code datasets...")
        scraper = PapersWithCodeScraper(limit=pwc_cfg.get("limit", 50))
        result = scraper.fetch()
        logger.info("Found %d datasets from PapersWithCode", len(result))
        return "paperswithcode", result

    def fetch_arxiv():
        arxiv_cfg = sources_cfg.get("arxiv", {})
        if not arxiv_cfg.get("enabled", True):
            return "arxiv", []
        logger.info("Fetching arXiv papers...")
        scraper = ArxivScraper(
            limit=arxiv_cfg.get("limit", 50),
            categories=arxiv_cfg.get("categories", ["cs.CL", "cs.CV", "cs.LG"]),
        )
        result = scraper.fetch()
        logger.info("Found %d papers from arXiv", len(result))
        return "arxiv", result

    def fetch_github():
        github_cfg = sources_cfg.get("github", {})
        if not github_cfg.get("enabled", True):
            return "github", []
        logger.info("Fetching GitHub trending repos...")
        token = github_cfg.get("token", "")
        if token.startswith("${"):
            token = expand_env_vars(token)
        scraper = GitHubScraper(
            limit=github_cfg.get("limit", 30),
            days=github_cfg.get("days", 7),
            token=token if token else None,
        )
        result = scraper.fetch()
        dataset_repos = [r for r in result if r.get("is_dataset")]
        logger.info("Found %d repos (%d dataset-related) from GitHub", len(result), len(dataset_repos))
        return "github", result

    def fetch_hf_papers():
        hf_papers_cfg = sources_cfg.get("hf_papers", {})
        if not hf_papers_cfg.get("enabled", True):
            return "hf_papers", []
        logger.info("Fetching HuggingFace daily papers...")
        scraper = HFPapersScraper(
            limit=hf_papers_cfg.get("limit", 50),
            days=hf_papers_cfg.get("days", 7),
        )
        result = scraper.fetch()
        dataset_papers = [p for p in result if p.get("is_dataset_paper")]
        logger.info("Found %d papers (%d dataset-related) from HF Papers", len(result), len(dataset_papers))
        return "hf_papers", result

    # Run all fetchers in parallel
    fetchers = [fetch_huggingface, fetch_paperswithcode, fetch_arxiv, fetch_github, fetch_hf_papers]

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(fn): fn.__name__ for fn in fetchers}

        for future in as_completed(futures):
            try:
                source_name, result = future.result()
                data[source_name] = result
            except Exception as e:
                fn_name = futures[future]
                logger.error("Error in %s: %s", fn_name, e)
                # Set empty result for failed source
                source_name = fn_name.replace("fetch_", "")
                data[source_name] = []

    return data


def apply_filters(data: dict, config: dict) -> dict:
    """Apply filters to all data sources.

    Args:
        data: Dictionary with data from each source.
        config: Configuration dictionary.

    Returns:
        Filtered data dictionary.
    """
    filters_cfg = config.get("filters", {})

    min_downloads = filters_cfg.get("min_downloads", 0)
    keywords = filters_cfg.get("keywords", [])
    days = filters_cfg.get("days", 7)

    filtered_data = {}
    for source, items in data.items():
        filtered = filter_datasets(
            items,
            min_downloads=min_downloads,
            keywords=keywords,
            days=days,
        )
        filtered_data[source] = filtered
        if len(filtered) != len(items):
            print(f"  {source}: {len(items)} -> {len(filtered)} after filtering")

    return filtered_data


def save_data(data: dict, config: dict) -> str:
    """Save data to JSON file.

    Args:
        data: Dictionary with data from each source.
        config: Configuration dictionary.

    Returns:
        Path to the saved file.
    """
    output_cfg = config.get("output", {})
    output_dir = output_cfg.get("json_dir", "data")

    os.makedirs(output_dir, exist_ok=True)

    date_str = datetime.now().strftime("%Y-%m-%d")
    filename = f"datasets_{date_str}.json"
    filepath = os.path.join(output_dir, filename)

    output = {
        "generated_at": datetime.now().isoformat(),
        "sources": data,
        "summary": {
            "huggingface_count": len(data.get("huggingface", [])),
            "paperswithcode_count": len(data.get("paperswithcode", [])),
            "arxiv_count": len(data.get("arxiv", [])),
            "github_count": len(data.get("github", [])),
            "hf_papers_count": len(data.get("hf_papers", [])),
        },
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Data saved to: {filepath}")
    return filepath


def send_notifications(
    data: dict,
    config: dict,
    trend_results: dict = None,
    opportunity_results: dict = None,
    domain_data: dict = None,
) -> None:
    """Send notifications via all enabled channels.

    Args:
        data: Dictionary with data from each source.
        config: Configuration dictionary.
        trend_results: Results from trend analysis.
        opportunity_results: Results from opportunity analysis.
        domain_data: Results from domain classification.
    """
    notifications_cfg = config.get("notifications", {})
    notifiers = create_notifiers(notifications_cfg, full_config=config)

    for notifier in notifiers:
        try:
            # BusinessIntelNotifier needs extra data
            if isinstance(notifier, BusinessIntelNotifier):
                notifier.notify(
                    data,
                    trend_results=trend_results,
                    opportunity_results=opportunity_results,
                    domain_data=domain_data,
                )
            else:
                notifier.notify(data)
        except Exception as e:
            print(f"Error in {notifier.__class__.__name__}: {e}")


def run_model_dataset_analysis(config: dict) -> dict:
    """Run model-dataset relationship analysis.

    Args:
        config: Configuration dictionary.

    Returns:
        Analysis results.
    """
    models_cfg = config.get("models", {})
    if not models_cfg.get("enabled", True):
        print("Model analysis disabled in config")
        return {}

    db = get_database(config)
    analyzer = ModelDatasetAnalyzer(db, config)

    print("\n" + "=" * 60)
    print("  Model-Dataset Relationship Analysis")
    print("=" * 60 + "\n")

    results = analyzer.analyze()

    # Print report
    report = analyzer.generate_report(results)
    print(report)

    return results


def run_trend_analysis(data: dict, config: dict, min_growth: float = None) -> dict:
    """Run trend analysis on fetched datasets.

    Args:
        data: Dictionary with fetched data.
        config: Configuration dictionary.
        min_growth: Optional minimum growth rate override.

    Returns:
        Trend analysis results.
    """
    db = get_database(config)
    analyzer = TrendAnalyzer(db, config)

    print("\n" + "=" * 60)
    print("  Dataset Trend Analysis")
    print("=" * 60 + "\n")

    # Combine all datasets for trend analysis
    all_datasets = data.get("huggingface", [])

    results = analyzer.analyze(all_datasets)

    # Print report
    report = analyzer.generate_report(results)
    print(report)

    return results


def run_opportunity_analysis(data: dict, config: dict) -> dict:
    """Run opportunity analysis to detect business signals.

    Args:
        data: Dictionary with fetched data.
        config: Configuration dictionary.

    Returns:
        Opportunity analysis results.
    """
    db = get_database(config)
    analyzer = OpportunityAnalyzer(db, config)

    print("\n" + "=" * 60)
    print("  Business Opportunity Analysis")
    print("=" * 60 + "\n")

    # Combine datasets
    datasets = data.get("huggingface", [])

    # Combine papers from all sources
    papers = []
    papers.extend(data.get("arxiv", []))
    papers.extend(data.get("hf_papers", []))

    results = analyzer.analyze(datasets, papers)

    # Print report
    report = analyzer.generate_report(results)
    print(report)

    return results


def run_domain_classification(data: dict, config: dict, focus: str = None) -> dict:
    """Classify all items by domain/focus area.

    Args:
        data: Dictionary with fetched data.
        config: Configuration dictionary.
        focus: Optional specific domain to focus on.

    Returns:
        Domain classification results.
    """
    focus_areas = config.get("focus_areas", {})
    if not focus_areas:
        return {}

    domain_filter = DomainFilter(focus_areas)

    print("\n" + "=" * 60)
    print("  Domain Classification")
    print("=" * 60 + "\n")

    # Classify all items
    all_items = []
    all_items.extend(data.get("huggingface", []))
    all_items.extend(data.get("arxiv", []))
    all_items.extend(data.get("hf_papers", []))
    all_items.extend(data.get("github", []))

    domain_data = domain_filter.classify_all(all_items)

    # Print summary
    for domain, items in domain_data.items():
        if domain != "uncategorized" and items:
            print(f"  {domain}: {len(items)} items")

    uncategorized = len(domain_data.get("uncategorized", []))
    print(f"  uncategorized: {uncategorized} items")

    # If focus specified, filter data
    if focus and focus in domain_data:
        print(f"\n  Focus mode: showing only '{focus}' domain")

    return domain_data


def run_value_analysis(data: dict, config: dict) -> dict:
    """Run v3 value scoring analysis.

    Args:
        data: Dictionary with fetched data.
        config: Configuration dictionary.

    Returns:
        Value analysis results.
    """
    print("\n" + "=" * 60)
    print("  High-Value Dataset Analysis (v3)")
    print("=" * 60 + "\n")

    aggregator = ValueAggregator()
    results = {}

    # Semantic Scholar - citation tracking
    ss_cfg = config.get("sources", {}).get("semantic_scholar", {})
    if ss_cfg.get("enabled", True):
        print("Fetching Semantic Scholar citations...")
        scraper = SemanticScholarScraper(
            limit=ss_cfg.get("limit", 100),
            months_back=ss_cfg.get("months_back", 6),
            min_citations=ss_cfg.get("min_citations", 20),
            min_monthly_growth=ss_cfg.get("min_monthly_growth", 10),
        )
        citation_papers = scraper.fetch_dataset_papers()
        print(f"  Found {len(citation_papers)} high-impact dataset papers")
        aggregator.add_semantic_scholar_data(citation_papers)
        results["citation_data"] = citation_papers

    # Model card analysis
    mc_cfg = config.get("value_analysis", {}).get("model_cards", {})
    if mc_cfg.get("enabled", True):
        print("Analyzing model cards...")
        analyzer = ModelCardAnalyzer(
            min_model_downloads=mc_cfg.get("min_downloads", 1000),
            model_limit=mc_cfg.get("limit", 500),
            min_dataset_usage=mc_cfg.get("min_usage", 3),
        )
        model_card_results = analyzer.analyze()
        print(f"  Analyzed {model_card_results.get('models_analyzed', 0)} models")
        print(f"  Found {len(model_card_results.get('valuable_datasets', []))} datasets with 3+ uses")
        aggregator.add_model_card_data(model_card_results)
        results["model_card_results"] = model_card_results

    # SOTA analysis
    sota_cfg = config.get("value_analysis", {}).get("sota", {})
    if sota_cfg.get("enabled", True):
        print("Analyzing SOTA associations...")
        scraper = PwCSOTAScraper(
            areas=sota_cfg.get("areas"),
            top_n=sota_cfg.get("top_n", 10),
        )
        sota_results = scraper.analyze_sota_datasets()
        print(f"  Found {sota_results.get('unique_datasets', 0)} datasets with SOTA associations")
        aggregator.add_sota_data(sota_results)
        results["sota_results"] = sota_results

    # Add HuggingFace data
    if data.get("huggingface"):
        aggregator.add_huggingface_data(data["huggingface"])

    # Get scored datasets
    min_score = config.get("value_analysis", {}).get("min_score", 0)
    scored_datasets = aggregator.get_scored_datasets(min_score)
    results["scored_datasets"] = scored_datasets

    # Print summary
    print("\nValue Analysis Summary:")
    high_value = len([d for d in scored_datasets if d.get("total_score", 0) >= 60])
    medium_value = len([d for d in scored_datasets if 40 <= d.get("total_score", 0) < 60])
    print(f"  High-value datasets (≥60): {high_value}")
    print(f"  Medium-value datasets (40-59): {medium_value}")
    print(f"  Total analyzed: {len(scored_datasets)}")

    # Generate and print report
    report = aggregator.generate_report(min_score=40)
    print("\n" + report)

    return results


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="AI Dataset Radar v3 - High-Value Dataset Discovery System"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--no-fetch",
        action="store_true",
        help="Skip fetching data from sources",
    )
    parser.add_argument(
        "--no-models",
        action="store_true",
        help="Skip model-dataset analysis",
    )
    parser.add_argument(
        "--no-trends",
        action="store_true",
        help="Skip trend analysis",
    )
    parser.add_argument(
        "--no-notify",
        action="store_true",
        help="Skip sending notifications",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: fetch data only, skip analysis",
    )

    # Business intelligence arguments (v2)
    parser.add_argument(
        "--focus",
        type=str,
        choices=["robotics", "rlhf", "multimodal", "code"],
        help="Filter by specific domain/focus area",
    )
    parser.add_argument(
        "--growth-only",
        action="store_true",
        help="Only show items with positive growth",
    )
    parser.add_argument(
        "--min-growth",
        type=float,
        default=None,
        help="Minimum growth rate to include (e.g., 0.5 for 50%%)",
    )
    parser.add_argument(
        "--opportunities",
        action="store_true",
        help="Focus on business opportunities (annotation signals, data factories)",
    )
    parser.add_argument(
        "--no-opportunities",
        action="store_true",
        help="Skip opportunity analysis",
    )

    # Value analysis arguments (v3)
    parser.add_argument(
        "--min-score",
        type=int,
        default=None,
        help="Minimum value score to include (0-100)",
    )
    parser.add_argument(
        "--domain",
        type=str,
        choices=["robotics", "nlp", "vision", "code", "rlhf"],
        help="Filter valuable datasets by domain",
    )
    parser.add_argument(
        "--top-institutions",
        action="store_true",
        help="Only show datasets from top institutions",
    )
    parser.add_argument(
        "--value-analysis",
        action="store_true",
        help="Run v3 value analysis (citation tracking, model cards, SOTA)",
    )
    parser.add_argument(
        "--no-value-analysis",
        action="store_true",
        help="Skip v3 value analysis",
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    print("=" * 60)
    print("  AI Dataset Radar v3")
    print("  High-Value Dataset Discovery System")
    print("=" * 60)
    print()

    # Load configuration
    config = load_config(args.config)

    # Override config with CLI arguments
    if args.min_growth is not None:
        if "analysis" not in config:
            config["analysis"] = {}
        config["analysis"]["min_growth_alert"] = args.min_growth

    # Initialize database
    db = get_database(config)
    print(f"Database: {config.get('database', {}).get('path', 'data/radar.db')}")
    print()

    # Fetch data from all sources
    if not args.no_fetch:
        print("Fetching data from sources...")
        data = fetch_all_data(config)
        print()

        # Apply filters
        print("Applying filters...")
        filtered_data = apply_filters(data, config)
        print()

        # Save data to JSON
        print("Saving data...")
        save_data(filtered_data, config)
        print()
    else:
        print("Skipping data fetch (--no-fetch)")
        filtered_data = {
            "huggingface": [],
            "paperswithcode": [],
            "arxiv": [],
            "github": [],
            "hf_papers": [],
        }
        print()

    # Quick mode - skip analysis
    if args.quick:
        print("Quick mode - skipping analysis")
        if not args.no_notify:
            print("Sending notifications...")
            send_notifications(filtered_data, config)
        print("\nDone!")
        return 0

    # Initialize results for notifiers
    trend_results = None
    opportunity_results = None
    domain_data = None

    # Run domain classification
    if config.get("focus_areas"):
        domain_data = run_domain_classification(filtered_data, config, focus=args.focus)

        # Enrich items with domain info for trend analysis
        if domain_data:
            domain_filter = DomainFilter(config.get("focus_areas", {}))
            for source_data in filtered_data.values():
                if isinstance(source_data, list):
                    domain_filter.enrich_items(source_data)

    # Run trend analysis (record daily stats)
    if not args.no_trends and filtered_data.get("huggingface"):
        trend_results = run_trend_analysis(
            filtered_data, config, min_growth=args.min_growth
        )

    # Run opportunity analysis
    if not args.no_opportunities and not args.quick:
        opportunity_results = run_opportunity_analysis(filtered_data, config)

    # Run model-dataset analysis
    if not args.no_models:
        model_results = run_model_dataset_analysis(config)

    # Run v3 value analysis
    value_results = None
    if args.value_analysis and not args.no_value_analysis:
        value_results = run_value_analysis(filtered_data, config)

        # Generate value report
        if value_results.get("scored_datasets"):
            report_path = generate_value_report(
                value_results["scored_datasets"],
                output_dir=config.get("output", {}).get("json_dir", "data"),
                sota_results=value_results.get("sota_results"),
                citation_data=value_results.get("citation_data"),
                model_card_results=value_results.get("model_card_results"),
            )
            print(f"\nValue report saved to: {report_path}")

        # Apply min-score filter
        if args.min_score is not None:
            scored = value_results.get("scored_datasets", [])
            filtered_scored = [d for d in scored if d.get("total_score", 0) >= args.min_score]
            print(f"\n--min-score={args.min_score}: {len(filtered_scored)} datasets meet threshold")

        # Apply top-institutions filter
        if args.top_institutions:
            scored = value_results.get("scored_datasets", [])
            top_inst = [d for d in scored if d.get("is_top_institution")]
            print(f"\n--top-institutions: {len(top_inst)} datasets from top institutions")

    # Filter by domain if --focus specified
    if args.focus and domain_data:
        focus_items = domain_data.get(args.focus, [])
        print(f"\n--focus={args.focus}: {len(focus_items)} items in this domain")

    # Filter by growth if --growth-only specified
    if args.growth_only and trend_results:
        rising = trend_results.get("top_growing_7d", [])
        print(f"\n--growth-only: {len(rising)} items with positive growth")

    # Opportunities-only mode
    if args.opportunities:
        print("\n--opportunities mode: focusing on business signals")
        if opportunity_results:
            opp_count = opportunity_results.get("summary", {}).get("annotation_opportunity_count", 0)
            factory_count = opportunity_results.get("summary", {}).get("data_factory_count", 0)
            print(f"  Annotation opportunities: {opp_count}")
            print(f"  Data factories detected: {factory_count}")

    # Send notifications
    if not args.no_notify:
        print("\nSending notifications...")
        send_notifications(
            filtered_data,
            config,
            trend_results=trend_results,
            opportunity_results=opportunity_results,
            domain_data=domain_data,
        )

    print("\nDone!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
