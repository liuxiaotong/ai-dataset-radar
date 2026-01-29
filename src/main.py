#!/usr/bin/env python3
"""Main entry point for AI Dataset Radar v2."""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import yaml

from scrapers import (
    HuggingFaceScraper,
    PapersWithCodeScraper,
    ArxivScraper,
    GitHubScraper,
    HFPapersScraper,
)
from filters import filter_datasets
from notifiers import create_notifiers, expand_env_vars
from db import get_database
from analyzers import ModelDatasetAnalyzer, TrendAnalyzer


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
    """Fetch data from all enabled sources.

    Args:
        config: Configuration dictionary.

    Returns:
        Dictionary with data from each source.
    """
    data = {}
    sources_cfg = config.get("sources", {})

    # Hugging Face
    hf_cfg = sources_cfg.get("huggingface", {})
    if hf_cfg.get("enabled", True):
        print("Fetching Hugging Face datasets...")
        scraper = HuggingFaceScraper(limit=hf_cfg.get("limit", 50))
        data["huggingface"] = scraper.fetch()
        print(f"  Found {len(data['huggingface'])} datasets")

    # Papers with Code
    pwc_cfg = sources_cfg.get("paperswithcode", {})
    if pwc_cfg.get("enabled", True):
        print("Fetching Papers with Code datasets...")
        scraper = PapersWithCodeScraper(limit=pwc_cfg.get("limit", 50))
        data["paperswithcode"] = scraper.fetch()
        print(f"  Found {len(data['paperswithcode'])} datasets")

    # arXiv
    arxiv_cfg = sources_cfg.get("arxiv", {})
    if arxiv_cfg.get("enabled", True):
        print("Fetching arXiv papers...")
        scraper = ArxivScraper(
            limit=arxiv_cfg.get("limit", 50),
            categories=arxiv_cfg.get("categories", ["cs.CL", "cs.CV", "cs.LG"]),
        )
        data["arxiv"] = scraper.fetch()
        print(f"  Found {len(data['arxiv'])} papers")

    # GitHub (early signal)
    github_cfg = sources_cfg.get("github", {})
    if github_cfg.get("enabled", True):
        print("Fetching GitHub trending repos...")
        token = github_cfg.get("token", "")
        if token.startswith("${"):
            token = expand_env_vars(token)
        scraper = GitHubScraper(
            limit=github_cfg.get("limit", 30),
            days=github_cfg.get("days", 7),
            token=token if token else None,
        )
        data["github"] = scraper.fetch()
        dataset_repos = [r for r in data["github"] if r.get("is_dataset")]
        print(f"  Found {len(data['github'])} repos ({len(dataset_repos)} dataset-related)")

    # HuggingFace Papers (early signal)
    hf_papers_cfg = sources_cfg.get("hf_papers", {})
    if hf_papers_cfg.get("enabled", True):
        print("Fetching HuggingFace daily papers...")
        scraper = HFPapersScraper(
            limit=hf_papers_cfg.get("limit", 50),
            days=hf_papers_cfg.get("days", 7),
        )
        data["hf_papers"] = scraper.fetch()
        dataset_papers = [p for p in data["hf_papers"] if p.get("is_dataset_paper")]
        print(f"  Found {len(data['hf_papers'])} papers ({len(dataset_papers)} dataset-related)")

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


def send_notifications(data: dict, config: dict) -> None:
    """Send notifications via all enabled channels.

    Args:
        data: Dictionary with data from each source.
        config: Configuration dictionary.
    """
    notifications_cfg = config.get("notifications", {})
    notifiers = create_notifiers(notifications_cfg)

    for notifier in notifiers:
        try:
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


def run_trend_analysis(data: dict, config: dict) -> dict:
    """Run trend analysis on fetched datasets.

    Args:
        data: Dictionary with fetched data.
        config: Configuration dictionary.

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


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="AI Dataset Radar v2 - Data Recipe Intelligence System"
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

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    print("=" * 60)
    print("  AI Dataset Radar v2")
    print("  Data Recipe Intelligence System")
    print("=" * 60)
    print()

    # Load configuration
    config = load_config(args.config)

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

    # Run trend analysis (record daily stats)
    if not args.no_trends and filtered_data.get("huggingface"):
        trend_results = run_trend_analysis(filtered_data, config)

    # Run model-dataset analysis
    if not args.no_models:
        model_results = run_model_dataset_analysis(config)

    # Send notifications
    if not args.no_notify:
        print("\nSending notifications...")
        send_notifications(filtered_data, config)

    print("\nDone!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
