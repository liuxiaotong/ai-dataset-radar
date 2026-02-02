#!/usr/bin/env python3
"""AI Dataset Radar v5 - Competitive Intelligence System.

Main entry point for the competitive intelligence workflow.
Integrates HuggingFace, GitHub, and Blog monitoring.
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

# Add src to path
src_dir = Path(__file__).parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

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
    """Load configuration from YAML file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def fetch_dataset_readmes(datasets: list[dict], hf_scraper: HuggingFaceScraper) -> list[dict]:
    """Fetch README content for datasets to improve classification.

    Args:
        datasets: List of datasets.
        hf_scraper: HuggingFace scraper instance.

    Returns:
        Datasets with card_data populated.
    """
    print("  Fetching dataset READMEs for better classification...")
    count = 0
    for ds in datasets[:30]:  # Limit to avoid rate limiting
        ds_id = ds.get("id", "")
        if ds_id and not ds.get("card_data"):
            try:
                card_data = hf_scraper.fetch_dataset_readme(ds_id)
                if card_data:
                    ds["card_data"] = card_data[:5000]  # Limit length
                    count += 1
                time.sleep(0.3)  # Rate limiting
            except Exception as e:
                print(f"    Warning: Could not fetch README for {ds_id}: {e}")

    print(f"    Fetched {count} READMEs")
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

    args = parser.parse_args()

    # Load config
    print("=" * 60)
    print("  AI Dataset Radar v5")
    print("  Competitive Intelligence System")
    print("=" * 60)
    print()

    config = load_config(args.config)

    # Initialize components
    org_tracker = OrgTracker(config)
    github_tracker = GitHubTracker(config)
    blog_tracker = BlogTracker(config)
    data_classifier = DataTypeClassifier(config)
    paper_filter = PaperFilter(config)
    report_generator = IntelReportGenerator(config)
    hf_scraper = HuggingFaceScraper(config)

    # 1. Track AI Labs on HuggingFace
    lab_activity = {"labs": {}}
    vendor_activity = {"vendors": {}}

    if not args.no_labs or not args.no_vendors:
        print("Tracking organizations on HuggingFace...")

        if not args.no_labs:
            lab_activity = {
                "labs": org_tracker.fetch_lab_activity(days=args.days)
            }

        if not args.no_vendors:
            vendor_activity = {
                "vendors": org_tracker.fetch_vendor_activity(days=args.days)
            }

    # 2. Track GitHub organizations
    github_activity = []
    if not args.no_github:
        print("\nTracking GitHub organizations...")
        github_data = github_tracker.fetch_all_orgs(days=args.days)
        github_activity = github_data.get("vendors", []) + github_data.get("labs", [])
        active_count = sum(1 for a in github_activity if a.get("repos_updated"))
        repo_count = sum(len(a.get("repos_updated", [])) for a in github_activity)
        print(f"  Found {active_count} active orgs with {repo_count} updated repos")

    # 3. Track company blogs
    blog_activity = []
    if not args.no_blogs:
        print("\nTracking company blogs...")
        blog_activity = blog_tracker.fetch_all_blogs(days=args.days)
        active_count = sum(1 for a in blog_activity if a.get("articles"))
        article_count = sum(len(a.get("articles", [])) for a in blog_activity)
        print(f"  Found {active_count} active blogs with {article_count} relevant articles")

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

    print(f"\nCollected {len(all_datasets)} datasets from tracked organizations")

    # 5. Fetch dataset READMEs for better classification
    if not args.no_readme and all_datasets:
        all_datasets = fetch_dataset_readmes(all_datasets, hf_scraper)

    # 6. Classify datasets
    print("Classifying datasets by training type...")
    datasets_by_type = data_classifier.group_by_type(all_datasets)

    summary = data_classifier.summarize(all_datasets)
    print(f"  Classified datasets: {summary['relevant']}/{summary['total']} relevant")
    print(f"  Other ratio: {summary['other_ratio']:.1%}")
    for dtype, count in summary["by_type"].items():
        if count > 0:
            print(f"    {dtype}: {count}")

    # 7. Fetch and filter papers
    papers = []
    if not args.no_papers:
        print("\nFetching relevant papers...")

        # arXiv
        arxiv_config = config.get("sources", {}).get("arxiv", {})
        if arxiv_config.get("enabled", True):
            print("  Fetching from arXiv...")
            arxiv_scraper = ArxivScraper(limit=50, config=config)
            arxiv_papers = arxiv_scraper.fetch()
            print(f"    Found {len(arxiv_papers)} papers")

            # Filter with paper filter
            arxiv_papers = paper_filter.filter_papers(arxiv_papers)
            print(f"    Relevant: {len(arxiv_papers)}")
            papers.extend(arxiv_papers)

        # HF Papers
        hf_config = config.get("sources", {}).get("hf_papers", {})
        if hf_config.get("enabled", True):
            print("  Fetching from HuggingFace Papers...")
            hf_papers_scraper = HFPapersScraper(
                limit=50,
                days=hf_config.get("days", 7),
            )
            hf_papers = hf_papers_scraper.fetch()
            print(f"    Found {len(hf_papers)} papers")

            # Filter with paper filter
            hf_papers = paper_filter.filter_papers(hf_papers)
            print(f"    Relevant: {len(hf_papers)}")
            papers.extend(hf_papers)

    # 8. Generate report
    print("\nGenerating intelligence report...")

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
        print(f"Report saved to: {output_path}")

        if args.json:
            json_path = output_path.with_suffix(".json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(
                    formatter._format_json_output(all_data),
                    f, ensure_ascii=False, indent=2, default=str
                )
            print(f"JSON data saved to: {json_path}")
    else:
        # Use DualOutputFormatter for default path
        md_path, json_path = formatter.save_reports(
            markdown_content=report,
            data=all_data,
            filename_prefix="intel_report"
        )
        print(f"Report saved to: {md_path}")
        print(f"JSON data saved to: {json_path}")

    # Print console summary
    print()
    print(report_generator.generate_console_summary(
        lab_activity, vendor_activity, datasets_by_type,
        github_activity, blog_activity
    ))

    print("\nDone!")


if __name__ == "__main__":
    main()
