#!/usr/bin/env python3
"""AI Dataset Radar v4 - Competitive Intelligence System.

Main entry point for the competitive intelligence workflow.
Focused on tracking US AI Labs and data vendors.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import yaml

# Add src to path
src_dir = Path(__file__).parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from trackers.org_tracker import OrgTracker
from analyzers.data_type_classifier import DataTypeClassifier
from intel_report import IntelReportGenerator
from scrapers.arxiv import ArxivScraper
from scrapers.hf_papers import HFPapersScraper


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def filter_relevant_papers(papers: list[dict], config: dict) -> list[dict]:
    """Filter papers to only those matching relevant keywords.

    Args:
        papers: List of papers to filter.
        config: Configuration with arxiv keywords.

    Returns:
        Filtered list of papers with matched keywords.
    """
    keywords = config.get("sources", {}).get("arxiv", {}).get("keywords", [])
    if not keywords:
        return papers

    keywords_lower = [kw.lower() for kw in keywords]
    relevant = []

    for paper in papers:
        title = paper.get("title", "").lower()
        summary = paper.get("summary", "") or paper.get("abstract", "")
        summary = summary.lower()
        text = f"{title} {summary}"

        matched = []
        for kw, kw_lower in zip(keywords, keywords_lower):
            if kw_lower in text:
                matched.append(kw)

        if matched:
            paper["_matched_keywords"] = matched
            relevant.append(paper)

    return relevant


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="AI Dataset Radar v4 - Competitive Intelligence System"
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
        "--no-papers",
        action="store_true",
        help="Skip paper fetching",
    )

    args = parser.parse_args()

    # Load config
    print("=" * 60)
    print("  AI Dataset Radar v4")
    print("  Competitive Intelligence System")
    print("=" * 60)
    print()

    config = load_config(args.config)

    # Initialize components
    org_tracker = OrgTracker(config)
    data_classifier = DataTypeClassifier(config)
    report_generator = IntelReportGenerator(config)

    # Track organizations
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

    # Collect all datasets for classification
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

    # Classify datasets
    print("Classifying datasets by training type...")
    datasets_by_type = data_classifier.group_by_type(all_datasets)

    summary = data_classifier.summarize(all_datasets)
    print(f"  Relevant datasets: {summary['relevant']}/{summary['total']}")
    for dtype, count in summary["by_type"].items():
        if count > 0:
            print(f"    {dtype}: {count}")

    # Fetch papers
    papers = []
    if not args.no_papers:
        print("\nFetching relevant papers...")

        # arXiv
        arxiv_config = config.get("sources", {}).get("arxiv", {})
        if arxiv_config.get("enabled", True):
            print("  Fetching from arXiv...")
            arxiv_scraper = ArxivScraper(limit=50)
            arxiv_papers = arxiv_scraper.fetch()
            print(f"    Found {len(arxiv_papers)} papers")

            # Filter relevant
            arxiv_papers = filter_relevant_papers(arxiv_papers, config)
            print(f"    Relevant: {len(arxiv_papers)}")
            papers.extend(arxiv_papers)

        # HF Papers
        hf_config = config.get("sources", {}).get("hf_papers", {})
        if hf_config.get("enabled", True):
            print("  Fetching from HuggingFace Papers...")
            hf_scraper = HFPapersScraper(
                limit=50,
                days=hf_config.get("days", 7),
            )
            hf_papers = hf_scraper.fetch()
            print(f"    Found {len(hf_papers)} papers")

            # Filter relevant
            hf_papers = filter_relevant_papers(hf_papers, config)
            print(f"    Relevant: {len(hf_papers)}")
            papers.extend(hf_papers)

    # Generate report
    print("\nGenerating intelligence report...")

    report = report_generator.generate(
        lab_activity=lab_activity,
        vendor_activity=vendor_activity,
        datasets_by_type=datasets_by_type,
        papers=papers,
    )

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_dir = Path(config.get("report", {}).get("output_dir", "data"))
        output_dir.mkdir(parents=True, exist_ok=True)
        date_str = datetime.now().strftime("%Y-%m-%d")
        output_path = output_dir / f"intel_report_{date_str}.md"

    # Save report
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"Report saved to: {output_path}")

    # Save JSON if requested
    if args.json:
        json_path = output_path.with_suffix(".json")
        data = {
            "generated_at": datetime.now().isoformat(),
            "period_days": args.days,
            "lab_activity": lab_activity,
            "vendor_activity": vendor_activity,
            "datasets_by_type": {
                k: [
                    {key: val for key, val in ds.items() if not key.startswith("_")}
                    for ds in v
                ]
                for k, v in datasets_by_type.items()
            },
            "papers": papers,
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)
        print(f"JSON data saved to: {json_path}")

    # Print console summary
    print()
    print(report_generator.generate_console_summary(
        lab_activity, vendor_activity, datasets_by_type
    ))

    print("\nDone!")


if __name__ == "__main__":
    main()
