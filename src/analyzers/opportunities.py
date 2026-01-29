"""Opportunity detection for business intelligence."""

import re
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

# Add parent directory to path for imports when running standalone
_src_dir = Path(__file__).parent.parent
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

from db import RadarDatabase


class OpportunityAnalyzer:
    """Analyze data for business opportunities.

    Detects:
    1. Data factories - authors/orgs publishing many datasets quickly
    2. Annotation signals - papers indicating data collection needs
    3. Organization activity - tracking major players
    """

    # Default annotation signals indicating potential business opportunities
    DEFAULT_ANNOTATION_SIGNALS = [
        "human annotation",
        "manually labeled",
        "crowdsourced",
        "we collected",
        "data collection",
        "benchmark",
        "human evaluation",
        "human preference",
        "RLHF",
        "evaluation dataset",
        "training data",
        "annotation guideline",
        "labeled data",
        "labelled data",
        "annotators",
        "annotation process",
        "quality control",
    ]

    def __init__(self, db: RadarDatabase, config: Optional[dict] = None):
        """Initialize the opportunity analyzer.

        Args:
            db: Database instance.
            config: Optional configuration dict.
        """
        self.db = db
        self.config = config or {}
        self.opportunities_config = self.config.get("opportunities", {})

        # Load annotation signals from config or use defaults
        self.annotation_signals = self.opportunities_config.get(
            "annotation_signals", self.DEFAULT_ANNOTATION_SIGNALS
        )
        self.annotation_signals_lower = [s.lower() for s in self.annotation_signals]

        # Data factory detection settings
        factory_config = self.opportunities_config.get("data_factory", {})
        self.factory_min_datasets = factory_config.get("min_datasets", 3)
        self.factory_days = factory_config.get("days", 7)

        # Load tracked organizations
        self.tracked_orgs = self.config.get("tracked_orgs", {})

    def detect_data_factories(self, datasets: list[dict]) -> list[dict]:
        """Detect authors/organizations publishing datasets at high frequency.

        A "data factory" is an author who has published 3+ datasets in the last 7 days.

        Args:
            datasets: List of dataset dictionaries with 'author' and 'created_at' fields.

        Returns:
            List of data factory detections with author info and their datasets.
        """
        # Group datasets by author
        author_datasets = defaultdict(list)
        cutoff = datetime.now() - timedelta(days=self.factory_days)

        for ds in datasets:
            author = ds.get("author", "").strip()
            if not author:
                continue

            # Check if dataset is recent
            created_at = ds.get("created_at")
            if created_at:
                try:
                    if isinstance(created_at, str):
                        # Parse ISO format date
                        ds_date = datetime.fromisoformat(
                            created_at.replace("Z", "+00:00")
                        ).replace(tzinfo=None)
                    else:
                        ds_date = created_at.replace(tzinfo=None)

                    if ds_date < cutoff:
                        continue
                except (ValueError, TypeError):
                    pass

            author_datasets[author].append(ds)

        # Find authors with multiple datasets
        factories = []
        for author, ds_list in author_datasets.items():
            if len(ds_list) >= self.factory_min_datasets:
                # Try to detect organization affiliation
                org = self._detect_org_from_author(author)

                factories.append({
                    "author": author,
                    "dataset_count": len(ds_list),
                    "datasets": ds_list,
                    "possible_org": org,
                    "period_days": self.factory_days,
                })

        # Sort by dataset count descending
        factories.sort(key=lambda x: x["dataset_count"], reverse=True)
        return factories

    def _detect_org_from_author(self, author: str) -> Optional[str]:
        """Try to detect organization from author name.

        Args:
            author: Author name string.

        Returns:
            Organization name if detected, None otherwise.
        """
        author_lower = author.lower()

        for org_name, aliases in self.tracked_orgs.items():
            for alias in aliases:
                if alias.lower() in author_lower:
                    return org_name

        return None

    def extract_annotation_signals(self, papers: list[dict]) -> list[dict]:
        """Extract papers that signal annotation/data collection needs.

        Args:
            papers: List of paper dictionaries with 'title', 'summary', 'abstract' fields.

        Returns:
            List of papers with detected signals.
        """
        opportunities = []

        for paper in papers:
            # Build text to search
            title = paper.get("title", "")
            summary = paper.get("summary", "") or paper.get("abstract", "")
            text = f"{title} {summary}".lower()

            # Find matching signals
            detected_signals = []
            for signal in self.annotation_signals_lower:
                if signal in text:
                    # Get the original case version
                    idx = self.annotation_signals_lower.index(signal)
                    detected_signals.append(self.annotation_signals[idx])

            if detected_signals:
                # Detect organization
                org = self._detect_org_from_paper(paper)

                opportunities.append({
                    "paper": paper,
                    "title": title,
                    "signals": detected_signals,
                    "signal_count": len(detected_signals),
                    "detected_org": org,
                    "url": paper.get("url", paper.get("arxiv_url", "")),
                    "arxiv_id": paper.get("arxiv_id", paper.get("id", "")),
                })

        # Sort by signal count descending
        opportunities.sort(key=lambda x: x["signal_count"], reverse=True)
        return opportunities

    def _detect_org_from_paper(self, paper: dict) -> Optional[str]:
        """Detect organization from paper metadata.

        Args:
            paper: Paper dictionary.

        Returns:
            Organization name if detected.
        """
        # Check authors
        authors = paper.get("authors", [])
        if isinstance(authors, list):
            authors_text = " ".join(str(a) for a in authors).lower()
        else:
            authors_text = str(authors).lower()

        # Check title and summary
        title = paper.get("title", "").lower()
        summary = (paper.get("summary", "") or paper.get("abstract", "")).lower()

        search_text = f"{authors_text} {title} {summary}"

        for org_name, aliases in self.tracked_orgs.items():
            for alias in aliases:
                if alias.lower() in search_text:
                    return org_name

        return None

    def track_organization_activity(
        self,
        datasets: list[dict],
        papers: list[dict],
    ) -> dict:
        """Track activity of monitored organizations.

        Args:
            datasets: List of datasets to analyze.
            papers: List of papers to analyze.

        Returns:
            Dictionary with organization activity summary.
        """
        org_activity = {}

        for org_name in self.tracked_orgs.keys():
            org_activity[org_name] = {
                "datasets": [],
                "papers": [],
                "total_items": 0,
            }

        # Process datasets
        for ds in datasets:
            org = self._detect_org_from_author(ds.get("author", ""))
            if org and org in org_activity:
                org_activity[org]["datasets"].append(ds)

        # Process papers
        for paper in papers:
            org = self._detect_org_from_paper(paper)
            if org and org in org_activity:
                org_activity[org]["papers"].append(paper)

        # Calculate totals
        for org_name in org_activity:
            org_activity[org_name]["total_items"] = (
                len(org_activity[org_name]["datasets"]) +
                len(org_activity[org_name]["papers"])
            )

        # Filter out orgs with no activity and sort by total items
        active_orgs = {
            org: data for org, data in org_activity.items()
            if data["total_items"] > 0
        }

        return active_orgs

    def analyze(
        self,
        datasets: list[dict],
        papers: list[dict],
    ) -> dict:
        """Run full opportunity analysis.

        Args:
            datasets: List of datasets to analyze.
            papers: List of papers to analyze (arxiv + hf_papers).

        Returns:
            Comprehensive opportunity analysis results.
        """
        print("Detecting data factories...")
        data_factories = self.detect_data_factories(datasets)
        print(f"  Found {len(data_factories)} potential data factories")

        print("Extracting annotation signals from papers...")
        annotation_opportunities = self.extract_annotation_signals(papers)
        print(f"  Found {len(annotation_opportunities)} papers with annotation signals")

        print("Tracking organization activity...")
        org_activity = self.track_organization_activity(datasets, papers)
        active_orgs = [org for org, data in org_activity.items() if data["total_items"] > 0]
        print(f"  Found activity from {len(active_orgs)} tracked organizations")

        return {
            "data_factories": data_factories,
            "annotation_opportunities": annotation_opportunities,
            "org_activity": org_activity,
            "summary": {
                "data_factory_count": len(data_factories),
                "annotation_opportunity_count": len(annotation_opportunities),
                "active_org_count": len(active_orgs),
            },
        }

    def generate_report(self, results: dict) -> str:
        """Generate a text report from opportunity analysis.

        Args:
            results: Results from analyze() method.

        Returns:
            Formatted text report.
        """
        lines = []
        lines.append("=" * 60)
        lines.append("  Business Opportunity Analysis")
        lines.append("=" * 60)
        lines.append("")

        summary = results.get("summary", {})
        lines.append(f"Data factories detected: {summary.get('data_factory_count', 0)}")
        lines.append(f"Papers with annotation signals: {summary.get('annotation_opportunity_count', 0)}")
        lines.append(f"Active tracked organizations: {summary.get('active_org_count', 0)}")
        lines.append("")

        # Data factories
        factories = results.get("data_factories", [])
        if factories:
            lines.append("-" * 60)
            lines.append("  Data Factories (High-frequency Dataset Publishers)")
            lines.append("-" * 60)

            for factory in factories[:5]:
                org_str = f" ({factory['possible_org']})" if factory.get("possible_org") else ""
                lines.append(f"\n  {factory['author']}{org_str}")
                lines.append(f"    Published {factory['dataset_count']} datasets in {factory['period_days']} days:")
                for ds in factory["datasets"][:3]:
                    lines.append(f"      - {ds.get('name', ds.get('id', 'Unknown'))}")
                if len(factory["datasets"]) > 3:
                    lines.append(f"      ... and {len(factory['datasets']) - 3} more")

        # Annotation opportunities
        opportunities = results.get("annotation_opportunities", [])
        if opportunities:
            lines.append("")
            lines.append("-" * 60)
            lines.append("  Papers with Annotation/Data Collection Signals")
            lines.append("-" * 60)

            for opp in opportunities[:10]:
                title = opp["title"][:60] + "..." if len(opp["title"]) > 60 else opp["title"]
                org_str = f" [{opp['detected_org']}]" if opp.get("detected_org") else ""
                lines.append(f"\n  {title}{org_str}")
                lines.append(f"    Signals: {', '.join(opp['signals'][:3])}")
                if opp.get("arxiv_id"):
                    lines.append(f"    arXiv: {opp['arxiv_id']}")

        # Organization activity
        org_activity = results.get("org_activity", {})
        active_orgs = [(org, data) for org, data in org_activity.items() if data["total_items"] > 0]
        if active_orgs:
            lines.append("")
            lines.append("-" * 60)
            lines.append("  Organization Activity")
            lines.append("-" * 60)

            # Sort by total items
            active_orgs.sort(key=lambda x: x[1]["total_items"], reverse=True)

            for org, data in active_orgs[:5]:
                lines.append(f"\n  {org.upper()}")
                if data["datasets"]:
                    lines.append(f"    Datasets: {len(data['datasets'])}")
                    for ds in data["datasets"][:2]:
                        lines.append(f"      - {ds.get('name', ds.get('id', 'Unknown'))}")
                if data["papers"]:
                    lines.append(f"    Papers: {len(data['papers'])}")
                    for p in data["papers"][:2]:
                        title = p.get("title", "Unknown")[:50]
                        lines.append(f"      - {title}...")

        lines.append("")
        lines.append("=" * 60)

        return "\n".join(lines)
