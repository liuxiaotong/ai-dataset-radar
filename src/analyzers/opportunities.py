"""Opportunity detection for business intelligence."""

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
from analyzers.author_filter import AuthorFilter
from analyzers.quality_scorer import QualityScorer
from analyzers.org_detector import OrgDetector


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

        # Initialize quality filters (new in v4)
        self.author_filter = AuthorFilter(self.config)
        self.quality_scorer = QualityScorer(self.config)
        self.org_detector = OrgDetector(self.config)

        # Quality filter settings
        quality_config = self.config.get("quality_filter", {})
        self.enable_quality_filter = quality_config.get("enabled", True)
        self.min_dataset_quality = quality_config.get("min_dataset_quality", 2)

    def detect_data_factories(self, datasets: list[dict]) -> dict:
        """Detect authors/organizations publishing datasets at high frequency.

        A "data factory" is an author who has published 3+ datasets in the last 7 days.
        Now includes quality filtering to remove spam accounts.

        Args:
            datasets: List of dataset dictionaries with 'author' and 'created_at' fields.

        Returns:
            Dict with 'org_factories', 'individual_factories', and 'filtered_stats'.
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
                        ds_date = datetime.fromisoformat(created_at.replace("Z", "+00:00")).replace(
                            tzinfo=None
                        )
                    else:
                        ds_date = created_at.replace(tzinfo=None)

                    if ds_date < cutoff:
                        continue
                except (ValueError, TypeError):
                    pass

            author_datasets[author].append(ds)

        # Filter to authors with minimum datasets
        qualified_authors = {
            author: ds_list
            for author, ds_list in author_datasets.items()
            if len(ds_list) >= self.factory_min_datasets
        }

        # Apply quality filtering if enabled
        filtered_out = []
        if self.enable_quality_filter:
            qualified_authors, filtered_out = self.author_filter.filter_authors(qualified_authors)

        # Separate into org-affiliated and individual publishers
        org_factories = []
        individual_factories = []

        for author, ds_list in qualified_authors.items():
            # Detect organization
            org_detection = self.org_detector.detect_from_dataset(ds_list[0] if ds_list else {})
            org = org_detection["organization"]

            # If not detected from dataset, try author name
            if not org:
                org = self.org_detector.detect_from_author(author)

            # Score datasets
            total_quality = 0
            for ds in ds_list:
                score_result = self.quality_scorer.score_dataset(ds)
                ds["_quality_score"] = score_result["total_score"]
                total_quality += score_result["total_score"]

            avg_quality = total_quality / len(ds_list) if ds_list else 0

            factory_info = {
                "author": author,
                "dataset_count": len(ds_list),
                "datasets": ds_list,
                "possible_org": org,
                "org_display": self.org_detector.get_org_display_name(org) if org else None,
                "period_days": self.factory_days,
                "avg_quality_score": round(avg_quality, 1),
                "quality_stars": self.quality_scorer.get_quality_stars(int(avg_quality)),
            }

            if org:
                org_factories.append(factory_info)
            else:
                individual_factories.append(factory_info)

        # Sort by dataset count descending
        org_factories.sort(key=lambda x: x["dataset_count"], reverse=True)
        individual_factories.sort(key=lambda x: (-x["avg_quality_score"], -x["dataset_count"]))

        return {
            "org_factories": org_factories,
            "individual_factories": individual_factories,
            "filtered_stats": {
                "filtered_count": len(filtered_out),
                "filtered_authors": filtered_out,
            },
        }

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

                opportunities.append(
                    {
                        "paper": paper,
                        "title": title,
                        "signals": detected_signals,
                        "signal_count": len(detected_signals),
                        "detected_org": org,
                        "url": paper.get("url", paper.get("arxiv_url", "")),
                        "arxiv_id": paper.get("arxiv_id", paper.get("id", "")),
                    }
                )

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
            org_activity[org_name]["total_items"] = len(org_activity[org_name]["datasets"]) + len(
                org_activity[org_name]["papers"]
            )

        # Filter out orgs with no activity and sort by total items
        active_orgs = {org: data for org, data in org_activity.items() if data["total_items"] > 0}

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
        factory_results = self.detect_data_factories(datasets)
        org_count = len(factory_results["org_factories"])
        ind_count = len(factory_results["individual_factories"])
        filtered_count = factory_results["filtered_stats"]["filtered_count"]
        print(
            f"  Found {org_count} org + {ind_count} individual factories (filtered {filtered_count} low-quality)"
        )

        print("Extracting annotation signals from papers...")
        annotation_opportunities = self.extract_annotation_signals(papers)
        print(f"  Found {len(annotation_opportunities)} papers with annotation signals")

        print("Tracking organization activity...")
        org_activity = self.track_organization_activity(datasets, papers)
        active_orgs = [org for org, data in org_activity.items() if data["total_items"] > 0]
        print(f"  Found activity from {len(active_orgs)} tracked organizations")

        return {
            "data_factories": factory_results,
            "annotation_opportunities": annotation_opportunities,
            "org_activity": org_activity,
            "summary": {
                "org_factory_count": org_count,
                "individual_factory_count": ind_count,
                "filtered_factory_count": filtered_count,
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
        org_count = summary.get("org_factory_count", 0)
        ind_count = summary.get("individual_factory_count", 0)
        filtered_count = summary.get("filtered_factory_count", 0)
        lines.append(f"Organization data factories: {org_count}")
        lines.append(f"Individual data factories: {ind_count}")
        lines.append(f"Low-quality accounts filtered: {filtered_count}")
        lines.append(
            f"Papers with annotation signals: {summary.get('annotation_opportunity_count', 0)}"
        )
        lines.append(f"Active tracked organizations: {summary.get('active_org_count', 0)}")
        lines.append("")

        # Data factories - new format
        factory_results = results.get("data_factories", {})

        # Organization factories
        org_factories = factory_results.get("org_factories", [])
        if org_factories:
            lines.append("-" * 60)
            lines.append("  Organization Data Factories")
            lines.append("-" * 60)

            for factory in org_factories[:5]:
                org_display = factory.get("org_display", factory.get("possible_org", "Unknown"))
                quality = factory.get("quality_stars", "")
                lines.append(f"\n  {org_display} ({factory['author']}) {quality}")
                lines.append(
                    f"    Published {factory['dataset_count']} datasets in {factory['period_days']} days"
                )
                lines.append(f"    Avg quality: {factory.get('avg_quality_score', 0)}/10")
                for ds in factory["datasets"][:3]:
                    ds_name = ds.get("name", ds.get("id", "Unknown"))
                    lines.append(f"      - {ds_name}")
                if len(factory["datasets"]) > 3:
                    lines.append(f"      ... and {len(factory['datasets']) - 3} more")
        else:
            lines.append("-" * 60)
            lines.append("  Organization Data Factories")
            lines.append("-" * 60)
            lines.append("\n  No organization factories this week")

        # Individual factories
        individual_factories = factory_results.get("individual_factories", [])
        if individual_factories:
            lines.append("")
            lines.append("-" * 60)
            lines.append("  Individual Data Factories (Worth Watching)")
            lines.append("-" * 60)

            for factory in individual_factories[:5]:
                quality = factory.get("quality_stars", "")
                lines.append(f"\n  {factory['author']} {quality}")
                ds_count = factory['dataset_count']
                quality_score = factory.get('avg_quality_score', 0)
                lines.append(f"    Published {ds_count} datasets | Quality: {quality_score}/10")
                for ds in factory["datasets"][:2]:
                    ds_name = ds.get("name", ds.get("id", "Unknown"))
                    lines.append(f"      - {ds_name}")
        else:
            lines.append("")
            lines.append("-" * 60)
            lines.append("  Individual Data Factories")
            lines.append("-" * 60)
            lines.append("\n  No quality individual factories this week")

        # Filtered accounts summary
        filtered_stats = factory_results.get("filtered_stats", {})
        if filtered_stats.get("filtered_count", 0) > 0:
            lines.append("")
            lines.append("-" * 60)
            lines.append("  Filtered Low-Quality Accounts")
            lines.append("-" * 60)
            lines.append(f"\n  Filtered {filtered_stats['filtered_count']} accounts")
            lines.append("  (Reasons: suspicious usernames, random dataset IDs, no metadata)")

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
