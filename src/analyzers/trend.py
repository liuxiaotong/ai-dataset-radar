"""Trend analysis for dataset growth patterns."""

import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# Add parent directory to path for imports when running standalone
_src_dir = Path(__file__).parent.parent
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

from db import RadarDatabase


class TrendAnalyzer:
    """Analyze dataset download trends and identify rising datasets.

    This analyzer:
    1. Records daily statistics for datasets
    2. Calculates growth rates over different time periods
    3. Identifies datasets with unusual growth patterns
    """

    def __init__(self, db: RadarDatabase, config: Optional[dict] = None):
        """Initialize the trend analyzer.

        Args:
            db: Database instance for storing and querying data.
            config: Optional configuration dict with analysis settings.
        """
        self.db = db
        self.config = config or {}
        self.analysis_config = self.config.get("analysis", {})
        self._last_recorded_ids: list[int] = []

    def record_daily_stats(self, datasets: list[dict]) -> int:
        """Record daily statistics for a batch of datasets.

        Args:
            datasets: List of dataset dictionaries with downloads/likes info.

        Returns:
            Number of datasets recorded.
        """
        recorded_ids = self.db.bulk_upsert_datasets_with_stats(datasets)
        self._last_recorded_ids = recorded_ids
        return len(recorded_ids)

    @property
    def last_recorded_ids(self) -> list[int]:
        """IDs of datasets that were updated in the last record pass."""
        return list(self._last_recorded_ids)

    def calculate_trends(
        self,
        days: Optional[list[int]] = None,
        dataset_ids: Optional[list[int]] = None,
    ) -> dict:
        """Calculate growth trends for all tracked datasets.

        Args:
            days: List of periods to calculate (default: [7, 30]).
            dataset_ids: Optional subset of dataset DB IDs to recalculate.

        Returns:
            Summary of trend calculations.
        """
        if days is None:
            days = self.analysis_config.get("trend_days", [7, 30])

        target_ids = dataset_ids or self._last_recorded_ids
        if target_ids:
            datasets = self.db.get_datasets_by_ids(target_ids)
        else:
            datasets = self.db.get_all_datasets(source="huggingface")
        today = datetime.now().strftime("%Y-%m-%d")

        calculated = 0
        with_growth = 0

        for ds in datasets:
            db_id = ds["id"]

            growth_7d = None
            growth_30d = None

            if 7 in days:
                growth_7d = self.db.calculate_growth_rate(db_id, days=7)

            if 30 in days:
                growth_30d = self.db.calculate_growth_rate(db_id, days=30)

            # Only record if we have at least one growth value
            if growth_7d is not None or growth_30d is not None:
                self.db.record_trend(
                    dataset_db_id=db_id,
                    downloads_7d_growth=growth_7d,
                    downloads_30d_growth=growth_30d,
                    date=today,
                )
                calculated += 1

                if (growth_7d is not None and growth_7d > 0) or (
                    growth_30d is not None and growth_30d > 0
                ):
                    with_growth += 1

        return {
            "total_datasets": len(datasets),
            "trends_calculated": calculated,
            "datasets_with_growth": with_growth,
            "date": today,
        }

    def get_rising_datasets(
        self,
        min_growth: Optional[float] = None,
        days: int = 7,
        limit: int = 20,
    ) -> list[dict]:
        """Get datasets with growth above threshold.

        Args:
            min_growth: Minimum growth rate (0.5 = 50%). Uses config if None.
            days: Use 7-day or 30-day growth.
            limit: Maximum results.

        Returns:
            List of rising datasets with growth info.
        """
        if min_growth is None:
            min_growth = self.analysis_config.get("min_growth_alert", 0.5)

        return self.db.get_rising_datasets(
            min_growth=min_growth,
            days=days,
            limit=limit,
        )

    def get_breakthrough_datasets(
        self,
        threshold: int = 1000,
        days: int = 7,
        limit: int = 10,
    ) -> list[dict]:
        """Get datasets that broke through from near-zero to significant downloads.

        Args:
            threshold: Download count to consider as "breakthrough".
            days: Lookback period.
            limit: Maximum results.

        Returns:
            List of breakthrough datasets.
        """
        return self.db.get_breakthrough_datasets(
            threshold=threshold,
            days=days,
            limit=limit,
        )

    def get_top_growing_datasets(
        self,
        days: int = 7,
        limit: int = 10,
        min_downloads: int = 0,
    ) -> list[dict]:
        """Get datasets sorted by growth rate with download info.

        Args:
            days: Period for growth calculation.
            limit: Maximum results.
            min_downloads: Minimum current downloads to include.

        Returns:
            List of datasets sorted by growth rate.
        """
        return self.db.get_top_growing_datasets(
            days=days,
            limit=limit,
            min_downloads=min_downloads,
        )

    def get_dataset_trend(self, dataset_id: str, source: str = "huggingface") -> Optional[dict]:
        """Get trend data for a specific dataset.

        Args:
            dataset_id: The dataset identifier.
            source: Data source (default: huggingface).

        Returns:
            Trend data or None if not found.
        """
        ds = self.db.get_dataset(source, dataset_id)
        if not ds:
            return None

        db_id = ds["id"]
        history = self.db.get_stats_history(db_id, days=30)
        growth_7d = self.db.calculate_growth_rate(db_id, days=7)
        growth_30d = self.db.calculate_growth_rate(db_id, days=30)

        return {
            "dataset": ds,
            "history": history,
            "growth_7d": growth_7d,
            "growth_30d": growth_30d,
        }

    def analyze(self, datasets: list[dict]) -> dict:
        """Run full trend analysis pipeline.

        Args:
            datasets: List of datasets to analyze (should have latest stats).

        Returns:
            Analysis results including rising and breakthrough datasets.
        """
        # Step 1: Record daily stats
        print("Recording daily statistics...")
        recorded = self.record_daily_stats(datasets)
        print(f"  Recorded stats for {recorded} datasets")

        # Step 2: Calculate trends
        print("Calculating growth trends...")
        trend_summary = self.calculate_trends(dataset_ids=self.last_recorded_ids)
        print(f"  Calculated trends for {trend_summary['trends_calculated']} datasets")

        # Step 3: Identify rising datasets
        print("Identifying rising datasets...")
        rising_7d = self.get_rising_datasets(days=7)
        rising_30d = self.get_rising_datasets(days=30)
        print(f"  Found {len(rising_7d)} rising (7-day), {len(rising_30d)} rising (30-day)")

        # Step 4: Get top growing datasets (sorted by growth rate with downloads)
        print("Finding top growing datasets...")
        top_growing_7d = self.get_top_growing_datasets(days=7, limit=10)
        print(f"  Found {len(top_growing_7d)} top growing datasets")

        # Step 5: Identify breakthrough datasets
        print("Detecting breakthrough datasets...")
        breakthroughs = self.get_breakthrough_datasets(threshold=1000, days=7)
        print(f"  Found {len(breakthroughs)} breakthrough datasets")

        return {
            "recorded_count": recorded,
            "trend_summary": trend_summary,
            "rising_7d": rising_7d,
            "rising_30d": rising_30d,
            "top_growing_7d": top_growing_7d,
            "breakthroughs": breakthroughs,
        }

    def generate_report(self, analysis_results: dict) -> str:
        """Generate a text report from trend analysis results.

        Args:
            analysis_results: Results from analyze() method.

        Returns:
            Formatted text report.
        """
        lines = []
        lines.append("=" * 60)
        lines.append("  Dataset Trend Analysis")
        lines.append("=" * 60)
        lines.append("")

        summary = analysis_results.get("trend_summary", {})
        lines.append(f"Total datasets tracked: {summary.get('total_datasets', 0)}")
        lines.append(f"Trends calculated: {summary.get('trends_calculated', 0)}")
        lines.append(f"Datasets with positive growth: {summary.get('datasets_with_growth', 0)}")
        lines.append("")

        # Top growing datasets (7-day) - sorted by growth rate with downloads
        top_growing = analysis_results.get("top_growing_7d", [])
        if top_growing:
            lines.append("-" * 60)
            lines.append("  Top Growing Datasets (7-day)")
            lines.append("-" * 60)

            for i, ds in enumerate(top_growing[:10], 1):
                growth = ds.get("growth", 0)
                growth_pct = f"{growth * 100:.1f}%" if growth != float("inf") else "New"
                downloads = ds.get("current_downloads", 0)
                lines.append(f"\n  {i}. {ds.get('name', ds.get('dataset_id', 'Unknown'))}")
                lines.append(f"     Growth: {growth_pct} | Downloads: {downloads:,}")
                lines.append(f"     URL: {ds.get('url', 'N/A')}")

        # Breakthrough datasets
        breakthroughs = analysis_results.get("breakthroughs", [])
        if breakthroughs:
            lines.append("")
            lines.append("-" * 60)
            lines.append("  Breakthrough Datasets (0 -> 1000+ downloads)")
            lines.append("-" * 60)

            for ds in breakthroughs[:5]:
                old = ds.get("old_downloads", 0) or 0
                current = ds.get("current_downloads", 0)
                increase = ds.get("download_increase", current - old)
                lines.append(f"\n  {ds.get('name', ds.get('dataset_id', 'Unknown'))}")
                lines.append(f"     {old:,} -> {current:,} (+{increase:,})")
                lines.append(f"     URL: {ds.get('url', 'N/A')}")

        # Rising datasets (7-day)
        rising_7d = analysis_results.get("rising_7d", [])
        if rising_7d:
            lines.append("")
            lines.append("-" * 60)
            lines.append("  Rising Datasets (7-day growth >= 50%)")
            lines.append("-" * 60)

            for ds in rising_7d[:10]:
                growth = ds.get("growth", 0)
                growth_pct = f"{growth * 100:.1f}%" if growth != float("inf") else "New"
                downloads = ds.get("current_downloads", 0) or 0
                lines.append(f"\n  {ds.get('name', ds.get('dataset_id', 'Unknown'))}")
                lines.append(f"    Growth: {growth_pct} | Downloads: {downloads:,}")
                lines.append(f"    URL: {ds.get('url', 'N/A')}")

        # Rising datasets (30-day)
        rising_30d = analysis_results.get("rising_30d", [])
        if rising_30d:
            lines.append("")
            lines.append("-" * 60)
            lines.append("  Rising Datasets (30-day growth >= 50%)")
            lines.append("-" * 60)

            for ds in rising_30d[:10]:
                growth = ds.get("growth", 0)
                growth_pct = f"{growth * 100:.1f}%" if growth != float("inf") else "New"
                downloads = ds.get("current_downloads", 0) or 0
                lines.append(f"\n  {ds.get('name', ds.get('dataset_id', 'Unknown'))}")
                lines.append(f"    Growth: {growth_pct} | Downloads: {downloads:,}")
                lines.append(f"    URL: {ds.get('url', 'N/A')}")

        if not rising_7d and not rising_30d and not top_growing:
            lines.append("")
            lines.append("No rising datasets found yet.")
            lines.append("(Need multiple days of data to calculate trends)")

        lines.append("")
        lines.append("=" * 60)

        return "\n".join(lines)
