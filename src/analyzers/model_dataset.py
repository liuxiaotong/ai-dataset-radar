"""Model-Dataset relationship analyzer."""

import sys
from pathlib import Path
from typing import Optional
from collections import defaultdict

# Add parent directory to path for imports when running standalone
_src_dir = Path(__file__).parent.parent
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

from db import RadarDatabase
from scrapers.huggingface import HuggingFaceScraper


class ModelDatasetAnalyzer:
    """Analyze relationships between models and datasets.

    This analyzer:
    1. Fetches trending models from Hugging Face
    2. Extracts datasets used by each model from Model Cards
    3. Builds a graph of model-dataset relationships
    4. Identifies high-value datasets (used by many popular models)
    """

    def __init__(self, db: RadarDatabase, config: Optional[dict] = None):
        """Initialize the analyzer.

        Args:
            db: Database instance for storing results.
            config: Optional configuration dict with model settings.
        """
        self.db = db
        self.config = config or {}
        self.scraper = HuggingFaceScraper()

    def analyze(self, limit: int = 100, min_downloads: int = 1000) -> dict:
        """Run the full model-dataset analysis.

        Args:
            limit: Maximum number of models to analyze.
            min_downloads: Minimum downloads for a model to be included.

        Returns:
            Analysis results including high-value datasets and statistics.
        """
        models_config = self.config.get("models", {})
        limit = models_config.get("limit", limit)
        min_downloads = models_config.get("min_downloads", min_downloads)

        print(f"Fetching top {limit} models (min downloads: {min_downloads:,})...")
        models = self.scraper.fetch_trending_models(
            limit=limit,
            min_downloads=min_downloads,
        )
        print(f"  Found {len(models)} models")

        # Track dataset usage
        dataset_usage = defaultdict(list)  # dataset_id -> list of models
        processed_models = 0
        total_links = 0

        print("Analyzing model cards for dataset references...")
        for i, model in enumerate(models):
            model_id = model.get("id", "")
            if not model_id:
                continue

            # Store model in database
            model_db_id = self.db.upsert_model(
                model_id=model_id,
                name=model.get("name", ""),
                author=model.get("author", ""),
                downloads=model.get("downloads", 0),
                likes=model.get("likes", 0),
                pipeline_tag=model.get("pipeline_tag", ""),
                url=model.get("url", ""),
            )

            # Fetch detailed model card
            model_card = self.scraper.fetch_model_card(model_id)
            if not model_card:
                continue

            # Extract dataset references
            datasets = self.scraper.extract_datasets_from_model(model_card)

            for dataset_id in datasets:
                dataset_usage[dataset_id].append(
                    {
                        "model_id": model_id,
                        "model_db_id": model_db_id,
                        "downloads": model.get("downloads", 0),
                    }
                )

                # Try to get dataset info and store in database
                dataset_info = self.scraper.fetch_dataset_info(dataset_id)
                if dataset_info:
                    dataset_db_id = self.db.upsert_dataset(
                        source="huggingface",
                        dataset_id=dataset_info.get("id", dataset_id),
                        name=dataset_info.get("name", dataset_id),
                        author=dataset_info.get("author", ""),
                        url=dataset_info.get("url", ""),
                        created_at=dataset_info.get("created_at"),
                    )

                    # Record the link
                    self.db.add_model_dataset_link(
                        model_db_id=model_db_id,
                        dataset_db_id=dataset_db_id,
                        relationship="training",
                    )
                    total_links += 1

            processed_models += 1
            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{len(models)} models...")

        # Analyze results
        print(f"\nAnalysis complete: {processed_models} models, {total_links} links")

        # Rank datasets by usage
        ranked_datasets = self._rank_datasets(dataset_usage)

        return {
            "models_analyzed": processed_models,
            "total_links": total_links,
            "unique_datasets": len(dataset_usage),
            "top_datasets": ranked_datasets[:20],
            "dataset_usage": dict(dataset_usage),
        }

    def _rank_datasets(self, dataset_usage: dict) -> list[dict]:
        """Rank datasets by usage and model quality.

        Args:
            dataset_usage: Dict mapping dataset_id to list of models.

        Returns:
            Sorted list of datasets with usage statistics.
        """
        ranked = []

        for dataset_id, models in dataset_usage.items():
            total_downloads = sum(m.get("downloads", 0) for m in models)
            model_count = len(models)

            ranked.append(
                {
                    "dataset_id": dataset_id,
                    "model_count": model_count,
                    "total_model_downloads": total_downloads,
                    "avg_model_downloads": total_downloads / model_count if model_count > 0 else 0,
                    "models": [m.get("model_id") for m in models],
                }
            )

        # Sort by model count first, then by total downloads
        ranked.sort(
            key=lambda x: (x["model_count"], x["total_model_downloads"]),
            reverse=True,
        )

        return ranked

    def get_high_value_datasets(self, min_models: int = 3) -> list[dict]:
        """Get datasets used by multiple popular models.

        Args:
            min_models: Minimum number of models that must use the dataset.

        Returns:
            List of high-value datasets from database.
        """
        return self.db.get_most_used_datasets(limit=50)

    def get_dataset_models(self, dataset_id: str) -> list[dict]:
        """Get all models that use a specific dataset.

        Args:
            dataset_id: The dataset ID.

        Returns:
            List of models using this dataset.
        """
        dataset = self.db.get_dataset("huggingface", dataset_id)
        if not dataset:
            return []

        return self.db.get_models_using_dataset(dataset["id"])

    def generate_report(self, analysis_results: dict) -> str:
        """Generate a text report from analysis results.

        Args:
            analysis_results: Results from analyze() method.

        Returns:
            Formatted text report.
        """
        lines = []
        lines.append("=" * 60)
        lines.append("  Model-Dataset Relationship Analysis")
        lines.append("=" * 60)
        lines.append("")

        lines.append(f"Models analyzed: {analysis_results.get('models_analyzed', 0)}")
        lines.append(f"Total links found: {analysis_results.get('total_links', 0)}")
        lines.append(f"Unique datasets: {analysis_results.get('unique_datasets', 0)}")
        lines.append("")

        lines.append("-" * 60)
        lines.append("  Top Datasets by Model Usage")
        lines.append("-" * 60)

        for i, ds in enumerate(analysis_results.get("top_datasets", [])[:15], 1):
            lines.append(f"\n{i}. {ds['dataset_id']}")
            lines.append(f"   Used by {ds['model_count']} models")
            lines.append(f"   Total model downloads: {ds['total_model_downloads']:,}")
            if ds.get("models"):
                model_sample = ds["models"][:3]
                lines.append(f"   Sample models: {', '.join(model_sample)}")

        lines.append("")
        lines.append("=" * 60)

        return "\n".join(lines)
