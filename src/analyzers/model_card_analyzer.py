"""Model Card Analyzer - Extract valuable datasets from popular models.

Analyzes HuggingFace model cards to find which datasets are used
by high-download models, enabling reverse discovery of valuable datasets.
"""

import re
import time
from collections import defaultdict
from datetime import datetime
from typing import Optional
import requests


class ModelCardAnalyzer:
    """Analyze model cards to discover valuable training datasets."""

    HF_API_URL = "https://huggingface.co/api"

    def __init__(
        self,
        min_model_downloads: int = 1000,
        model_limit: int = 500,
        min_dataset_usage: int = 3,
    ):
        """Initialize the Model Card Analyzer.

        Args:
            min_model_downloads: Minimum downloads for model to be analyzed.
            model_limit: Maximum number of models to analyze.
            min_dataset_usage: Minimum times a dataset must be used to be included.
        """
        self.min_model_downloads = min_model_downloads
        self.model_limit = model_limit
        self.min_dataset_usage = min_dataset_usage
        self.session = requests.Session()
        self.session.headers["User-Agent"] = "AI-Dataset-Radar/3.0"

    def fetch_top_models(self) -> list[dict]:
        """Fetch top models by download count from HuggingFace.

        Returns:
            List of model metadata dictionaries.
        """
        url = f"{self.HF_API_URL}/models"
        params = {
            "sort": "downloads",
            "direction": -1,
            "limit": self.model_limit,
        }

        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            models = response.json()
        except requests.RequestException as e:
            print(f"Error fetching models: {e}")
            return []

        # Filter by minimum downloads
        filtered = []
        for model in models:
            downloads = model.get("downloads", 0)
            if downloads >= self.min_model_downloads:
                filtered.append(
                    {
                        "id": model.get("id", ""),
                        "author": model.get("author", ""),
                        "downloads": downloads,
                        "likes": model.get("likes", 0),
                        "pipeline_tag": model.get("pipeline_tag", ""),
                        "tags": model.get("tags", []),
                        "created_at": model.get("createdAt", ""),
                    }
                )

        return filtered

    def fetch_model_card(self, model_id: str) -> Optional[str]:
        """Fetch the README/model card content for a model.

        Args:
            model_id: HuggingFace model ID (e.g., "meta-llama/Llama-2-7b").

        Returns:
            Model card content as string, or None if not found.
        """
        # Try to fetch README.md
        url = f"https://huggingface.co/{model_id}/raw/main/README.md"

        try:
            response = self.session.get(url, timeout=15)
            if response.status_code == 200:
                return response.text
        except requests.RequestException:
            pass  # Network errors expected for some models, return None

        return None

    def extract_datasets_from_card(self, model_id: str, card_content: str) -> list[dict]:
        """Extract dataset references from model card content.

        Args:
            model_id: Model identifier.
            card_content: Raw model card text.

        Returns:
            List of extracted dataset references.
        """
        datasets = []
        content_lower = card_content.lower()

        # Pattern 1: YAML frontmatter datasets field
        yaml_match = re.search(r"^---\n(.*?)\n---", card_content, re.DOTALL)
        if yaml_match:
            yaml_content = yaml_match.group(1)
            # Look for datasets: field
            datasets_match = re.search(r"datasets:\s*\n((?:\s*-\s*.+\n)+)", yaml_content)
            if datasets_match:
                dataset_lines = datasets_match.group(1)
                for line in dataset_lines.split("\n"):
                    line = line.strip()
                    if line.startswith("-"):
                        ds_name = line[1:].strip()
                        if ds_name:
                            datasets.append(
                                {
                                    "name": ds_name,
                                    "source": "yaml_metadata",
                                    "model_id": model_id,
                                }
                            )

        # Pattern 2: "trained on" mentions
        trained_patterns = [
            r"trained\s+on\s+(?:the\s+)?([A-Za-z0-9\-_/]+(?:\s+dataset)?)",
            r"fine-?tuned\s+on\s+(?:the\s+)?([A-Za-z0-9\-_/]+(?:\s+dataset)?)",
            r"training\s+data(?:set)?[:\s]+([A-Za-z0-9\-_/]+)",
        ]

        for pattern in trained_patterns:
            matches = re.findall(pattern, content_lower)
            for match in matches:
                ds_name = match.strip()
                # Clean up common suffixes
                ds_name = re.sub(r"\s+dataset$", "", ds_name)
                if ds_name and len(ds_name) > 2:
                    datasets.append(
                        {
                            "name": ds_name,
                            "source": "text_mention",
                            "model_id": model_id,
                        }
                    )

        # Pattern 3: HuggingFace dataset links
        hf_dataset_pattern = r"huggingface\.co/datasets/([A-Za-z0-9\-_/]+)"
        matches = re.findall(hf_dataset_pattern, card_content)
        for match in matches:
            datasets.append(
                {
                    "name": match,
                    "source": "hf_link",
                    "model_id": model_id,
                }
            )

        # Pattern 4: Known dataset names
        known_datasets = [
            "openwebtext",
            "c4",
            "the pile",
            "redpajama",
            "dolma",
            "wikipedia",
            "bookcorpus",
            "common crawl",
            "laion",
            "squad",
            "glue",
            "superglue",
            "mmlu",
            "hellaswag",
            "alpaca",
            "sharegpt",
            "wizardlm",
            "evol-instruct",
            "coco",
            "imagenet",
            "vqa",
            "visual genome",
            "code_search_net",
            "the stack",
            "starcoderdata",
            "openassistant",
            "anthropic-hh",
            "ultrachat",
        ]

        for ds in known_datasets:
            if ds in content_lower:
                datasets.append(
                    {
                        "name": ds,
                        "source": "known_dataset",
                        "model_id": model_id,
                    }
                )

        # Deduplicate
        seen = set()
        unique = []
        for ds in datasets:
            key = ds["name"].lower().replace("-", "_").replace(" ", "_")
            if key not in seen:
                seen.add(key)
                unique.append(ds)

        return unique

    def analyze(self) -> dict:
        """Run full analysis of model cards.

        Returns:
            Analysis results with dataset usage statistics.
        """
        print("Fetching top models from HuggingFace...")
        models = self.fetch_top_models()
        print(f"  Found {len(models)} models with >={self.min_model_downloads} downloads")

        # Track dataset usage
        dataset_usage = defaultdict(
            lambda: {
                "count": 0,
                "models": [],
                "total_model_downloads": 0,
            }
        )

        analyzed = 0
        for model in models:
            model_id = model["id"]

            # Rate limiting
            if analyzed > 0 and analyzed % 50 == 0:
                print(f"  Analyzed {analyzed}/{len(models)} models...")
                time.sleep(1)

            card_content = self.fetch_model_card(model_id)
            if not card_content:
                continue

            datasets = self.extract_datasets_from_card(model_id, card_content)

            for ds in datasets:
                ds_name = ds["name"].lower()
                dataset_usage[ds_name]["count"] += 1
                dataset_usage[ds_name]["models"].append(
                    {
                        "id": model_id,
                        "downloads": model["downloads"],
                        "pipeline_tag": model.get("pipeline_tag", ""),
                    }
                )
                dataset_usage[ds_name]["total_model_downloads"] += model["downloads"]

            analyzed += 1
            time.sleep(0.1)  # Rate limiting

        # Filter by minimum usage
        valuable_datasets = []
        for ds_name, usage in dataset_usage.items():
            if usage["count"] >= self.min_dataset_usage:
                valuable_datasets.append(
                    {
                        "name": ds_name,
                        "usage_count": usage["count"],
                        "total_model_downloads": usage["total_model_downloads"],
                        "models": usage["models"][:10],  # Top 10 models
                        "top_model": usage["models"][0]["id"] if usage["models"] else None,
                    }
                )

        # Sort by usage count
        valuable_datasets.sort(key=lambda x: x["usage_count"], reverse=True)

        return {
            "models_analyzed": analyzed,
            "datasets_found": len(dataset_usage),
            "valuable_datasets": valuable_datasets,
            "analysis_date": datetime.now().isoformat(),
        }

    def get_dataset_models(self, dataset_name: str) -> list[dict]:
        """Get all models that use a specific dataset.

        Args:
            dataset_name: Name of the dataset to search for.

        Returns:
            List of models using this dataset.
        """
        url = f"{self.HF_API_URL}/models"
        params = {
            "search": dataset_name,
            "sort": "downloads",
            "direction": -1,
            "limit": 100,
        }

        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            models = response.json()

            result = []
            for model in models:
                # Check if dataset is actually used (not just mentioned)
                tags = model.get("tags", [])
                model_id = model.get("id", "")

                # Basic check - could be enhanced with model card analysis
                if any(dataset_name.lower() in str(tag).lower() for tag in tags):
                    result.append(
                        {
                            "id": model_id,
                            "downloads": model.get("downloads", 0),
                            "likes": model.get("likes", 0),
                            "pipeline_tag": model.get("pipeline_tag", ""),
                        }
                    )

            return result
        except requests.RequestException as e:
            print(f"Error searching models: {e}")
            return []

    def generate_report(self, results: dict) -> str:
        """Generate a human-readable report from analysis results.

        Args:
            results: Analysis results dictionary.

        Returns:
            Formatted report string.
        """
        lines = [
            "=" * 60,
            "Model Card Analysis Report",
            "=" * 60,
            "",
            f"Models analyzed: {results['models_analyzed']}",
            f"Unique datasets found: {results['datasets_found']}",
            f"Datasets with >={self.min_dataset_usage} uses: {len(results['valuable_datasets'])}",
            "",
            "Top Datasets by Model Usage:",
            "-" * 40,
        ]

        for i, ds in enumerate(results["valuable_datasets"][:20], 1):
            lines.append(
                f"{i:2}. {ds['name']:<30} "
                f"({ds['usage_count']} models, "
                f"{ds['total_model_downloads']:,} total downloads)"
            )
            if ds.get("top_model"):
                lines.append(f"      Top model: {ds['top_model']}")

        return "\n".join(lines)
