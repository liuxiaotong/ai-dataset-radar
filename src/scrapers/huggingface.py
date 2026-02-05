"""Hugging Face datasets and models scraper."""

import re
import requests
from datetime import datetime
from typing import Optional

from .base import BaseScraper
from .registry import register_scraper

from utils.logging_config import get_logger
from utils.cache import get_cache

logger = get_logger(__name__)

# Cache TTL: 24 hours for README content
README_CACHE_TTL = 86400


@register_scraper("huggingface")
class HuggingFaceScraper(BaseScraper):
    """Scraper for Hugging Face Hub datasets and models."""

    name = "huggingface"
    source_type = "dataset_registry"

    DATASETS_URL = "https://huggingface.co/api/datasets"
    MODELS_URL = "https://huggingface.co/api/models"
    BASE_URL = "https://huggingface.co/api/datasets"  # Keep for backward compatibility

    def __init__(self, config: dict = None, limit: int = 50):
        super().__init__(config)
        self.limit = limit
        self.headers = {
            "User-Agent": "AI-Dataset-Radar/1.0"
        }

    def scrape(self, config: dict = None) -> list[dict]:
        """Scrape datasets from Hugging Face Hub.

        Args:
            config: Optional runtime configuration.

        Returns:
            List of dataset dictionaries.
        """
        return self.fetch()

    def fetch(self) -> list[dict]:
        """Fetch latest datasets from Hugging Face Hub.

        Returns:
            List of dataset information dictionaries.
        """
        params = {
            "limit": self.limit,
            "sort": "createdAt",
            "direction": -1,  # Descending (newest first)
            "full": "true",
        }

        try:
            response = requests.get(self.BASE_URL, params=params, timeout=30)
            response.raise_for_status()
            datasets = response.json()
        except requests.RequestException as e:
            logger.info("Error fetching Hugging Face datasets: %s", e)
            return []

        results = []
        for ds in datasets:
            result = self._parse_dataset(ds)
            if result:
                results.append(result)

        return results

    def _parse_dataset(self, ds: dict) -> Optional[dict]:
        """Parse a dataset entry from the API response.

        Args:
            ds: Raw dataset dictionary from API.

        Returns:
            Parsed dataset info or None if parsing fails.
        """
        try:
            created_at = ds.get("createdAt", "")
            if created_at:
                created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            else:
                created_at = None

            last_modified = ds.get("lastModified", "")
            if last_modified:
                last_modified = datetime.fromisoformat(
                    last_modified.replace("Z", "+00:00")
                )
            else:
                last_modified = None

            dataset_id = ds.get("id", "")
            dataset_url = f"https://huggingface.co/datasets/{dataset_id}"

            return {
                "source": "huggingface",
                "id": dataset_id,
                "name": dataset_id.split("/")[-1],
                "author": ds.get("author", ""),
                "downloads": ds.get("downloads", 0),
                "likes": ds.get("likes", 0),
                "tags": ds.get("tags", []),
                "task_categories": ds.get("taskCategories", []),
                "languages": ds.get("languages", []),
                "license": ds.get("license", ""),
                "size_category": ds.get("sizeCategory", ""),
                "created_at": created_at.isoformat() if created_at else None,
                "last_modified": last_modified.isoformat() if last_modified else None,
                "card_data": ds.get("cardData", {}),
                "description": ds.get("description", ""),
                "url": dataset_url,
                "source_url": dataset_url,
            }
        except Exception as e:
            logger.info("Error parsing dataset %s: %s", ds.get('id', 'unknown'), e)
            return None

    def fetch_trending_models(
        self,
        limit: int = 100,
        min_downloads: int = 1000,
    ) -> list[dict]:
        """Fetch trending models from Hugging Face Hub.

        Args:
            limit: Maximum number of models to fetch.
            min_downloads: Minimum download count filter.

        Returns:
            List of model information dictionaries.
        """
        params = {
            "limit": limit,
            "sort": "downloads",
            "direction": -1,  # Descending
            "full": "true",
        }

        try:
            response = requests.get(self.MODELS_URL, params=params, timeout=30)
            response.raise_for_status()
            models = response.json()
        except requests.RequestException as e:
            logger.info("Error fetching Hugging Face models: %s", e)
            return []

        results = []
        for model in models:
            parsed = self._parse_model(model)
            if parsed and parsed.get("downloads", 0) >= min_downloads:
                results.append(parsed)

        return results

    def _parse_model(self, model: dict) -> Optional[dict]:
        """Parse a model entry from the API response.

        Args:
            model: Raw model dictionary from API.

        Returns:
            Parsed model info or None if parsing fails.
        """
        try:
            created_at = model.get("createdAt", "")
            if created_at:
                created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            else:
                created_at = None

            return {
                "source": "huggingface",
                "id": model.get("id", ""),
                "name": model.get("id", "").split("/")[-1],
                "author": model.get("author", ""),
                "downloads": model.get("downloads", 0),
                "likes": model.get("likes", 0),
                "pipeline_tag": model.get("pipeline_tag", ""),
                "tags": model.get("tags", []),
                "created_at": created_at.isoformat() if created_at else None,
                "url": f"https://huggingface.co/{model.get('id', '')}",
            }
        except Exception as e:
            logger.info("Error parsing model %s: %s", model.get('id', 'unknown'), e)
            return None

    def fetch_model_card(self, model_id: str) -> Optional[dict]:
        """Fetch detailed model card information.

        Args:
            model_id: The model ID (e.g., 'meta-llama/Llama-2-7b').

        Returns:
            Model card data including README content, or None if failed.
        """
        url = f"{self.MODELS_URL}/{model_id}"

        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            model_data = response.json()
        except requests.RequestException as e:
            logger.info("Error fetching model card for %s: %s", model_id, e)
            return None

        # Also try to fetch the README content
        readme_url = f"https://huggingface.co/{model_id}/raw/main/README.md"
        readme_content = ""
        try:
            readme_response = requests.get(readme_url, timeout=15)
            if readme_response.status_code == 200:
                readme_content = readme_response.text
        except requests.RequestException:
            pass  # README is optional

        parsed = self._parse_model(model_data)
        if parsed:
            parsed["readme"] = readme_content
            parsed["card_data"] = model_data.get("cardData", {})

        return parsed

    def extract_datasets_from_model(self, model_data: dict) -> list[str]:
        """Extract dataset references from a model's metadata and README.

        Args:
            model_data: Model data including card_data and readme.

        Returns:
            List of dataset IDs referenced by the model.
        """
        datasets = set()

        # 1. Extract from cardData.datasets field
        card_data = model_data.get("card_data", {})
        if isinstance(card_data, dict):
            card_datasets = card_data.get("datasets", [])
            if isinstance(card_datasets, list):
                for ds in card_datasets:
                    if isinstance(ds, str):
                        datasets.add(ds)

        # 2. Extract from tags (some models have dataset-tagged)
        tags = model_data.get("tags", [])
        for tag in tags:
            if isinstance(tag, str) and tag.startswith("dataset:"):
                dataset_id = tag.replace("dataset:", "")
                datasets.add(dataset_id)

        # 3. Extract from README content
        readme = model_data.get("readme", "")
        if readme:
            # Pattern: huggingface.co/datasets/XXX
            hf_pattern = r"huggingface\.co/datasets/([a-zA-Z0-9_-]+/[a-zA-Z0-9_-]+|[a-zA-Z0-9_-]+)"
            for match in re.finditer(hf_pattern, readme):
                datasets.add(match.group(1))

            # Pattern: datasets/XXX in markdown links
            link_pattern = r"\[.*?\]\(https?://huggingface\.co/datasets/([^)\s]+)\)"
            for match in re.finditer(link_pattern, readme):
                datasets.add(match.group(1))

            # Pattern: Trained on XXX dataset (common phrasing)
            trained_pattern = r"[Tt]rained\s+on\s+(?:the\s+)?([A-Za-z0-9_-]+(?:/[A-Za-z0-9_-]+)?)\s+dataset"
            for match in re.finditer(trained_pattern, readme):
                candidate = match.group(1)
                # Filter out common false positives
                if candidate.lower() not in ["a", "an", "the", "this", "our"]:
                    datasets.add(candidate)

        return list(datasets)

    def fetch_dataset_info(self, dataset_id: str) -> Optional[dict]:
        """Fetch information about a specific dataset.

        Args:
            dataset_id: The dataset ID (e.g., 'squad' or 'huggingface/squad').

        Returns:
            Dataset info or None if not found.
        """
        url = f"{self.DATASETS_URL}/{dataset_id}"

        try:
            response = requests.get(url, timeout=15)
            if response.status_code == 404:
                return None
            response.raise_for_status()
            return self._parse_dataset(response.json())
        except requests.RequestException as e:
            logger.info("Error fetching dataset %s: %s", dataset_id, e)
            return None

    def fetch_dataset_readme(self, dataset_id: str) -> Optional[str]:
        """Fetch the README content for a dataset.

        Uses file-based cache with 24-hour TTL to avoid repeated API calls.

        Args:
            dataset_id: The dataset ID (e.g., 'allenai/dolma').

        Returns:
            README content string or None if not found.
        """
        # Check cache first
        cache = get_cache()
        cache_key = f"hf:readme:{dataset_id}"
        cached = cache.get(cache_key)
        if cached is not None:
            logger.debug("Cache hit for README: %s", dataset_id)
            return cached

        # Try the API endpoint first
        url = f"{self.DATASETS_URL}/{dataset_id}"
        result = None

        try:
            response = requests.get(url, headers=self.headers, timeout=15)
            if response.status_code == 200:
                data = response.json()
                # cardData often contains the README-like content
                card_data = data.get("cardData", "")
                if card_data:
                    result = str(card_data)
                else:
                    # Try description
                    description = data.get("description", "")
                    if description:
                        result = description
        except requests.RequestException:
            pass

        # Fallback: try to fetch raw README.md
        if not result:
            readme_url = f"https://huggingface.co/datasets/{dataset_id}/raw/main/README.md"
            try:
                response = requests.get(readme_url, headers=self.headers, timeout=15)
                if response.status_code == 200:
                    result = response.text
            except requests.RequestException:
                pass

        # Cache the result (even if None, to avoid repeated requests)
        if result:
            cache.set(cache_key, result, README_CACHE_TTL)

        return result
