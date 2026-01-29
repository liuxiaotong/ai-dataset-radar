"""Hugging Face datasets scraper."""

import requests
from datetime import datetime
from typing import Optional


class HuggingFaceScraper:
    """Scraper for Hugging Face Hub datasets."""

    BASE_URL = "https://huggingface.co/api/datasets"

    def __init__(self, limit: int = 50):
        self.limit = limit

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
            print(f"Error fetching Hugging Face datasets: {e}")
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

            return {
                "source": "huggingface",
                "id": ds.get("id", ""),
                "name": ds.get("id", "").split("/")[-1],
                "author": ds.get("author", ""),
                "downloads": ds.get("downloads", 0),
                "likes": ds.get("likes", 0),
                "tags": ds.get("tags", []),
                "description": ds.get("description", ""),
                "created_at": created_at.isoformat() if created_at else None,
                "url": f"https://huggingface.co/datasets/{ds.get('id', '')}",
            }
        except Exception as e:
            print(f"Error parsing dataset {ds.get('id', 'unknown')}: {e}")
            return None
