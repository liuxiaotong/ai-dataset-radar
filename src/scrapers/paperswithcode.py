"""Papers with Code benchmarks scraper."""

import requests
from datetime import datetime
from typing import Optional


class PapersWithCodeScraper:
    """Scraper for Papers with Code benchmarks/datasets."""

    BASE_URL = "https://paperswithcode.com/api/v1"

    def __init__(self, limit: int = 50):
        self.limit = limit

    def fetch(self) -> list[dict]:
        """Fetch latest datasets from Papers with Code.

        Returns:
            List of dataset information dictionaries.
        """
        results = []

        # Fetch datasets
        datasets = self._fetch_datasets()
        results.extend(datasets)

        return results

    def _fetch_datasets(self) -> list[dict]:
        """Fetch datasets from Papers with Code API."""
        url = f"{self.BASE_URL}/datasets/"
        params = {
            "items_per_page": self.limit,
            "page": 1,
        }

        headers = {
            "Accept": "application/json",
            "User-Agent": "AI-Dataset-Radar/1.0",
        }

        try:
            response = requests.get(url, params=params, headers=headers, timeout=30)
            response.raise_for_status()

            # Check if response is JSON
            content_type = response.headers.get("Content-Type", "")
            if "application/json" not in content_type:
                print(f"Papers with Code API returned non-JSON response: {content_type}")
                return []

            data = response.json()
        except requests.RequestException as e:
            print(f"Error fetching Papers with Code datasets: {e}")
            return []
        except ValueError as e:
            print(f"Error parsing Papers with Code response: {e}")
            return []

        results = []
        for ds in data.get("results", []):
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
            # Papers with Code API may not always have date info
            introduced_date = ds.get("introduced_date")
            if introduced_date:
                try:
                    created_at = datetime.strptime(introduced_date, "%Y-%m-%d")
                    created_at = created_at.isoformat()
                except ValueError:
                    created_at = None
            else:
                created_at = None

            return {
                "source": "paperswithcode",
                "id": ds.get("id", ""),
                "name": ds.get("name", ""),
                "full_name": ds.get("full_name", ""),
                "description": ds.get("description", ""),
                "paper_count": ds.get("num_papers", 0),
                "homepage": ds.get("homepage", ""),
                "modalities": ds.get("modalities", []),
                "languages": ds.get("languages", []),
                "created_at": created_at,
                "url": ds.get("url", ""),
            }
        except Exception as e:
            print(f"Error parsing dataset {ds.get('name', 'unknown')}: {e}")
            return None
