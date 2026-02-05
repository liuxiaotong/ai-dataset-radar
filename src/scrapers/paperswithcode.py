"""Papers with Code benchmarks scraper.

API Documentation: https://paperswithcode.com/api/v1/docs/
"""

import time
import random
from datetime import datetime
from typing import Optional
import requests

from .base import BaseScraper
from .registry import register_scraper

from utils.logging_config import get_logger

logger = get_logger(__name__)


@register_scraper("paperswithcode")
class PapersWithCodeScraper(BaseScraper):
    """Scraper for Papers with Code benchmarks/datasets."""

    name = "paperswithcode"
    source_type = "dataset_registry"

    BASE_URL = "https://paperswithcode.com/api/v1"

    def __init__(self, config: dict = None, limit: int = 50):
        super().__init__(config)
        self.limit = limit
        self.session = requests.Session()
        self.session.headers.update({
            "Accept": "application/json",
            "User-Agent": "AI-Dataset-Radar/3.0 (https://github.com/liuxiaotong/ai-dataset-radar)",
        })
        self._last_request_time = 0
        self._base_delay = 1.0

    def scrape(self, config: dict = None) -> list[dict]:
        """Scrape datasets from Papers with Code.

        Args:
            config: Optional runtime configuration.

        Returns:
            List of dataset dictionaries.
        """
        return self.fetch()

    def _rate_limit_wait(self) -> None:
        """Wait to respect rate limits."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self._base_delay:
            time.sleep(self._base_delay - elapsed + random.uniform(0.1, 0.3))
        self._last_request_time = time.time()

    def _request_with_retry(
        self,
        url: str,
        params: dict,
        max_retries: int = 3,
    ) -> Optional[dict]:
        """Make a request with retry logic.

        Args:
            url: Request URL.
            params: Query parameters.
            max_retries: Maximum retry attempts.

        Returns:
            JSON response data or None on failure.
        """
        for attempt in range(max_retries + 1):
            self._rate_limit_wait()

            try:
                response = self.session.get(url, params=params, timeout=30)

                # Check content type
                content_type = response.headers.get("Content-Type", "")
                if "application/json" not in content_type:
                    if attempt < max_retries:
                        logger.info("  Non-JSON response, retrying... (attempt %s)", attempt + 1)
                        time.sleep(2 ** attempt)
                        continue
                    logger.info("  Papers with Code API returned non-JSON: %s", content_type[:50])
                    return None

                if response.status_code == 200:
                    return response.json()

                elif response.status_code == 429:
                    wait_time = (2 ** attempt) * 3 + random.uniform(1, 2)
                    if attempt < max_retries:
                        logger.info("  Rate limited, waiting %.1fs...", wait_time)
                        time.sleep(wait_time)
                        continue
                    return None

                elif response.status_code >= 500:
                    if attempt < max_retries:
                        time.sleep(2 ** attempt)
                        continue

                response.raise_for_status()

            except requests.exceptions.Timeout:
                if attempt < max_retries:
                    time.sleep(2 ** attempt)
                    continue
                logger.info("  Request timeout")
                return None

            except requests.RequestException as e:
                if attempt < max_retries:
                    time.sleep(2 ** attempt)
                    continue
                logger.info("  Request error: %s", e)
                return None

            except ValueError as e:
                logger.info("  JSON parse error: %s", e)
                return None

        return None

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

        data = self._request_with_retry(url, params)
        if not data:
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
            logger.info("Error parsing dataset %s: %s", ds.get('name', 'unknown'), e)
            return None

    def search_datasets(self, query: str, limit: int = 20) -> list[dict]:
        """Search for datasets by name.

        Args:
            query: Search query.
            limit: Maximum results.

        Returns:
            List of matching datasets.
        """
        url = f"{self.BASE_URL}/datasets/"
        params = {
            "q": query,
            "items_per_page": limit,
        }

        data = self._request_with_retry(url, params)
        if not data:
            return []

        results = []
        for ds in data.get("results", []):
            result = self._parse_dataset(ds)
            if result:
                results.append(result)

        return results

    def get_dataset_papers(self, dataset_id: str, limit: int = 20) -> list[dict]:
        """Get papers using a specific dataset.

        Args:
            dataset_id: Dataset ID from Papers with Code.
            limit: Maximum papers to return.

        Returns:
            List of paper dictionaries.
        """
        url = f"{self.BASE_URL}/datasets/{dataset_id}/papers/"
        params = {"items_per_page": limit}

        data = self._request_with_retry(url, params)
        if not data:
            return []

        papers = []
        for item in data.get("results", []):
            papers.append({
                "id": item.get("id"),
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "arxiv_id": item.get("arxiv_id"),
                "abstract": item.get("abstract", ""),
            })

        return papers
