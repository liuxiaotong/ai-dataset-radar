"""PapersWithCode datasets scraper.

PapersWithCode is the leading platform for tracking ML papers with code,
datasets, and benchmark results. It provides a public REST API with no
authentication required.

API docs: https://paperswithcode.com/api/v1/docs/
"""

import requests
from datetime import datetime
from typing import Optional

from .base import BaseScraper
from .registry import register_scraper

from utils.logging_config import get_logger

logger = get_logger(__name__)


@register_scraper("paperswithcode")
class PapersWithCodeScraper(BaseScraper):
    """Scraper for PapersWithCode datasets.

    Monitors dataset releases on paperswithcode.com, including
    benchmark datasets used in ML research papers.
    """

    name = "paperswithcode"
    source_type = "dataset_registry"

    DATASETS_API = "https://paperswithcode.com/api/v1/datasets/"

    def __init__(self, config: dict = None, limit: int = 50):
        super().__init__(config)
        self.limit = limit
        self.headers = {
            "User-Agent": "AI-Dataset-Radar/1.0",
            "Accept": "application/json",
        }

    def scrape(self, config: dict = None) -> list[dict]:
        """Scrape datasets from PapersWithCode.

        Args:
            config: Optional runtime configuration with 'limit' and 'ordering'.

        Returns:
            List of dataset dictionaries.
        """
        cfg = config or self.config or {}
        limit = cfg.get("limit", self.limit)
        ordering = cfg.get("ordering", "-paper_count")

        return self._fetch_datasets(limit=limit, ordering=ordering)

    def _fetch_datasets(self, limit: int = 50, ordering: str = "-paper_count") -> list[dict]:
        """Fetch datasets from PapersWithCode API.

        Args:
            limit: Maximum number of datasets to fetch.
            ordering: Sort order (e.g. '-paper_count', '-created_date').

        Returns:
            List of normalized dataset dicts.
        """
        results = []
        page = 1
        page_size = min(limit, 50)

        while len(results) < limit:
            try:
                resp = requests.get(
                    self.DATASETS_API,
                    headers=self.headers,
                    params={
                        "page": page,
                        "page_size": page_size,
                        "ordering": ordering,
                    },
                    timeout=30,
                )
                resp.raise_for_status()
                data = resp.json()
            except requests.RequestException as e:
                logger.error(f"PapersWithCode API error (page {page}): {e}")
                break

            items = data.get("results", [])
            if not items:
                break

            for item in items:
                normalized = self._normalize(item)
                if normalized:
                    results.append(normalized)

            if not data.get("next"):
                break

            page += 1
            if len(results) >= limit:
                break

        logger.info(f"PapersWithCode: fetched {len(results)} datasets")
        return self.deduplicate(results[:limit])

    def _normalize(self, item: dict) -> Optional[dict]:
        """Normalize a PapersWithCode dataset item.

        Args:
            item: Raw API response item.

        Returns:
            Normalized dataset dict, or None if invalid.
        """
        dataset_id = item.get("id") or item.get("url", "").rstrip("/").split("/")[-1]
        if not dataset_id:
            return None

        name = item.get("name", "")
        url = item.get("url", "")
        if url and not url.startswith("http"):
            url = f"https://paperswithcode.com{url}"

        return {
            "id": f"pwc_{dataset_id}",
            "name": name,
            "source": "paperswithcode",
            "source_type": self.source_type,
            "url": url,
            "description": item.get("description") or "",
            "paper_count": item.get("paper_count", 0),
            "modalities": item.get("modalities", []),
            "languages": item.get("languages", []),
            "tasks": [t.get("task", "") for t in item.get("tasks", []) if t.get("task")],
            "homepage": item.get("homepage") or "",
            "license": item.get("license") or "",
            "created_at": item.get("created_date") or "",
            "scraped_at": datetime.utcnow().isoformat() + "Z",
        }

    def _get_unique_key(self, item: dict) -> str:
        return item.get("id", item.get("url", ""))
