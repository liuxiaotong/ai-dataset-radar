"""Kaggle datasets scraper.

Monitors Kaggle for new and trending AI/ML datasets using the
Kaggle REST API. Requires KAGGLE_USERNAME and KAGGLE_KEY environment
variables (from kaggle.json API token).

Gracefully skips if credentials are not configured.
"""

import base64
import os
from datetime import datetime, timedelta, timezone
from typing import Optional

from .base import BaseScraper
from .registry import register_scraper

from utils.async_http import AsyncHTTPClient
from utils.logging_config import get_logger

logger = get_logger(__name__)


@register_scraper("kaggle")
class KaggleScraper(BaseScraper):
    """Scraper for Kaggle datasets."""

    name = "kaggle"
    source_type = "dataset_registry"

    API_BASE = "https://www.kaggle.com/api/v1"

    def __init__(
        self,
        config: dict = None,
        limit: int = 50,
        http_client: AsyncHTTPClient = None,
    ):
        super().__init__(config)
        self.limit = limit
        self._http = http_client or AsyncHTTPClient()

        # Read credentials from env
        username = os.environ.get("KAGGLE_USERNAME", "")
        key = os.environ.get("KAGGLE_KEY", "")
        self._auth_header = None
        if username and key:
            token = base64.b64encode(f"{username}:{key}".encode()).decode()
            self._auth_header = f"Basic {token}"

        # Keywords from config
        kaggle_cfg = (config or {}).get("sources", {}).get("kaggle", {})
        self.keywords = kaggle_cfg.get("keywords", [
            "llm training", "instruction tuning", "rlhf",
            "preference", "text generation", "NLP benchmark",
        ])
        self.enabled = kaggle_cfg.get("enabled", True)

    async def scrape(self, config: dict = None) -> list[dict]:
        """Scrape datasets from Kaggle.

        Returns:
            List of dataset dictionaries.
        """
        return await self.fetch()

    async def fetch(self, days: int = 7) -> list[dict]:
        """Fetch datasets matching configured keywords.

        Args:
            days: Look back period in days (used for filtering).

        Returns:
            List of dataset info dicts.
        """
        if not self.enabled:
            return []

        if not self._auth_header:
            logger.warning("No KAGGLE_USERNAME/KAGGLE_KEY — skipping Kaggle")
            return []

        cutoff = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=days)
        headers = {
            "Authorization": self._auth_header,
            "User-Agent": "AI-Dataset-Radar/6.0",
        }

        seen_refs = set()
        results = []

        for keyword in self.keywords:
            url = f"{self.API_BASE}/datasets/list"
            params = {
                "search": keyword,
                "sortBy": "updated",
                "filetype": "all",
                "page": 1,
                "pageSize": min(self.limit, 20),
            }

            try:
                data = await self._http.get_json(
                    url, params=params, headers=headers, max_retries=2
                )
            except Exception as e:
                logger.warning("Kaggle search error for '%s': %s", keyword, e)
                continue

            if not data or not isinstance(data, list):
                continue

            for ds in data:
                ref = ds.get("ref", "")
                if not ref or ref in seen_refs:
                    continue
                seen_refs.add(ref)

                parsed = self._parse_dataset(ds)
                if parsed:
                    # Filter by date if available
                    last_updated = parsed.get("last_modified")
                    if last_updated:
                        try:
                            dt = datetime.fromisoformat(
                                last_updated.replace("Z", "+00:00")
                            ).replace(tzinfo=None)
                            if dt < cutoff:
                                continue
                        except (ValueError, AttributeError):
                            pass
                    results.append(parsed)

        logger.info("Kaggle: found %d datasets from %d keywords", len(results), len(self.keywords))
        return self.deduplicate(results)

    def _parse_dataset(self, ds: dict) -> Optional[dict]:
        """Parse a Kaggle dataset API response item.

        Args:
            ds: Raw dataset dict from Kaggle API.

        Returns:
            Standardized dataset dict, or None.
        """
        ref = ds.get("ref", "")
        if not ref:
            return None

        title = ds.get("title", ref.split("/")[-1] if "/" in ref else ref)
        owner = ref.split("/")[0] if "/" in ref else ""

        return {
            "source": "kaggle",
            "id": ref,
            "name": title,
            "author": owner,
            "description": ds.get("subtitle", "") or ds.get("description", ""),
            "url": f"https://www.kaggle.com/datasets/{ref}",
            "source_url": f"https://www.kaggle.com/datasets/{ref}",
            "downloads": ds.get("downloadCount", 0),
            "views": ds.get("viewCount", 0),
            "votes": ds.get("voteCount", 0),
            "usability_rating": ds.get("usabilityRating", 0),
            "license": ds.get("licenseName", ""),
            "tags": [t.get("name", "") for t in ds.get("tags", []) if isinstance(t, dict)],
            "size_bytes": ds.get("totalBytes", 0),
            "last_modified": ds.get("lastUpdated", ""),
            "created_at": ds.get("creatorUrl", ""),
        }
