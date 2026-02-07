"""ModelScope (魔搭社区) datasets scraper.

ModelScope is China's leading model and dataset hub, operated by Alibaba DAMO Academy.
It is the Chinese equivalent of Hugging Face and a critical source for tracking
Chinese AI labs' dataset releases.

API docs: https://modelscope.cn/docs/ModelScope%20Hub%E4%BD%BF%E7%94%A8%E6%96%87%E6%A1%A3
"""

import requests
from datetime import datetime
from typing import Optional

from .base import BaseScraper
from .registry import register_scraper

from utils.logging_config import get_logger

logger = get_logger(__name__)


@register_scraper("modelscope")
class ModelScopeScraper(BaseScraper):
    """Scraper for ModelScope Hub datasets.

    Monitors dataset releases on modelscope.cn, particularly from
    Chinese AI labs (Qwen, DeepSeek, Zhipu, BAAI, etc.).
    """

    name = "modelscope"
    source_type = "dataset_registry"

    DATASETS_API = "https://modelscope.cn/api/v1/datasets"
    MODELS_API = "https://modelscope.cn/api/v1/models"

    def __init__(self, config: dict = None, limit: int = 50):
        super().__init__(config)
        self.limit = limit
        self.headers = {
            "User-Agent": "AI-Dataset-Radar/1.0",
            "Accept": "application/json",
        }

    def scrape(self, config: dict = None) -> list[dict]:
        """Scrape datasets from ModelScope Hub.

        Args:
            config: Optional runtime configuration with 'watch_orgs' list.

        Returns:
            List of dataset dictionaries.
        """
        cfg = config or self.config or {}
        watch_orgs = cfg.get("watch_orgs", [])

        if watch_orgs:
            return self._fetch_org_datasets(watch_orgs)
        return self._fetch_recent()

    def _fetch_recent(self) -> list[dict]:
        """Fetch recently created datasets.

        Returns:
            List of dataset info dictionaries.
        """
        params = {
            "PageSize": self.limit,
            "PageNumber": 1,
            "SortBy": "GmtCreate",
        }

        try:
            response = requests.get(
                self.DATASETS_API,
                params=params,
                headers=self.headers,
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as e:
            logger.info("Error fetching ModelScope datasets: %s", e)
            return []

        items = data.get("Data", {}).get("Datasets", [])
        if not items:
            # Try alternative response structure
            items = data.get("data", {}).get("datasets", [])

        results = []
        for ds in items:
            parsed = self._parse_dataset(ds)
            if parsed:
                results.append(parsed)

        return results

    def _fetch_org_datasets(self, orgs: list[str]) -> list[dict]:
        """Fetch datasets from specific organizations.

        Args:
            orgs: List of organization/user names on ModelScope.

        Returns:
            Combined list of dataset info dictionaries.
        """
        import time

        results = []
        for org in orgs:
            params = {
                "PageSize": 20,
                "PageNumber": 1,
                "SortBy": "GmtModified",
                "Owner": org,
            }

            try:
                response = requests.get(
                    self.DATASETS_API,
                    params=params,
                    headers=self.headers,
                    timeout=30,
                )
                response.raise_for_status()
                data = response.json()
            except requests.RequestException as e:
                logger.info("Error fetching ModelScope datasets for %s: %s", org, e)
                continue

            items = data.get("Data", {}).get("Datasets", [])
            if not items:
                items = data.get("data", {}).get("datasets", [])

            for ds in items:
                parsed = self._parse_dataset(ds, org=org)
                if parsed:
                    results.append(parsed)

            # Rate limiting
            time.sleep(0.5)

        return self.deduplicate(results)

    def _parse_dataset(self, ds: dict, org: str = "") -> Optional[dict]:
        """Parse a dataset entry from the API response.

        Args:
            ds: Raw dataset dictionary from API.
            org: Organization name override.

        Returns:
            Parsed dataset info or None if parsing fails.
        """
        try:
            dataset_name = ds.get("Name", ds.get("name", ""))
            owner = ds.get("Owner", ds.get("owner", org))
            dataset_id = f"{owner}/{dataset_name}" if owner else dataset_name

            created_at = ds.get("GmtCreate", ds.get("gmt_create", ""))
            if created_at and isinstance(created_at, str):
                try:
                    created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                except ValueError:
                    created_at = None
            else:
                created_at = None

            last_modified = ds.get("GmtModified", ds.get("gmt_modified", ""))
            if last_modified and isinstance(last_modified, str):
                try:
                    last_modified = datetime.fromisoformat(last_modified.replace("Z", "+00:00"))
                except ValueError:
                    last_modified = None
            else:
                last_modified = None

            dataset_url = f"https://modelscope.cn/datasets/{dataset_id}"

            return {
                "source": "modelscope",
                "id": dataset_id,
                "name": dataset_name,
                "author": owner,
                "downloads": ds.get("Downloads", ds.get("downloads", 0)),
                "likes": ds.get("Likes", ds.get("likes", 0)),
                "tags": ds.get("Tags", ds.get("tags", [])) or [],
                "description": ds.get(
                    "ChineseDescription", ds.get("Description", ds.get("description", ""))
                ),
                "license": ds.get("License", ds.get("license", "")),
                "created_at": created_at.isoformat() if created_at else None,
                "last_modified": last_modified.isoformat() if last_modified else None,
                "url": dataset_url,
                "source_url": dataset_url,
            }
        except Exception as e:
            logger.info("Error parsing ModelScope dataset %s: %s", ds.get("Name", "unknown"), e)
            return None
