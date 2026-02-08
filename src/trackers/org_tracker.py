"""Organization tracker for monitoring specific orgs on HuggingFace.

Tracks AI Labs and data vendors by fetching their datasets and models
from HuggingFace API.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Optional

from utils.logging_config import get_logger
from utils.cache import get_cache
from utils.async_http import AsyncHTTPClient, AsyncRateLimiter

logger = get_logger(__name__)


class OrgTracker:
    """Track specific organizations on HuggingFace.

    Monitors:
    - AI Labs (OpenAI, Anthropic, Meta, etc.)
    - Data vendors (Scale AI, Surge AI, etc.)
    """

    HF_API_URL = "https://huggingface.co/api"

    def __init__(self, config: dict, http_client: AsyncHTTPClient = None):
        """Initialize the organization tracker.

        Args:
            config: Configuration dict with watched_orgs and watched_vendors.
            http_client: Optional shared AsyncHTTPClient instance.
        """
        self.config = config
        self._http = http_client or AsyncHTTPClient(
            headers={"User-Agent": "AI-Dataset-Radar/4.0"},
        )

        # Parse watched organizations
        self.watched_orgs = self._parse_orgs(config.get("watched_orgs", {}))
        self.watched_vendors = self._parse_vendors(config.get("watched_vendors", {}))

        # Async rate limiting (cooperative, no lock needed)
        self._rate_limiter = AsyncRateLimiter(calls_per_second=10)
        self._cache = get_cache()

    def _parse_orgs(self, orgs_config: dict) -> dict:
        """Parse organization configuration into flat structure.

        Args:
            orgs_config: Nested org config from YAML.

        Returns:
            Flat dict: org_name -> {hf_ids, keywords, category, priority}
        """
        result = {}
        for category, orgs in orgs_config.items():
            if not isinstance(orgs, dict):
                continue
            for org_name, org_info in orgs.items():
                if not isinstance(org_info, dict):
                    continue
                result[org_name] = {
                    "hf_ids": org_info.get("hf_ids", []),
                    "keywords": org_info.get("keywords", []),
                    "category": category,
                    "priority": org_info.get("priority", "medium"),
                }
        return result

    def _parse_vendors(self, vendors_config: dict) -> dict:
        """Parse vendor configuration into flat structure."""
        result = {}
        for tier, vendors in vendors_config.items():
            if not isinstance(vendors, dict):
                continue
            for vendor_name, vendor_info in vendors.items():
                if not isinstance(vendor_info, dict):
                    continue
                result[vendor_name] = {
                    "hf_ids": vendor_info.get("hf_ids", []),
                    "github": vendor_info.get("github", []),
                    "keywords": vendor_info.get("keywords", []),
                    "tier": tier,
                    "blog_url": vendor_info.get("blog_url"),
                }
        return result

    async def _rate_limit(self):
        """Apply cooperative async rate limiting between requests."""
        await self._rate_limiter.acquire()

    async def _request_with_retry(
        self, url: str, params: dict, cache_key: str, description: str, max_retries: int = 3
    ) -> list[dict]:
        """Make an HF API request with retry and cache fallback.

        Args:
            url: API endpoint URL.
            params: Query parameters.
            cache_key: Cache key for storing/retrieving results.
            description: Description for logging.
            max_retries: Maximum retry attempts.

        Returns:
            List of result dictionaries.
        """
        # Synchronous cache check (FileCache is sync)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        await self._rate_limit()

        result = await self._http.get_json(url, params=params, max_retries=max_retries)
        if result is not None:
            # Synchronous cache set (FileCache is sync)
            self._cache.set(cache_key, result, ttl=3600)
            return result

        logger.warning("All retries failed for %s", description)
        return []

    async def _fetch_org_datasets(self, org_id: str, limit: int = 100) -> list[dict]:
        """Fetch datasets from a specific organization (with caching + retry).

        Args:
            org_id: HuggingFace organization ID.
            limit: Maximum datasets to fetch.

        Returns:
            List of dataset dictionaries.
        """
        return await self._request_with_retry(
            url=f"{self.HF_API_URL}/datasets",
            params={"author": org_id, "limit": limit, "sort": "lastModified", "direction": -1},
            cache_key=f"hf:datasets:{org_id}:{limit}",
            description=f"datasets/{org_id}",
        )

    async def _fetch_org_models(self, org_id: str, limit: int = 50) -> list[dict]:
        """Fetch models from a specific organization (with caching + retry).

        Args:
            org_id: HuggingFace organization ID.
            limit: Maximum models to fetch.

        Returns:
            List of model dictionaries.
        """
        return await self._request_with_retry(
            url=f"{self.HF_API_URL}/models",
            params={"author": org_id, "limit": limit, "sort": "lastModified", "direction": -1},
            cache_key=f"hf:models:{org_id}:{limit}",
            description=f"models/{org_id}",
        )

    async def _fetch_single_org(self, org_name: str, org_info: dict, cutoff: datetime) -> Optional[tuple]:
        """Fetch datasets and models for a single org.

        Args:
            org_name: Organization name.
            org_info: Organization config info.
            cutoff: Cutoff datetime for recency filtering.

        Returns:
            Tuple of (category, org_name, org_data) or None.
        """
        category = org_info["category"]
        org_data = {
            "datasets": [],
            "models": [],
            "priority": org_info["priority"],
        }

        for hf_id in org_info["hf_ids"]:
            datasets = await self._fetch_org_datasets(hf_id)
            for ds in datasets:
                ds["_org"] = org_name
                ds["_hf_id"] = hf_id
                ds["created_at"] = ds.get("createdAt", "")
                ds["last_modified"] = ds.get("lastModified", "")
                modified = ds.get("lastModified", "")
                if modified:
                    try:
                        mod_date = datetime.fromisoformat(modified.replace("Z", "+00:00"))
                        if mod_date.replace(tzinfo=None) >= cutoff:
                            org_data["datasets"].append(ds)
                    except (ValueError, TypeError):
                        org_data["datasets"].append(ds)
                else:
                    org_data["datasets"].append(ds)

            models = await self._fetch_org_models(hf_id)
            for model in models:
                model["_org"] = org_name
                model["_hf_id"] = hf_id
                model["created_at"] = model.get("createdAt", "")
                model["last_modified"] = model.get("lastModified", "")
                modified = model.get("lastModified", "")
                if modified:
                    try:
                        mod_date = datetime.fromisoformat(modified.replace("Z", "+00:00"))
                        if mod_date.replace(tzinfo=None) >= cutoff:
                            org_data["models"].append(model)
                    except (ValueError, TypeError):
                        org_data["models"].append(model)

        if org_data["datasets"] or org_data["models"]:
            return (category, org_name, org_data)
        return None

    async def fetch_lab_activity(self, days: int = 7) -> dict:
        """Fetch recent activity from all watched AI labs (concurrent with semaphore).

        Args:
            days: Look back period in days.

        Returns:
            Dict organized by category -> org -> {datasets, models}
        """
        cutoff = datetime.now() - timedelta(days=days)
        results = {
            "frontier_labs": {},
            "emerging_labs": {},
            "research_labs": {},
        }

        total_orgs = len(self.watched_orgs)
        logger.info("  Tracking %s AI labs...", total_orgs)

        sem = asyncio.Semaphore(10)

        async def _bounded(name, info):
            async with sem:
                return await self._fetch_single_org(name, info, cutoff)

        tasks = [_bounded(name, info) for name, info in self.watched_orgs.items()]
        results_raw = await asyncio.gather(*tasks, return_exceptions=True)

        done_count = 0
        for name, result in zip(self.watched_orgs.keys(), results_raw):
            done_count += 1
            if isinstance(result, Exception):
                logger.warning("Error fetching org %s: %s", name, result)
                continue
            if result:
                category, org_name, org_data = result
                if category not in results:
                    results[category] = {}
                results[category][org_name] = org_data
            if done_count % 10 == 0:
                logger.info("    Processed %s/%s orgs...", done_count, total_orgs)

        return results

    async def fetch_vendor_activity(self, days: int = 7) -> dict:
        """Fetch recent activity from watched data vendors (concurrent with semaphore).

        Args:
            days: Look back period in days.

        Returns:
            Dict organized by tier -> vendor -> {datasets, repos}
        """
        cutoff = datetime.now() - timedelta(days=days)
        results = {
            "premium": {},
            "specialized": {},
        }

        total_vendors = len(self.watched_vendors)
        logger.info("  Tracking %s data vendors...", total_vendors)

        sem = asyncio.Semaphore(10)

        async def _fetch_vendor(vendor_name, vendor_info):
            async with sem:
                vendor_data = {
                    "datasets": [],
                    "blog_url": vendor_info.get("blog_url"),
                }
                for hf_id in vendor_info["hf_ids"]:
                    datasets = await self._fetch_org_datasets(hf_id)
                    for ds in datasets:
                        ds["_vendor"] = vendor_name
                        ds["_hf_id"] = hf_id
                        ds["created_at"] = ds.get("createdAt", "")
                        ds["last_modified"] = ds.get("lastModified", "")
                        modified = ds.get("lastModified", "")
                        if modified:
                            try:
                                mod_date = datetime.fromisoformat(modified.replace("Z", "+00:00"))
                                if mod_date.replace(tzinfo=None) >= cutoff:
                                    vendor_data["datasets"].append(ds)
                            except (ValueError, TypeError):
                                vendor_data["datasets"].append(ds)
                return vendor_info["tier"], vendor_name, vendor_data

        tasks = [
            _fetch_vendor(name, info)
            for name, info in self.watched_vendors.items()
        ]
        results_raw = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results_raw:
            if isinstance(result, Exception):
                logger.warning("Error fetching vendor: %s", result)
                continue
            tier, vendor_name, vendor_data = result
            if vendor_data["datasets"]:
                if tier not in results:
                    results[tier] = {}
                results[tier][vendor_name] = vendor_data

        return results

    async def fetch_all(self, days: int = 7) -> dict:
        """Fetch all tracked activity.

        Args:
            days: Look back period in days.

        Returns:
            Combined results for labs and vendors.
        """
        logger.info("Fetching AI lab activity...")
        labs = await self.fetch_lab_activity(days)

        logger.info("Fetching vendor activity...")
        vendors = await self.fetch_vendor_activity(days)

        # Count items
        lab_datasets = sum(
            len(org_data["datasets"])
            for category in labs.values()
            for org_data in category.values()
        )
        lab_models = sum(
            len(org_data["models"]) for category in labs.values() for org_data in category.values()
        )
        vendor_datasets = sum(
            len(vendor_data["datasets"])
            for tier in vendors.values()
            for vendor_data in tier.values()
        )

        logger.info("  Labs: %s datasets, %s models", lab_datasets, lab_models)
        logger.info("  Vendors: %s datasets", vendor_datasets)

        return {
            "labs": labs,
            "vendors": vendors,
            "summary": {
                "lab_datasets": lab_datasets,
                "lab_models": lab_models,
                "vendor_datasets": vendor_datasets,
                "period_days": days,
            },
        }

    def get_all_hf_ids(self) -> list[str]:
        """Get all HuggingFace IDs being tracked.

        Returns:
            List of all org/vendor HF IDs.
        """
        ids = []
        for org_info in self.watched_orgs.values():
            ids.extend(org_info["hf_ids"])
        for vendor_info in self.watched_vendors.values():
            ids.extend(vendor_info["hf_ids"])
        return list(set(ids))
