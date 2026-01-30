"""Organization tracker for monitoring specific orgs on HuggingFace.

Tracks AI Labs and data vendors by fetching their datasets and models
from HuggingFace API.
"""

import time
import requests
from datetime import datetime, timedelta
from typing import Optional


class OrgTracker:
    """Track specific organizations on HuggingFace.

    Monitors:
    - AI Labs (OpenAI, Anthropic, Meta, etc.)
    - Data vendors (Scale AI, Surge AI, etc.)
    """

    HF_API_URL = "https://huggingface.co/api"

    def __init__(self, config: dict):
        """Initialize the organization tracker.

        Args:
            config: Configuration dict with watched_orgs and watched_vendors.
        """
        self.config = config
        self.session = requests.Session()
        self.session.headers["User-Agent"] = "AI-Dataset-Radar/4.0"

        # Parse watched organizations
        self.watched_orgs = self._parse_orgs(config.get("watched_orgs", {}))
        self.watched_vendors = self._parse_vendors(config.get("watched_vendors", {}))

        # Rate limiting
        self._last_request = 0
        self._request_delay = 0.5  # seconds

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

    def _rate_limit(self):
        """Apply rate limiting between requests."""
        elapsed = time.time() - self._last_request
        if elapsed < self._request_delay:
            time.sleep(self._request_delay - elapsed)
        self._last_request = time.time()

    def _fetch_org_datasets(self, org_id: str, limit: int = 100) -> list[dict]:
        """Fetch datasets from a specific organization.

        Args:
            org_id: HuggingFace organization ID.
            limit: Maximum datasets to fetch.

        Returns:
            List of dataset dictionaries.
        """
        self._rate_limit()

        url = f"{self.HF_API_URL}/datasets"
        params = {
            "author": org_id,
            "limit": limit,
            "sort": "lastModified",
            "direction": -1,
        }

        try:
            response = self.session.get(url, params=params, timeout=30)
            if response.status_code == 200:
                return response.json()
            else:
                return []
        except requests.RequestException:
            return []

    def _fetch_org_models(self, org_id: str, limit: int = 50) -> list[dict]:
        """Fetch models from a specific organization.

        Args:
            org_id: HuggingFace organization ID.
            limit: Maximum models to fetch.

        Returns:
            List of model dictionaries.
        """
        self._rate_limit()

        url = f"{self.HF_API_URL}/models"
        params = {
            "author": org_id,
            "limit": limit,
            "sort": "lastModified",
            "direction": -1,
        }

        try:
            response = self.session.get(url, params=params, timeout=30)
            if response.status_code == 200:
                return response.json()
            else:
                return []
        except requests.RequestException:
            return []

    def fetch_lab_activity(self, days: int = 7) -> dict:
        """Fetch recent activity from all watched AI labs.

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
        print(f"  Tracking {total_orgs} AI labs...")

        for i, (org_name, org_info) in enumerate(self.watched_orgs.items()):
            category = org_info["category"]
            if category not in results:
                results[category] = {}

            org_data = {
                "datasets": [],
                "models": [],
                "priority": org_info["priority"],
            }

            # Fetch from all HF IDs for this org
            for hf_id in org_info["hf_ids"]:
                # Fetch datasets
                datasets = self._fetch_org_datasets(hf_id)
                for ds in datasets:
                    ds["_org"] = org_name
                    ds["_hf_id"] = hf_id
                    # Check if recent
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

                # Fetch models (to analyze training data)
                models = self._fetch_org_models(hf_id)
                for model in models:
                    model["_org"] = org_name
                    model["_hf_id"] = hf_id
                    modified = model.get("lastModified", "")
                    if modified:
                        try:
                            mod_date = datetime.fromisoformat(modified.replace("Z", "+00:00"))
                            if mod_date.replace(tzinfo=None) >= cutoff:
                                org_data["models"].append(model)
                        except (ValueError, TypeError):
                            org_data["models"].append(model)

            if org_data["datasets"] or org_data["models"]:
                results[category][org_name] = org_data

            if (i + 1) % 5 == 0:
                print(f"    Processed {i + 1}/{total_orgs} orgs...")

        return results

    def fetch_vendor_activity(self, days: int = 7) -> dict:
        """Fetch recent activity from watched data vendors.

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
        print(f"  Tracking {total_vendors} data vendors...")

        for i, (vendor_name, vendor_info) in enumerate(self.watched_vendors.items()):
            tier = vendor_info["tier"]
            if tier not in results:
                results[tier] = {}

            vendor_data = {
                "datasets": [],
                "blog_url": vendor_info.get("blog_url"),
            }

            # Fetch from HF
            for hf_id in vendor_info["hf_ids"]:
                datasets = self._fetch_org_datasets(hf_id)
                for ds in datasets:
                    ds["_vendor"] = vendor_name
                    ds["_hf_id"] = hf_id
                    modified = ds.get("lastModified", "")
                    if modified:
                        try:
                            mod_date = datetime.fromisoformat(modified.replace("Z", "+00:00"))
                            if mod_date.replace(tzinfo=None) >= cutoff:
                                vendor_data["datasets"].append(ds)
                        except (ValueError, TypeError):
                            vendor_data["datasets"].append(ds)

            if vendor_data["datasets"]:
                results[tier][vendor_name] = vendor_data

        return results

    def fetch_all(self, days: int = 7) -> dict:
        """Fetch all tracked activity.

        Args:
            days: Look back period in days.

        Returns:
            Combined results for labs and vendors.
        """
        print("Fetching AI lab activity...")
        labs = self.fetch_lab_activity(days)

        print("Fetching vendor activity...")
        vendors = self.fetch_vendor_activity(days)

        # Count items
        lab_datasets = sum(
            len(org_data["datasets"])
            for category in labs.values()
            for org_data in category.values()
        )
        lab_models = sum(
            len(org_data["models"])
            for category in labs.values()
            for org_data in category.values()
        )
        vendor_datasets = sum(
            len(vendor_data["datasets"])
            for tier in vendors.values()
            for vendor_data in tier.values()
        )

        print(f"  Labs: {lab_datasets} datasets, {lab_models} models")
        print(f"  Vendors: {vendor_datasets} datasets")

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
