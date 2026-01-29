"""Filtering logic for AI Dataset Radar."""

from datetime import datetime, timedelta, timezone
from typing import Optional
from dateutil import parser as date_parser


class DatasetFilter:
    """Filter datasets based on various criteria."""

    def __init__(
        self,
        min_downloads: int = 0,
        keywords: Optional[list[str]] = None,
        days: int = 7,
    ):
        """Initialize the filter.

        Args:
            min_downloads: Minimum download count (for Hugging Face datasets).
            keywords: List of keywords to filter by (matches name, description, tags).
            days: Only include items from the last N days.
        """
        self.min_downloads = min_downloads
        self.keywords = [k.lower() for k in (keywords or [])]
        self.days = days
        self.cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)

    def filter_all(self, items: list[dict]) -> list[dict]:
        """Apply all filters to a list of items.

        Args:
            items: List of dataset/paper dictionaries.

        Returns:
            Filtered list of items.
        """
        filtered = []
        for item in items:
            if self._passes_all_filters(item):
                filtered.append(item)
        return filtered

    def _passes_all_filters(self, item: dict) -> bool:
        """Check if an item passes all filter criteria.

        Args:
            item: Dataset or paper dictionary.

        Returns:
            True if item passes all filters.
        """
        if not self._passes_date_filter(item):
            return False
        if not self._passes_downloads_filter(item):
            return False
        if not self._passes_keyword_filter(item):
            return False
        return True

    def _passes_date_filter(self, item: dict) -> bool:
        """Check if item is within the date range.

        Args:
            item: Dataset or paper dictionary.

        Returns:
            True if item is recent enough or has no date.
        """
        created_at = item.get("created_at")
        if not created_at:
            # If no date, include it (can't filter)
            return True

        try:
            if isinstance(created_at, str):
                item_date = date_parser.parse(created_at)
            elif isinstance(created_at, datetime):
                item_date = created_at
            else:
                return True

            # Make sure item_date is timezone-aware
            if item_date.tzinfo is None:
                item_date = item_date.replace(tzinfo=timezone.utc)

            return item_date >= self.cutoff_date
        except (ValueError, TypeError):
            # If date parsing fails, include the item
            return True

    def _passes_downloads_filter(self, item: dict) -> bool:
        """Check if item meets minimum download threshold.

        Args:
            item: Dataset or paper dictionary.

        Returns:
            True if item meets download threshold or doesn't have downloads.
        """
        # Only apply to Hugging Face datasets
        if item.get("source") != "huggingface":
            return True

        downloads = item.get("downloads", 0)
        return downloads >= self.min_downloads

    def _passes_keyword_filter(self, item: dict) -> bool:
        """Check if item matches any of the keywords.

        Args:
            item: Dataset or paper dictionary.

        Returns:
            True if item matches keywords or no keywords specified.
        """
        if not self.keywords:
            return True

        # Build searchable text from various fields
        searchable_parts = []

        # Name fields
        for field in ["name", "title", "full_name", "id"]:
            if field in item and item[field]:
                searchable_parts.append(str(item[field]))

        # Description fields
        for field in ["description", "summary"]:
            if field in item and item[field]:
                searchable_parts.append(str(item[field]))

        # Tags and categories
        for field in ["tags", "categories", "modalities", "languages"]:
            if field in item and item[field]:
                if isinstance(item[field], list):
                    searchable_parts.extend([str(t) for t in item[field]])
                else:
                    searchable_parts.append(str(item[field]))

        searchable_text = " ".join(searchable_parts).lower()

        # Check if any keyword matches
        return any(keyword in searchable_text for keyword in self.keywords)


def filter_datasets(
    items: list[dict],
    min_downloads: int = 0,
    keywords: Optional[list[str]] = None,
    days: int = 7,
) -> list[dict]:
    """Convenience function to filter datasets.

    Args:
        items: List of dataset/paper dictionaries.
        min_downloads: Minimum download count.
        keywords: Keywords to filter by.
        days: Only include items from last N days.

    Returns:
        Filtered list of items.
    """
    filter_obj = DatasetFilter(
        min_downloads=min_downloads,
        keywords=keywords,
        days=days,
    )
    return filter_obj.filter_all(items)


class DomainFilter:
    """Filter and classify items by domain/focus area."""

    def __init__(self, focus_areas: dict):
        """Initialize the domain filter.

        Args:
            focus_areas: Dictionary of focus areas from config, each with:
                - enabled: bool
                - keywords: list of keywords
                - hf_tags: list of Hugging Face tags to match
        """
        self.focus_areas = focus_areas
        self._compiled_areas = self._compile_areas()

    def _compile_areas(self) -> dict:
        """Compile focus areas into lowercase keyword sets for faster matching."""
        compiled = {}
        for area_name, area_config in self.focus_areas.items():
            if not area_config.get("enabled", True):
                continue
            compiled[area_name] = {
                "keywords": [k.lower() for k in area_config.get("keywords", [])],
                "hf_tags": [t.lower() for t in area_config.get("hf_tags", [])],
            }
        return compiled

    def classify_item(self, item: dict) -> list[str]:
        """Classify an item into matching focus areas.

        Args:
            item: Dataset or paper dictionary.

        Returns:
            List of matching focus area names.
        """
        matches = []

        # Build searchable text
        searchable_parts = []
        for field in ["name", "title", "full_name", "id", "description", "summary"]:
            if field in item and item[field]:
                searchable_parts.append(str(item[field]))

        searchable_text = " ".join(searchable_parts).lower()

        # Get item tags
        item_tags = []
        for field in ["tags", "categories", "task_categories"]:
            if field in item and item[field]:
                if isinstance(item[field], list):
                    item_tags.extend([str(t).lower() for t in item[field]])
                else:
                    item_tags.append(str(item[field]).lower())
        item_tags_text = " ".join(item_tags)

        # Check each focus area
        for area_name, area_data in self._compiled_areas.items():
            # Check keywords in searchable text
            if any(kw in searchable_text for kw in area_data["keywords"]):
                matches.append(area_name)
                continue

            # Check HF tags
            for hf_tag in area_data["hf_tags"]:
                if hf_tag in item_tags_text:
                    matches.append(area_name)
                    break

        return matches

    def filter_by_domain(self, items: list[dict], domain: str) -> list[dict]:
        """Filter items to only those matching a specific domain.

        Args:
            items: List of items to filter.
            domain: Domain name to filter by.

        Returns:
            Filtered list of items.
        """
        if domain not in self._compiled_areas:
            return []

        return [item for item in items if domain in self.classify_item(item)]

    def classify_all(self, items: list[dict]) -> dict:
        """Classify all items and return grouped by domain.

        Args:
            items: List of items to classify.

        Returns:
            Dictionary with domain names as keys and lists of matching items as values.
        """
        result = {area: [] for area in self._compiled_areas}
        result["uncategorized"] = []

        for item in items:
            domains = self.classify_item(item)
            if domains:
                for domain in domains:
                    result[domain].append(item)
            else:
                result["uncategorized"].append(item)

        return result

    def enrich_items(self, items: list[dict]) -> list[dict]:
        """Add domain classifications to each item.

        Args:
            items: List of items to enrich.

        Returns:
            Items with added 'domains' field.
        """
        for item in items:
            item["domains"] = self.classify_item(item)
        return items


class OrganizationFilter:
    """Detect and filter items by organization."""

    def __init__(self, tracked_orgs: dict):
        """Initialize the organization filter.

        Args:
            tracked_orgs: Dictionary mapping org names to lists of aliases.
        """
        self.tracked_orgs = tracked_orgs
        self._compiled_orgs = {
            org: [alias.lower() for alias in aliases]
            for org, aliases in tracked_orgs.items()
        }

    def detect_org(self, item: dict) -> Optional[str]:
        """Detect if an item is associated with a tracked organization.

        Args:
            item: Dataset or paper dictionary.

        Returns:
            Organization name if detected, None otherwise.
        """
        # Build searchable text from author, name, description
        searchable_parts = []
        for field in ["author", "name", "title", "description", "summary"]:
            if field in item and item[field]:
                searchable_parts.append(str(item[field]))

        # Also check authors list
        if "authors" in item and item["authors"]:
            if isinstance(item["authors"], list):
                searchable_parts.extend([str(a) for a in item["authors"]])
            else:
                searchable_parts.append(str(item["authors"]))

        searchable_text = " ".join(searchable_parts).lower()

        # Check each organization
        for org_name, aliases in self._compiled_orgs.items():
            if any(alias in searchable_text for alias in aliases):
                return org_name

        return None

    def classify_all(self, items: list[dict]) -> dict:
        """Classify items by organization.

        Args:
            items: List of items to classify.

        Returns:
            Dictionary with org names as keys and lists of items as values.
        """
        result = {org: [] for org in self._compiled_orgs}
        result["other"] = []

        for item in items:
            org = self.detect_org(item)
            if org:
                result[org].append(item)
                item["detected_org"] = org
            else:
                result["other"].append(item)

        return result

    def enrich_items(self, items: list[dict]) -> list[dict]:
        """Add organization detection to each item.

        Args:
            items: List of items to enrich.

        Returns:
            Items with added 'detected_org' field.
        """
        for item in items:
            item["detected_org"] = self.detect_org(item)
        return items
