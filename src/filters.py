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
