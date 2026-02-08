"""Base scraper class for plugin-based architecture."""

from abc import ABC, abstractmethod
from typing import Literal, Optional


SourceType = Literal["dataset_registry", "code_host", "blog", "paper"]


class BaseScraper(ABC):
    """Abstract base class for all scrapers.

    Provides a common interface and shared functionality for data scrapers.
    All scrapers should inherit from this class and implement the scrape() method.
    """

    name: str = "base"
    source_type: SourceType = "dataset_registry"

    def __init__(self, config: dict = None):
        """Initialize the scraper.

        Args:
            config: Optional configuration dictionary.
        """
        self.config = config or {}

    @abstractmethod
    async def scrape(self, config: dict = None) -> list[dict]:
        """Fetch data from source.

        Args:
            config: Optional runtime configuration to override defaults.

        Returns:
            List of standardized dictionaries containing scraped data.
        """
        pass

    def deduplicate(self, items: list[dict]) -> list[dict]:
        """Remove duplicates by unique key.

        Args:
            items: List of item dictionaries.

        Returns:
            Deduplicated list of items.
        """
        seen = set()
        result = []
        for item in items:
            key = self._get_unique_key(item)
            if key not in seen:
                seen.add(key)
                result.append(item)
        return result

    def _get_unique_key(self, item: dict) -> str:
        """Get unique identifier for an item.

        Override this method in subclasses for custom deduplication logic.

        Args:
            item: Item dictionary.

        Returns:
            Unique string key for the item.
        """
        return item.get("id") or item.get("url") or str(item)

    def validate_item(self, item: dict) -> bool:
        """Validate an item has required fields.

        Args:
            item: Item dictionary to validate.

        Returns:
            True if item is valid, False otherwise.
        """
        required_fields = ["source", "id"]
        return all(item.get(field) for field in required_fields)

    def filter_items(
        self, items: list[dict], keywords: Optional[list[str]] = None, min_score: int = 0
    ) -> list[dict]:
        """Filter items by keywords or score.

        Args:
            items: List of items to filter.
            keywords: Optional keywords to match in name/description.
            min_score: Minimum score threshold (for items with scores).

        Returns:
            Filtered list of items.
        """
        result = items

        if keywords:
            filtered = []
            for item in result:
                text = " ".join(
                    [
                        str(item.get("name", "")),
                        str(item.get("description", "")),
                        " ".join(item.get("tags", [])),
                    ]
                ).lower()
                if any(kw.lower() in text for kw in keywords):
                    filtered.append(item)
            result = filtered

        if min_score > 0:
            result = [item for item in result if item.get("score", 0) >= min_score]

        return result
