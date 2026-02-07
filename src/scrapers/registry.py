"""Scraper registry for plugin-based architecture."""

from typing import TYPE_CHECKING, Callable, Optional

if TYPE_CHECKING:
    from .base import BaseScraper


# Global registry of scrapers
_SCRAPER_REGISTRY: dict[str, type["BaseScraper"]] = {}


def register_scraper(name: str) -> Callable:
    """Decorator to register a scraper class.

    Usage:
        @register_scraper("huggingface")
        class HuggingFaceScraper(BaseScraper):
            ...

    Args:
        name: Unique name for the scraper.

    Returns:
        Decorator function.
    """

    def decorator(cls: type["BaseScraper"]) -> type["BaseScraper"]:
        if name in _SCRAPER_REGISTRY:
            raise ValueError(f"Scraper '{name}' is already registered")
        cls.name = name
        _SCRAPER_REGISTRY[name] = cls
        return cls

    return decorator


def get_scraper(name: str, config: dict = None) -> Optional["BaseScraper"]:
    """Get a scraper instance by name.

    Args:
        name: Name of the scraper.
        config: Optional configuration to pass to the scraper.

    Returns:
        Scraper instance or None if not found.
    """
    cls = _SCRAPER_REGISTRY.get(name)
    if cls is None:
        return None
    return cls(config=config)


def get_all_scrapers(config: dict = None) -> dict[str, "BaseScraper"]:
    """Get instances of all registered scrapers.

    Args:
        config: Optional configuration to pass to all scrapers.

    Returns:
        Dictionary mapping scraper names to instances.
    """
    return {name: cls(config=config) for name, cls in _SCRAPER_REGISTRY.items()}


def list_scrapers() -> list[str]:
    """List all registered scraper names.

    Returns:
        List of scraper names.
    """
    return list(_SCRAPER_REGISTRY.keys())


def get_scrapers_by_type(source_type: str, config: dict = None) -> dict[str, "BaseScraper"]:
    """Get scrapers filtered by source type.

    Args:
        source_type: Source type to filter by (e.g., "dataset_registry", "paper").
        config: Optional configuration to pass to scrapers.

    Returns:
        Dictionary mapping scraper names to instances.
    """
    return {
        name: cls(config=config)
        for name, cls in _SCRAPER_REGISTRY.items()
        if getattr(cls, "source_type", None) == source_type
    }


def clear_registry() -> None:
    """Clear the scraper registry. Primarily for testing."""
    _SCRAPER_REGISTRY.clear()
