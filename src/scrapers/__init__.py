"""Data source scrapers for AI Dataset Radar.

This module provides a plugin-based scraper architecture with:
- BaseScraper: Abstract base class for all scrapers
- Registry: Functions to register and retrieve scrapers
- Built-in scrapers for various data sources
"""

# Base class and registry (import first)
from .base import BaseScraper, SourceType
from .registry import (
    register_scraper,
    get_scraper,
    get_all_scrapers,
    list_scrapers,
    get_scrapers_by_type,
    clear_registry,
)

# Import all scrapers to trigger registration
from .huggingface import HuggingFaceScraper
from .paperswithcode import PapersWithCodeScraper
from .arxiv import ArxivScraper
from .github import GitHubScraper
from .github_org import GitHubOrgScraper
from .blog_rss import BlogRSSScraper
from .hf_papers import HFPapersScraper
from .semantic_scholar import SemanticScholarScraper
from .pwc_sota import PwCSOTAScraper

__all__ = [
    # Base and registry
    "BaseScraper",
    "SourceType",
    "register_scraper",
    "get_scraper",
    "get_all_scrapers",
    "list_scrapers",
    "get_scrapers_by_type",
    "clear_registry",
    # Scrapers
    "HuggingFaceScraper",
    "PapersWithCodeScraper",
    "ArxivScraper",
    "GitHubScraper",
    "GitHubOrgScraper",
    "BlogRSSScraper",
    "HFPapersScraper",
    "SemanticScholarScraper",
    "PwCSOTAScraper",
]
