"""Data source scrapers for AI Dataset Radar."""

from .huggingface import HuggingFaceScraper
from .paperswithcode import PapersWithCodeScraper
from .arxiv import ArxivScraper
from .github import GitHubScraper
from .hf_papers import HFPapersScraper

__all__ = [
    "HuggingFaceScraper",
    "PapersWithCodeScraper",
    "ArxivScraper",
    "GitHubScraper",
    "HFPapersScraper",
]
