"""arXiv papers scraper for RLHF and data annotation related papers.

Enhanced scraper with tighter search queries focused on:
- RLHF / preference learning
- Data annotation methodology
- Training data quality
"""

import feedparser
import urllib.parse
from datetime import datetime
from typing import Optional

from .base import BaseScraper
from .registry import register_scraper

from utils.async_http import AsyncHTTPClient
from utils.logging_config import get_logger

logger = get_logger(__name__)


@register_scraper("arxiv")
class ArxivScraper(BaseScraper):
    """Scraper for arXiv papers related to RLHF and data annotation."""

    name = "arxiv"
    source_type = "paper"

    BASE_URL = "http://export.arxiv.org/api/query"

    # Search terms focused on RLHF and data annotation
    SEARCH_TERMS = [
        # RLHF related
        '"human feedback"',
        '"RLHF"',
        '"preference learning"',
        '"reward model"',
        # Data annotation
        '"data annotation"',
        '"annotation guideline"',
        '"crowdsourcing"',
        # Instruction tuning
        '"instruction tuning"',
        '"instruction following"',
        # Data quality
        '"data curation"',
        '"synthetic data"',
    ]

    def __init__(
        self, limit: int = 50, categories: Optional[list[str]] = None, config: dict = None,
        http_client: AsyncHTTPClient = None,
    ):
        """Initialize arXiv scraper.

        Args:
            limit: Maximum number of papers to fetch.
            categories: arXiv categories to search.
            config: Optional configuration dict.
            http_client: Optional shared AsyncHTTPClient instance.
        """
        super().__init__(config)
        self.limit = limit
        self.categories = categories or ["cs.CL", "cs.LG", "cs.AI"]
        self._http = http_client or AsyncHTTPClient()

    async def scrape(self, config: dict = None) -> list[dict]:
        """Scrape papers from arXiv.

        Args:
            config: Optional runtime configuration.

        Returns:
            List of paper dictionaries.
        """
        return await self.fetch()

    async def fetch(self) -> list[dict]:
        """Fetch latest RLHF/annotation related papers from arXiv.

        Returns:
            List of paper information dictionaries.
        """
        # Build search query
        cat_query = " OR ".join([f"cat:{cat}" for cat in self.categories])
        terms_query = " OR ".join([f"ti:{term} OR abs:{term}" for term in self.SEARCH_TERMS])
        query = f"({cat_query}) AND ({terms_query})"

        params = {
            "search_query": query,
            "start": 0,
            "max_results": self.limit,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }

        url = f"{self.BASE_URL}?{urllib.parse.urlencode(params)}"

        try:
            text = await self._http.get_text(url)
            if not text:
                return []
            feed = feedparser.parse(text)
        except Exception as e:
            logger.warning("Error fetching arXiv papers: %s", e)
            return []

        if feed.bozo and feed.bozo_exception:
            logger.info("Warning: Feed parsing issue: %s", feed.bozo_exception)

        results = []
        for entry in feed.entries:
            result = self._parse_entry(entry)
            if result:
                results.append(result)

        return results

    async def fetch_by_keywords(self, keywords: list[str]) -> list[dict]:
        """Fetch papers matching specific keywords.

        Args:
            keywords: List of search keywords.

        Returns:
            List of paper dicts.
        """
        cat_query = " OR ".join([f"cat:{cat}" for cat in self.categories])
        terms = " OR ".join([f'ti:"{kw}" OR abs:"{kw}"' for kw in keywords])
        query = f"({cat_query}) AND ({terms})"

        params = {
            "search_query": query,
            "start": 0,
            "max_results": self.limit,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }

        url = f"{self.BASE_URL}?{urllib.parse.urlencode(params)}"

        try:
            text = await self._http.get_text(url)
            if not text:
                return []
            feed = feedparser.parse(text)
        except Exception as e:
            logger.warning("Error fetching arXiv papers: %s", e)
            return []

        results = []
        for entry in feed.entries:
            result = self._parse_entry(entry)
            if result:
                results.append(result)

        return results

    def _parse_entry(self, entry: dict) -> Optional[dict]:
        """Parse an entry from the arXiv feed."""
        try:
            arxiv_id = entry.id.split("/abs/")[-1]

            published = entry.get("published", "")
            if published:
                try:
                    pub_date = datetime.strptime(published, "%Y-%m-%dT%H:%M:%SZ")
                    published = pub_date.isoformat()
                except ValueError:
                    pass

            categories = [tag.term for tag in entry.get("tags", [])]
            authors = [author.name for author in entry.get("authors", [])]

            pdf_link = None
            for link in entry.get("links", []):
                if link.get("type") == "application/pdf":
                    pdf_link = link.get("href")
                    break

            return {
                "source": "arxiv",
                "id": arxiv_id,
                "title": entry.get("title", "").replace("\n", " ").strip(),
                "authors": authors,
                "summary": entry.get("summary", "").replace("\n", " ").strip(),
                "abstract": entry.get("summary", "").replace("\n", " ").strip(),
                "categories": categories,
                "created_at": published,
                "url": f"https://arxiv.org/abs/{arxiv_id}",
                "pdf_url": pdf_link,
            }
        except Exception as e:
            logger.info("Error parsing arXiv entry: %s", e)
            return None
