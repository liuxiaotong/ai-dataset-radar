"""Semantic Scholar API scraper for high-impact dataset papers.

API: https://api.semanticscholar.org/graph/v1/paper/search
Rate Limits: 100 req/5min without key, higher with API key.
Gracefully works without credentials.
"""

import os
from datetime import datetime, timedelta, timezone
from typing import Optional

from .base import BaseScraper
from .registry import register_scraper

from utils.async_http import AsyncHTTPClient
from utils.logging_config import get_logger

logger = get_logger(__name__)


@register_scraper("semantic_scholar")
class SemanticScholarScraper(BaseScraper):
    """Scraper for high-citation dataset papers via Semantic Scholar."""

    name = "semantic_scholar"
    source_type = "paper"

    BASE_URL = "https://api.semanticscholar.org/graph/v1"
    FIELDS = "paperId,title,abstract,authors,year,citationCount,publicationDate,externalIds,url,venue"

    def __init__(
        self,
        config: dict = None,
        limit: int = 100,
        http_client: AsyncHTTPClient = None,
    ):
        super().__init__(config)
        self.limit = limit
        self._http = http_client or AsyncHTTPClient()

        ss_cfg = (config or {}).get("sources", {}).get("semantic_scholar", {})
        self.enabled = ss_cfg.get("enabled", True)
        self.keywords = ss_cfg.get("keywords", [
            "dataset benchmark machine learning",
            "training data large language model",
            "dataset corpus NLP",
            "benchmark evaluation AI",
        ])

        api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "")
        self._headers = {"User-Agent": "AI-Dataset-Radar/6.0"}
        if api_key:
            self._headers["x-api-key"] = api_key

    async def scrape(self, config: dict = None) -> list[dict]:
        return await self.fetch()

    async def fetch(self, days: int = 30) -> list[dict]:
        """Fetch papers matching configured keywords.

        Args:
            days: Look back period in days.

        Returns:
            List of paper info dicts.
        """
        if not self.enabled:
            return []

        start_date = (
            datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=days)
        ).strftime("%Y-%m-%d")

        seen_ids = set()
        results = []

        for keyword in self.keywords:
            try:
                data = await self._http.get_json(
                    f"{self.BASE_URL}/paper/search",
                    params={
                        "query": keyword,
                        "limit": min(self.limit, 50),
                        "fields": self.FIELDS,
                        "publicationDateOrYear": f"{start_date}:",
                    },
                    headers=self._headers,
                    max_retries=2,
                )
            except Exception as e:
                logger.warning("Semantic Scholar search error for '%s': %s", keyword, e)
                continue

            if not data or not isinstance(data, dict):
                continue

            for item in data.get("data", []):
                paper_id = item.get("paperId")
                if not paper_id or paper_id in seen_ids:
                    continue
                seen_ids.add(paper_id)

                parsed = self._parse_paper(item)
                if parsed:
                    results.append(parsed)

        logger.info(
            "Semantic Scholar: found %d papers from %d queries",
            len(results), len(self.keywords),
        )
        return results

    def _parse_paper(self, item: dict) -> Optional[dict]:
        """Parse a paper from API response."""
        paper_id = item.get("paperId")
        if not paper_id:
            return None

        external_ids = item.get("externalIds") or {}
        authors = [
            a.get("name", "")
            for a in (item.get("authors") or [])
            if a.get("name")
        ]

        citation_count = item.get("citationCount") or 0
        pub_date = item.get("publicationDate") or ""

        return {
            "source": "semantic_scholar",
            "id": paper_id,
            "title": item.get("title", ""),
            "abstract": item.get("abstract") or "",
            "authors": authors,
            "year": item.get("year"),
            "citation_count": citation_count,
            "publication_date": pub_date,
            "arxiv_id": external_ids.get("ArXiv"),
            "doi": external_ids.get("DOI"),
            "url": item.get("url") or f"https://www.semanticscholar.org/paper/{paper_id}",
            "venue": item.get("venue") or "",
            "created_at": pub_date or "",
        }
