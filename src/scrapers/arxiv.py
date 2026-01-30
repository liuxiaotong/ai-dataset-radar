"""arXiv papers scraper for RLHF and data annotation related papers.

Enhanced v5 scraper with tighter search queries focused on:
- RLHF / preference learning
- Data annotation methodology
- Training data quality
"""

import feedparser
import urllib.parse
from datetime import datetime
from typing import Optional


class ArxivScraper:
    """Scraper for arXiv papers related to RLHF and data annotation."""

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
        self,
        limit: int = 50,
        categories: Optional[list[str]] = None,
        config: dict = None
    ):
        """Initialize arXiv scraper.

        Args:
            limit: Maximum number of papers to fetch.
            categories: arXiv categories to search.
            config: Optional configuration dict.
        """
        self.limit = limit
        self.categories = categories or ["cs.CL", "cs.LG", "cs.AI"]
        self.config = config or {}

    def fetch(self) -> list[dict]:
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
            feed = feedparser.parse(url)
        except Exception as e:
            print(f"Error fetching arXiv papers: {e}")
            return []

        if feed.bozo and feed.bozo_exception:
            print(f"Warning: Feed parsing issue: {feed.bozo_exception}")

        results = []
        for entry in feed.entries:
            result = self._parse_entry(entry)
            if result:
                results.append(result)

        return results

    def fetch_by_keywords(self, keywords: list[str]) -> list[dict]:
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
            feed = feedparser.parse(url)
        except Exception as e:
            print(f"Error fetching arXiv papers: {e}")
            return []

        results = []
        for entry in feed.entries:
            result = self._parse_entry(entry)
            if result:
                results.append(result)

        return results

    def _parse_entry(self, entry: dict) -> Optional[dict]:
        """Parse an entry from the arXiv feed.

        Args:
            entry: Raw entry from feedparser.

        Returns:
            Parsed paper info or None if parsing fails.
        """
        try:
            # Extract arXiv ID from the entry ID URL
            arxiv_id = entry.id.split("/abs/")[-1]

            # Parse publication date
            published = entry.get("published", "")
            if published:
                try:
                    pub_date = datetime.strptime(published, "%Y-%m-%dT%H:%M:%SZ")
                    published = pub_date.isoformat()
                except ValueError:
                    pass

            # Extract categories
            categories = [tag.term for tag in entry.get("tags", [])]

            # Extract authors
            authors = [author.name for author in entry.get("authors", [])]

            # Get PDF link
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
            print(f"Error parsing arXiv entry: {e}")
            return None
