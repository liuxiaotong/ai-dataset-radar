"""HuggingFace Daily Papers scraper for early dataset discovery."""

import re
from datetime import datetime
from typing import Optional

from bs4 import BeautifulSoup

from utils.async_http import AsyncHTTPClient
from utils.logging_config import get_logger

logger = get_logger(__name__)


class HFPapersScraper:
    """Scraper for HuggingFace Daily Papers.

    HuggingFace curates trending AI papers daily, which often include
    new dataset announcements before they appear on HuggingFace Hub.
    """

    PAPERS_API = "https://huggingface.co/api/daily_papers"
    PAPERS_PAGE = "https://huggingface.co/papers"

    # Keywords indicating dataset-related papers
    DATASET_KEYWORDS = [
        "dataset",
        "benchmark",
        "corpus",
        "annotation",
        "training data",
        "evaluation",
        "instruction",
        "fine-tuning",
        "labeled",
        "labelled",
        "data collection",
        "human feedback",
        "rlhf",
        "preference",
        "alignment",
    ]

    def __init__(
        self,
        limit: int = 50,
        days: int = 7,
        http_client: AsyncHTTPClient = None,
    ):
        """Initialize the scraper.

        Args:
            limit: Maximum number of papers to fetch.
            days: Look back period in days.
            http_client: Shared async HTTP client (created if not provided).
        """
        self.limit = limit
        self.days = days
        self._http = http_client or AsyncHTTPClient()

    async def fetch(self) -> list[dict]:
        """Fetch papers from HuggingFace Daily Papers.

        Returns:
            List of paper information dictionaries.
        """
        papers = []

        # Try API first
        api_papers = await self._fetch_from_api()
        if api_papers:
            papers.extend(api_papers)

        # Supplement with page scraping if needed
        if len(papers) < self.limit:
            page_papers = await self._fetch_from_page()
            seen_ids = {p["id"] for p in papers}
            for paper in page_papers:
                if paper["id"] not in seen_ids:
                    papers.append(paper)
                    if len(papers) >= self.limit:
                        break

        # Filter for dataset-related papers and mark them
        for paper in papers:
            paper["is_dataset_paper"] = self._is_dataset_related(paper)

        return papers[: self.limit]

    async def _fetch_from_api(self) -> list[dict]:
        """Fetch papers from HuggingFace API.

        Returns:
            List of parsed paper dictionaries.
        """
        try:
            data = await self._http.get_json(self.PAPERS_API)

            if data is None:
                return []

            papers = []

            for item in data:
                parsed = self._parse_api_paper(item)
                if parsed:
                    papers.append(parsed)

            return papers

        except Exception as e:
            logger.info("Error parsing HF papers API response: %s", e)
            return []

    async def _fetch_from_page(self) -> list[dict]:
        """Fetch papers by scraping the HuggingFace papers page.

        Returns:
            List of parsed paper dictionaries.
        """
        try:
            html = await self._http.get_text(self.PAPERS_PAGE)

            if html is None:
                return []

            soup = BeautifulSoup(html, "html.parser")
            papers = []

            # Find paper cards - adjust selector based on actual page structure
            for article in soup.select("article"):
                paper = self._parse_page_article(article)
                if paper:
                    papers.append(paper)

            return papers

        except Exception as e:
            logger.info("Error parsing HF papers page: %s", e)
            return []

    def _parse_api_paper(self, item: dict) -> Optional[dict]:
        """Parse a paper from API response.

        Args:
            item: Raw paper data from API.

        Returns:
            Parsed paper info or None.
        """
        try:
            paper = item.get("paper", {})
            arxiv_id = paper.get("id", "")

            published_at = paper.get("publishedAt", "")
            if published_at:
                try:
                    published_at = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
                except (ValueError, TypeError):
                    published_at = None

            return {
                "source": "hf_papers",
                "id": arxiv_id,
                "arxiv_id": arxiv_id,
                "title": paper.get("title", ""),
                "summary": paper.get("summary", ""),
                "authors": [a.get("name", "") for a in paper.get("authors", [])],
                "upvotes": item.get("paper", {}).get("upvotes", 0),
                "comments": item.get("numComments", 0),
                "published_at": published_at.isoformat() if published_at else None,
                "url": f"https://huggingface.co/papers/{arxiv_id}",
                "arxiv_url": f"https://arxiv.org/abs/{arxiv_id}",
            }
        except Exception as e:
            logger.info("Error parsing HF paper: %s", e)
            return None

    def _parse_page_article(self, article) -> Optional[dict]:
        """Parse a paper from HTML article element.

        Args:
            article: BeautifulSoup article element.

        Returns:
            Parsed paper info or None.
        """
        try:
            # Get title and link
            title_elem = article.select_one("h3 a, h2 a, a.text-lg")
            if not title_elem:
                return None

            title = title_elem.get_text(strip=True)
            href = title_elem.get("href", "")

            # Extract arxiv ID from href
            arxiv_match = re.search(r"/papers/(\d+\.\d+)", href)
            if not arxiv_match:
                return None

            arxiv_id = arxiv_match.group(1)

            # Get summary/abstract if available
            summary_elem = article.select_one("p")
            summary = summary_elem.get_text(strip=True) if summary_elem else ""

            # Get upvotes if available
            upvotes = 0
            upvote_elem = article.select_one('[class*="upvote"], [class*="like"]')
            if upvote_elem:
                try:
                    upvotes = int(upvote_elem.get_text(strip=True).replace(",", ""))
                except ValueError:
                    pass

            # Extract date from <time> element
            published_at = None
            time_elem = article.select_one("time")
            if time_elem:
                datetime_attr = time_elem.get("datetime", "")
                if datetime_attr:
                    published_at = datetime_attr[:10]

            return {
                "source": "hf_papers",
                "id": arxiv_id,
                "arxiv_id": arxiv_id,
                "title": title,
                "summary": summary,
                "authors": [],
                "upvotes": upvotes,
                "comments": 0,
                "published_at": published_at,
                "url": f"https://huggingface.co/papers/{arxiv_id}",
                "arxiv_url": f"https://arxiv.org/abs/{arxiv_id}",
            }
        except Exception as e:
            logger.info("Error parsing HF paper article: %s", e)
            return None

    def _is_dataset_related(self, paper: dict) -> bool:
        """Check if a paper is related to datasets.

        Args:
            paper: Paper dict with title and summary.

        Returns:
            True if likely dataset-related.
        """
        text = f"{paper.get('title', '')} {paper.get('summary', '')}".lower()

        for keyword in self.DATASET_KEYWORDS:
            if keyword in text:
                return True

        return False

    async def fetch_paper_details(self, arxiv_id: str) -> Optional[dict]:
        """Fetch detailed information about a specific paper.

        Args:
            arxiv_id: arXiv paper ID.

        Returns:
            Paper details or None.
        """
        url = f"https://huggingface.co/api/papers/{arxiv_id}"

        try:
            data = await self._http.get_json(url)
            if data is not None:
                return {
                    "source": "hf_papers",
                    "id": arxiv_id,
                    "arxiv_id": arxiv_id,
                    "title": data.get("title", ""),
                    "summary": data.get("summary", ""),
                    "authors": [a.get("name", "") for a in data.get("authors", [])],
                    "upvotes": data.get("upvotes", 0),
                    "url": f"https://huggingface.co/papers/{arxiv_id}",
                    "arxiv_url": f"https://arxiv.org/abs/{arxiv_id}",
                }
            return None
        except Exception as e:
            logger.info("Error fetching paper %s: %s", arxiv_id, e)
            return None

    async def get_dataset_papers(self) -> list[dict]:
        """Fetch only dataset-related papers.

        Returns:
            List of dataset-related papers.
        """
        all_papers = await self.fetch()
        return [p for p in all_papers if p.get("is_dataset_paper", False)]
