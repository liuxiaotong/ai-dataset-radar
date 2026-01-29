"""HuggingFace Daily Papers scraper for early dataset discovery."""

import re
import requests
from datetime import datetime, timedelta
from typing import Optional
from bs4 import BeautifulSoup


class HFPapersScraper:
    """Scraper for HuggingFace Daily Papers.

    HuggingFace curates trending AI papers daily, which often include
    new dataset announcements before they appear on HuggingFace Hub.
    """

    PAPERS_API = "https://huggingface.co/api/daily_papers"
    PAPERS_PAGE = "https://huggingface.co/papers"

    # Keywords indicating dataset-related papers
    DATASET_KEYWORDS = [
        "dataset", "benchmark", "corpus", "annotation",
        "training data", "evaluation", "instruction",
        "fine-tuning", "labeled", "labelled", "data collection",
        "human feedback", "rlhf", "preference", "alignment",
    ]

    def __init__(self, limit: int = 50, days: int = 7):
        """Initialize the scraper.

        Args:
            limit: Maximum number of papers to fetch.
            days: Look back period in days.
        """
        self.limit = limit
        self.days = days

    def fetch(self) -> list[dict]:
        """Fetch papers from HuggingFace Daily Papers.

        Returns:
            List of paper information dictionaries.
        """
        papers = []

        # Try API first
        api_papers = self._fetch_from_api()
        if api_papers:
            papers.extend(api_papers)

        # Supplement with page scraping if needed
        if len(papers) < self.limit:
            page_papers = self._fetch_from_page()
            seen_ids = {p["id"] for p in papers}
            for paper in page_papers:
                if paper["id"] not in seen_ids:
                    papers.append(paper)
                    if len(papers) >= self.limit:
                        break

        # Filter for dataset-related papers and mark them
        for paper in papers:
            paper["is_dataset_paper"] = self._is_dataset_related(paper)

        return papers[:self.limit]

    def _fetch_from_api(self) -> list[dict]:
        """Fetch papers from HuggingFace API.

        Returns:
            List of parsed paper dictionaries.
        """
        try:
            response = requests.get(
                self.PAPERS_API,
                timeout=30
            )

            if response.status_code != 200:
                return []

            data = response.json()
            papers = []

            for item in data:
                parsed = self._parse_api_paper(item)
                if parsed:
                    papers.append(parsed)

            return papers

        except requests.RequestException as e:
            print(f"Error fetching HF papers API: {e}")
            return []
        except Exception as e:
            print(f"Error parsing HF papers API response: {e}")
            return []

    def _fetch_from_page(self) -> list[dict]:
        """Fetch papers by scraping the HuggingFace papers page.

        Returns:
            List of parsed paper dictionaries.
        """
        try:
            response = requests.get(
                self.PAPERS_PAGE,
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=30
            )
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")
            papers = []

            # Find paper cards - adjust selector based on actual page structure
            for article in soup.select("article"):
                paper = self._parse_page_article(article)
                if paper:
                    papers.append(paper)

            return papers

        except requests.RequestException as e:
            print(f"Error fetching HF papers page: {e}")
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
                except:
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
            print(f"Error parsing HF paper: {e}")
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

            return {
                "source": "hf_papers",
                "id": arxiv_id,
                "arxiv_id": arxiv_id,
                "title": title,
                "summary": summary,
                "authors": [],
                "upvotes": upvotes,
                "comments": 0,
                "published_at": None,
                "url": f"https://huggingface.co/papers/{arxiv_id}",
                "arxiv_url": f"https://arxiv.org/abs/{arxiv_id}",
            }
        except Exception as e:
            print(f"Error parsing HF paper article: {e}")
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

    def fetch_paper_details(self, arxiv_id: str) -> Optional[dict]:
        """Fetch detailed information about a specific paper.

        Args:
            arxiv_id: arXiv paper ID.

        Returns:
            Paper details or None.
        """
        url = f"https://huggingface.co/api/papers/{arxiv_id}"

        try:
            response = requests.get(url, timeout=15)
            if response.status_code == 200:
                data = response.json()
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
        except requests.RequestException as e:
            print(f"Error fetching paper {arxiv_id}: {e}")
            return None

    def get_dataset_papers(self) -> list[dict]:
        """Fetch only dataset-related papers.

        Returns:
            List of dataset-related papers.
        """
        all_papers = self.fetch()
        return [p for p in all_papers if p.get("is_dataset_paper", False)]
