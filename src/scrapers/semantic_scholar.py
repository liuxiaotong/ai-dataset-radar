"""Semantic Scholar API scraper for high-impact dataset papers.

API Documentation: https://api.semanticscholar.org/api-docs/graph

Rate Limits:
- Without API key: 100 requests per 5 minutes
- With API key: 1 request per second (higher limits available)
"""

import time
import random
from datetime import datetime, timedelta
from typing import Optional
import requests

from utils.logging_config import get_logger

logger = get_logger(__name__)


class SemanticScholarScraper:
    """Scraper for high-citation dataset papers using Semantic Scholar API."""

    BASE_URL = "https://api.semanticscholar.org/graph/v1"

    def __init__(
        self,
        limit: int = 100,
        months_back: int = 6,
        min_citations: int = 20,
        min_monthly_growth: int = 10,
        api_key: Optional[str] = None,
    ):
        """Initialize the Semantic Scholar scraper.

        Args:
            limit: Maximum number of papers to return.
            months_back: How many months back to search.
            min_citations: Minimum citation count filter.
            min_monthly_growth: Minimum monthly citation growth filter.
            api_key: Optional API key for higher rate limits.
        """
        self.limit = limit
        self.months_back = months_back
        self.min_citations = min_citations
        self.min_monthly_growth = min_monthly_growth
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers["User-Agent"] = "AI-Dataset-Radar/3.0"
        if api_key:
            self.session.headers["x-api-key"] = api_key

        # Rate limiting settings
        self._base_delay = 1.0 if api_key else 3.5  # seconds
        self._max_retries = 3
        self._last_request_time = 0

    def _rate_limit_wait(self) -> None:
        """Wait to respect rate limits."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self._base_delay:
            time.sleep(self._base_delay - elapsed + random.uniform(0.1, 0.5))
        self._last_request_time = time.time()

    def _request_with_retry(
        self,
        url: str,
        params: dict,
        max_retries: Optional[int] = None,
    ) -> Optional[dict]:
        """Make a request with exponential backoff retry.

        Args:
            url: Request URL.
            params: Query parameters.
            max_retries: Maximum retry attempts.

        Returns:
            JSON response data or None on failure.
        """
        max_retries = max_retries or self._max_retries

        for attempt in range(max_retries + 1):
            self._rate_limit_wait()

            try:
                response = self.session.get(url, params=params, timeout=30)

                if response.status_code == 200:
                    return response.json()

                elif response.status_code == 429:
                    # Rate limited - exponential backoff
                    wait_time = (2**attempt) * 5 + random.uniform(1, 3)
                    if attempt < max_retries:
                        logger.info(
                            "  Rate limited, waiting %.1fs (attempt %s/%s)",
                            wait_time,
                            attempt + 1,
                            max_retries + 1,
                        )
                        time.sleep(wait_time)
                        continue
                    else:
                        logger.info("  Rate limit exceeded after %s attempts", max_retries + 1)
                        return None

                elif response.status_code == 504:
                    # Gateway timeout - retry with backoff
                    wait_time = (2**attempt) * 2
                    if attempt < max_retries:
                        logger.info("  Gateway timeout, retrying in %.1fs...", wait_time)
                        time.sleep(wait_time)
                        continue

                else:
                    response.raise_for_status()

            except requests.exceptions.Timeout:
                if attempt < max_retries:
                    logger.info("  Request timeout, retrying...")
                    time.sleep(2**attempt)
                    continue
                logger.info("  Request timeout after all retries")
                return None

            except requests.RequestException as e:
                if attempt < max_retries:
                    time.sleep(2**attempt)
                    continue
                logger.info("  Request error: %s", e)
                return None

        return None

    def fetch(self) -> list[dict]:
        """Fetch high-impact dataset papers.

        Returns:
            List of paper information dictionaries with citation data.
        """
        all_papers = []

        # Search terms for dataset papers
        search_queries = [
            "dataset benchmark machine learning",
            "dataset corpus NLP",
            "benchmark evaluation AI",
            "training data large language model",
            "dataset robotics manipulation",
        ]

        for i, query in enumerate(search_queries):
            logger.info("  Searching: %s... (%s/%s)", query[:40], i + 1, len(search_queries))
            papers = self._search_papers(query)
            all_papers.extend(papers)

        # Deduplicate by paper ID
        seen_ids = set()
        unique_papers = []
        for paper in all_papers:
            if paper["id"] not in seen_ids:
                seen_ids.add(paper["id"])
                unique_papers.append(paper)

        logger.info("  Total unique papers: %s", len(unique_papers))

        # Filter by citation criteria
        filtered = self._filter_by_impact(unique_papers)
        logger.info("  Papers meeting impact criteria: %s", len(filtered))

        # Sort by value (citation count + growth)
        filtered.sort(
            key=lambda p: p.get("citation_count", 0) + p.get("citation_monthly_growth", 0) * 10,
            reverse=True,
        )

        return filtered[: self.limit]

    def _search_papers(self, query: str, limit: int = 50) -> list[dict]:
        """Search for papers matching query.

        Args:
            query: Search query string.
            limit: Max results per query.

        Returns:
            List of paper dictionaries.
        """
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.months_back * 30)
        year_range = f"{start_date.year}-{end_date.year}"

        url = f"{self.BASE_URL}/paper/search"
        params = {
            "query": query,
            "limit": limit,
            "fields": "paperId,title,abstract,authors,year,citationCount,publicationDate,externalIds,url,fieldsOfStudy,venue",
            "year": year_range,
        }

        data = self._request_with_retry(url, params)
        if not data:
            return []

        papers = []
        for item in data.get("data", []):
            paper = self._parse_paper(item)
            if paper:
                papers.append(paper)

        return papers

    def _parse_paper(self, item: dict) -> Optional[dict]:
        """Parse a paper from API response.

        Args:
            item: Raw paper data from API.

        Returns:
            Parsed paper dictionary or None.
        """
        try:
            paper_id = item.get("paperId")
            if not paper_id:
                return None

            # Extract external IDs
            external_ids = item.get("externalIds", {}) or {}
            arxiv_id = external_ids.get("ArXiv")
            doi = external_ids.get("DOI")

            # Parse authors
            authors = []
            for author in item.get("authors", []) or []:
                if author.get("name"):
                    authors.append(author["name"])

            # Parse publication date
            pub_date = item.get("publicationDate")
            if pub_date:
                try:
                    pub_datetime = datetime.strptime(pub_date, "%Y-%m-%d")
                    pub_date = pub_datetime.isoformat()
                except ValueError:
                    pass

            # Calculate citation growth rate (rough estimate)
            citation_count = item.get("citationCount", 0) or 0
            year = item.get("year")
            monthly_growth = 0

            if year and citation_count > 0:
                months_since_pub = max(1, (datetime.now().year - year) * 12)
                if pub_date:
                    try:
                        pub_dt = datetime.fromisoformat(pub_date.replace("Z", "+00:00"))
                        months_since_pub = max(
                            1, (datetime.now() - pub_dt.replace(tzinfo=None)).days // 30
                        )
                    except (ValueError, TypeError):
                        pass
                monthly_growth = citation_count / months_since_pub

            # Extract fields of study
            fields = item.get("fieldsOfStudy", []) or []

            return {
                "source": "semantic_scholar",
                "id": paper_id,
                "title": item.get("title", ""),
                "abstract": item.get("abstract", ""),
                "authors": authors,
                "year": year,
                "citation_count": citation_count,
                "citation_monthly_growth": round(monthly_growth, 2),
                "publication_date": pub_date,
                "arxiv_id": arxiv_id,
                "doi": doi,
                "url": item.get("url", f"https://www.semanticscholar.org/paper/{paper_id}"),
                "venue": item.get("venue", ""),
                "fields_of_study": fields,
                "created_at": pub_date or datetime.now().isoformat(),
            }
        except Exception as e:
            logger.info("Error parsing paper: %s", e)
            return None

    def _filter_by_impact(self, papers: list[dict]) -> list[dict]:
        """Filter papers by citation impact criteria.

        Args:
            papers: List of paper dictionaries.

        Returns:
            Filtered list meeting citation thresholds.
        """
        filtered = []
        for paper in papers:
            citations = paper.get("citation_count", 0)
            monthly_growth = paper.get("citation_monthly_growth", 0)

            # Include if citations > threshold OR monthly growth > threshold
            if citations >= self.min_citations or monthly_growth >= self.min_monthly_growth:
                filtered.append(paper)

        return filtered

    def get_paper_details(self, paper_id: str) -> Optional[dict]:
        """Get detailed information for a specific paper.

        Args:
            paper_id: Semantic Scholar paper ID.

        Returns:
            Detailed paper information or None.
        """
        url = f"{self.BASE_URL}/paper/{paper_id}"
        params = {
            "fields": "paperId,title,abstract,authors,year,citationCount,citations,references,publicationDate,externalIds,url,fieldsOfStudy,venue,tldr",
        }

        data = self._request_with_retry(url, params)
        if data:
            return self._parse_paper(data)
        return None

    def get_citations(self, paper_id: str, limit: int = 100) -> list[dict]:
        """Get papers that cite the given paper.

        Args:
            paper_id: Semantic Scholar paper ID.
            limit: Maximum citations to return.

        Returns:
            List of citing papers.
        """
        url = f"{self.BASE_URL}/paper/{paper_id}/citations"
        params = {
            "fields": "paperId,title,authors,year,citationCount",
            "limit": limit,
        }

        data = self._request_with_retry(url, params)
        if not data:
            return []

        citations = []
        for item in data.get("data", []):
            citing_paper = item.get("citingPaper", {})
            if citing_paper.get("paperId"):
                citations.append(
                    {
                        "id": citing_paper["paperId"],
                        "title": citing_paper.get("title", ""),
                        "year": citing_paper.get("year"),
                        "citation_count": citing_paper.get("citationCount", 0),
                    }
                )
        return citations

    def extract_dataset_info(self, paper: dict) -> Optional[dict]:
        """Extract dataset information from paper abstract/title.

        Args:
            paper: Paper dictionary.

        Returns:
            Dataset information if detected, else None.
        """
        title = (paper.get("title", "") or "").lower()
        abstract = (paper.get("abstract", "") or "").lower()
        text = f"{title} {abstract}"

        # Dataset indicators
        dataset_keywords = [
            "dataset",
            "benchmark",
            "corpus",
            "collection",
            "training data",
            "evaluation set",
            "test set",
        ]

        is_dataset_paper = any(kw in text for kw in dataset_keywords)
        if not is_dataset_paper:
            return None

        # Try to extract dataset name from title
        dataset_name = None
        title_original = paper.get("title", "")

        # Common patterns: "XXX: A Dataset for...", "The XXX Dataset", "XXX Benchmark"
        import re

        patterns = [
            r"^([A-Z][A-Za-z0-9\-]+):\s*[Aa]",  # "DatasetName: A..."
            r"[Tt]he\s+([A-Z][A-Za-z0-9\-]+)\s+[Dd]ataset",  # "The XXX Dataset"
            r"([A-Z][A-Za-z0-9\-]+)\s+[Bb]enchmark",  # "XXX Benchmark"
            r"([A-Z][A-Za-z0-9\-]+)\s+[Cc]orpus",  # "XXX Corpus"
        ]

        for pattern in patterns:
            match = re.search(pattern, title_original)
            if match:
                dataset_name = match.group(1)
                break

        return {
            "paper_id": paper.get("id"),
            "paper_title": paper.get("title"),
            "dataset_name": dataset_name,
            "citation_count": paper.get("citation_count", 0),
            "citation_monthly_growth": paper.get("citation_monthly_growth", 0),
            "arxiv_id": paper.get("arxiv_id"),
            "url": paper.get("url"),
            "authors": paper.get("authors", []),
            "year": paper.get("year"),
        }

    def fetch_dataset_papers(self) -> list[dict]:
        """Fetch papers and extract dataset information.

        Returns:
            List of dataset information dictionaries.
        """
        papers = self.fetch()

        datasets = []
        for paper in papers:
            dataset_info = self.extract_dataset_info(paper)
            if dataset_info:
                datasets.append(dataset_info)

        return datasets
