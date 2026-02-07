"""GitHub repository scraper for dataset discovery."""

import requests
from datetime import datetime, timedelta
from typing import Optional
from bs4 import BeautifulSoup

from .base import BaseScraper
from .registry import register_scraper

from utils.logging_config import get_logger

logger = get_logger(__name__)


@register_scraper("github")
class GitHubScraper(BaseScraper):
    """Scraper for GitHub repositories related to datasets.

    Monitors:
    1. GitHub Search API for new dataset repos
    2. GitHub Trending page for popular new repos
    """

    name = "github"
    source_type = "code_host"

    SEARCH_API = "https://api.github.com/search/repositories"
    TRENDING_URL = "https://github.com/trending"

    # Keywords that indicate a dataset repository
    DATASET_KEYWORDS = [
        "dataset",
        "datasets",
        "benchmark",
        "corpus",
        "training-data",
        "evaluation",
        "annotation",
        "labeled",
        "labelled",
        "fine-tuning",
        "instruction",
    ]

    # Topics to search for
    DATASET_TOPICS = [
        "dataset",
        "datasets",
        "machine-learning-dataset",
        "nlp-dataset",
        "benchmark",
        "llm",
        "fine-tuning",
    ]

    # Keywords for relevance scoring
    RELEVANCE_KEYWORDS = [
        "dataset",
        "annotation",
        "benchmark",
        "rlhf",
        "evaluation",
        "training-data",
        "preference",
        "instruction",
        "fine-tuning",
    ]

    def __init__(
        self, config: dict = None, limit: int = 50, days: int = 7, token: Optional[str] = None
    ):
        """Initialize the scraper.

        Args:
            config: Optional configuration dict.
            limit: Maximum number of repos to fetch.
            days: Look back period in days.
            token: Optional GitHub API token for higher rate limits.
        """
        super().__init__(config)
        self.limit = limit
        self.days = days
        self.headers = {"Accept": "application/vnd.github.v3+json"}
        if token:
            self.headers["Authorization"] = f"token {token}"
        # Load relevance keywords from config if available
        self.relevance_keywords = self.config.get("relevance_keywords") or self.RELEVANCE_KEYWORDS

    def scrape(self, config: dict = None) -> list[dict]:
        """Scrape repositories from GitHub.

        Args:
            config: Optional runtime configuration.

        Returns:
            List of repository dictionaries with relevance field.
        """
        repos = self.fetch()
        # Add relevance scoring to all repos
        for repo in repos:
            if "relevance" not in repo:
                repo["relevance"] = self._calculate_relevance(repo)
        return repos

    def _calculate_relevance(self, repo: dict, keywords: Optional[list[str]] = None) -> str:
        """Calculate relevance score for a repository.

        Args:
            repo: Repository dictionary.
            keywords: Keywords to check against.

        Returns:
            "high" if 2+ keyword matches or dataset-related topic, else "low".
        """
        keywords = keywords or self.relevance_keywords
        matches = 0

        # Check name
        name = (repo.get("name") or "").lower()
        for kw in keywords:
            if kw.lower() in name:
                matches += 1

        # Check description
        description = (repo.get("description") or "").lower()
        for kw in keywords:
            if kw.lower() in description:
                matches += 1

        # Check topics
        topics = [t.lower() for t in repo.get("topics", [])]
        dataset_topics = {"dataset", "datasets", "benchmark", "training-data"}
        if any(t in dataset_topics for t in topics):
            return "high"

        for kw in keywords:
            if kw.lower() in topics:
                matches += 1

        return "high" if matches >= 2 else "low"

    def fetch(self) -> list[dict]:
        """Fetch dataset-related repositories.

        Returns:
            List of repository information dictionaries.
        """
        results = []
        seen_repos = set()

        # Strategy 1: Search API with keywords
        for keyword in ["dataset", "benchmark", "training data", "instruction tuning"]:
            repos = self._search_repos(keyword)
            for repo in repos:
                if repo["id"] not in seen_repos:
                    seen_repos.add(repo["id"])
                    results.append(repo)
                    if len(results) >= self.limit:
                        break
            if len(results) >= self.limit:
                break

        # Strategy 2: Trending repos (supplement)
        if len(results) < self.limit:
            trending = self._fetch_trending()
            for repo in trending:
                if repo["id"] not in seen_repos:
                    # Filter for dataset-related
                    if self._is_dataset_related(repo):
                        seen_repos.add(repo["id"])
                        results.append(repo)
                        if len(results) >= self.limit:
                            break

        return results[: self.limit]

    def _search_repos(self, keyword: str) -> list[dict]:
        """Search for repositories using GitHub Search API.

        Args:
            keyword: Search keyword.

        Returns:
            List of parsed repository dictionaries.
        """
        # Calculate date range
        since_date = (datetime.now() - timedelta(days=self.days)).strftime("%Y-%m-%d")

        # Build query: keyword + created recently + high activity
        query = f"{keyword} in:name,description,readme created:>{since_date}"

        params = {
            "q": query,
            "sort": "stars",
            "order": "desc",
            "per_page": min(self.limit, 30),
        }

        try:
            response = requests.get(
                self.SEARCH_API, params=params, headers=self.headers, timeout=30
            )
            response.raise_for_status()
            data = response.json()

            repos = []
            for item in data.get("items", []):
                parsed = self._parse_repo(item)
                if parsed:
                    repos.append(parsed)
            return repos

        except requests.RequestException:
            logger.info("Error searching GitHub for '{keyword}': {e}")
            return []

    def _fetch_trending(self) -> list[dict]:
        """Fetch trending repositories from GitHub Trending page.

        Returns:
            List of parsed repository dictionaries.
        """
        try:
            # Fetch trending for Python (most ML repos)
            response = requests.get(
                f"{self.TRENDING_URL}/python",
                params={"since": "weekly"},
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=30,
            )
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")
            repos = []

            # Parse trending repo articles
            for article in soup.select("article.Box-row"):
                repo = self._parse_trending_article(article)
                if repo:
                    repos.append(repo)

            return repos

        except requests.RequestException as e:
            logger.info("Error fetching GitHub trending: %s", e)
            return []

    def _parse_repo(self, item: dict) -> Optional[dict]:
        """Parse a repository from GitHub API response.

        Args:
            item: Raw repository data from API.

        Returns:
            Parsed repository info or None.
        """
        try:
            created_at = item.get("created_at", "")
            if created_at:
                created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            else:
                created_at = None

            return {
                "source": "github",
                "id": item.get("full_name", ""),
                "name": item.get("name", ""),
                "full_name": item.get("full_name", ""),
                "author": item.get("owner", {}).get("login", ""),
                "description": item.get("description", "") or "",
                "stars": item.get("stargazers_count", 0),
                "forks": item.get("forks_count", 0),
                "language": item.get("language", ""),
                "topics": item.get("topics", []),
                "created_at": created_at.isoformat() if created_at else None,
                "url": item.get("html_url", ""),
                "is_dataset": self._is_dataset_related(
                    {
                        "name": item.get("name", ""),
                        "description": item.get("description", "") or "",
                        "topics": item.get("topics", []),
                    }
                ),
            }
        except Exception as e:
            logger.info("Error parsing repo %s: %s", item.get("full_name", "unknown"), e)
            return None

    def _parse_trending_article(self, article) -> Optional[dict]:
        """Parse a trending repo from HTML article element.

        Args:
            article: BeautifulSoup article element.

        Returns:
            Parsed repository info or None.
        """
        try:
            # Get repo name from h2 > a
            name_link = article.select_one("h2 a")
            if not name_link:
                return None

            full_name = name_link.get("href", "").strip("/")
            if not full_name:
                return None

            parts = full_name.split("/")
            if len(parts) != 2:
                return None

            author, name = parts

            # Get description
            desc_elem = article.select_one("p")
            description = desc_elem.get_text(strip=True) if desc_elem else ""

            # Get stars
            stars = 0
            stars_elem = article.select_one('a[href$="/stargazers"]')
            if stars_elem:
                stars_text = stars_elem.get_text(strip=True).replace(",", "")
                try:
                    stars = int(stars_text)
                except ValueError:
                    pass

            # Get language
            language = ""
            lang_elem = article.select_one('[itemprop="programmingLanguage"]')
            if lang_elem:
                language = lang_elem.get_text(strip=True)

            return {
                "source": "github_trending",
                "id": full_name,
                "name": name,
                "full_name": full_name,
                "author": author,
                "description": description,
                "stars": stars,
                "forks": 0,
                "language": language,
                "topics": [],
                "created_at": None,
                "url": f"https://github.com/{full_name}",
                "is_dataset": self._is_dataset_related(
                    {
                        "name": name,
                        "description": description,
                        "topics": [],
                    }
                ),
            }
        except Exception as e:
            logger.info("Error parsing trending article: %s", e)
            return None

    def _is_dataset_related(self, repo: dict) -> bool:
        """Check if a repository is related to datasets.

        Args:
            repo: Repository dict with name, description, topics.

        Returns:
            True if likely dataset-related.
        """
        # Check topics
        topics = [t.lower() for t in repo.get("topics", [])]
        for topic in self.DATASET_TOPICS:
            if topic in topics:
                return True

        # Check name and description
        text = f"{repo.get('name', '')} {repo.get('description', '')}".lower()

        for keyword in self.DATASET_KEYWORDS:
            if keyword in text:
                return True

        return False

    def fetch_repo_details(self, full_name: str) -> Optional[dict]:
        """Fetch detailed information about a specific repository.

        Args:
            full_name: Repository full name (owner/repo).

        Returns:
            Repository details or None.
        """
        url = f"https://api.github.com/repos/{full_name}"

        try:
            response = requests.get(url, headers=self.headers, timeout=15)
            response.raise_for_status()
            return self._parse_repo(response.json())
        except requests.RequestException as e:
            logger.info("Error fetching repo %s: %s", full_name, e)
            return None

    def fetch_readme(self, full_name: str) -> Optional[str]:
        """Fetch README content for a repository.

        Args:
            full_name: Repository full name (owner/repo).

        Returns:
            README content or None.
        """
        url = f"https://api.github.com/repos/{full_name}/readme"

        try:
            response = requests.get(
                url, headers={**self.headers, "Accept": "application/vnd.github.raw"}, timeout=15
            )
            if response.status_code == 200:
                return response.text
            return None
        except requests.RequestException:
            return None
