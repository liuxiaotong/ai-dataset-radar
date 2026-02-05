"""GitHub organization scraper for monitoring specific orgs.

Monitors activity from configured GitHub organizations,
tracking new repositories and updates.
"""

import os
import time
import requests
from datetime import datetime, timedelta
from typing import Optional

from .base import BaseScraper
from .registry import register_scraper

from utils.logging_config import get_logger

logger = get_logger(__name__)


@register_scraper("github_org")
class GitHubOrgScraper(BaseScraper):
    """Scraper for monitoring GitHub organization activity."""

    name = "github_org"
    source_type = "code_host"

    API_BASE = "https://api.github.com"

    # Default relevance keywords
    DEFAULT_RELEVANCE_KEYWORDS = [
        "dataset", "annotation", "benchmark", "rlhf", "evaluation",
        "preference", "instruction", "fine-tuning", "training",
    ]

    def __init__(self, config: dict = None, token: Optional[str] = None):
        """Initialize the GitHub org scraper.

        Args:
            config: Configuration with watch_orgs and relevance_keywords.
            token: GitHub API token (or from env GITHUB_TOKEN).
        """
        super().__init__(config)
        self.token = token or os.environ.get("GITHUB_TOKEN")
        self.headers = {"Accept": "application/vnd.github.v3+json"}
        if self.token:
            self.headers["Authorization"] = f"token {self.token}"

        # Configuration
        self.watch_orgs = self.config.get("watch_orgs", [])
        self.relevance_keywords = self.config.get(
            "relevance_keywords", self.DEFAULT_RELEVANCE_KEYWORDS
        )
        self._last_request_time = 0

    def _rate_limit_wait(self) -> None:
        """Wait to respect rate limits."""
        elapsed = time.time() - self._last_request_time
        if elapsed < 1.0:
            time.sleep(1.0 - elapsed)
        self._last_request_time = time.time()

    def scrape(self, config: dict = None) -> list[dict]:
        """Scrape repos from all watched organizations.

        Args:
            config: Optional runtime config with watch_orgs override.

        Returns:
            List of repository dictionaries.
        """
        runtime_config = config or {}
        orgs = runtime_config.get("watch_orgs") or self.watch_orgs

        if not orgs:
            logger.info("  No GitHub organizations configured to watch")
            return []

        all_repos = []
        for org in orgs:
            repos = self.fetch_org_repos(org)
            all_repos.extend(repos)

        return self.deduplicate(all_repos)

    def fetch_org_repos(
        self,
        org: str,
        days: int = 7,
        limit: int = 30
    ) -> list[dict]:
        """Fetch recently updated repos from an organization.

        Args:
            org: GitHub organization name.
            days: Look back period in days.
            limit: Maximum repos per org.

        Returns:
            List of repository dictionaries.
        """
        self._rate_limit_wait()

        url = f"{self.API_BASE}/orgs/{org}/repos"
        params = {
            "sort": "updated",
            "direction": "desc",
            "per_page": min(limit, 100),
        }

        try:
            response = requests.get(
                url, params=params, headers=self.headers, timeout=30
            )
            response.raise_for_status()
            repos_data = response.json()
        except requests.RequestException as e:
            logger.info("    Error fetching repos for %s: %s", org, e)
            return []

        cutoff = datetime.utcnow() - timedelta(days=days)
        results = []

        for repo in repos_data:
            parsed = self._parse_repo(repo, org)
            if not parsed:
                continue

            # Filter by update date
            updated_at = parsed.get("last_updated")
            if updated_at:
                try:
                    updated_dt = datetime.fromisoformat(
                        updated_at.replace("Z", "+00:00")
                    )
                    if updated_dt.replace(tzinfo=None) < cutoff:
                        continue
                except (ValueError, AttributeError):
                    pass

            # Add relevance score
            parsed["relevance"] = self._calculate_relevance(parsed)
            results.append(parsed)

        return results

    def _parse_repo(self, repo: dict, org: str) -> Optional[dict]:
        """Parse a repository from GitHub API response.

        Args:
            repo: Raw repository data.
            org: Organization name.

        Returns:
            Parsed repository dict or None.
        """
        try:
            created_at = repo.get("created_at", "")
            updated_at = repo.get("updated_at", "") or repo.get("pushed_at", "")

            return {
                "source": "github_org",
                "id": repo.get("full_name", ""),
                "name": repo.get("name", ""),
                "full_name": repo.get("full_name", ""),
                "org": org,
                "author": repo.get("owner", {}).get("login", ""),
                "description": repo.get("description", "") or "",
                "stars": repo.get("stargazers_count", 0),
                "forks": repo.get("forks_count", 0),
                "language": repo.get("language", ""),
                "topics": repo.get("topics", []),
                "created_at": created_at,
                "last_updated": updated_at,
                "url": repo.get("html_url", ""),
            }
        except Exception as e:
            logger.info("    Error parsing repo: %s", e)
            return None

    def _calculate_relevance(self, repo: dict) -> str:
        """Calculate relevance for a repository.

        Args:
            repo: Repository dictionary.

        Returns:
            "high" if relevant to AI data, "low" otherwise.
        """
        matches = 0
        keywords = self.relevance_keywords

        # Check name
        name = repo.get("name", "").lower()
        for kw in keywords:
            if kw.lower() in name:
                matches += 1

        # Check description
        description = repo.get("description", "").lower()
        for kw in keywords:
            if kw.lower() in description:
                matches += 1

        # Check topics
        topics = [t.lower() for t in repo.get("topics", [])]
        for kw in keywords:
            if kw.lower() in topics:
                matches += 1

        # High relevance if dataset-related topic or 2+ keyword matches
        dataset_topics = {"dataset", "datasets", "benchmark", "training-data"}
        if any(t in dataset_topics for t in topics):
            return "high"

        return "high" if matches >= 2 else "low"

    def fetch(self) -> list[dict]:
        """Fetch repos (alias for scrape for backward compatibility).

        Returns:
            List of repository dictionaries.
        """
        return self.scrape()
