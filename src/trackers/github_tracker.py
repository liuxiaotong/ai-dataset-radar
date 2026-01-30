"""GitHub organization tracker for monitoring data vendor activities.

Monitors GitHub organizations for:
- Recently updated repositories
- Signal keywords in descriptions and READMEs
- Data annotation and RLHF related projects
"""

import os
import re
import time
from datetime import datetime, timedelta
from typing import Optional

import requests


# Signal keywords for detecting relevant repos
SIGNAL_KEYWORDS = [
    "rlhf", "reward", "preference", "annotation", "labeling", "label",
    "evaluation", "benchmark", "human-feedback", "sft", "instruction",
    "fine-tuning", "fine-tune", "data-quality", "dataset", "training-data",
    "llm", "language-model", "crowdsourcing", "human-in-the-loop",
    "active-learning", "data-collection", "synthetic-data"
]


class GitHubTracker:
    """Track GitHub organization activities."""

    BASE_URL = "https://api.github.com"

    def __init__(self, config: dict):
        """Initialize GitHub tracker.

        Args:
            config: Configuration dict with github settings.
        """
        self.config = config
        github_config = config.get("github", {})

        # Get token from config or environment
        self.token = github_config.get("token") or os.environ.get("GITHUB_TOKEN")

        self.headers = {
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "AI-Dataset-Radar/1.0"
        }
        if self.token:
            self.headers["Authorization"] = f"token {self.token}"

        # Get org lists from config
        self.vendor_orgs = github_config.get("orgs", {}).get("data_vendors", [])
        self.lab_orgs = github_config.get("orgs", {}).get("ai_labs", [])

    def _make_request(self, url: str, params: dict = None) -> Optional[dict]:
        """Make a GitHub API request with rate limit handling."""
        try:
            resp = requests.get(url, headers=self.headers, params=params, timeout=30)

            if resp.status_code == 403:
                # Rate limited
                reset_time = resp.headers.get("X-RateLimit-Reset")
                if reset_time:
                    wait_time = int(reset_time) - int(time.time())
                    print(f"  GitHub rate limited, reset in {wait_time}s")
                return None

            if resp.status_code == 404:
                return None

            resp.raise_for_status()
            return resp.json()

        except requests.RequestException as e:
            print(f"  GitHub API error: {e}")
            return None

    def get_org_repos(self, org_name: str, days: int = 7) -> list[dict]:
        """Get recently updated repos for an organization.

        Args:
            org_name: GitHub organization name.
            days: Look back period in days.

        Returns:
            List of repo dicts with activity info.
        """
        url = f"{self.BASE_URL}/orgs/{org_name}/repos"
        params = {
            "sort": "updated",
            "direction": "desc",
            "per_page": 30
        }

        data = self._make_request(url, params)
        if not data:
            return []

        cutoff = datetime.utcnow() - timedelta(days=days)
        recent_repos = []

        for repo in data:
            updated_at = datetime.strptime(
                repo["updated_at"], "%Y-%m-%dT%H:%M:%SZ"
            )

            if updated_at < cutoff:
                continue

            repo_info = {
                "name": repo["name"],
                "full_name": repo["full_name"],
                "description": repo.get("description") or "",
                "url": repo["html_url"],
                "updated_at": repo["updated_at"][:10],
                "stars": repo["stargazers_count"],
                "language": repo.get("language"),
                "topics": repo.get("topics", []),
            }

            # Extract signals from description and topics
            repo_info["signals"] = self._extract_signals(repo_info)

            recent_repos.append(repo_info)

        return recent_repos

    def _extract_signals(self, repo: dict) -> list[str]:
        """Extract signal keywords from repo info.

        Args:
            repo: Repository info dict.

        Returns:
            List of matched signal keywords.
        """
        text = " ".join([
            repo.get("name", ""),
            repo.get("description", ""),
            " ".join(repo.get("topics", []))
        ]).lower()

        signals = []
        for keyword in SIGNAL_KEYWORDS:
            # Handle hyphenated keywords
            keyword_pattern = keyword.replace("-", "[- ]?")
            if re.search(keyword_pattern, text):
                signals.append(keyword)

        return list(set(signals))  # Deduplicate

    def get_repo_readme(self, full_name: str) -> Optional[str]:
        """Get README content for a repo.

        Args:
            full_name: Full repo name (org/repo).

        Returns:
            README content or None.
        """
        url = f"{self.BASE_URL}/repos/{full_name}/readme"

        # Request raw content
        headers = self.headers.copy()
        headers["Accept"] = "application/vnd.github.v3.raw"

        try:
            resp = requests.get(url, headers=headers, timeout=30)
            if resp.status_code == 200:
                return resp.text[:5000]  # Limit length
        except requests.RequestException:
            pass

        return None

    def get_org_activity(self, org_name: str, days: int = 7) -> dict:
        """Get complete activity summary for an organization.

        Args:
            org_name: GitHub organization name.
            days: Look back period in days.

        Returns:
            Activity summary dict.
        """
        repos = self.get_org_repos(org_name, days)

        # Filter to repos with signals or high stars
        relevant_repos = [
            r for r in repos
            if r["signals"] or r["stars"] >= 100
        ]

        return {
            "org": org_name,
            "repos_count": len(repos),
            "repos_updated": relevant_repos,
            "has_activity": len(relevant_repos) > 0
        }

    def fetch_vendor_activity(self, days: int = 7) -> list[dict]:
        """Fetch activity for all configured vendor organizations.

        Args:
            days: Look back period in days.

        Returns:
            List of org activity summaries.
        """
        results = []

        print(f"  Tracking {len(self.vendor_orgs)} vendor GitHub orgs...")

        for org in self.vendor_orgs:
            activity = self.get_org_activity(org, days)
            if activity["has_activity"]:
                results.append(activity)
            time.sleep(0.5)  # Rate limit courtesy

        return results

    def fetch_lab_activity(self, days: int = 7) -> list[dict]:
        """Fetch activity for all configured AI lab organizations.

        Args:
            days: Look back period in days.

        Returns:
            List of org activity summaries.
        """
        results = []

        print(f"  Tracking {len(self.lab_orgs)} lab GitHub orgs...")

        for org in self.lab_orgs:
            activity = self.get_org_activity(org, days)
            if activity["has_activity"]:
                results.append(activity)
            time.sleep(0.5)  # Rate limit courtesy

        return results

    def fetch_all_orgs(self, days: int = 7) -> dict:
        """Fetch activity for all configured organizations.

        Args:
            days: Look back period in days.

        Returns:
            Dict with vendor and lab activities.
        """
        return {
            "vendors": self.fetch_vendor_activity(days),
            "labs": self.fetch_lab_activity(days)
        }
