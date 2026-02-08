"""GitHub organization tracker for monitoring data vendor activities.

Monitors GitHub organizations for:
- Recently updated repositories
- Signal keywords in descriptions and READMEs
- Data annotation and RLHF related projects
"""

import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from typing import Optional

import requests

from utils.logging_config import get_logger

logger = get_logger(__name__)


# Default signal keywords for detecting relevant repos
DEFAULT_SIGNAL_KEYWORDS = [
    "rlhf",
    "reward",
    "preference",
    "annotation",
    "labeling",
    "label",
    "evaluation",
    "benchmark",
    "human-feedback",
    "sft",
    "instruction",
    "fine-tuning",
    "fine-tune",
    "data-quality",
    "dataset",
    "training-data",
    "llm",
    "language-model",
    "crowdsourcing",
    "human-in-the-loop",
    "active-learning",
    "data-collection",
    "synthetic-data",
    # Scaling & Data Quality
    "scaling-law",
    "data-curation",
    "data-filtering",
    "decontamination",
    "deduplication",
    "data-pipeline",
    # Methods & Techniques
    "dpo",
    "direct-preference",
    "chain-of-thought",
    "distillation",
    "curriculum-learning",
    "contrastive-learning",
    "data-augmentation",
    # Modalities & Use Cases
    "function-calling",
    "tool-use",
    "code-generation",
    "multimodal",
    "vision-language",
    "embodied",
    "robotics",
    "reward-model",
    "alignment",
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
        sources_github = config.get("sources", {}).get("github", {})

        # Get token from config or environment
        # Handle ${VAR} syntax in config that wasn't expanded
        token = github_config.get("token", "")
        if token and token.startswith("${") and token.endswith("}"):
            # Extract variable name and get from environment
            var_name = token[2:-1]
            token = os.environ.get(var_name, "")
        self.token = token or os.environ.get("GITHUB_TOKEN", "")

        self.headers = {
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "AI-Dataset-Radar/1.0",
        }
        # Only add Authorization header if we have a valid token
        if self.token and not self.token.startswith("${"):
            self.headers["Authorization"] = f"token {self.token}"

        # Use Session for connection pooling across requests
        self.session = requests.Session()
        self.session.headers.update(self.headers)

        # Get org lists from config
        self.vendor_orgs = github_config.get("orgs", {}).get("data_vendors", [])
        self.lab_orgs = github_config.get("orgs", {}).get("ai_labs", [])

        # Get relevance keywords from config (sources.github.relevance_keywords)
        self.relevance_keywords = (
            sources_github.get("relevance_keywords") or DEFAULT_SIGNAL_KEYWORDS
        )

    def _make_request(self, url: str, params: dict = None, max_retries: int = 3) -> Optional[dict]:
        """Make a GitHub API request with rate limit handling and retry."""
        for attempt in range(max_retries):
            try:
                resp = self.session.get(url, params=params, timeout=30)

                if resp.status_code == 403:
                    # Rate limited
                    reset_time = resp.headers.get("X-RateLimit-Reset")
                    if reset_time:
                        wait_time = max(int(reset_time) - int(time.time()), 1)
                        logger.info("  GitHub rate limited, reset in %ss", wait_time)
                    return None

                if resp.status_code == 404:
                    return None

                if resp.status_code >= 500:
                    # Server error — retry
                    if attempt < max_retries - 1:
                        wait = 2 ** (attempt + 1)
                        logger.warning("  GitHub %d, retry in %ds", resp.status_code, wait)
                        time.sleep(wait)
                        continue
                    return None

                resp.raise_for_status()
                return resp.json()

            except requests.RequestException as e:
                if attempt < max_retries - 1:
                    wait = 2 ** (attempt + 1)
                    logger.warning("  GitHub API error, retry in %ds: %s", wait, e)
                    time.sleep(wait)
                else:
                    logger.info("  GitHub API error (all retries failed): %s", e)
                    return None
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
        params = {"sort": "updated", "direction": "desc", "per_page": 30}

        data = self._make_request(url, params)
        if not data:
            return []

        cutoff = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=days)
        recent_repos = []

        for repo in data:
            updated_at = datetime.strptime(repo["updated_at"], "%Y-%m-%dT%H:%M:%SZ")

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

            # Extract relevance signals from description and topics
            relevance_signals = self._extract_relevance_signals(repo_info)
            repo_info["relevance_signals"] = relevance_signals
            repo_info["relevance"] = self._calculate_relevance(relevance_signals, repo_info)

            # Keep legacy signals field for backward compat
            repo_info["signals"] = self._extract_signals(repo_info)

            recent_repos.append(repo_info)

        return recent_repos

    def _extract_signals(self, repo: dict) -> list[str]:
        """Extract signal keywords from repo info (legacy method).

        Args:
            repo: Repository info dict.

        Returns:
            List of matched signal keywords.
        """
        text = " ".join(
            [repo.get("name", ""), repo.get("description", ""), " ".join(repo.get("topics", []))]
        ).lower()

        signals = []
        for keyword in DEFAULT_SIGNAL_KEYWORDS:
            # Handle hyphenated keywords
            keyword_pattern = keyword.replace("-", "[- ]?")
            if re.search(keyword_pattern, text):
                signals.append(keyword)

        return list(set(signals))  # Deduplicate

    def _extract_relevance_signals(self, repo: dict) -> list[str]:
        """Extract relevance signals using config keywords.

        Matches config relevance_keywords against repo name, description, and topics.
        Supports partial matching and case-insensitive comparison.

        Args:
            repo: Repository info dict.

        Returns:
            List of matched keywords.
        """
        name = (repo.get("name") or "").lower()
        description = (repo.get("description") or "").lower()
        topics = [t.lower() for t in repo.get("topics", [])]
        topics_text = " ".join(topics)

        matched = set()

        for keyword in self.relevance_keywords:
            kw_lower = keyword.lower()
            # Handle hyphenated keywords - match with or without hyphen
            kw_pattern = kw_lower.replace("-", "[- ]?")

            # Check in name
            if re.search(kw_pattern, name):
                matched.add(keyword)
                continue

            # Check in description
            if re.search(kw_pattern, description):
                matched.add(keyword)
                continue

            # Check in topics (exact or partial)
            if kw_lower in topics or re.search(kw_pattern, topics_text):
                matched.add(keyword)
                continue

        return sorted(list(matched))

    # Negative patterns — repos matching these are likely noise
    NEGATIVE_PATTERNS = [
        "example",
        "tutorial",
        "template",
        "demo",
        "starter",
        "awesome-",
        "mirror",
        "fork-of",
    ]

    def _calculate_relevance(self, signals: list[str], repo: dict = None) -> str:
        """Calculate relevance level using weighted scoring.

        Score = (keyword_matches × 10) + (stars / 100) + recency_boost
        - high: score >= 20
        - medium: score >= 5
        - low: score < 5

        Args:
            signals: List of matched keywords.
            repo: Optional repo dict for star/recency weighting.

        Returns:
            "high", "medium", or "low".
        """
        score = len(signals) * 10

        if repo:
            # Star weight
            score += repo.get("stars", 0) / 100

            # Recency boost: updated within last 3 days
            updated = repo.get("updated_at", "")
            if updated:
                try:
                    days_ago = (datetime.now(timezone.utc).replace(tzinfo=None) - datetime.strptime(updated, "%Y-%m-%d")).days
                    if days_ago <= 3:
                        score += 5
                except ValueError:
                    pass

            # Negative pattern penalty
            name = (repo.get("name") or "").lower()
            for neg in self.NEGATIVE_PATTERNS:
                if neg in name:
                    score -= 15
                    break

        if score >= 20:
            return "high"
        elif score >= 5:
            return "medium"
        else:
            return "low"

    def get_repo_readme(self, full_name: str) -> Optional[str]:
        """Get README content for a repo.

        Args:
            full_name: Full repo name (org/repo).

        Returns:
            README content or None.
        """
        url = f"{self.BASE_URL}/repos/{full_name}/readme"

        # Request raw content - use direct request with overridden Accept header
        try:
            resp = self.session.get(
                url, headers={"Accept": "application/vnd.github.v3.raw"}, timeout=30
            )
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
        relevant_repos = [r for r in repos if r["signals"] or r["stars"] >= 100]

        return {
            "org": org_name,
            "repos_count": len(repos),
            "repos_updated": relevant_repos,
            "has_activity": len(relevant_repos) > 0,
        }

    def _fetch_orgs_parallel(self, orgs: list[str], days: int) -> list[dict]:
        """Fetch activity for a list of orgs in parallel.

        Args:
            orgs: List of GitHub org names.
            days: Look back period in days.

        Returns:
            List of org activity summaries (only active ones).
        """
        results = []
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(self.get_org_activity, org, days): org for org in orgs}
            for future in futures:
                try:
                    activity = future.result()
                    if activity["has_activity"]:
                        results.append(activity)
                except Exception as e:
                    logger.warning("Error fetching GitHub org %s: %s", futures[future], e)
        return results

    def fetch_vendor_activity(self, days: int = 7) -> list[dict]:
        """Fetch activity for all configured vendor organizations (parallelized).

        Args:
            days: Look back period in days.

        Returns:
            List of org activity summaries.
        """
        logger.info("  Tracking %s vendor GitHub orgs...", len(self.vendor_orgs))
        return self._fetch_orgs_parallel(self.vendor_orgs, days)

    def fetch_lab_activity(self, days: int = 7) -> list[dict]:
        """Fetch activity for all configured AI lab organizations (parallelized).

        Args:
            days: Look back period in days.

        Returns:
            List of org activity summaries.
        """
        logger.info("  Tracking %s lab GitHub orgs...", len(self.lab_orgs))
        return self._fetch_orgs_parallel(self.lab_orgs, days)

    def fetch_all_orgs(self, days: int = 7) -> dict:
        """Fetch activity for all configured organizations (parallelized).

        All org categories are fetched concurrently.

        Args:
            days: Look back period in days.

        Returns:
            Dict with vendor and lab activities.
        """
        all_orgs = list(set(self.vendor_orgs + self.lab_orgs))
        logger.info("  Tracking %s total GitHub orgs...", len(all_orgs))

        all_results = self._fetch_orgs_parallel(all_orgs, days)

        # Split back into vendor vs lab
        vendor_set = set(self.vendor_orgs)
        lab_set = set(self.lab_orgs)

        return {
            "vendors": [r for r in all_results if r["org"] in vendor_set],
            "labs": [r for r in all_results if r["org"] in lab_set],
        }
