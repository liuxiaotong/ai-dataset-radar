"""GitHub Trending tracker.

Monitors trending GitHub repositories for AI/dataset signals
using the unofficial ghapi.huchen.dev API (no auth required).
"""

from datetime import datetime, timezone
from typing import Optional

from utils.async_http import AsyncHTTPClient
from utils.logging_config import get_logger

logger = get_logger(__name__)

SIGNAL_KEYWORDS = [
    "dataset", "benchmark", "training data", "fine-tune", "finetune",
    "llm", "language model", "nlp", "rlhf", "instruction",
    "synthetic data", "alignment", "reward model", "evaluation",
    "embedding", "tokenizer", "corpus", "annotation", "labeling",
    "multimodal", "vision-language", "diffusion", "speech",
]


class GitHubTrendingTracker:
    """Tracks trending GitHub repos with AI/dataset relevance."""

    API_URL = "https://ghapi.huchen.dev/repositories"

    def __init__(self, config: dict = None, http_client: AsyncHTTPClient = None):
        config = config or {}
        cfg = config.get("github_trending", {})
        self.enabled = cfg.get("enabled", True)
        self.languages = cfg.get("languages", ["python", "jupyter-notebook", ""])
        self.since = cfg.get("since", "weekly")
        self.min_stars = cfg.get("min_stars", 10)
        self._http = http_client
        self._owns_http = http_client is None
        if self._owns_http:
            self._http = AsyncHTTPClient()

    async def fetch_all(
        self,
        days: int = 7,
        source_watermarks: dict = None,
    ) -> dict:
        """Fetch trending repos and filter for AI/dataset signals.

        Args:
            days: Not directly used (GitHub Trending has its own periods).
            source_watermarks: Optional watermarks dict.

        Returns:
            Dict with "repos" and "metadata".
        """
        if not self.enabled:
            return {"repos": [], "metadata": {"languages_searched": 0}}

        seen_urls = set()
        all_repos = []

        for lang in self.languages:
            try:
                params = {"since": self.since}
                if lang:
                    params["language"] = lang
                data = await self._http.get_json(
                    self.API_URL, params=params, max_retries=2,
                )
            except Exception as e:
                logger.warning("GitHub Trending error for lang='%s': %s", lang, e)
                continue

            if not data or not isinstance(data, list):
                continue

            for repo in data:
                url = repo.get("url", "")
                if not url or url in seen_urls:
                    continue
                seen_urls.add(url)

                stars = repo.get("stars", 0) or 0
                if stars < self.min_stars:
                    continue

                signals = self._extract_signals(repo)
                if not signals:
                    continue

                all_repos.append({
                    "author": repo.get("author", ""),
                    "name": repo.get("name", ""),
                    "url": url,
                    "description": repo.get("description") or "",
                    "language": repo.get("language") or lang,
                    "stars": stars,
                    "forks": repo.get("forks", 0) or 0,
                    "currentPeriodStars": repo.get("currentPeriodStars", 0) or 0,
                    "signals": signals,
                    "date": datetime.now(timezone.utc).replace(tzinfo=None).strftime("%Y-%m-%d"),
                })

        # Sort by current period stars descending
        all_repos.sort(key=lambda r: r.get("currentPeriodStars", 0), reverse=True)

        logger.info(
            "GitHub Trending: %d relevant repos from %d languages",
            len(all_repos), len(self.languages),
        )
        return {
            "repos": all_repos,
            "metadata": {
                "languages_searched": len(self.languages),
                "total_repos": len(all_repos),
            },
        }

    def _extract_signals(self, repo: dict) -> list[str]:
        """Extract AI/dataset signal keywords from repo name + description."""
        text = " ".join([
            repo.get("name", ""),
            repo.get("description") or "",
        ]).lower()
        return [kw for kw in SIGNAL_KEYWORDS if kw in text]

    async def close(self):
        if self._owns_http and self._http:
            await self._http.close()
            self._http = None
