"""Reddit tracker for monitoring AI dataset discussions.

Monitors AI/ML subreddits for dataset-related posts using Reddit's
public JSON API (no authentication required).

Tracked signals: dataset releases, benchmark announcements, training data
discussions, RLHF/DPO/SFT resources, model releases with data components.
"""

import asyncio
import re
from datetime import datetime, timedelta, timezone

from utils.async_http import AsyncHTTPClient
from utils.logging_config import get_logger

logger = get_logger("reddit_tracker")

# Reuse same signal keywords as X tracker for consistency
SIGNAL_KEYWORDS = [
    "dataset", "training data", "benchmark", "evaluation",
    "open source", "open-source", "releasing", "we release",
    "announcing", "introducing", "new paper", "new model",
    "fine-tuning", "fine-tune", "rlhf", "human feedback",
    "instruction tuning", "alignment", "synthetic data",
    "pre-training", "pretraining", "language model", "llm",
    "multimodal", "vision-language", "reasoning",
    "scaling law", "data quality", "data curation", "decontamination",
    "dpo", "direct preference", "reward model",
    "distillation", "knowledge distillation",
    "function calling", "tool use", "code generation",
    "embodied", "robotics data", "agent data",
    "开源", "发布", "数据集", "模型", "论文", "训练", "合成数据", "标注",
]

DEFAULT_SUBREDDITS = [
    "MachineLearning",
    "LanguageTechnology",
    "LocalLLaMA",
    "dataset",
    "deeplearning",
    "artificial",
]


class RedditTracker:
    """Track Reddit subreddits for AI dataset discussions.

    Uses Reddit's public JSON API (no auth required).
    Rate limited to 1 request/second with proper User-Agent.
    """

    REDDIT_BASE = "https://www.reddit.com"
    USER_AGENT = "AI-Dataset-Radar/6.0 (competitive-intelligence; +https://github.com/liuxiaotong/ai-dataset-radar)"

    def __init__(self, config: dict, http_client: AsyncHTTPClient = None):
        """Initialize Reddit tracker.

        Args:
            config: Configuration dict with reddit_tracker settings.
            http_client: Optional shared AsyncHTTPClient instance.
        """
        self.config = config
        reddit_config = config.get("reddit_tracker", {})

        self.subreddits = reddit_config.get("subreddits", DEFAULT_SUBREDDITS)
        self.search_keywords = reddit_config.get("search_keywords", [
            "dataset release", "training data", "new benchmark",
        ])
        self.min_score = reddit_config.get("min_score", 5)

        self._http = http_client
        self._owns_http = http_client is None

    async def _ensure_http(self):
        """Lazily create HTTP client if not provided."""
        if self._http is None:
            self._http = AsyncHTTPClient()

    async def _fetch_subreddit(self, subreddit: str, days: int = 7) -> list[dict]:
        """Fetch recent posts from a subreddit.

        Args:
            subreddit: Subreddit name (without r/).
            days: Look back period in days.

        Returns:
            List of post dicts with signal filtering applied.
        """
        await self._ensure_http()

        url = f"{self.REDDIT_BASE}/r/{subreddit}/new.json"
        params = {"limit": 100, "raw_json": 1}
        headers = {"User-Agent": self.USER_AGENT}

        try:
            data = await self._http.get_json(url, params=params, headers=headers, max_retries=2)
        except Exception as e:
            logger.warning("Error fetching r/%s: %s", subreddit, e)
            return []

        if not data:
            return []

        cutoff = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=days)
        posts = []

        children = data.get("data", {}).get("children", [])
        for item in children:
            post_data = item.get("data", {})
            if not post_data:
                continue

            # Parse creation time (Reddit uses Unix timestamps)
            created_utc = post_data.get("created_utc", 0)
            if created_utc:
                try:
                    created = datetime.fromtimestamp(created_utc, tz=timezone.utc).replace(tzinfo=None)
                except (OSError, ValueError):
                    created = None
            else:
                created = None

            if created and created < cutoff:
                continue

            title = post_data.get("title", "").strip()
            selftext = post_data.get("selftext", "").strip()
            score = post_data.get("score", 0)
            permalink = post_data.get("permalink", "")

            post = {
                "subreddit": subreddit,
                "title": title,
                "selftext": selftext[:500],
                "url": f"https://www.reddit.com{permalink}" if permalink else "",
                "score": score,
                "num_comments": post_data.get("num_comments", 0),
                "date": created.strftime("%Y-%m-%d") if created else "",
                "author": post_data.get("author", ""),
            }

            post["signals"] = self._extract_signals(post)
            posts.append(post)

        return posts

    def _extract_signals(self, post: dict) -> list[str]:
        """Extract signal keywords from post title and body.

        Args:
            post: Post dict with title and selftext fields.

        Returns:
            List of matched signal keywords.
        """
        text = (post.get("title", "") + " " + post.get("selftext", "")).lower()
        signals = []
        for keyword in SIGNAL_KEYWORDS:
            if keyword.lower() in text:
                signals.append(keyword)
        return list(set(signals))

    def _parse_date_value(self, date_str: str | None) -> datetime | None:
        if not date_str:
            return None
        try:
            return datetime.strptime(date_str[:10], "%Y-%m-%d")
        except ValueError:
            return None

    async def fetch_all(
        self,
        days: int = 7,
        source_watermarks: dict[str, str] | None = None,
    ) -> dict:
        """Fetch posts from all configured subreddits concurrently.

        Args:
            days: Look back period in days.

        Returns:
            Dict with posts list and metadata.
        """
        await self._ensure_http()

        logger.info("Tracking %d subreddits...", len(self.subreddits))

        results = {
            "posts": [],
            "metadata": {
                "subreddits_checked": len(self.subreddits),
                "total_posts": 0,
                "relevant_posts": 0,
            },
        }

        if not self.subreddits:
            return results

        # Fetch subreddits concurrently with bounded parallelism
        sem = asyncio.Semaphore(5)  # Reddit rate limit ~60 req/min

        async def _bounded(sub):
            async with sem:
                # Small delay between requests for rate limiting
                await asyncio.sleep(0.5)
                return await self._fetch_subreddit(sub, days)

        raw_results = await asyncio.gather(
            *[_bounded(s) for s in self.subreddits], return_exceptions=True
        )

        all_posts = []
        for i, result in enumerate(raw_results):
            if isinstance(result, Exception):
                logger.warning("Error fetching r/%s: %s", self.subreddits[i], result)
            elif result:
                all_posts.extend(result)

        filtered_posts = []
        for post in all_posts:
            sub = post.get("subreddit", "")
            min_ts = source_watermarks.get(sub) if source_watermarks else None
            min_dt = self._parse_date_value(min_ts) if min_ts else None
            post_dt = self._parse_date_value(post.get("date"))
            if min_dt and post_dt and post_dt <= min_dt:
                continue
            filtered_posts.append(post)

        results["metadata"]["total_posts"] = len(filtered_posts)

        # Filter to relevant posts (have signals) and meet min score
        relevant = [
            p for p in filtered_posts
            if p.get("signals") and p.get("score", 0) >= self.min_score
        ]

        # Sort by score descending
        relevant.sort(key=lambda p: p.get("score", 0), reverse=True)

        results["posts"] = relevant
        results["metadata"]["relevant_posts"] = len(relevant)

        logger.info(
            "Found %d relevant posts from %d total (min_score=%d)",
            len(relevant), len(filtered_posts), self.min_score,
        )

        return results

    async def close(self):
        """Close the HTTP client if we own it."""
        if self._owns_http and self._http is not None:
            await self._http.close()
            self._http = None
