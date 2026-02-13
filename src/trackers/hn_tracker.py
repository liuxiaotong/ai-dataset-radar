"""Hacker News tracker for monitoring AI dataset discussions.

Uses the Algolia HN Search API (no authentication required) to find
AI/dataset-related stories and discussions on Hacker News.

Tracked signals: dataset releases, benchmark announcements, training data
discussions, RLHF resources, model releases with data components.
"""

import asyncio
from datetime import datetime, timedelta, timezone

from utils.async_http import AsyncHTTPClient
from utils.logging_config import get_logger

logger = get_logger("hn_tracker")

# Reuse same signal keywords as Reddit/X tracker for consistency
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
]

DEFAULT_SEARCH_QUERIES = [
    "dataset",
    "training data",
    "AI benchmark",
    "open source model",
    "RLHF",
    "LLM",
]


class HNTracker:
    """Track Hacker News for AI dataset discussions.

    Uses the Algolia HN Search API (free, no auth, no rate limits).
    """

    ALGOLIA_BASE = "https://hn.algolia.com/api/v1"

    def __init__(self, config: dict, http_client: AsyncHTTPClient = None):
        """Initialize HN tracker.

        Args:
            config: Configuration dict with hn_tracker settings.
            http_client: Optional shared AsyncHTTPClient instance.
        """
        self.config = config
        hn_config = config.get("hn_tracker", {})

        self.search_keywords = hn_config.get("search_keywords", DEFAULT_SEARCH_QUERIES)
        self.min_points = hn_config.get("min_points", 10)

        self._http = http_client
        self._owns_http = http_client is None

    async def _ensure_http(self):
        """Lazily create HTTP client if not provided."""
        if self._http is None:
            self._http = AsyncHTTPClient()

    async def _search_stories(self, query: str, days: int = 7) -> list[dict]:
        """Search HN stories via Algolia API.

        Args:
            query: Search query string.
            days: Look back period in days.

        Returns:
            List of story dicts.
        """
        await self._ensure_http()

        cutoff = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=days)
        cutoff_ts = int(cutoff.replace(tzinfo=timezone.utc).timestamp())

        url = f"{self.ALGOLIA_BASE}/search_by_date"
        params = {
            "query": query,
            "tags": "story",
            "numericFilters": f"created_at_i>{cutoff_ts}",
            "hitsPerPage": 50,
        }

        try:
            data = await self._http.get_json(url, params=params, max_retries=2)
        except Exception as e:
            logger.warning("Error searching HN for '%s': %s", query, e)
            return []

        if not data:
            return []

        stories = []
        for hit in data.get("hits", []):
            title = hit.get("title") or ""
            story_url = hit.get("url") or ""
            hn_url = f"https://news.ycombinator.com/item?id={hit.get('objectID', '')}"

            created_at = hit.get("created_at", "")
            try:
                created = datetime.fromisoformat(created_at.replace("Z", "+00:00")).replace(tzinfo=None)
            except (ValueError, AttributeError):
                created = None

            story = {
                "title": title.strip(),
                "url": story_url,
                "hn_url": hn_url,
                "author": hit.get("author", ""),
                "points": hit.get("points") or 0,
                "num_comments": hit.get("num_comments") or 0,
                "date": created.strftime("%Y-%m-%d") if created else "",
                "objectID": hit.get("objectID", ""),
            }

            story["signals"] = self._extract_signals(story)
            stories.append(story)

        return stories

    def _extract_signals(self, story: dict) -> list[str]:
        """Extract signal keywords from story title.

        Args:
            story: Story dict with title field.

        Returns:
            List of matched signal keywords.
        """
        text = story.get("title", "").lower()
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
        """Search HN for all configured keywords concurrently.

        Args:
            days: Look back period in days.
            source_watermarks: Optional dict of keyword -> last seen date.

        Returns:
            Dict with stories list and metadata.
        """
        await self._ensure_http()

        logger.info("Searching %d HN keywords...", len(self.search_keywords))

        results = {
            "stories": [],
            "metadata": {
                "keywords_searched": len(self.search_keywords),
                "total_stories": 0,
                "relevant_stories": 0,
            },
        }

        if not self.search_keywords:
            return results

        # Search keywords concurrently with bounded parallelism
        sem = asyncio.Semaphore(5)

        async def _bounded(query):
            async with sem:
                return await self._search_stories(query, days)

        raw_results = await asyncio.gather(
            *[_bounded(q) for q in self.search_keywords], return_exceptions=True
        )

        # Deduplicate by objectID
        seen_ids = set()
        all_stories = []
        for result in raw_results:
            if isinstance(result, Exception):
                logger.warning("HN search error: %s", result)
                continue
            for story in (result or []):
                oid = story.get("objectID")
                if oid and oid not in seen_ids:
                    seen_ids.add(oid)
                    all_stories.append(story)

        # Apply watermark filtering
        global_wm = self._parse_date_value(
            source_watermarks.get("hn") if source_watermarks else None
        )
        if global_wm:
            all_stories = [
                s for s in all_stories
                if not self._parse_date_value(s.get("date"))
                or self._parse_date_value(s.get("date")) > global_wm
            ]

        results["metadata"]["total_stories"] = len(all_stories)

        # Filter by min_points and having signals
        relevant = [
            s for s in all_stories
            if s.get("signals") and s.get("points", 0) >= self.min_points
        ]

        # Sort by points descending
        relevant.sort(key=lambda s: s.get("points", 0), reverse=True)

        results["stories"] = relevant
        results["metadata"]["relevant_stories"] = len(relevant)

        logger.info(
            "Found %d relevant HN stories from %d total (min_points=%d)",
            len(relevant), len(all_stories), self.min_points,
        )

        return results

    async def close(self):
        """Close the HTTP client if we own it."""
        if self._owns_http and self._http is not None:
            await self._http.close()
            self._http = None
