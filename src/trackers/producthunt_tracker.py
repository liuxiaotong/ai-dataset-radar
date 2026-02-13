"""Product Hunt tracker.

Monitors Product Hunt for AI/dataset-related product launches
via the public RSS feed (no auth required).
"""

from datetime import datetime, timedelta, timezone

import feedparser

from utils.async_http import AsyncHTTPClient
from utils.logging_config import get_logger

logger = get_logger(__name__)

SIGNAL_KEYWORDS = [
    "dataset", "training data", "fine-tune", "finetune",
    "llm", "language model", "ai", "machine learning",
    "nlp", "synthetic data", "annotation", "labeling",
    "embedding", "vector", "rag", "agent", "copilot",
    "automation", "data pipeline", "data engineering",
]

FEED_URL = "https://www.producthunt.com/feed"


class ProductHuntTracker:
    """Tracks AI/dataset-related launches on Product Hunt."""

    def __init__(self, config: dict = None, http_client: AsyncHTTPClient = None):
        config = config or {}
        cfg = config.get("producthunt_tracker", {})
        self.enabled = cfg.get("enabled", True)
        self.search_keywords = cfg.get(
            "search_keywords",
            ["dataset", "llm", "ai", "machine learning", "nlp", "synthetic data"],
        )
        self._http = http_client
        self._owns_http = http_client is None
        if self._owns_http:
            self._http = AsyncHTTPClient()

    async def fetch_all(
        self,
        days: int = 7,
        source_watermarks: dict = None,
    ) -> dict:
        """Fetch Product Hunt feed and filter for AI/dataset signals.

        Args:
            days: Look back period in days.
            source_watermarks: Optional watermarks dict.

        Returns:
            Dict with "products" and "metadata".
        """
        if not self.enabled:
            return {"products": [], "metadata": {"total_products": 0}}

        cutoff = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=days)

        # Apply watermark if available
        wm_str = (source_watermarks or {}).get("producthunt")
        if wm_str:
            try:
                wm_dt = datetime.fromisoformat(wm_str.replace("Z", "+00:00")).replace(tzinfo=None)
                if wm_dt > cutoff:
                    cutoff = wm_dt
            except (ValueError, AttributeError):
                pass

        try:
            text = await self._http.get_text(FEED_URL, max_retries=2)
        except Exception as e:
            logger.warning("Product Hunt feed error: %s", e)
            return {"products": [], "metadata": {"total_products": 0}}

        feed = feedparser.parse(text)
        if not feed.entries:
            return {"products": [], "metadata": {"total_products": 0}}

        products = []
        for entry in feed.entries:
            # Parse date
            published = entry.get("published_parsed") or entry.get("updated_parsed")
            if published:
                dt = datetime(*published[:6])
            else:
                continue

            if dt < cutoff:
                continue

            title = entry.get("title", "")
            summary = entry.get("summary", "")
            signals = self._extract_signals(title, summary)
            if not signals:
                continue

            products.append({
                "title": title,
                "tagline": summary[:300] if summary else "",
                "url": entry.get("link", ""),
                "author": entry.get("author", ""),
                "date": dt.strftime("%Y-%m-%d"),
                "signals": signals,
            })

        logger.info("Product Hunt: %d relevant products", len(products))
        return {
            "products": products,
            "metadata": {
                "total_products": len(products),
            },
        }

    def _extract_signals(self, title: str, description: str) -> list[str]:
        """Extract AI/dataset signal keywords from title + description."""
        text = f"{title} {description}".lower()
        return [kw for kw in SIGNAL_KEYWORDS if kw in text]

    async def close(self):
        if self._owns_http and self._http:
            await self._http.close()
            self._http = None
