"""RSS/Atom feed scraper for monitoring blogs.

Fetches and parses RSS/Atom feeds from configured sources,
with URL-based deduplication.
"""

import re
import time
from datetime import datetime, timedelta
from typing import Optional
from urllib.parse import urlparse, urlunparse, parse_qs, urlencode

import feedparser
import requests
from bs4 import BeautifulSoup

from .base import BaseScraper
from .registry import register_scraper

from utils.logging_config import get_logger

logger = get_logger(__name__)


@register_scraper("blog_rss")
class BlogRSSScraper(BaseScraper):
    """Scraper for RSS/Atom blog feeds."""

    name = "blog_rss"
    source_type = "blog"

    # Signal keywords for detecting relevant articles
    SIGNAL_KEYWORDS = [
        "rlhf", "human feedback", "preference", "annotation", "labeling",
        "data quality", "evaluation", "benchmark", "dataset", "training data",
        "fine-tuning", "instruction", "crowdsourcing", "data collection",
        "synthetic data", "reward model", "alignment", "llm", "language model",
        "product launch", "release", "announcing", "introducing"
    ]

    def __init__(self, config: dict = None):
        """Initialize the RSS scraper.

        Args:
            config: Configuration with feeds list.
                    Each feed: {name: str, url: str}
        """
        super().__init__(config)
        self.feeds = self.config.get("feeds", [])
        self.headers = {
            "User-Agent": "Mozilla/5.0 (compatible; AI-Dataset-Radar/1.0)"
        }

    def scrape(self, config: dict = None) -> list[dict]:
        """Scrape articles from all configured feeds.

        Args:
            config: Optional runtime config with feeds override.

        Returns:
            List of article dictionaries, deduplicated by URL.
        """
        runtime_config = config or {}
        feeds = runtime_config.get("feeds") or self.feeds
        days = runtime_config.get("days", 7)

        if not feeds:
            logger.info("  No RSS feeds configured")
            return []

        all_articles = []
        for feed_config in feeds:
            articles = self.fetch_feed(feed_config, days)
            all_articles.extend(articles)
            time.sleep(0.5)  # Politeness delay

        # Deduplicate by normalized URL
        return self.deduplicate(all_articles)

    def fetch_feed(self, feed_config: dict, days: int = 7) -> list[dict]:
        """Fetch articles from a single RSS/Atom feed.

        Args:
            feed_config: Feed configuration {name, url}.
            days: Look back period in days.

        Returns:
            List of article dictionaries.
        """
        name = feed_config.get("name", "Unknown")
        url = feed_config.get("url", "")

        if not url:
            return []

        try:
            feed = feedparser.parse(url)
        except Exception as e:
            logger.info("    RSS parse error for %s: %s", name, e)
            return []

        if feed.bozo and feed.bozo_exception:
            # Feed had parsing issues but may still have entries
            pass

        cutoff = datetime.utcnow() - timedelta(days=days)
        articles = []

        for entry in feed.entries[:30]:  # Limit entries per feed
            article = self._parse_entry(entry, name, url)
            if not article:
                continue

            # Filter by date if available
            published = article.get("published_dt")
            if published and published < cutoff:
                continue

            # Extract signals
            article["signals"] = self._extract_signals(article)
            articles.append(article)

        return articles

    def _parse_entry(
        self,
        entry: dict,
        source_name: str,
        feed_url: str
    ) -> Optional[dict]:
        """Parse a feed entry into article dict.

        Args:
            entry: feedparser entry object.
            source_name: Name of the source blog.
            feed_url: URL of the feed.

        Returns:
            Article dictionary or None.
        """
        try:
            # Parse published date
            published = None
            published_str = ""
            if hasattr(entry, "published_parsed") and entry.published_parsed:
                published = datetime(*entry.published_parsed[:6])
                published_str = published.strftime("%Y-%m-%d")
            elif hasattr(entry, "updated_parsed") and entry.updated_parsed:
                published = datetime(*entry.updated_parsed[:6])
                published_str = published.strftime("%Y-%m-%d")

            # Get article URL
            article_url = entry.get("link", "")

            # Get summary/description
            summary = entry.get("summary", "") or entry.get("description", "")
            summary = self._clean_html(summary)[:500]

            return {
                "source": "blog_rss",
                "source_name": source_name,
                "id": article_url,
                "title": entry.get("title", "").strip(),
                "url": article_url,
                "date": published_str,
                "published_dt": published,
                "summary": summary,
                "author": entry.get("author", ""),
            }
        except Exception as e:
            logger.info("    Error parsing entry: %s", e)
            return None

    def _clean_html(self, text: str) -> str:
        """Remove HTML tags from text.

        Args:
            text: HTML content.

        Returns:
            Clean text.
        """
        if not text:
            return ""
        soup = BeautifulSoup(text, "html.parser")
        return soup.get_text(strip=True)

    def _extract_signals(self, article: dict) -> list[str]:
        """Extract signal keywords from article.

        Args:
            article: Article dictionary.

        Returns:
            List of matched keywords.
        """
        text = " ".join([
            article.get("title", ""),
            article.get("summary", "")
        ]).lower()

        signals = []
        for keyword in self.SIGNAL_KEYWORDS:
            if keyword.lower() in text:
                signals.append(keyword)

        return list(set(signals))

    def _get_unique_key(self, item: dict) -> str:
        """Get unique key for deduplication (normalized URL).

        Args:
            item: Article dictionary.

        Returns:
            Normalized URL as unique key.
        """
        url = item.get("url", "")
        return self._normalize_url(url)

    def _normalize_url(self, url: str) -> str:
        """Normalize URL for deduplication.

        Removes trailing slashes, fragments, and common tracking params.

        Args:
            url: URL to normalize.

        Returns:
            Normalized URL.
        """
        if not url:
            return ""

        try:
            parsed = urlparse(url)

            # Remove fragment
            parsed = parsed._replace(fragment="")

            # Remove tracking query params
            tracking_params = {
                "utm_source", "utm_medium", "utm_campaign",
                "utm_content", "utm_term", "ref", "source"
            }
            if parsed.query:
                params = parse_qs(parsed.query, keep_blank_values=True)
                filtered = {
                    k: v for k, v in params.items()
                    if k.lower() not in tracking_params
                }
                new_query = urlencode(filtered, doseq=True) if filtered else ""
                parsed = parsed._replace(query=new_query)

            # Normalize path (remove trailing slash)
            path = parsed.path.rstrip("/") or "/"
            parsed = parsed._replace(path=path)

            return urlunparse(parsed).lower()
        except Exception:
            return url.lower()

    def fetch(self) -> list[dict]:
        """Fetch articles (alias for scrape for backward compatibility).

        Returns:
            List of article dictionaries.
        """
        return self.scrape()
