"""Blog and RSS feed tracker for monitoring company announcements.

Monitors company blogs for:
- New articles about products, research, datasets
- Signal keywords related to AI data and annotation
"""

import re
import time
from datetime import datetime, timedelta
from typing import Optional
from urllib.parse import urljoin

import feedparser
import requests
from bs4 import BeautifulSoup


# Signal keywords for detecting relevant articles
SIGNAL_KEYWORDS = [
    "rlhf", "human feedback", "preference", "annotation", "labeling",
    "data quality", "evaluation", "benchmark", "dataset", "training data",
    "fine-tuning", "instruction", "crowdsourcing", "data collection",
    "synthetic data", "reward model", "alignment", "llm", "language model",
    "product launch", "release", "announcing", "introducing"
]


class BlogTracker:
    """Track blog and RSS feed updates."""

    def __init__(self, config: dict):
        """Initialize blog tracker.

        Args:
            config: Configuration dict with blog settings.
        """
        self.config = config
        # Check multiple possible config locations
        self.blogs = (
            config.get("data_vendors", {}).get("blogs", []) or
            config.get("watched_vendors", {}).get("blogs", []) or
            config.get("blogs", [])
        )

        self.headers = {
            "User-Agent": "Mozilla/5.0 (compatible; AI-Dataset-Radar/1.0)"
        }

    def fetch_rss(self, url: str, days: int = 7) -> list[dict]:
        """Parse RSS feed and extract recent articles.

        Args:
            url: RSS feed URL.
            days: Look back period in days.

        Returns:
            List of article dicts.
        """
        try:
            feed = feedparser.parse(url)
        except Exception as e:
            print(f"    RSS parse error for {url}: {e}")
            return []

        cutoff = datetime.utcnow() - timedelta(days=days)
        articles = []

        for entry in feed.entries[:20]:  # Limit entries
            # Parse published date
            published = None
            if hasattr(entry, "published_parsed") and entry.published_parsed:
                published = datetime(*entry.published_parsed[:6])
            elif hasattr(entry, "updated_parsed") and entry.updated_parsed:
                published = datetime(*entry.updated_parsed[:6])

            if published and published < cutoff:
                continue

            article = {
                "title": entry.get("title", ""),
                "url": entry.get("link", ""),
                "date": published.strftime("%Y-%m-%d") if published else "",
                "summary": self._clean_html(
                    entry.get("summary", "") or entry.get("description", "")
                )[:300],
            }

            article["signals"] = self._extract_signals(article)
            articles.append(article)

        return articles

    def scrape_blog(
        self, url: str, selector: str, days: int = 7
    ) -> list[dict]:
        """Scrape blog page for articles.

        Args:
            url: Blog page URL.
            selector: CSS selector for article elements.
            days: Look back period in days.

        Returns:
            List of article dicts.
        """
        try:
            resp = requests.get(url, headers=self.headers, timeout=30)
            resp.raise_for_status()
        except requests.RequestException as e:
            print(f"    Scrape error for {url}: {e}")
            return []

        soup = BeautifulSoup(resp.text, "html.parser")
        articles = []

        # Try to find articles with the selector
        elements = soup.select(selector)[:10]

        for elem in elements:
            # Extract title
            title_elem = elem.select_one("h1, h2, h3, .title, [class*='title']")
            title = title_elem.get_text(strip=True) if title_elem else ""

            # Extract link
            link_elem = elem.select_one("a[href]") or elem.find_parent("a")
            if link_elem:
                link = link_elem.get("href", "")
                if link and not link.startswith("http"):
                    link = urljoin(url, link)
            else:
                link = ""

            # Extract summary
            summary_elem = elem.select_one(
                "p, .summary, .excerpt, [class*='excerpt'], [class*='description']"
            )
            summary = summary_elem.get_text(strip=True)[:300] if summary_elem else ""

            # Extract date if available
            date_elem = elem.select_one(
                "time, .date, [class*='date'], [datetime]"
            )
            date = ""
            if date_elem:
                date = date_elem.get("datetime", "") or date_elem.get_text(strip=True)
                date = self._parse_date(date)

            if title and link:
                article = {
                    "title": title,
                    "url": link,
                    "date": date,
                    "summary": summary,
                }
                article["signals"] = self._extract_signals(article)
                articles.append(article)

        return articles

    def _clean_html(self, text: str) -> str:
        """Remove HTML tags from text."""
        soup = BeautifulSoup(text, "html.parser")
        return soup.get_text(strip=True)

    def _parse_date(self, date_str: str) -> str:
        """Try to parse date string into YYYY-MM-DD format."""
        if not date_str:
            return ""

        # Try common formats
        formats = [
            "%Y-%m-%d",
            "%Y-%m-%dT%H:%M:%S",
            "%B %d, %Y",
            "%b %d, %Y",
            "%d %B %Y",
            "%d %b %Y",
        ]

        for fmt in formats:
            try:
                dt = datetime.strptime(date_str.strip()[:20], fmt)
                return dt.strftime("%Y-%m-%d")
            except ValueError:
                continue

        return ""

    def _extract_signals(self, article: dict) -> list[str]:
        """Extract signal keywords from article.

        Args:
            article: Article dict with title and summary.

        Returns:
            List of matched signal keywords.
        """
        text = " ".join([
            article.get("title", ""),
            article.get("summary", "")
        ]).lower()

        signals = []
        for keyword in SIGNAL_KEYWORDS:
            if keyword in text:
                signals.append(keyword)

        return list(set(signals))

    def fetch_blog(self, blog_config: dict, days: int = 7) -> dict:
        """Fetch articles from a single blog.

        Args:
            blog_config: Blog configuration dict.
            days: Look back period in days.

        Returns:
            Blog activity dict.
        """
        name = blog_config.get("name", "Unknown")
        url = blog_config.get("url", "")
        blog_type = blog_config.get("type", "rss")

        if blog_type == "rss":
            articles = self.fetch_rss(url, days)
        else:
            selector = blog_config.get("selector", "article")
            articles = self.scrape_blog(url, selector, days)

        # Filter to articles with signals
        relevant = [a for a in articles if a.get("signals")]

        return {
            "source": name,
            "url": url,
            "articles": relevant,
            "total_articles": len(articles),
            "has_activity": len(relevant) > 0
        }

    def fetch_all_blogs(self, days: int = 7) -> list[dict]:
        """Fetch articles from all configured blogs.

        Args:
            days: Look back period in days.

        Returns:
            List of blog activity dicts.
        """
        results = []

        print(f"  Tracking {len(self.blogs)} company blogs...")

        for blog_config in self.blogs:
            print(f"    Checking {blog_config.get('name', 'Unknown')}...")
            activity = self.fetch_blog(blog_config, days)
            if activity["has_activity"] or activity["total_articles"] > 0:
                results.append(activity)
            time.sleep(1)  # Politeness delay

        return results


# Mapping from blog source name to vendor key
BLOG_VENDOR_MAP = {
    "Scale AI": "scale_ai",
    "Snorkel AI": "snorkel_ai",
    "Argilla": "argilla",
    "Labelbox": "labelbox",
    "Anthropic Research": "anthropic",
    "OpenAI": "openai",
}


def map_blog_to_vendor(source_name: str) -> str:
    """Map blog source name to vendor key.

    Args:
        source_name: Blog source display name.

    Returns:
        Vendor key for aggregation.
    """
    return BLOG_VENDOR_MAP.get(source_name, source_name.lower().replace(" ", "_"))
