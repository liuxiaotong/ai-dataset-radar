"""Blog and RSS feed tracker for monitoring company announcements.

Monitors company blogs for:
- New articles about products, research, datasets
- Signal keywords related to AI data and annotation
"""

import re
import time
from datetime import datetime, timedelta
from typing import Optional
from urllib.parse import urljoin, urlparse, urlunparse, parse_qs, urlencode

import feedparser
import requests
from bs4 import BeautifulSoup


# Signal keywords for detecting relevant articles
SIGNAL_KEYWORDS = [
    # Data & Training
    "rlhf", "human feedback", "preference", "annotation", "labeling",
    "data quality", "evaluation", "benchmark", "dataset", "training data",
    "fine-tuning", "instruction", "crowdsourcing", "data collection",
    "synthetic data", "reward model", "alignment", "llm", "language model",
    # Research announcements
    "product launch", "release", "announcing", "introducing",
    # AI Research topics (Chinese labs)
    "reasoning", "context", "learning", "model", "multimodal", "vision",
    "agent", "tool", "code", "math", "science", "knowledge",
    "open source", "开源", "发布", "数据集", "训练", "模型",
]

# Common RSS feed paths to try
RSS_PATHS = [
    "/feed",
    "/feed.xml",
    "/rss",
    "/rss.xml",
    "/atom.xml",
    "/blog/feed",
    "/blog/feed.xml",
    "/blog/rss",
    "/blog/rss.xml",
    "/index.xml",
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
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }

    def _discover_rss_feed(self, base_url: str) -> Optional[str]:
        """Try to discover RSS feed URL for a blog.

        Args:
            base_url: Blog base URL.

        Returns:
            RSS feed URL if found, None otherwise.
        """
        parsed = urlparse(base_url)
        base = f"{parsed.scheme}://{parsed.netloc}"

        # Try common RSS paths
        for path in RSS_PATHS:
            feed_url = base + path
            try:
                resp = requests.head(feed_url, headers=self.headers, timeout=5, allow_redirects=True)
                if resp.status_code == 200:
                    content_type = resp.headers.get("Content-Type", "")
                    if any(t in content_type for t in ["xml", "rss", "atom", "feed"]):
                        return feed_url
                    # Try GET to check content
                    resp = requests.get(feed_url, headers=self.headers, timeout=5)
                    if resp.status_code == 200 and ("<?xml" in resp.text[:100] or "<rss" in resp.text[:200] or "<feed" in resp.text[:200]):
                        return feed_url
            except requests.RequestException:
                continue

        # Try to find feed link in HTML
        try:
            resp = requests.get(base_url, headers=self.headers, timeout=10)
            if resp.status_code == 200:
                soup = BeautifulSoup(resp.text, "html.parser")
                # Look for RSS/Atom link tags
                for link in soup.find_all("link", rel=["alternate", "feed"]):
                    link_type = link.get("type", "")
                    if "rss" in link_type or "atom" in link_type or "xml" in link_type:
                        href = link.get("href", "")
                        if href:
                            if not href.startswith("http"):
                                href = urljoin(base_url, href)
                            return href
        except requests.RequestException:
            pass

        return None

    def fetch_rss(self, url: str, days: int = 7) -> tuple[list[dict], Optional[str]]:
        """Parse RSS feed and extract recent articles.

        Args:
            url: RSS feed URL.
            days: Look back period in days.

        Returns:
            Tuple of (articles list, error message or None).
        """
        try:
            feed = feedparser.parse(url)
        except Exception as e:
            return [], f"RSS parse error: {e}"

        if feed.bozo and feed.bozo_exception:
            # Check if we got any entries despite the error
            if not feed.entries:
                return [], f"Feed error: {feed.bozo_exception}"

        if not feed.entries:
            return [], "No entries found in feed"

        cutoff = datetime.utcnow() - timedelta(days=days)
        articles = []

        for entry in feed.entries[:20]:  # Limit entries
            # Parse published date
            published = None
            if hasattr(entry, "published_parsed") and entry.published_parsed:
                try:
                    published = datetime(*entry.published_parsed[:6])
                except (TypeError, ValueError):
                    pass
            if not published and hasattr(entry, "updated_parsed") and entry.updated_parsed:
                try:
                    published = datetime(*entry.updated_parsed[:6])
                except (TypeError, ValueError):
                    pass

            # Skip articles outside the time window
            if published and published < cutoff:
                continue

            # Get title
            title = entry.get("title", "").strip()
            if not title:
                continue

            # Get URL
            article_url = entry.get("link", "")
            if not article_url:
                continue

            # Get summary - prefer summary over description, clean HTML
            summary_raw = entry.get("summary", "") or entry.get("description", "") or ""
            summary = self._clean_html(summary_raw)
            # Validate summary is not just a date or navigation text
            summary = self._validate_summary(summary, title)

            article = {
                "title": title,
                "url": article_url,
                "date": published.strftime("%Y-%m-%d") if published else "",
                "summary": summary[:300] if summary else "",
            }

            article["signals"] = self._extract_signals(article)
            articles.append(article)

        return articles, None

    def _validate_summary(self, summary: str, title: str) -> str:
        """Validate and clean summary text.

        Args:
            summary: Raw summary text.
            title: Article title for comparison.

        Returns:
            Cleaned summary or empty string if invalid.
        """
        if not summary:
            return ""

        # Remove if it's just a date
        date_patterns = [
            r"^\w+\s+\d{1,2},?\s+\d{4}$",  # "January 15, 2024"
            r"^\d{4}-\d{2}-\d{2}$",  # "2024-01-15"
            r"^\d{1,2}/\d{1,2}/\d{4}$",  # "01/15/2024"
        ]
        for pattern in date_patterns:
            if re.match(pattern, summary.strip()):
                return ""

        # Remove if it's navigation/menu text
        nav_indicators = [
            "github", "hugging face", "modelscope", "demo", "discord",
            "navigation", "menu", "home", "about", "contact",
            "skip to content", "toggle menu"
        ]
        summary_lower = summary.lower()
        nav_matches = sum(1 for ind in nav_indicators if ind in summary_lower)
        if nav_matches >= 3 or (len(summary) < 50 and nav_matches >= 2):
            return ""

        # Remove if summary is same as title
        if summary.strip().lower() == title.strip().lower():
            return ""

        return summary

    def scrape_blog(
        self, url: str, selector: str, days: int = 7
    ) -> tuple[list[dict], Optional[str]]:
        """Scrape blog page for articles.

        Args:
            url: Blog page URL.
            selector: CSS selector for article elements.
            days: Look back period in days.

        Returns:
            Tuple of (articles list, error message or None).
        """
        try:
            resp = requests.get(url, headers=self.headers, timeout=30)
            resp.raise_for_status()
        except requests.RequestException as e:
            return [], f"HTTP error: {e}"

        soup = BeautifulSoup(resp.text, "html.parser")
        articles = []
        cutoff = datetime.utcnow() - timedelta(days=days)

        # Try to find articles with the selector
        elements = soup.select(selector)[:15]

        if not elements:
            return [], f"No elements found with selector: {selector}"

        for elem in elements:
            # Extract title - try multiple approaches
            title = ""
            for title_sel in ["h1", "h2", "h3", ".title", "[class*='title']", "a"]:
                title_elem = elem.select_one(title_sel)
                if title_elem:
                    title = title_elem.get_text(strip=True)
                    if title and len(title) > 10:  # Skip very short titles
                        break

            if not title:
                continue

            # Extract link
            link = ""
            link_elem = elem.select_one("a[href]")
            if not link_elem:
                link_elem = elem.find_parent("a")
            if link_elem:
                link = link_elem.get("href", "")
                if link and not link.startswith("http"):
                    link = urljoin(url, link)

            if not link:
                continue

            # Extract date
            date_str = ""
            date_elem = elem.select_one("time, .date, [class*='date'], [datetime]")
            if date_elem:
                date_str = date_elem.get("datetime", "") or date_elem.get_text(strip=True)
                date_str = self._parse_date(date_str)

            # Filter by date if we have one
            if date_str:
                try:
                    article_date = datetime.strptime(date_str, "%Y-%m-%d")
                    if article_date < cutoff:
                        continue
                except ValueError:
                    pass

            # Extract summary
            summary = ""
            for sum_sel in ["p", ".summary", ".excerpt", "[class*='excerpt']", "[class*='description']"]:
                sum_elem = elem.select_one(sum_sel)
                if sum_elem:
                    summary = sum_elem.get_text(strip=True)
                    summary = self._validate_summary(summary, title)
                    if summary:
                        break

            article = {
                "title": title,
                "url": link,
                "date": date_str,
                "summary": summary[:300] if summary else "",
            }

            article["signals"] = self._extract_signals(article)
            articles.append(article)

        if not articles:
            return [], "No valid articles extracted"

        return articles, None

    def scrape_with_browser(
        self, url: str, selector: str, days: int = 7
    ) -> tuple[list[dict], Optional[str]]:
        """Scrape JavaScript-rendered page using Playwright.

        Args:
            url: Page URL.
            selector: CSS selector for article elements.
            days: Look back period in days.

        Returns:
            Tuple of (articles list, error message or None).
        """
        try:
            from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout
        except ImportError:
            return [], "Playwright not installed. Run: pip install playwright && playwright install chromium"

        articles = []
        cutoff = datetime.utcnow() - timedelta(days=days)

        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                page = browser.new_page()
                page.set_default_timeout(30000)

                # Navigate and wait for content
                page.goto(url, wait_until="networkidle")
                page.wait_for_timeout(2000)  # Extra wait for dynamic content

                # Find all article elements
                elements = page.query_selector_all(selector)
                if not elements:
                    browser.close()
                    return [], f"No elements found with selector: {selector}"

                # First pass: collect metadata from list page
                items_data = []
                for i, elem in enumerate(elements[:15]):
                    # Extract title
                    title = ""
                    for title_sel in [".blog-title", "h1", "h2", "h3", ".title", "[class*='title']"]:
                        title_elem = elem.query_selector(title_sel)
                        if title_elem:
                            title = title_elem.inner_text().strip()
                            if title and len(title) > 5:
                                break

                    if not title:
                        continue

                    # Extract date
                    date_str = ""
                    date_elem = elem.query_selector(".blog-item-date, time, .date, [class*='date'], [datetime]")
                    if date_elem:
                        date_str = date_elem.get_attribute("datetime") or date_elem.inner_text().strip()
                        date_str = self._parse_date(date_str)

                    # Check if there's a direct link
                    link = ""
                    link_elem = elem.query_selector("a[href]")
                    if link_elem:
                        link = link_elem.get_attribute("href") or ""
                        if link and not link.startswith("http"):
                            link = urljoin(url, link)

                    # Extract summary
                    summary = ""
                    for sum_sel in [".blog-desc", "p", ".summary", ".excerpt", "[class*='desc']"]:
                        sum_elem = elem.query_selector(sum_sel)
                        if sum_elem:
                            summary = sum_elem.inner_text().strip()
                            summary = self._validate_summary(summary, title)
                            if summary:
                                break

                    items_data.append({
                        "index": i,
                        "title": title,
                        "date": date_str,
                        "link": link,
                        "summary": summary,
                    })

                # Second pass: for items without links, click to discover URL
                for item in items_data:
                    if not item["link"]:
                        try:
                            # Re-query elements (page state may have changed)
                            elements = page.query_selector_all(selector)
                            if item["index"] < len(elements):
                                elem = elements[item["index"]]
                                # Click and wait for navigation
                                try:
                                    with page.expect_navigation(timeout=5000):
                                        elem.click()
                                    # Capture the new URL
                                    item["link"] = page.url
                                    # Go back to list page
                                    page.go_back(wait_until="networkidle")
                                    page.wait_for_timeout(1000)
                                except PlaywrightTimeout:
                                    # No navigation occurred, use list page URL
                                    item["link"] = url
                        except Exception:
                            item["link"] = url

                browser.close()

                # Build article list
                for item in items_data:
                    article = {
                        "title": item["title"],
                        "url": item["link"] or url,
                        "date": item["date"],
                        "summary": item["summary"][:300] if item["summary"] else "",
                    }
                    article["signals"] = self._extract_signals(article)
                    articles.append(article)

            if not articles:
                return [], "No valid articles extracted from browser render"

            return articles, None

        except Exception as e:
            return [], f"Browser scrape error: {e}"

    def _clean_html(self, text: str) -> str:
        """Remove HTML tags from text."""
        if not text:
            return ""
        soup = BeautifulSoup(text, "html.parser")
        return soup.get_text(strip=True)

    def _parse_date(self, date_str: str) -> str:
        """Try to parse date string into YYYY-MM-DD format."""
        if not date_str:
            return ""

        # Clean the string
        date_str = date_str.strip()

        # Try common formats
        formats = [
            "%Y-%m-%d",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%dT%H:%M:%S%z",
            "%B %d, %Y",
            "%b %d, %Y",
            "%d %B %Y",
            "%d %b %Y",
            "%m/%d/%Y",
            "%d/%m/%Y",
        ]

        for fmt in formats:
            try:
                dt = datetime.strptime(date_str[:30], fmt)
                return dt.strftime("%Y-%m-%d")
            except ValueError:
                continue

        # Try to extract date with regex
        date_match = re.search(r"(\d{4})-(\d{2})-(\d{2})", date_str)
        if date_match:
            return date_match.group(0)

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
            Blog activity dict with status.
        """
        name = blog_config.get("name", "Unknown")
        url = blog_config.get("url", "")
        blog_type = blog_config.get("type", "auto")
        rss_url = blog_config.get("rss_url")

        result = {
            "source": name,
            "url": url,
            "articles": [],
            "total_articles": 0,
            "has_activity": False,
            "status": "success",
            "error": None,
        }

        if not url:
            result["status"] = "scrape_failed"
            result["error"] = "No URL configured"
            return result

        articles = []
        error = None

        # Strategy 1: Use explicit RSS URL if provided
        if rss_url:
            articles, error = self.fetch_rss(rss_url, days)
            if articles:
                result["feed_url"] = rss_url

        # Strategy 2: Try to discover RSS feed
        if not articles and blog_type in ("auto", "rss"):
            discovered_feed = self._discover_rss_feed(url)
            if discovered_feed:
                articles, error = self.fetch_rss(discovered_feed, days)
                if articles:
                    result["feed_url"] = discovered_feed

        # Strategy 3: Try direct RSS URL patterns
        if not articles and blog_type in ("auto", "rss"):
            for rss_path in ["/feed", "/feed.xml", "/rss.xml", "/atom.xml"]:
                parsed = urlparse(url)
                test_url = f"{parsed.scheme}://{parsed.netloc}{rss_path}"
                articles, error = self.fetch_rss(test_url, days)
                if articles:
                    result["feed_url"] = test_url
                    break

        # Strategy 4: Fall back to HTML scraping
        if not articles and blog_type in ("auto", "scrape"):
            selector = blog_config.get("selector", "article, [class*='post'], [class*='blog']")
            articles, error = self.scrape_blog(url, selector, days)

        # Strategy 5: Use Playwright for JavaScript-rendered pages
        if not articles and blog_type in ("auto", "scrape", "browser"):
            selector = blog_config.get("selector", "article, [class*='post'], [class*='blog']")
            articles, error = self.scrape_with_browser(url, selector, days)
            if articles:
                result["render_method"] = "browser"

        # Process results
        if articles:
            result["total_articles"] = len(articles)
            # Filter to articles with signals
            relevant = [a for a in articles if a.get("signals")]
            result["articles"] = relevant
            result["has_activity"] = len(relevant) > 0
            result["status"] = "success"
            result["error"] = None
        else:
            result["status"] = "scrape_failed"
            result["error"] = error or "No articles found"

        return result

    def fetch_all_blogs(self, days: int = 7) -> list[dict]:
        """Fetch articles from all configured blogs.

        Args:
            days: Look back period in days.

        Returns:
            List of blog activity dicts, deduplicated by URL.
        """
        results = []
        seen_urls = set()

        print(f"  Tracking {len(self.blogs)} company blogs...")

        for blog_config in self.blogs:
            print(f"    Checking {blog_config.get('name', 'Unknown')}...")
            activity = self.fetch_blog(blog_config, days)

            # Deduplicate articles by normalized URL
            unique_articles = []
            for article in activity.get("articles", []):
                normalized_url = self._normalize_url(article.get("url", ""))
                if normalized_url and normalized_url not in seen_urls:
                    seen_urls.add(normalized_url)
                    unique_articles.append(article)

            activity["articles"] = unique_articles
            if unique_articles:
                activity["has_activity"] = True

            # Always include in results (even failures for transparency)
            results.append(activity)
            time.sleep(1)  # Politeness delay

        return results

    def _normalize_url(self, url: str) -> str:
        """Normalize URL for deduplication.

        Removes trailing slashes, fragments, and tracking params.

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
