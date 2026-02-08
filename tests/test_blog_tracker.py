"""Tests for BlogTracker class."""

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from time import struct_time

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from trackers.blog_tracker import BlogTracker, SIGNAL_KEYWORDS, map_blog_to_vendor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_http_mock():
    """Create an AsyncMock that mimics AsyncHTTPClient."""
    mock = AsyncMock()
    mock.get_text = AsyncMock(return_value=None)
    mock.head = AsyncMock(return_value=None)
    mock.get_json = AsyncMock(return_value=None)
    mock.close = AsyncMock()
    return mock


def _recent_date():
    """Return a struct_time for yesterday (always within default 7-day window)."""
    dt = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=1)
    return dt.timetuple()


def _old_date():
    """Return a struct_time for 30 days ago (outside default 7-day window)."""
    dt = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=30)
    return dt.timetuple()


def _make_feed_entry(title="Test Article", link="https://example.com/post-1",
                     summary="A post about RLHF and dataset quality.",
                     published_parsed=None):
    """Build a mock feedparser entry."""
    entry = MagicMock()
    entry.get = lambda k, default="": {
        "title": title,
        "link": link,
        "summary": summary,
        "description": "",
    }.get(k, default)
    entry.published_parsed = published_parsed if published_parsed is not None else _recent_date()
    entry.updated_parsed = None
    # feedparser entries respond to hasattr for published_parsed/updated_parsed
    return entry


# ===========================================================================
# 1. Initialization
# ===========================================================================

class TestBlogTrackerInit:
    """Tests for BlogTracker initialization."""

    def test_init_default_config(self):
        """Default empty config results in empty blog list."""
        tracker = BlogTracker({})
        assert tracker.blogs == []
        assert tracker._rss_cache == {}
        assert tracker._http is not None

    def test_init_with_blogs_in_data_vendors(self):
        """Blogs pulled from config['data_vendors']['blogs']."""
        blogs = [{"name": "Blog A", "url": "https://a.com/blog"}]
        config = {"data_vendors": {"blogs": blogs}}
        tracker = BlogTracker(config)
        assert tracker.blogs == blogs

    def test_init_with_blogs_in_watched_vendors(self):
        """Blogs pulled from config['watched_vendors']['blogs']."""
        blogs = [{"name": "Blog B", "url": "https://b.com/blog"}]
        config = {"watched_vendors": {"blogs": blogs}}
        tracker = BlogTracker(config)
        assert tracker.blogs == blogs

    def test_init_with_blogs_top_level(self):
        """Blogs pulled from config['blogs'] as fallback."""
        blogs = [{"name": "Blog C", "url": "https://c.com/blog"}]
        config = {"blogs": blogs}
        tracker = BlogTracker(config)
        assert tracker.blogs == blogs

    def test_init_custom_http_client(self):
        """Injected http_client is used instead of creating a new one."""
        mock_http = _make_http_mock()
        tracker = BlogTracker({}, http_client=mock_http)
        assert tracker._http is mock_http


# ===========================================================================
# 2. _normalize_url
# ===========================================================================

class TestNormalizeUrl:
    """Tests for URL normalization."""

    def test_empty_url(self):
        tracker = BlogTracker({})
        assert tracker._normalize_url("") == ""

    def test_strips_fragment(self):
        tracker = BlogTracker({})
        result = tracker._normalize_url("https://example.com/post#comments")
        assert "#" not in result

    def test_strips_utm_params(self):
        tracker = BlogTracker({})
        url = "https://example.com/post?utm_source=twitter&utm_medium=social&id=42"
        result = tracker._normalize_url(url)
        assert "utm_source" not in result
        assert "utm_medium" not in result
        # Non-tracking param preserved
        assert "id=42" in result

    def test_strips_trailing_slash(self):
        tracker = BlogTracker({})
        result = tracker._normalize_url("https://example.com/post/")
        assert result.endswith("/post")

    def test_lowercases(self):
        tracker = BlogTracker({})
        result = tracker._normalize_url("https://Example.COM/Post")
        assert result == result.lower()

    def test_preserves_non_tracking_query(self):
        tracker = BlogTracker({})
        result = tracker._normalize_url("https://example.com/search?q=datasets&page=2")
        assert "q=datasets" in result
        assert "page=2" in result


# ===========================================================================
# 3. _extract_signals
# ===========================================================================

class TestExtractSignals:
    """Tests for signal keyword extraction from articles."""

    def test_matches_keywords_in_title(self):
        tracker = BlogTracker({})
        article = {"title": "New RLHF Dataset Released", "summary": ""}
        signals = tracker._extract_signals(article)
        assert "rlhf" in signals
        assert "dataset" in signals

    def test_matches_keywords_in_summary(self):
        tracker = BlogTracker({})
        article = {"title": "Update", "summary": "We improved our annotation pipeline for data quality."}
        signals = tracker._extract_signals(article)
        assert "annotation" in signals
        assert "data quality" in signals

    def test_no_signals_for_irrelevant_content(self):
        tracker = BlogTracker({})
        article = {"title": "Company Picnic Photos", "summary": "Fun at the park."}
        signals = tracker._extract_signals(article)
        assert signals == []

    def test_signals_are_deduplicated(self):
        tracker = BlogTracker({})
        article = {"title": "dataset dataset", "summary": "dataset"}
        signals = tracker._extract_signals(article)
        assert signals.count("dataset") == 1

    def test_chinese_keywords(self):
        tracker = BlogTracker({})
        article = {"title": "最新开源模型发布", "summary": "合成数据训练"}
        signals = tracker._extract_signals(article)
        assert "开源" in signals
        assert "发布" in signals
        assert "合成数据" in signals


# ===========================================================================
# 4. _validate_summary
# ===========================================================================

class TestValidateSummary:
    """Tests for summary validation."""

    def test_empty_summary(self):
        tracker = BlogTracker({})
        assert tracker._validate_summary("", "Title") == ""

    def test_date_only_summary_rejected(self):
        tracker = BlogTracker({})
        assert tracker._validate_summary("2024-06-15", "Title") == ""
        assert tracker._validate_summary("June 15, 2024", "Title") == ""

    def test_summary_same_as_title_rejected(self):
        tracker = BlogTracker({})
        assert tracker._validate_summary("My Title", "My Title") == ""

    def test_nav_text_rejected(self):
        tracker = BlogTracker({})
        nav_text = "GitHub Hugging Face ModelScope Demo Discord"
        assert tracker._validate_summary(nav_text, "Title") == ""

    def test_valid_summary_kept(self):
        tracker = BlogTracker({})
        good = "We are excited to announce a new benchmark for evaluating reasoning."
        assert tracker._validate_summary(good, "Title") == good


# ===========================================================================
# 5. _parse_date and _extract_date_from_url
# ===========================================================================

class TestDateParsing:
    """Tests for date parsing helpers."""

    def test_parse_iso_date(self):
        tracker = BlogTracker({})
        assert tracker._parse_date("2024-06-15") == "2024-06-15"

    def test_parse_iso_datetime(self):
        tracker = BlogTracker({})
        assert tracker._parse_date("2024-06-15T10:30:00Z") == "2024-06-15"

    def test_parse_human_readable(self):
        tracker = BlogTracker({})
        assert tracker._parse_date("June 15, 2024") == "2024-06-15"

    def test_parse_empty(self):
        tracker = BlogTracker({})
        assert tracker._parse_date("") == ""

    def test_extract_date_from_url_full(self):
        tracker = BlogTracker({})
        assert tracker._extract_date_from_url("https://blog.com/2024/6/15/post") == "2024-06-15"

    def test_extract_date_from_url_year_month(self):
        tracker = BlogTracker({})
        assert tracker._extract_date_from_url("https://blog.com/2024/3/") == "2024-03-01"

    def test_extract_date_from_url_none(self):
        tracker = BlogTracker({})
        assert tracker._extract_date_from_url("https://blog.com/about") == ""


# ===========================================================================
# 6. fetch_rss (async)
# ===========================================================================

class TestFetchRss:
    """Tests for RSS feed fetching."""

    async def test_successful_rss_fetch(self):
        """Recent article with signals is returned."""
        mock_http = _make_http_mock()
        tracker = BlogTracker({}, http_client=mock_http)

        yesterday = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=1)
        rss_xml = f"""<?xml version="1.0"?>
        <rss version="2.0">
          <channel>
            <item>
              <title>New RLHF Dataset</title>
              <link>https://example.com/rlhf-post</link>
              <description>We release a new RLHF dataset for alignment.</description>
              <pubDate>{yesterday.strftime('%a, %d %b %Y %H:%M:%S')} GMT</pubDate>
            </item>
          </channel>
        </rss>"""
        mock_http.get_text.return_value = rss_xml

        articles, error = await tracker.fetch_rss("https://example.com/feed", days=7)
        assert error is None
        assert len(articles) >= 1
        assert articles[0]["title"] == "New RLHF Dataset"
        assert "rlhf" in articles[0]["signals"]

    async def test_rss_fetch_empty_feed(self):
        """Empty feed returns error message."""
        mock_http = _make_http_mock()
        tracker = BlogTracker({}, http_client=mock_http)

        rss_xml = """<?xml version="1.0"?>
        <rss version="2.0"><channel></channel></rss>"""
        mock_http.get_text.return_value = rss_xml

        articles, error = await tracker.fetch_rss("https://example.com/feed")
        assert articles == []
        assert error == "No entries found in feed"

    async def test_rss_fetch_http_failure(self):
        """HTTP failure (None text) returns error."""
        mock_http = _make_http_mock()
        tracker = BlogTracker({}, http_client=mock_http)
        mock_http.get_text.return_value = None

        articles, error = await tracker.fetch_rss("https://example.com/feed")
        assert articles == []
        assert "Failed to fetch" in error

    async def test_rss_fetch_exception(self):
        """Exception during fetch returns error."""
        mock_http = _make_http_mock()
        tracker = BlogTracker({}, http_client=mock_http)
        mock_http.get_text.side_effect = Exception("Connection refused")

        articles, error = await tracker.fetch_rss("https://example.com/feed")
        assert articles == []
        assert "RSS parse error" in error

    async def test_rss_old_articles_filtered(self):
        """Articles older than the lookback window are excluded."""
        mock_http = _make_http_mock()
        tracker = BlogTracker({}, http_client=mock_http)

        old_date = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=30)
        rss_xml = f"""<?xml version="1.0"?>
        <rss version="2.0">
          <channel>
            <item>
              <title>Old Dataset Post</title>
              <link>https://example.com/old-post</link>
              <description>Old content about datasets.</description>
              <pubDate>{old_date.strftime('%a, %d %b %Y %H:%M:%S')} GMT</pubDate>
            </item>
          </channel>
        </rss>"""
        mock_http.get_text.return_value = rss_xml

        articles, error = await tracker.fetch_rss("https://example.com/feed", days=7)
        assert articles == []


# ===========================================================================
# 7. _discover_rss_feed (async)
# ===========================================================================

class TestDiscoverRssFeed:
    """Tests for RSS feed discovery."""

    async def test_discover_via_path_probe(self):
        """Finds feed when HEAD returns 200 and content looks like XML."""
        mock_http = _make_http_mock()
        tracker = BlogTracker({}, http_client=mock_http)

        # HEAD returns 200 for the /feed path
        async def head_side_effect(url, **kwargs):
            if url.endswith("/feed"):
                return 200
            return 404

        mock_http.head.side_effect = head_side_effect

        # get_text returns XML-like content for the /feed path
        async def get_text_side_effect(url, **kwargs):
            if url.endswith("/feed"):
                return '<?xml version="1.0"?><rss><channel></channel></rss>'
            return "<html><body>Hello</body></html>"

        mock_http.get_text.side_effect = get_text_side_effect

        result = await tracker._discover_rss_feed("https://example.com")
        assert result == "https://example.com/feed"

    async def test_discover_cached(self):
        """Cached result is returned without making HTTP calls."""
        mock_http = _make_http_mock()
        tracker = BlogTracker({}, http_client=mock_http)
        tracker._rss_cache["https://example.com"] = "https://example.com/feed.xml"

        result = await tracker._discover_rss_feed("https://example.com")
        assert result == "https://example.com/feed.xml"
        mock_http.head.assert_not_awaited()

    async def test_discover_via_html_link_tag(self):
        """Falls back to parsing HTML <link rel='alternate'> tag."""
        mock_http = _make_http_mock()
        tracker = BlogTracker({}, http_client=mock_http)

        # All path probes fail
        mock_http.head.return_value = 404

        # HTML page has an RSS link tag
        html = """<html><head>
        <link rel="alternate" type="application/rss+xml" href="/blog/feed.xml" />
        </head><body></body></html>"""
        mock_http.get_text.return_value = html

        result = await tracker._discover_rss_feed("https://example.com")
        assert result == "https://example.com/blog/feed.xml"

    async def test_discover_no_feed_found(self):
        """Returns None when no feed can be discovered."""
        mock_http = _make_http_mock()
        tracker = BlogTracker({}, http_client=mock_http)

        mock_http.head.return_value = 404
        mock_http.get_text.return_value = "<html><body>No feeds here</body></html>"

        result = await tracker._discover_rss_feed("https://example.com")
        assert result is None
        # Result is cached as None
        assert tracker._rss_cache["https://example.com"] is None


# ===========================================================================
# 8. fetch_all_blogs (async)
# ===========================================================================

class TestFetchAllBlogs:
    """Tests for concurrent blog fetching."""

    async def test_fetch_all_empty_config(self):
        """No blogs configured returns empty list."""
        mock_http = _make_http_mock()
        tracker = BlogTracker({}, http_client=mock_http)
        results = await tracker.fetch_all_blogs()
        assert results == []

    async def test_fetch_all_concurrent(self):
        """Multiple blogs are fetched and results collected."""
        mock_http = _make_http_mock()
        blogs = [
            {"name": "Blog A", "url": "https://a.com/blog", "type": "rss", "rss_url": "https://a.com/feed"},
            {"name": "Blog B", "url": "https://b.com/blog", "type": "rss", "rss_url": "https://b.com/feed"},
        ]
        config = {"blogs": blogs}
        tracker = BlogTracker(config, http_client=mock_http)

        yesterday = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=1)
        rss_xml = f"""<?xml version="1.0"?>
        <rss version="2.0">
          <channel>
            <item>
              <title>RLHF Dataset Release</title>
              <link>https://a.com/post-1</link>
              <description>Announcing a new dataset.</description>
              <pubDate>{yesterday.strftime('%a, %d %b %Y %H:%M:%S')} GMT</pubDate>
            </item>
          </channel>
        </rss>"""
        mock_http.get_text.return_value = rss_xml
        mock_http.head.return_value = 404  # no auto-discovery needed

        results = await tracker.fetch_all_blogs(days=7)
        assert len(results) == 2

    async def test_fetch_all_deduplicates_urls(self):
        """Duplicate article URLs across blogs are deduplicated."""
        mock_http = _make_http_mock()
        blogs = [
            {"name": "Blog A", "url": "https://a.com", "type": "rss", "rss_url": "https://a.com/feed"},
            {"name": "Blog B", "url": "https://b.com", "type": "rss", "rss_url": "https://b.com/feed"},
        ]
        config = {"blogs": blogs}
        tracker = BlogTracker(config, http_client=mock_http)

        yesterday = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=1)
        # Both blogs return articles pointing to the SAME URL
        rss_xml = f"""<?xml version="1.0"?>
        <rss version="2.0">
          <channel>
            <item>
              <title>Shared Dataset Post</title>
              <link>https://shared.com/post</link>
              <description>A dataset article.</description>
              <pubDate>{yesterday.strftime('%a, %d %b %Y %H:%M:%S')} GMT</pubDate>
            </item>
          </channel>
        </rss>"""
        mock_http.get_text.return_value = rss_xml
        mock_http.head.return_value = 404

        results = await tracker.fetch_all_blogs(days=7)
        # Collect all articles across all blogs
        all_articles = []
        for r in results:
            all_articles.extend(r.get("articles", []))
        # Only one copy should survive dedup
        assert len(all_articles) == 1

    async def test_fetch_all_handles_individual_errors(self):
        """An exception in one blog does not prevent others from returning."""
        mock_http = _make_http_mock()
        blogs = [
            {"name": "Good Blog", "url": "https://good.com", "type": "rss", "rss_url": "https://good.com/feed"},
            {"name": "Bad Blog", "url": "https://bad.com", "type": "rss", "rss_url": "https://bad.com/feed"},
        ]
        config = {"blogs": blogs}
        tracker = BlogTracker(config, http_client=mock_http)

        yesterday = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=1)
        good_rss = f"""<?xml version="1.0"?>
        <rss version="2.0">
          <channel>
            <item>
              <title>Dataset Quality Improvements</title>
              <link>https://good.com/post</link>
              <description>Better data quality through annotation.</description>
              <pubDate>{yesterday.strftime('%a, %d %b %Y %H:%M:%S')} GMT</pubDate>
            </item>
          </channel>
        </rss>"""

        call_count = 0

        async def selective_get_text(url, **kwargs):
            nonlocal call_count
            call_count += 1
            if "bad.com" in url:
                return None
            return good_rss

        mock_http.get_text.side_effect = selective_get_text
        mock_http.head.return_value = 404

        results = await tracker.fetch_all_blogs(days=7)
        # Both blogs should appear in results (one success, one failed)
        assert len(results) == 2
        statuses = {r["source"]: r["status"] for r in results}
        assert statuses["Good Blog"] == "success"
        assert statuses["Bad Blog"] == "scrape_failed"


# ===========================================================================
# 9. fetch_blog (async)
# ===========================================================================

class TestFetchBlog:
    """Tests for single blog fetching logic."""

    async def test_fetch_blog_no_url(self):
        """Blog config with no URL returns error."""
        mock_http = _make_http_mock()
        tracker = BlogTracker({}, http_client=mock_http)
        result = await tracker.fetch_blog({"name": "Empty"})
        assert result["status"] == "scrape_failed"
        assert result["error"] == "No URL configured"

    async def test_fetch_blog_with_rss_url(self):
        """Blog with explicit rss_url uses it directly."""
        mock_http = _make_http_mock()
        tracker = BlogTracker({}, http_client=mock_http)

        yesterday = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=1)
        rss_xml = f"""<?xml version="1.0"?>
        <rss version="2.0">
          <channel>
            <item>
              <title>Alignment Research Update</title>
              <link>https://lab.com/alignment</link>
              <description>New alignment dataset released.</description>
              <pubDate>{yesterday.strftime('%a, %d %b %Y %H:%M:%S')} GMT</pubDate>
            </item>
          </channel>
        </rss>"""
        mock_http.get_text.return_value = rss_xml

        blog_cfg = {
            "name": "AI Lab",
            "url": "https://lab.com/blog",
            "rss_url": "https://lab.com/feed.xml",
        }
        result = await tracker.fetch_blog(blog_cfg)
        assert result["status"] == "success"
        assert result["total_articles"] >= 1
        assert result["feed_url"] == "https://lab.com/feed.xml"


# ===========================================================================
# 10. map_blog_to_vendor
# ===========================================================================

class TestMapBlogToVendor:
    """Tests for the blog-to-vendor name mapping utility."""

    def test_known_mapping(self):
        assert map_blog_to_vendor("Scale AI") == "scale_ai"
        assert map_blog_to_vendor("Anthropic Research") == "anthropic"

    def test_unknown_mapping_normalized(self):
        assert map_blog_to_vendor("Weights Biases") == "weights_biases"

    def test_clean_html(self):
        """_clean_html strips tags."""
        tracker = BlogTracker({})
        result = tracker._clean_html("<p>Hello <b>world</b></p>")
        # BeautifulSoup.get_text(strip=True) may collapse inter-tag spaces
        assert "Hello" in result
        assert "world" in result
        assert "<" not in result
        assert tracker._clean_html("") == ""
