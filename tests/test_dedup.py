"""Tests for deduplication functionality."""

import sys
from pathlib import Path

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from scrapers.base import BaseScraper
from scrapers.blog_rss import BlogRSSScraper


class TestBaseScraper:
    """Tests for BaseScraper.deduplicate()."""

    def test_deduplicate_by_id(self):
        """Test deduplication using ID field."""

        class TestScraper(BaseScraper):
            def scrape(self, config=None):
                return []

        scraper = TestScraper()

        items = [
            {"id": "a", "name": "Item A"},
            {"id": "b", "name": "Item B"},
            {"id": "a", "name": "Item A Duplicate"},  # Duplicate
            {"id": "c", "name": "Item C"},
        ]

        result = scraper.deduplicate(items)

        assert len(result) == 3
        assert result[0]["id"] == "a"
        assert result[1]["id"] == "b"
        assert result[2]["id"] == "c"

    def test_deduplicate_by_url_fallback(self):
        """Test deduplication falls back to URL when no ID."""

        class TestScraper(BaseScraper):
            def scrape(self, config=None):
                return []

        scraper = TestScraper()

        items = [
            {"url": "https://example.com/a", "name": "Item A"},
            {"url": "https://example.com/b", "name": "Item B"},
            {"url": "https://example.com/a", "name": "Duplicate A"},
        ]

        result = scraper.deduplicate(items)

        assert len(result) == 2
        assert result[0]["url"] == "https://example.com/a"
        assert result[1]["url"] == "https://example.com/b"

    def test_deduplicate_empty_list(self):
        """Test deduplication with empty list."""

        class TestScraper(BaseScraper):
            def scrape(self, config=None):
                return []

        scraper = TestScraper()
        result = scraper.deduplicate([])
        assert result == []

    def test_deduplicate_preserves_order(self):
        """Test that deduplication preserves original order."""

        class TestScraper(BaseScraper):
            def scrape(self, config=None):
                return []

        scraper = TestScraper()

        items = [
            {"id": "c", "order": 1},
            {"id": "a", "order": 2},
            {"id": "b", "order": 3},
            {"id": "a", "order": 4},  # Duplicate of second item
        ]

        result = scraper.deduplicate(items)

        assert len(result) == 3
        assert [item["id"] for item in result] == ["c", "a", "b"]
        assert [item["order"] for item in result] == [1, 2, 3]


class TestBlogRSSDedup:
    """Tests for BlogRSSScraper URL deduplication."""

    @pytest.fixture
    def scraper(self):
        return BlogRSSScraper()

    def test_normalize_url_removes_trailing_slash(self, scraper):
        """Test that trailing slashes are removed."""
        url = "https://example.com/blog/post/"
        normalized = scraper._normalize_url(url)
        assert normalized == "https://example.com/blog/post"

    def test_normalize_url_removes_fragment(self, scraper):
        """Test that URL fragments are removed."""
        url = "https://example.com/blog/post#section"
        normalized = scraper._normalize_url(url)
        assert normalized == "https://example.com/blog/post"

    def test_normalize_url_removes_utm_params(self, scraper):
        """Test that UTM tracking params are removed."""
        url = "https://example.com/blog/post?utm_source=twitter&utm_medium=social"
        normalized = scraper._normalize_url(url)
        assert normalized == "https://example.com/blog/post"

    def test_normalize_url_preserves_non_tracking_params(self, scraper):
        """Test that non-tracking query params are preserved."""
        url = "https://example.com/search?q=test&page=2"
        normalized = scraper._normalize_url(url)
        assert "q=test" in normalized
        assert "page=2" in normalized

    def test_normalize_url_mixed_params(self, scraper):
        """Test URL with both tracking and non-tracking params."""
        url = "https://example.com/page?id=123&utm_source=email&ref=newsletter"
        normalized = scraper._normalize_url(url)
        assert "id=123" in normalized
        assert "utm_source" not in normalized
        assert "ref=" not in normalized

    def test_normalize_url_lowercase(self, scraper):
        """Test that URLs are lowercased."""
        url = "HTTPS://EXAMPLE.COM/Blog/Post"
        normalized = scraper._normalize_url(url)
        assert normalized == "https://example.com/blog/post"

    def test_normalize_url_empty(self, scraper):
        """Test normalization of empty URL."""
        assert scraper._normalize_url("") == ""

    def test_deduplicate_by_normalized_url(self, scraper):
        """Test that articles are deduplicated by normalized URL."""
        articles = [
            {"url": "https://example.com/post1", "title": "Post 1"},
            {"url": "https://example.com/post2/", "title": "Post 2"},
            {"url": "https://example.com/post1?utm_source=twitter", "title": "Post 1 Duplicate"},
            {"url": "https://example.com/post2", "title": "Post 2 Duplicate"},
        ]

        result = scraper.deduplicate(articles)

        assert len(result) == 2
        assert result[0]["title"] == "Post 1"
        assert result[1]["title"] == "Post 2"


class TestURLNormalizationEdgeCases:
    """Edge case tests for URL normalization."""

    @pytest.fixture
    def scraper(self):
        return BlogRSSScraper()

    def test_url_with_port(self, scraper):
        """Test URL with port number."""
        url = "https://example.com:8080/blog/post/"
        normalized = scraper._normalize_url(url)
        assert "8080" in normalized
        assert not normalized.endswith("/")

    def test_url_with_username_password(self, scraper):
        """Test URL with credentials (should be preserved)."""
        url = "https://user:pass@example.com/blog/post"
        normalized = scraper._normalize_url(url)
        assert "user:pass" in normalized

    def test_url_root_path(self, scraper):
        """Test URL with just root path."""
        url = "https://example.com/"
        normalized = scraper._normalize_url(url)
        assert normalized == "https://example.com/"

    def test_invalid_url_returns_lowercased(self, scraper):
        """Test that invalid URLs are at least lowercased."""
        url = "NOT_A_URL"
        normalized = scraper._normalize_url(url)
        assert normalized == "not_a_url"
