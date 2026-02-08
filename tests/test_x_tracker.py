"""Tests for X/Twitter tracker."""

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from trackers.x_tracker import XTracker, SIGNAL_KEYWORDS


class TestXTrackerInit:
    """Tests for XTracker initialization."""

    def test_init_defaults(self):
        """Test initialization with empty config."""
        tracker = XTracker({})
        assert tracker.backend == "rsshub"
        assert tracker.rsshub_urls == ["https://rsshub.app"]
        assert tracker.bearer_token == ""
        assert tracker.accounts == []
        assert tracker.search_keywords == []

    def test_init_rsshub_backend(self):
        """Test initialization with RSSHub config."""
        config = {
            "x_tracker": {
                "backend": "rsshub",
                "rsshub_url": "https://my-rsshub.example.com/",
                "accounts": ["OpenAI", "AnthropicAI"],
            }
        }
        tracker = XTracker(config)
        assert tracker.backend == "rsshub"
        assert tracker.rsshub_urls == ["https://my-rsshub.example.com"]
        assert len(tracker.accounts) == 2

    def test_init_rsshub_urls_list(self):
        """Test initialization with multiple RSSHub URLs."""
        config = {
            "x_tracker": {
                "rsshub_urls": [
                    "https://rsshub1.example.com",
                    "https://rsshub2.example.com/",
                ],
            }
        }
        tracker = XTracker(config)
        assert tracker.rsshub_urls == [
            "https://rsshub1.example.com",
            "https://rsshub2.example.com",
        ]

    def test_init_auto_backend(self):
        """Test initialization with auto backend."""
        config = {
            "x_tracker": {
                "backend": "auto",
                "bearer_token": "test_token",
            }
        }
        tracker = XTracker(config)
        assert tracker.backend == "auto"
        assert tracker.bearer_token == "test_token"

    def test_init_api_backend(self):
        """Test initialization with API config."""
        config = {
            "x_tracker": {
                "backend": "api",
                "bearer_token": "test_token_123",
                "accounts": ["OpenAI"],
                "search_keywords": ["dataset release"],
            }
        }
        tracker = XTracker(config)
        assert tracker.backend == "api"
        assert tracker.bearer_token == "test_token_123"
        assert tracker.search_keywords == ["dataset release"]

    def test_init_backward_compat_single_url(self):
        """Test that old rsshub_url config still works."""
        config = {
            "x_tracker": {
                "rsshub_url": "https://rsshub.app///",
            }
        }
        tracker = XTracker(config)
        assert tracker.rsshub_urls == ["https://rsshub.app"]


class TestExtractSignals:
    """Tests for signal keyword extraction."""

    @pytest.fixture
    def tracker(self):
        return XTracker({})

    def test_extract_dataset_signal(self, tracker):
        """Test extracting dataset-related signals."""
        tweet = {"text": "We're releasing a new dataset for fine-tuning language models"}
        signals = tracker._extract_signals(tweet)
        assert "dataset" in signals
        assert "fine-tuning" in signals or "fine-tune" in signals
        assert "language model" in signals

    def test_extract_chinese_signals(self, tracker):
        """Test extracting Chinese keyword signals."""
        tweet = {"text": "我们发布了一个新的开源数据集用于模型训练"}
        signals = tracker._extract_signals(tweet)
        assert "发布" in signals
        assert "开源" in signals
        assert "数据集" in signals
        assert "训练" in signals

    def test_no_signals_for_irrelevant_tweet(self, tracker):
        """Test that irrelevant tweets return no signals."""
        tweet = {"text": "Just had a great lunch at the cafe"}
        signals = tracker._extract_signals(tweet)
        assert signals == []

    def test_case_insensitive_matching(self, tracker):
        """Test that keyword matching is case insensitive."""
        tweet = {"text": "NEW DATASET released for RLHF research"}
        signals = tracker._extract_signals(tweet)
        assert "dataset" in signals
        assert "rlhf" in signals

    def test_deduplication(self, tracker):
        """Test that signals are deduplicated."""
        tweet = {"text": "dataset dataset dataset"}
        signals = tracker._extract_signals(tweet)
        assert signals.count("dataset") == 1

    def test_empty_text(self, tracker):
        """Test handling of empty text."""
        tweet = {"text": ""}
        signals = tracker._extract_signals(tweet)
        assert signals == []

    def test_missing_text(self, tracker):
        """Test handling of missing text field."""
        tweet = {}
        signals = tracker._extract_signals(tweet)
        assert signals == []


class TestFetchRSSHubFeed:
    """Tests for RSSHub feed fetching."""

    @pytest.fixture
    def tracker(self):
        tracker = XTracker(
            {
                "x_tracker": {
                    "backend": "rsshub",
                    "rsshub_url": "https://rsshub.app",
                }
            }
        )
        # Pre-inject a mock HTTP client so _ensure_http doesn't create a real one
        tracker._http = MagicMock()
        tracker._owns_http = False
        return tracker

    @patch("trackers.x_tracker.feedparser.parse")
    async def test_fetch_rsshub_feed_success(self, mock_parse, tracker):
        """Test successful RSSHub feed fetch."""
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        mock_entry = MagicMock()
        mock_entry.published_parsed = now.timetuple()[:6] + (0, 0, 0)
        mock_entry.get = lambda key, default="": {
            "title": "We release a new dataset for RLHF",
            "summary": "Announcing our open-source dataset for RLHF training",
            "link": "https://x.com/test/status/123",
        }.get(key, default)

        mock_feed = MagicMock()
        mock_feed.entries = [mock_entry]
        mock_parse.return_value = mock_feed

        tracker._http.head = AsyncMock(return_value=200)
        tracker._http.get_text = AsyncMock(return_value="<rss>mock</rss>")

        tweets = await tracker._fetch_rsshub_feed("testuser", days=7)

        assert len(tweets) == 1
        assert tweets[0]["username"] == "testuser"
        assert tweets[0]["source"] == "x_rsshub"
        assert "dataset" in tweets[0]["signals"]
        mock_parse.assert_called_once_with("<rss>mock</rss>")

    @patch("trackers.x_tracker.feedparser.parse")
    async def test_fetch_rsshub_feed_empty(self, mock_parse, tracker):
        """Test RSSHub feed with no entries."""
        mock_feed = MagicMock()
        mock_feed.entries = []
        mock_parse.return_value = mock_feed

        tracker._http.head = AsyncMock(return_value=200)
        tracker._http.get_text = AsyncMock(return_value="<rss></rss>")

        tweets = await tracker._fetch_rsshub_feed("testuser")
        assert tweets == []

    @patch("trackers.x_tracker.feedparser.parse")
    async def test_fetch_rsshub_feed_old_entries_filtered(self, mock_parse, tracker):
        """Test that entries older than cutoff are filtered out."""
        old_date = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=30)
        mock_entry = MagicMock()
        mock_entry.published_parsed = old_date.timetuple()[:6] + (0, 0, 0)
        mock_entry.get = lambda key, default="": {
            "title": "Old dataset release",
            "summary": "This is an old tweet about a dataset",
            "link": "https://x.com/test/status/old",
        }.get(key, default)

        mock_feed = MagicMock()
        mock_feed.entries = [mock_entry]
        mock_parse.return_value = mock_feed

        tracker._http.head = AsyncMock(return_value=200)
        tracker._http.get_text = AsyncMock(return_value="<rss>mock</rss>")

        tweets = await tracker._fetch_rsshub_feed("testuser", days=7)
        assert tweets == []

    async def test_fetch_rsshub_feed_http_error(self, tracker):
        """Test handling of HTTP request errors."""
        tracker._http.head = AsyncMock(return_value=200)
        tracker._http.get_text = AsyncMock(return_value=None)

        tweets = await tracker._fetch_rsshub_feed("testuser")
        assert tweets == []

    @patch("trackers.x_tracker.feedparser.parse")
    async def test_fetch_rsshub_html_cleaned(self, mock_parse, tracker):
        """Test that HTML tags are cleaned from summary."""
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        mock_entry = MagicMock()
        mock_entry.published_parsed = now.timetuple()[:6] + (0, 0, 0)
        mock_entry.get = lambda key, default="": {
            "title": "Dataset announcement",
            "summary": "<p>We are <b>releasing</b> a new <a href='#'>dataset</a></p>",
            "link": "https://x.com/test/status/456",
        }.get(key, default)

        mock_feed = MagicMock()
        mock_feed.entries = [mock_entry]
        mock_parse.return_value = mock_feed

        tracker._http.head = AsyncMock(return_value=200)
        tracker._http.get_text = AsyncMock(return_value="<rss>mock</rss>")

        tweets = await tracker._fetch_rsshub_feed("testuser", days=7)
        assert len(tweets) == 1
        assert "<" not in tweets[0]["text"]
        assert ">" not in tweets[0]["text"]

    async def test_fetch_rsshub_redirect_detected(self, tracker):
        """Test that HTTP redirects (like rsshub.app -> google.com/404) are rejected."""
        tracker._http.head = AsyncMock(return_value=302)

        tweets = await tracker._fetch_rsshub_feed("testuser")
        assert tweets == []
        assert tracker._rsshub_consecutive_fails == 1


class TestFetchAPIUserTweets:
    """Tests for X API v2 user tweets fetching."""

    @pytest.fixture
    def tracker(self):
        tracker = XTracker(
            {
                "x_tracker": {
                    "backend": "api",
                    "bearer_token": "test_bearer_token",
                }
            }
        )
        tracker._http = MagicMock()
        tracker._owns_http = False
        return tracker

    async def test_no_bearer_token_returns_empty(self):
        """Test that missing bearer token returns empty list."""
        tracker = XTracker({"x_tracker": {"backend": "api"}})
        tweets = await tracker._fetch_api_user_tweets("testuser")
        assert tweets == []

    async def test_fetch_api_user_tweets_success(self, tracker):
        """Test successful API user tweets fetch."""
        user_data = {"data": {"id": "12345"}}
        tweets_data = {
            "data": [
                {
                    "id": "111",
                    "text": "We are releasing a new open-source dataset for RLHF",
                    "created_at": "2026-02-06T10:00:00Z",
                    "public_metrics": {"like_count": 100, "retweet_count": 50},
                }
            ]
        }

        tracker._http.get_json = AsyncMock(side_effect=[user_data, tweets_data])

        tweets = await tracker._fetch_api_user_tweets("testuser", days=7)
        assert len(tweets) == 1
        assert tweets[0]["source"] == "x_api"
        assert tweets[0]["username"] == "testuser"
        assert "dataset" in tweets[0]["signals"]

    async def test_fetch_api_user_lookup_fails(self, tracker):
        """Test handling of user lookup failure."""
        tracker._http.get_json = AsyncMock(return_value=None)

        tweets = await tracker._fetch_api_user_tweets("nonexistent")
        assert tweets == []

    async def test_fetch_api_rate_limited(self, tracker):
        """Test handling of rate limiting."""
        user_data = {"data": {"id": "12345"}}
        tracker._http.get_json = AsyncMock(side_effect=[user_data, None])

        tweets = await tracker._fetch_api_user_tweets("testuser")
        assert tweets == []

    async def test_fetch_api_network_error(self, tracker):
        """Test handling of network errors."""
        tracker._http.get_json = AsyncMock(side_effect=Exception("Network error"))

        tweets = await tracker._fetch_api_user_tweets("testuser")
        assert tweets == []


class TestFetchAPISearch:
    """Tests for X API v2 search."""

    @pytest.fixture
    def tracker(self):
        tracker = XTracker(
            {
                "x_tracker": {
                    "backend": "api",
                    "bearer_token": "test_token",
                }
            }
        )
        tracker._http = MagicMock()
        tracker._owns_http = False
        return tracker

    async def test_no_bearer_token_returns_empty(self):
        """Test that missing bearer token returns empty list."""
        tracker = XTracker({})
        tweets = await tracker._fetch_api_search("dataset release")
        assert tweets == []

    async def test_search_success(self, tracker):
        """Test successful search."""
        search_data = {
            "data": [
                {
                    "id": "222",
                    "text": "New benchmark dataset for evaluation",
                    "created_at": "2026-02-06",
                    "author_id": "999",
                }
            ]
        }
        tracker._http.get_json = AsyncMock(return_value=search_data)

        tweets = await tracker._fetch_api_search("dataset benchmark", days=7)
        assert len(tweets) == 1
        assert tweets[0]["source"] == "x_api_search"
        assert tweets[0]["query"] == "dataset benchmark"

    async def test_search_api_error(self, tracker):
        """Test handling of search API error."""
        tracker._http.get_json = AsyncMock(return_value=None)

        tweets = await tracker._fetch_api_search("test query")
        assert tweets == []


class TestFetchAccount:
    """Tests for fetch_account method."""

    @pytest.fixture
    def tracker(self):
        tracker = XTracker(
            {
                "x_tracker": {
                    "backend": "rsshub",
                    "rsshub_url": "https://rsshub.app",
                }
            }
        )
        tracker._http = MagicMock()
        tracker._owns_http = False
        return tracker

    @patch.object(XTracker, "_fetch_rsshub_feed", new_callable=AsyncMock)
    async def test_fetch_account_rsshub(self, mock_fetch, tracker):
        """Test fetching account via RSSHub backend."""
        mock_fetch.return_value = [
            {"text": "New dataset release", "signals": ["dataset"]},
            {"text": "Had lunch today", "signals": []},
        ]

        result = await tracker.fetch_account("OpenAI", days=7)
        assert result["username"] == "OpenAI"
        assert result["total_tweets"] == 2
        assert len(result["relevant_tweets"]) == 1
        assert result["has_activity"] is True

    @patch.object(XTracker, "_fetch_rsshub_feed", new_callable=AsyncMock)
    async def test_fetch_account_no_relevant(self, mock_fetch, tracker):
        """Test account with no relevant tweets."""
        mock_fetch.return_value = [
            {"text": "Just a normal tweet", "signals": []},
        ]

        result = await tracker.fetch_account("testuser")
        assert result["total_tweets"] == 1
        assert result["relevant_tweets"] == []
        assert result["has_activity"] is False

    @patch.object(XTracker, "_fetch_api_user_tweets", new_callable=AsyncMock)
    async def test_fetch_account_api_backend(self, mock_fetch):
        """Test fetching account via API backend."""
        tracker = XTracker(
            {
                "x_tracker": {
                    "backend": "api",
                    "bearer_token": "token123",
                }
            }
        )
        tracker._http = MagicMock()
        tracker._owns_http = False
        mock_fetch.return_value = [
            {"text": "Releasing open-source model", "signals": ["open source"]},
        ]

        result = await tracker.fetch_account("testuser")
        assert result["has_activity"] is True
        mock_fetch.assert_called_once()

    async def test_fetch_account_strips_at_symbol(self, tracker):
        """Test that @ is stripped from username."""
        with patch.object(tracker, "_fetch_rsshub_feed", new_callable=AsyncMock, return_value=[]) as mock:
            await tracker.fetch_account("@OpenAI")
            mock.assert_called_once_with("OpenAI", 7)


class TestFetchAll:
    """Tests for fetch_all method."""

    @pytest.fixture
    def tracker(self):
        tracker = XTracker(
            {
                "x_tracker": {
                    "backend": "rsshub",
                    "accounts": ["OpenAI", "AnthropicAI"],
                }
            }
        )
        tracker._http = MagicMock()
        tracker._owns_http = False
        return tracker

    @patch.object(XTracker, "fetch_account", new_callable=AsyncMock)
    async def test_fetch_all_parallel(self, mock_fetch, tracker):
        """Test parallel fetching of all accounts."""
        mock_fetch.side_effect = [
            {
                "username": "OpenAI",
                "total_tweets": 5,
                "relevant_tweets": [{"text": "dataset", "signals": ["dataset"]}],
                "has_activity": True,
            },
            {
                "username": "AnthropicAI",
                "total_tweets": 3,
                "relevant_tweets": [],
                "has_activity": False,
            },
        ]

        result = await tracker.fetch_all(days=7)
        assert len(result["accounts"]) == 1  # Only active accounts
        assert result["accounts"][0]["username"] == "OpenAI"
        assert result["search_results"] == []

    @patch.object(XTracker, "fetch_account", new_callable=AsyncMock)
    async def test_fetch_all_empty_accounts(self, mock_fetch):
        """Test fetch_all with no configured accounts."""
        tracker = XTracker({})
        tracker._http = MagicMock()
        tracker._owns_http = False
        result = await tracker.fetch_all()
        assert result["accounts"] == []
        assert result["search_results"] == []
        mock_fetch.assert_not_called()

    @patch.object(XTracker, "_fetch_api_search", new_callable=AsyncMock)
    @patch.object(XTracker, "fetch_account", new_callable=AsyncMock)
    async def test_fetch_all_with_search(self, mock_fetch, mock_search):
        """Test fetch_all with API search keywords."""
        tracker = XTracker(
            {
                "x_tracker": {
                    "backend": "api",
                    "bearer_token": "token",
                    "accounts": ["OpenAI"],
                    "search_keywords": ["dataset release"],
                }
            }
        )
        tracker._http = MagicMock()
        tracker._owns_http = False
        mock_fetch.return_value = {
            "username": "OpenAI",
            "total_tweets": 0,
            "relevant_tweets": [],
            "has_activity": False,
        }
        mock_search.return_value = [{"text": "New dataset release!", "signals": ["dataset"]}]

        result = await tracker.fetch_all()
        assert len(result["search_results"]) == 1
        mock_search.assert_called_once()

    @patch.object(XTracker, "fetch_account", new_callable=AsyncMock)
    async def test_fetch_all_handles_errors(self, mock_fetch, tracker):
        """Test that errors from individual accounts are handled gracefully."""
        mock_fetch.side_effect = Exception("Connection failed")

        result = await tracker.fetch_all()
        assert result["accounts"] == []


class TestSignalKeywords:
    """Tests for signal keywords configuration."""

    def test_signal_keywords_not_empty(self):
        """Test that signal keywords list is not empty."""
        assert len(SIGNAL_KEYWORDS) > 0

    def test_signal_keywords_contains_core_terms(self):
        """Test that core AI terms are in signal keywords."""
        core_terms = ["dataset", "benchmark", "open source", "rlhf", "llm"]
        for term in core_terms:
            assert term in SIGNAL_KEYWORDS, f"Missing core term: {term}"

    def test_signal_keywords_contains_chinese(self):
        """Test that Chinese keywords are included."""
        chinese_terms = ["开源", "数据集", "模型"]
        for term in chinese_terms:
            assert term in SIGNAL_KEYWORDS, f"Missing Chinese term: {term}"


class TestMultiInstanceFallback:
    """Tests for multi-RSSHub-instance fallback and auto backend."""

    @patch("trackers.x_tracker.feedparser.parse")
    async def test_rsshub_fallback_to_second_instance(self, mock_parse):
        """First instance fails (get_text returns None), second succeeds."""
        tracker = XTracker({
            "x_tracker": {
                "rsshub_urls": [
                    "https://bad-instance.example.com",
                    "https://good-instance.example.com",
                ],
            }
        })
        tracker._http = MagicMock()
        tracker._owns_http = False

        now = datetime.now(timezone.utc).replace(tzinfo=None)
        mock_entry = MagicMock()
        mock_entry.published_parsed = now.timetuple()[:6] + (0, 0, 0)
        mock_entry.get = lambda key, default="": {
            "title": "New dataset",
            "summary": "A dataset release",
            "link": "https://x.com/test/status/1",
        }.get(key, default)
        mock_feed = MagicMock()
        mock_feed.entries = [mock_entry]
        mock_parse.return_value = mock_feed

        # First call to head returns 200 (no redirect), get_text returns None (fail)
        # Second call to head returns 200 (no redirect), get_text returns RSS content
        tracker._http.head = AsyncMock(return_value=200)
        tracker._http.get_text = AsyncMock(side_effect=[None, "<rss>mock</rss>"])

        tweets = await tracker._fetch_rsshub_feed("testuser", days=7)
        assert len(tweets) == 1
        assert tracker._last_working_rsshub == "https://good-instance.example.com"

    async def test_rsshub_all_fail_increments_counter(self):
        """All RSSHub instances fail -> consecutive fail counter increases."""
        tracker = XTracker({
            "x_tracker": {
                "rsshub_urls": [
                    "https://dead1.example.com",
                    "https://dead2.example.com",
                ],
            }
        })
        tracker._http = MagicMock()
        tracker._owns_http = False
        tracker._http.head = AsyncMock(return_value=200)
        tracker._http.get_text = AsyncMock(return_value=None)

        tweets = await tracker._fetch_rsshub_feed("testuser")
        assert tweets == []
        assert tracker._rsshub_consecutive_fails == 1

    async def test_rsshub_threshold_skips_immediately(self):
        """Once fail threshold reached, _fetch_rsshub_feed returns [] without HTTP calls."""
        tracker = XTracker({})
        tracker._http = MagicMock()
        tracker._owns_http = False
        tracker._rsshub_consecutive_fails = tracker._rsshub_fail_threshold
        tracker._http.head = AsyncMock()

        tweets = await tracker._fetch_rsshub_feed("testuser")
        assert tweets == []
        tracker._http.head.assert_not_called()

    async def test_rsshub_success_resets_consecutive_fails(self):
        """A successful fetch resets the consecutive failure counter."""
        tracker = XTracker({
            "x_tracker": {
                "rsshub_urls": ["https://good.example.com"],
            }
        })
        tracker._http = MagicMock()
        tracker._owns_http = False
        tracker._rsshub_consecutive_fails = 3

        now = datetime.now(timezone.utc).replace(tzinfo=None)
        mock_entry = MagicMock()
        mock_entry.published_parsed = now.timetuple()[:6] + (0, 0, 0)
        mock_entry.get = lambda key, default="": {
            "title": "Test", "summary": "Test", "link": "https://x.com/t/1",
        }.get(key, default)
        mock_feed = MagicMock()
        mock_feed.entries = [mock_entry]

        tracker._http.head = AsyncMock(return_value=200)
        tracker._http.get_text = AsyncMock(return_value="<rss>mock</rss>")

        with patch("trackers.x_tracker.feedparser.parse", return_value=mock_feed):
            tweets = await tracker._fetch_rsshub_feed("testuser", days=7)
            assert len(tweets) == 1
            assert tracker._rsshub_consecutive_fails == 0
            assert tracker._rsshub_success_count == 1

    @patch.object(XTracker, "_fetch_api_user_tweets", new_callable=AsyncMock)
    @patch.object(XTracker, "_fetch_rsshub_feed", new_callable=AsyncMock)
    async def test_auto_backend_rsshub_then_api(self, mock_rss, mock_api):
        """auto mode: RSSHub all fail -> fallback to X API."""
        tracker = XTracker({
            "x_tracker": {
                "backend": "auto",
                "bearer_token": "test_token",
                "accounts": ["testuser"],
            }
        })
        tracker._http = MagicMock()
        tracker._owns_http = False
        # RSSHub returns nothing and threshold reached
        mock_rss.return_value = []
        tracker._rsshub_consecutive_fails = tracker._rsshub_fail_threshold

        mock_api.return_value = [
            {"text": "New dataset via API", "signals": ["dataset"]},
        ]

        result = await tracker.fetch_account("testuser", days=7)
        assert result["has_activity"] is True
        assert len(result["relevant_tweets"]) == 1
        mock_api.assert_called_once()

    @patch.object(XTracker, "_fetch_rsshub_feed", new_callable=AsyncMock)
    async def test_auto_backend_rsshub_success_no_api(self, mock_rss):
        """auto mode: RSSHub succeeds -> no API call."""
        tracker = XTracker({
            "x_tracker": {
                "backend": "auto",
                "bearer_token": "test_token",
            }
        })
        tracker._http = MagicMock()
        tracker._owns_http = False
        mock_rss.return_value = [
            {"text": "Dataset release", "signals": ["dataset"]},
        ]

        result = await tracker.fetch_account("testuser", days=7)
        assert result["has_activity"] is True
        mock_rss.assert_called_once()

    @patch.object(XTracker, "_fetch_rsshub_feed", new_callable=AsyncMock)
    async def test_auto_no_api_token_graceful(self, mock_rss):
        """auto mode: RSSHub fails + no API token -> returns empty gracefully."""
        tracker = XTracker({
            "x_tracker": {
                "backend": "auto",
                "bearer_token": "",
            }
        })
        tracker._http = MagicMock()
        tracker._owns_http = False
        mock_rss.return_value = []
        tracker._rsshub_consecutive_fails = tracker._rsshub_fail_threshold

        result = await tracker.fetch_account("testuser")
        assert result["has_activity"] is False
        assert result["relevant_tweets"] == []
