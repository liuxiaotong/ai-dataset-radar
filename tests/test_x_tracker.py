"""Tests for X/Twitter tracker."""

import sys
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock, PropertyMock

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
        assert tracker.rsshub_base == "https://rsshub.app"
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
        assert tracker.rsshub_base == "https://my-rsshub.example.com"
        assert len(tracker.accounts) == 2

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

    def test_init_rsshub_url_strips_trailing_slash(self):
        """Test that trailing slash is stripped from RSSHub URL."""
        config = {
            "x_tracker": {
                "rsshub_url": "https://rsshub.app///",
            }
        }
        tracker = XTracker(config)
        assert tracker.rsshub_base == "https://rsshub.app"


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
        return XTracker({
            "x_tracker": {
                "backend": "rsshub",
                "rsshub_url": "https://rsshub.app",
            }
        })

    @patch("trackers.x_tracker.feedparser.parse")
    def test_fetch_rsshub_feed_success(self, mock_parse, tracker):
        """Test successful RSSHub feed fetch."""
        now = datetime.utcnow()
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

        tweets = tracker._fetch_rsshub_feed("testuser", days=7)

        assert len(tweets) == 1
        assert tweets[0]["username"] == "testuser"
        assert tweets[0]["source"] == "x_rsshub"
        assert "dataset" in tweets[0]["signals"]
        mock_parse.assert_called_once_with("https://rsshub.app/twitter/user/testuser")

    @patch("trackers.x_tracker.feedparser.parse")
    def test_fetch_rsshub_feed_empty(self, mock_parse, tracker):
        """Test RSSHub feed with no entries."""
        mock_feed = MagicMock()
        mock_feed.entries = []
        mock_parse.return_value = mock_feed

        tweets = tracker._fetch_rsshub_feed("testuser")
        assert tweets == []

    @patch("trackers.x_tracker.feedparser.parse")
    def test_fetch_rsshub_feed_old_entries_filtered(self, mock_parse, tracker):
        """Test that entries older than cutoff are filtered out."""
        old_date = datetime.utcnow() - timedelta(days=30)
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

        tweets = tracker._fetch_rsshub_feed("testuser", days=7)
        assert tweets == []

    @patch("trackers.x_tracker.feedparser.parse")
    def test_fetch_rsshub_feed_parse_error(self, mock_parse, tracker):
        """Test handling of feed parse errors."""
        mock_parse.side_effect = Exception("Feed parse error")

        tweets = tracker._fetch_rsshub_feed("testuser")
        assert tweets == []

    @patch("trackers.x_tracker.feedparser.parse")
    def test_fetch_rsshub_html_cleaned(self, mock_parse, tracker):
        """Test that HTML tags are cleaned from summary."""
        now = datetime.utcnow()
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

        tweets = tracker._fetch_rsshub_feed("testuser", days=7)
        assert len(tweets) == 1
        assert "<" not in tweets[0]["text"]
        assert ">" not in tweets[0]["text"]


class TestFetchAPIUserTweets:
    """Tests for X API v2 user tweets fetching."""

    @pytest.fixture
    def tracker(self):
        return XTracker({
            "x_tracker": {
                "backend": "api",
                "bearer_token": "test_bearer_token",
            }
        })

    def test_no_bearer_token_returns_empty(self):
        """Test that missing bearer token returns empty list."""
        tracker = XTracker({"x_tracker": {"backend": "api"}})
        tweets = tracker._fetch_api_user_tweets("testuser")
        assert tweets == []

    @patch("trackers.x_tracker.requests.get")
    def test_fetch_api_user_tweets_success(self, mock_get, tracker):
        """Test successful API user tweets fetch."""
        # Mock user lookup
        user_resp = MagicMock()
        user_resp.status_code = 200
        user_resp.json.return_value = {"data": {"id": "12345"}}

        # Mock tweets fetch
        tweets_resp = MagicMock()
        tweets_resp.status_code = 200
        tweets_resp.json.return_value = {
            "data": [
                {
                    "id": "111",
                    "text": "We are releasing a new open-source dataset for RLHF",
                    "created_at": "2026-02-06T10:00:00Z",
                    "public_metrics": {"like_count": 100, "retweet_count": 50},
                }
            ]
        }

        mock_get.side_effect = [user_resp, tweets_resp]

        tweets = tracker._fetch_api_user_tweets("testuser", days=7)
        assert len(tweets) == 1
        assert tweets[0]["source"] == "x_api"
        assert tweets[0]["username"] == "testuser"
        assert "dataset" in tweets[0]["signals"]

    @patch("trackers.x_tracker.requests.get")
    def test_fetch_api_user_lookup_fails(self, mock_get, tracker):
        """Test handling of user lookup failure."""
        resp = MagicMock()
        resp.status_code = 404
        mock_get.return_value = resp

        tweets = tracker._fetch_api_user_tweets("nonexistent")
        assert tweets == []

    @patch("trackers.x_tracker.requests.get")
    def test_fetch_api_rate_limited(self, mock_get, tracker):
        """Test handling of rate limiting."""
        user_resp = MagicMock()
        user_resp.status_code = 200
        user_resp.json.return_value = {"data": {"id": "12345"}}

        rate_resp = MagicMock()
        rate_resp.status_code = 429

        mock_get.side_effect = [user_resp, rate_resp]

        tweets = tracker._fetch_api_user_tweets("testuser")
        assert tweets == []

    @patch("trackers.x_tracker.requests.get")
    def test_fetch_api_network_error(self, mock_get, tracker):
        """Test handling of network errors."""
        import requests as req
        mock_get.side_effect = req.RequestException("Network error")

        tweets = tracker._fetch_api_user_tweets("testuser")
        assert tweets == []


class TestFetchAPISearch:
    """Tests for X API v2 search."""

    @pytest.fixture
    def tracker(self):
        return XTracker({
            "x_tracker": {
                "backend": "api",
                "bearer_token": "test_token",
            }
        })

    def test_no_bearer_token_returns_empty(self):
        """Test that missing bearer token returns empty list."""
        tracker = XTracker({})
        tweets = tracker._fetch_api_search("dataset release")
        assert tweets == []

    @patch("trackers.x_tracker.requests.get")
    def test_search_success(self, mock_get, tracker):
        """Test successful search."""
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {
            "data": [
                {
                    "id": "222",
                    "text": "New benchmark dataset for evaluation",
                    "created_at": "2026-02-06",
                    "author_id": "999",
                }
            ]
        }
        mock_get.return_value = resp

        tweets = tracker._fetch_api_search("dataset benchmark", days=7)
        assert len(tweets) == 1
        assert tweets[0]["source"] == "x_api_search"
        assert tweets[0]["query"] == "dataset benchmark"

    @patch("trackers.x_tracker.requests.get")
    def test_search_api_error(self, mock_get, tracker):
        """Test handling of search API error."""
        resp = MagicMock()
        resp.status_code = 500
        mock_get.return_value = resp

        tweets = tracker._fetch_api_search("test query")
        assert tweets == []


class TestFetchAccount:
    """Tests for fetch_account method."""

    @pytest.fixture
    def tracker(self):
        return XTracker({
            "x_tracker": {
                "backend": "rsshub",
                "rsshub_url": "https://rsshub.app",
            }
        })

    @patch.object(XTracker, "_fetch_rsshub_feed")
    def test_fetch_account_rsshub(self, mock_fetch, tracker):
        """Test fetching account via RSSHub backend."""
        mock_fetch.return_value = [
            {"text": "New dataset release", "signals": ["dataset"]},
            {"text": "Had lunch today", "signals": []},
        ]

        result = tracker.fetch_account("OpenAI", days=7)
        assert result["username"] == "OpenAI"
        assert result["total_tweets"] == 2
        assert len(result["relevant_tweets"]) == 1
        assert result["has_activity"] is True

    @patch.object(XTracker, "_fetch_rsshub_feed")
    def test_fetch_account_no_relevant(self, mock_fetch, tracker):
        """Test account with no relevant tweets."""
        mock_fetch.return_value = [
            {"text": "Just a normal tweet", "signals": []},
        ]

        result = tracker.fetch_account("testuser")
        assert result["total_tweets"] == 1
        assert result["relevant_tweets"] == []
        assert result["has_activity"] is False

    @patch.object(XTracker, "_fetch_api_user_tweets")
    def test_fetch_account_api_backend(self, mock_fetch):
        """Test fetching account via API backend."""
        tracker = XTracker({
            "x_tracker": {
                "backend": "api",
                "bearer_token": "token123",
            }
        })
        mock_fetch.return_value = [
            {"text": "Releasing open-source model", "signals": ["open source"]},
        ]

        result = tracker.fetch_account("testuser")
        assert result["has_activity"] is True
        mock_fetch.assert_called_once()

    def test_fetch_account_strips_at_symbol(self, tracker):
        """Test that @ is stripped from username."""
        with patch.object(tracker, "_fetch_rsshub_feed", return_value=[]) as mock:
            tracker.fetch_account("@OpenAI")
            mock.assert_called_once_with("OpenAI", 7)


class TestFetchAll:
    """Tests for fetch_all method."""

    @pytest.fixture
    def tracker(self):
        return XTracker({
            "x_tracker": {
                "backend": "rsshub",
                "accounts": ["OpenAI", "AnthropicAI"],
            }
        })

    @patch.object(XTracker, "fetch_account")
    def test_fetch_all_parallel(self, mock_fetch, tracker):
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

        result = tracker.fetch_all(days=7)
        assert len(result["accounts"]) == 1  # Only active accounts
        assert result["accounts"][0]["username"] == "OpenAI"
        assert result["search_results"] == []

    @patch.object(XTracker, "fetch_account")
    def test_fetch_all_empty_accounts(self, mock_fetch):
        """Test fetch_all with no configured accounts."""
        tracker = XTracker({})
        result = tracker.fetch_all()
        assert result["accounts"] == []
        assert result["search_results"] == []
        mock_fetch.assert_not_called()

    @patch.object(XTracker, "_fetch_api_search")
    @patch.object(XTracker, "fetch_account")
    def test_fetch_all_with_search(self, mock_fetch, mock_search):
        """Test fetch_all with API search keywords."""
        tracker = XTracker({
            "x_tracker": {
                "backend": "api",
                "bearer_token": "token",
                "accounts": ["OpenAI"],
                "search_keywords": ["dataset release"],
            }
        })
        mock_fetch.return_value = {
            "username": "OpenAI",
            "total_tweets": 0,
            "relevant_tweets": [],
            "has_activity": False,
        }
        mock_search.return_value = [
            {"text": "New dataset release!", "signals": ["dataset"]}
        ]

        result = tracker.fetch_all()
        assert len(result["search_results"]) == 1
        mock_search.assert_called_once()

    @patch.object(XTracker, "fetch_account")
    def test_fetch_all_handles_errors(self, mock_fetch, tracker):
        """Test that errors from individual accounts are handled gracefully."""
        mock_fetch.side_effect = Exception("Connection failed")

        result = tracker.fetch_all()
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
