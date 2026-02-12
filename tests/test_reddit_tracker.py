"""Tests for Reddit tracker."""

import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from trackers.reddit_tracker import RedditTracker, SIGNAL_KEYWORDS


@pytest.fixture
def config():
    return {
        "reddit_tracker": {
            "enabled": True,
            "subreddits": ["MachineLearning", "LocalLLaMA"],
            "min_score": 5,
            "search_keywords": ["dataset release"],
        }
    }


@pytest.fixture
def mock_http():
    http = AsyncMock()
    http.get_json = AsyncMock()
    http.close = AsyncMock()
    return http


@pytest.fixture
def tracker(config, mock_http):
    return RedditTracker(config, http_client=mock_http)


# ─── Init tests ─────────────────────────────────────────────────────────────

class TestRedditTrackerInit:
    def test_defaults(self):
        t = RedditTracker({})
        assert len(t.subreddits) > 0
        assert "MachineLearning" in t.subreddits
        assert t.min_score == 5

    def test_config_override(self, config):
        t = RedditTracker(config)
        assert t.subreddits == ["MachineLearning", "LocalLLaMA"]
        assert t.min_score == 5

    def test_owns_http_when_none(self):
        t = RedditTracker({})
        assert t._owns_http is True

    def test_shared_http(self, mock_http):
        t = RedditTracker({}, http_client=mock_http)
        assert t._owns_http is False
        assert t._http is mock_http


# ─── Signal extraction tests ────────────────────────────────────────────────

class TestExtractSignals:
    def test_matches_keywords(self, tracker):
        post = {"title": "New RLHF dataset released", "selftext": ""}
        signals = tracker._extract_signals(post)
        assert "dataset" in signals
        assert "rlhf" in signals

    def test_no_match(self, tracker):
        post = {"title": "How to cook pasta", "selftext": "Boil water"}
        signals = tracker._extract_signals(post)
        assert signals == []

    def test_case_insensitive(self, tracker):
        post = {"title": "DATASET RELEASE", "selftext": ""}
        signals = tracker._extract_signals(post)
        assert "dataset" in signals

    def test_searches_both_fields(self, tracker):
        post = {"title": "Check this out", "selftext": "New synthetic data approach"}
        signals = tracker._extract_signals(post)
        assert "synthetic data" in signals

    def test_deduplicates(self, tracker):
        post = {"title": "dataset dataset dataset", "selftext": ""}
        signals = tracker._extract_signals(post)
        assert signals.count("dataset") == 1


# ─── Fetch subreddit tests ──────────────────────────────────────────────────

SAMPLE_REDDIT_RESPONSE = {
    "data": {
        "children": [
            {
                "data": {
                    "title": "New RLHF dataset from OpenAI",
                    "selftext": "Check out this new training data",
                    "score": 150,
                    "num_comments": 42,
                    "created_utc": 1770508800,  # 2026-02-08 approx
                    "author": "ai_researcher",
                    "permalink": "/r/MachineLearning/comments/abc123/new_rlhf/",
                }
            },
            {
                "data": {
                    "title": "My cat is cute",
                    "selftext": "Look at my cat",
                    "score": 500,
                    "num_comments": 100,
                    "created_utc": 1770508800,
                    "author": "cat_lover",
                    "permalink": "/r/MachineLearning/comments/def456/cat/",
                }
            },
            {
                "data": {
                    "title": "Released benchmark for LLM evaluation",
                    "selftext": "We open-source a new benchmark",
                    "score": 3,
                    "num_comments": 2,
                    "created_utc": 1770508800,
                    "author": "benchmarker",
                    "permalink": "/r/MachineLearning/comments/ghi789/bench/",
                }
            },
        ]
    }
}


class TestFetchSubreddit:
    @pytest.mark.asyncio
    async def test_basic_fetch(self, tracker, mock_http):
        mock_http.get_json.return_value = SAMPLE_REDDIT_RESPONSE
        posts = await tracker._fetch_subreddit("MachineLearning", days=365)
        assert len(posts) == 3
        assert posts[0]["subreddit"] == "MachineLearning"
        assert posts[0]["title"] == "New RLHF dataset from OpenAI"
        assert "rlhf" in posts[0]["signals"]

    @pytest.mark.asyncio
    async def test_empty_response(self, tracker, mock_http):
        mock_http.get_json.return_value = {"data": {"children": []}}
        posts = await tracker._fetch_subreddit("empty", days=7)
        assert posts == []

    @pytest.mark.asyncio
    async def test_none_response(self, tracker, mock_http):
        mock_http.get_json.return_value = None
        posts = await tracker._fetch_subreddit("broken", days=7)
        assert posts == []

    @pytest.mark.asyncio
    async def test_api_error(self, tracker, mock_http):
        mock_http.get_json.side_effect = Exception("API error")
        posts = await tracker._fetch_subreddit("error", days=7)
        assert posts == []

    @pytest.mark.asyncio
    async def test_url_format(self, tracker, mock_http):
        mock_http.get_json.return_value = SAMPLE_REDDIT_RESPONSE
        posts = await tracker._fetch_subreddit("MachineLearning", days=365)
        assert posts[0]["url"].startswith("https://www.reddit.com/r/")


# ─── Fetch all tests ────────────────────────────────────────────────────────

class TestFetchAll:
    @pytest.mark.asyncio
    async def test_fetch_all_structure(self, tracker, mock_http):
        mock_http.get_json.return_value = SAMPLE_REDDIT_RESPONSE
        result = await tracker.fetch_all(days=365)
        assert "posts" in result
        assert "metadata" in result
        assert "subreddits_checked" in result["metadata"]
        assert result["metadata"]["subreddits_checked"] == 2

    @pytest.mark.asyncio
    async def test_filters_by_min_score(self, tracker, mock_http):
        mock_http.get_json.return_value = SAMPLE_REDDIT_RESPONSE
        result = await tracker.fetch_all(days=365)
        # Post with score=3 should be filtered (min_score=5)
        # Only posts with signals AND score >= 5 are kept
        for post in result["posts"]:
            assert post["score"] >= 5

    @pytest.mark.asyncio
    async def test_sorts_by_score(self, tracker, mock_http):
        mock_http.get_json.return_value = SAMPLE_REDDIT_RESPONSE
        result = await tracker.fetch_all(days=365)
        scores = [p["score"] for p in result["posts"]]
        assert scores == sorted(scores, reverse=True)

    @pytest.mark.asyncio
    async def test_respects_watermark(self, tracker, mock_http):
        mock_http.get_json.return_value = SAMPLE_REDDIT_RESPONSE
        result = await tracker.fetch_all(
            days=365, source_watermarks={"MachineLearning": "2026-02-09"}
        )
        # All MachineLearning posts should be skipped, leaving only LocalLLaMA entries
        assert all(post["subreddit"] == "LocalLLaMA" for post in result["posts"])

    @pytest.mark.asyncio
    async def test_empty_subreddits(self, mock_http):
        tracker = RedditTracker({"reddit_tracker": {"subreddits": []}}, http_client=mock_http)
        result = await tracker.fetch_all(days=7)
        assert result["posts"] == []
        assert result["metadata"]["subreddits_checked"] == 0

    @pytest.mark.asyncio
    async def test_handles_exceptions(self, tracker, mock_http):
        mock_http.get_json.side_effect = Exception("Network error")
        result = await tracker.fetch_all(days=7)
        assert result["posts"] == []
        assert result["metadata"]["total_posts"] == 0


# ─── Close tests ────────────────────────────────────────────────────────────

class TestClose:
    @pytest.mark.asyncio
    async def test_close_owned_http(self):
        tracker = RedditTracker({})
        mock = AsyncMock()
        tracker._http = mock
        tracker._owns_http = True
        await tracker.close()
        mock.close.assert_awaited_once()
        assert tracker._http is None

    @pytest.mark.asyncio
    async def test_no_close_shared_http(self, tracker, mock_http):
        await tracker.close()
        mock_http.close.assert_not_awaited()
