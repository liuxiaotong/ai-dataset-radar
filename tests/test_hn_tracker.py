"""Tests for Hacker News tracker."""

import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from trackers.hn_tracker import HNTracker, SIGNAL_KEYWORDS


@pytest.fixture
def config():
    return {
        "hn_tracker": {
            "enabled": True,
            "min_points": 10,
            "search_keywords": ["dataset", "LLM"],
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
    return HNTracker(config, http_client=mock_http)


SAMPLE_ALGOLIA_RESPONSE = {
    "hits": [
        {
            "objectID": "111",
            "title": "New dataset release for LLM training",
            "url": "https://example.com/dataset",
            "author": "researcher1",
            "points": 150,
            "num_comments": 42,
            "created_at": "2026-02-12T10:00:00.000Z",
        },
        {
            "objectID": "222",
            "title": "Cooking pasta at home",
            "url": "https://example.com/pasta",
            "author": "chef1",
            "points": 25,
            "num_comments": 5,
            "created_at": "2026-02-11T08:00:00.000Z",
        },
        {
            "objectID": "333",
            "title": "Open source benchmark for reasoning",
            "url": "https://example.com/bench",
            "author": "mldev",
            "points": 5,
            "num_comments": 3,
            "created_at": "2026-02-10T12:00:00.000Z",
        },
    ],
}


# ─── Init tests ─────────────────────────────────────────────────────────────

class TestHNTrackerInit:
    def test_defaults(self):
        t = HNTracker({})
        assert t.min_points == 10
        assert len(t.search_keywords) > 0

    def test_config_override(self, config):
        t = HNTracker(config)
        assert t.search_keywords == ["dataset", "LLM"]
        assert t.min_points == 10

    def test_owns_http_when_none(self):
        t = HNTracker({})
        assert t._owns_http is True

    def test_shared_http(self, mock_http):
        t = HNTracker({}, http_client=mock_http)
        assert t._owns_http is False
        assert t._http is mock_http


# ─── Signal extraction tests ────────────────────────────────────────────────

class TestExtractSignals:
    def test_matches_keywords(self, tracker):
        story = {"title": "New dataset release for LLM training"}
        signals = tracker._extract_signals(story)
        assert "dataset" in signals

    def test_no_match(self, tracker):
        story = {"title": "How to cook pasta"}
        signals = tracker._extract_signals(story)
        assert signals == []

    def test_case_insensitive(self, tracker):
        story = {"title": "DATASET RELEASE"}
        signals = tracker._extract_signals(story)
        assert "dataset" in signals


# ─── Fetch all tests ────────────────────────────────────────────────────────

class TestFetchAll:
    @pytest.mark.asyncio
    async def test_fetch_all_structure(self, tracker, mock_http):
        mock_http.get_json.return_value = SAMPLE_ALGOLIA_RESPONSE
        result = await tracker.fetch_all(days=365)
        assert "stories" in result
        assert "metadata" in result
        assert "keywords_searched" in result["metadata"]
        assert result["metadata"]["keywords_searched"] == 2

    @pytest.mark.asyncio
    async def test_filters_by_min_points(self, tracker, mock_http):
        mock_http.get_json.return_value = SAMPLE_ALGOLIA_RESPONSE
        result = await tracker.fetch_all(days=365)
        for story in result["stories"]:
            assert story["points"] >= 10

    @pytest.mark.asyncio
    async def test_sorts_by_points(self, tracker, mock_http):
        mock_http.get_json.return_value = SAMPLE_ALGOLIA_RESPONSE
        result = await tracker.fetch_all(days=365)
        points = [s["points"] for s in result["stories"]]
        assert points == sorted(points, reverse=True)

    @pytest.mark.asyncio
    async def test_deduplicates_across_queries(self, tracker, mock_http):
        """Same story returned by multiple keyword queries should appear once."""
        mock_http.get_json.return_value = SAMPLE_ALGOLIA_RESPONSE
        result = await tracker.fetch_all(days=365)
        ids = [s["objectID"] for s in result["stories"]]
        assert len(ids) == len(set(ids))

    @pytest.mark.asyncio
    async def test_respects_watermark(self, tracker, mock_http):
        mock_http.get_json.return_value = SAMPLE_ALGOLIA_RESPONSE
        result = await tracker.fetch_all(
            days=365, source_watermarks={"hn": "2026-02-12"}
        )
        # Only stories after 2026-02-12 should remain
        for story in result["stories"]:
            if story.get("date"):
                assert story["date"] > "2026-02-12"

    @pytest.mark.asyncio
    async def test_empty_keywords(self, mock_http):
        tracker = HNTracker({"hn_tracker": {"search_keywords": []}}, http_client=mock_http)
        result = await tracker.fetch_all(days=7)
        assert result["stories"] == []
        assert result["metadata"]["keywords_searched"] == 0

    @pytest.mark.asyncio
    async def test_handles_exceptions(self, tracker, mock_http):
        mock_http.get_json.side_effect = Exception("Network error")
        result = await tracker.fetch_all(days=7)
        assert result["stories"] == []
        assert result["metadata"]["relevant_stories"] == 0


# ─── Close tests ────────────────────────────────────────────────────────────

class TestClose:
    @pytest.mark.asyncio
    async def test_close_owned_http(self):
        tracker = HNTracker({})
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
