"""Tests for Product Hunt tracker."""

import sys
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from trackers.producthunt_tracker import ProductHuntTracker

SAMPLE_FEED_XML = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>Product Hunt</title>
  <entry>
    <title>AI Dataset Builder - Create training datasets with AI</title>
    <summary>Build custom LLM training datasets from any data source</summary>
    <link href="https://www.producthunt.com/posts/ai-dataset-builder"/>
    <author><name>founder1</name></author>
    <published>2026-02-12T10:00:00Z</published>
  </entry>
  <entry>
    <title>CookMaster - Smart recipe planner</title>
    <summary>Plan your meals with smart suggestions</summary>
    <link href="https://www.producthunt.com/posts/cookmaster"/>
    <author><name>chef1</name></author>
    <published>2026-02-11T08:00:00Z</published>
  </entry>
  <entry>
    <title>VectorDB Pro - Embeddings made simple</title>
    <summary>A vector database for RAG and embedding workflows</summary>
    <link href="https://www.producthunt.com/posts/vectordb-pro"/>
    <author><name>dev1</name></author>
    <published>2026-02-10T12:00:00Z</published>
  </entry>
  <entry>
    <title>Old AI Tool</title>
    <summary>An old AI tool for machine learning</summary>
    <link href="https://www.producthunt.com/posts/old-ai-tool"/>
    <author><name>dev2</name></author>
    <published>2025-01-01T00:00:00Z</published>
  </entry>
</feed>"""


@pytest.fixture
def config():
    return {
        "producthunt_tracker": {
            "enabled": True,
        }
    }


@pytest.fixture
def mock_http():
    http = AsyncMock()
    http.get_text = AsyncMock(return_value=SAMPLE_FEED_XML)
    http.close = AsyncMock()
    return http


@pytest.fixture
def tracker(config, mock_http):
    return ProductHuntTracker(config, http_client=mock_http)


class TestInit:
    def test_defaults(self):
        t = ProductHuntTracker()
        assert t.enabled is True
        assert t._owns_http is True
        assert len(t.search_keywords) > 0

    def test_config_override(self):
        t = ProductHuntTracker({"producthunt_tracker": {"enabled": False}})
        assert t.enabled is False

    def test_shared_http(self, mock_http):
        t = ProductHuntTracker({}, http_client=mock_http)
        assert t._owns_http is False
        assert t._http is mock_http


class TestExtractSignals:
    def test_matches(self, tracker):
        signals = tracker._extract_signals("AI Dataset Builder", "training datasets")
        assert "ai" in signals
        assert "dataset" in signals

    def test_no_match(self, tracker):
        signals = tracker._extract_signals("CookMaster", "plan your meals")
        assert signals == []


class TestFetchAll:
    @pytest.mark.asyncio
    async def test_structure(self, tracker):
        result = await tracker.fetch_all(days=365)
        assert "products" in result
        assert "metadata" in result

    @pytest.mark.asyncio
    async def test_filters_by_signals(self, tracker):
        result = await tracker.fetch_all(days=365)
        titles = [p["title"] for p in result["products"]]
        # CookMaster has no AI signals → filtered
        assert "CookMaster - Smart recipe planner" not in titles
        # AI Dataset Builder should be present
        assert any("AI Dataset Builder" in t for t in titles)

    @pytest.mark.asyncio
    async def test_filters_by_date(self, tracker):
        result = await tracker.fetch_all(days=30)
        # "Old AI Tool" from 2025 should be filtered out
        titles = [p["title"] for p in result["products"]]
        assert "Old AI Tool" not in titles

    @pytest.mark.asyncio
    async def test_respects_watermark(self, tracker):
        result = await tracker.fetch_all(
            days=365,
            source_watermarks={"producthunt": "2026-02-12"},
        )
        for product in result["products"]:
            assert product["date"] >= "2026-02-12"

    @pytest.mark.asyncio
    async def test_disabled(self, mock_http):
        tracker = ProductHuntTracker(
            {"producthunt_tracker": {"enabled": False}},
            http_client=mock_http,
        )
        result = await tracker.fetch_all(days=7)
        assert result["products"] == []

    @pytest.mark.asyncio
    async def test_handles_error(self, tracker, mock_http):
        mock_http.get_text.side_effect = Exception("Network error")
        result = await tracker.fetch_all(days=7)
        assert result["products"] == []


class TestClose:
    @pytest.mark.asyncio
    async def test_close_owned_http(self):
        tracker = ProductHuntTracker({})
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
