"""Tests for GitHub Trending tracker."""

import sys
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from trackers.github_trending_tracker import GitHubTrendingTracker, SIGNAL_KEYWORDS


SAMPLE_TRENDING_RESPONSE = [
    {
        "author": "openai",
        "name": "dataset-tools",
        "url": "https://github.com/openai/dataset-tools",
        "description": "Tools for building LLM training datasets",
        "language": "Python",
        "stars": 2500,
        "forks": 300,
        "currentPeriodStars": 800,
    },
    {
        "author": "chef",
        "name": "recipe-app",
        "url": "https://github.com/chef/recipe-app",
        "description": "A cooking recipe application",
        "language": "JavaScript",
        "stars": 500,
        "forks": 50,
        "currentPeriodStars": 100,
    },
    {
        "author": "researcher",
        "name": "tiny-benchmark",
        "url": "https://github.com/researcher/tiny-benchmark",
        "description": "Benchmark for evaluation tasks",
        "language": "Python",
        "stars": 5,
        "forks": 1,
        "currentPeriodStars": 3,
    },
]


@pytest.fixture
def config():
    return {
        "github_trending": {
            "enabled": True,
            "languages": ["python"],
            "since": "weekly",
            "min_stars": 10,
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
    return GitHubTrendingTracker(config, http_client=mock_http)


class TestInit:
    def test_defaults(self):
        t = GitHubTrendingTracker()
        assert t.enabled is True
        assert t.min_stars == 10
        assert len(t.languages) > 0
        assert t._owns_http is True

    def test_config_override(self, config):
        t = GitHubTrendingTracker(config)
        assert t.languages == ["python"]
        assert t.since == "weekly"
        assert t.min_stars == 10

    def test_shared_http(self, mock_http):
        t = GitHubTrendingTracker({}, http_client=mock_http)
        assert t._owns_http is False
        assert t._http is mock_http


class TestExtractSignals:
    def test_matches_keywords(self, tracker):
        repo = {"name": "dataset-tools", "description": "LLM training tools"}
        signals = tracker._extract_signals(repo)
        assert "dataset" in signals
        assert "llm" in signals

    def test_no_match(self, tracker):
        repo = {"name": "recipe-app", "description": "cooking recipes"}
        signals = tracker._extract_signals(repo)
        assert signals == []


class TestFetchAll:
    @pytest.mark.asyncio
    async def test_structure(self, tracker, mock_http):
        mock_http.get_json.return_value = SAMPLE_TRENDING_RESPONSE
        result = await tracker.fetch_all(days=7)
        assert "repos" in result
        assert "metadata" in result
        assert "languages_searched" in result["metadata"]

    @pytest.mark.asyncio
    async def test_filters_by_signals(self, tracker, mock_http):
        mock_http.get_json.return_value = SAMPLE_TRENDING_RESPONSE
        result = await tracker.fetch_all(days=7)
        # "recipe-app" has no AI signals → filtered out
        names = [r["name"] for r in result["repos"]]
        assert "recipe-app" not in names

    @pytest.mark.asyncio
    async def test_filters_by_min_stars(self, tracker, mock_http):
        mock_http.get_json.return_value = SAMPLE_TRENDING_RESPONSE
        result = await tracker.fetch_all(days=7)
        # "tiny-benchmark" has 5 stars < 10 min → filtered out
        names = [r["name"] for r in result["repos"]]
        assert "tiny-benchmark" not in names

    @pytest.mark.asyncio
    async def test_sorts_by_period_stars(self, tracker, mock_http):
        mock_http.get_json.return_value = SAMPLE_TRENDING_RESPONSE
        result = await tracker.fetch_all(days=7)
        stars = [r["currentPeriodStars"] for r in result["repos"]]
        assert stars == sorted(stars, reverse=True)

    @pytest.mark.asyncio
    async def test_deduplicates_across_languages(self, mock_http):
        tracker = GitHubTrendingTracker(
            {"github_trending": {"languages": ["python", "jupyter-notebook"]}},
            http_client=mock_http,
        )
        mock_http.get_json.return_value = SAMPLE_TRENDING_RESPONSE
        result = await tracker.fetch_all(days=7)
        urls = [r["url"] for r in result["repos"]]
        assert len(urls) == len(set(urls))

    @pytest.mark.asyncio
    async def test_disabled(self, mock_http):
        tracker = GitHubTrendingTracker(
            {"github_trending": {"enabled": False}},
            http_client=mock_http,
        )
        result = await tracker.fetch_all(days=7)
        assert result["repos"] == []

    @pytest.mark.asyncio
    async def test_handles_error(self, tracker, mock_http):
        mock_http.get_json.side_effect = Exception("Network error")
        result = await tracker.fetch_all(days=7)
        assert result["repos"] == []


class TestClose:
    @pytest.mark.asyncio
    async def test_close_owned_http(self):
        tracker = GitHubTrendingTracker({})
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
