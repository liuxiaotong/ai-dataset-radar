"""Tests for Kaggle scraper."""

import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scrapers.kaggle_scraper import KaggleScraper


SAMPLE_KAGGLE_RESPONSE = [
    {
        "ref": "user1/llm-training-data",
        "title": "LLM Training Dataset",
        "subtitle": "A dataset for training language models",
        "downloadCount": 5000,
        "viewCount": 20000,
        "voteCount": 150,
        "usabilityRating": 0.88,
        "licenseName": "CC0-1.0",
        "tags": [{"name": "nlp"}, {"name": "llm"}],
        "totalBytes": 1073741824,
        "lastUpdated": "2026-02-12T10:00:00Z",
    },
    {
        "ref": "user2/rlhf-preferences",
        "title": "RLHF Preference Pairs",
        "subtitle": "Human preference data for RLHF",
        "downloadCount": 2000,
        "viewCount": 8000,
        "voteCount": 75,
        "usabilityRating": 0.92,
        "licenseName": "Apache 2.0",
        "tags": [{"name": "rlhf"}, {"name": "alignment"}],
        "totalBytes": 536870912,
        "lastUpdated": "2026-02-10T08:00:00Z",
    },
]


class TestKaggleScraper:
    @pytest.fixture
    def scraper(self):
        """Create a scraper with mock credentials."""
        with patch.dict("os.environ", {"KAGGLE_USERNAME": "testuser", "KAGGLE_KEY": "testkey"}):
            return KaggleScraper(
                config={"sources": {"kaggle": {"enabled": True, "keywords": ["llm", "rlhf"]}}},
            )

    @pytest.fixture
    def scraper_no_auth(self):
        """Create a scraper without credentials."""
        with patch.dict("os.environ", {}, clear=True):
            return KaggleScraper(
                config={"sources": {"kaggle": {"enabled": True, "keywords": ["llm"]}}},
            )

    def test_init_with_auth(self, scraper):
        assert scraper._auth_header is not None
        assert scraper._auth_header.startswith("Basic ")
        assert scraper.name == "kaggle"
        assert scraper.source_type == "dataset_registry"

    def test_init_without_auth(self, scraper_no_auth):
        assert scraper_no_auth._auth_header is None

    def test_parse_dataset(self, scraper):
        parsed = scraper._parse_dataset(SAMPLE_KAGGLE_RESPONSE[0])
        assert parsed["source"] == "kaggle"
        assert parsed["id"] == "user1/llm-training-data"
        assert parsed["name"] == "LLM Training Dataset"
        assert parsed["author"] == "user1"
        assert parsed["downloads"] == 5000
        assert "nlp" in parsed["tags"]
        assert "kaggle.com" in parsed["url"]

    def test_parse_dataset_empty_ref(self, scraper):
        assert scraper._parse_dataset({"ref": ""}) is None
        assert scraper._parse_dataset({}) is None

    @pytest.mark.asyncio
    async def test_fetch_no_auth_skips(self, scraper_no_auth):
        """Without credentials, fetch should return empty list."""
        result = await scraper_no_auth.fetch()
        assert result == []

    @pytest.mark.asyncio
    async def test_fetch_disabled(self):
        """Disabled scraper returns empty list."""
        with patch.dict("os.environ", {"KAGGLE_USERNAME": "u", "KAGGLE_KEY": "k"}):
            scraper = KaggleScraper(
                config={"sources": {"kaggle": {"enabled": False}}},
            )
        result = await scraper.fetch()
        assert result == []

    @pytest.mark.asyncio
    async def test_fetch_success(self, scraper):
        mock_http = AsyncMock()
        mock_http.get_json = AsyncMock(return_value=SAMPLE_KAGGLE_RESPONSE)
        scraper._http = mock_http

        result = await scraper.fetch(days=365)
        assert len(result) == 2
        assert result[0]["source"] == "kaggle"
        assert result[1]["id"] == "user2/rlhf-preferences"

    @pytest.mark.asyncio
    async def test_fetch_deduplicates(self, scraper):
        """Same dataset returned by multiple keywords should appear once."""
        mock_http = AsyncMock()
        mock_http.get_json = AsyncMock(return_value=SAMPLE_KAGGLE_RESPONSE)
        scraper._http = mock_http

        result = await scraper.fetch(days=365)
        ids = [ds["id"] for ds in result]
        assert len(ids) == len(set(ids))

    @pytest.mark.asyncio
    async def test_fetch_handles_api_error(self, scraper):
        mock_http = AsyncMock()
        mock_http.get_json = AsyncMock(side_effect=Exception("API error"))
        scraper._http = mock_http

        result = await scraper.fetch(days=7)
        assert result == []

    def test_scraper_registered(self):
        from scrapers.registry import get_scraper

        scraper = get_scraper("kaggle")
        assert scraper is not None
        assert scraper.name == "kaggle"
        assert scraper.source_type == "dataset_registry"
