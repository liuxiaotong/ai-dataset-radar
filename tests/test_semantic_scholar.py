"""Tests for Semantic Scholar scraper."""

import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scrapers.semantic_scholar import SemanticScholarScraper


SAMPLE_RESPONSE = {
    "total": 2,
    "data": [
        {
            "paperId": "abc123",
            "title": "A Large-Scale Dataset for LLM Training",
            "abstract": "We present a new dataset...",
            "authors": [{"name": "Alice"}, {"name": "Bob"}],
            "year": 2026,
            "citationCount": 42,
            "publicationDate": "2026-02-10",
            "externalIds": {"ArXiv": "2602.12345", "DOI": "10.1234/test"},
            "url": "https://www.semanticscholar.org/paper/abc123",
            "venue": "NeurIPS",
        },
        {
            "paperId": "def456",
            "title": "Benchmark for Reasoning Tasks",
            "abstract": "This benchmark evaluates...",
            "authors": [{"name": "Charlie"}],
            "year": 2026,
            "citationCount": 15,
            "publicationDate": "2026-02-08",
            "externalIds": {},
            "url": None,
            "venue": "",
        },
    ],
}


class TestSemanticScholarScraper:
    @pytest.fixture
    def scraper(self):
        return SemanticScholarScraper(
            config={"sources": {"semantic_scholar": {"enabled": True, "keywords": ["test query"]}}},
        )

    def test_init_defaults(self):
        s = SemanticScholarScraper()
        assert s.name == "semantic_scholar"
        assert s.source_type == "paper"
        assert s.enabled is True
        assert len(s.keywords) > 0

    def test_init_config(self):
        s = SemanticScholarScraper(
            config={"sources": {"semantic_scholar": {"enabled": False, "keywords": ["a", "b"]}}},
        )
        assert s.enabled is False
        assert s.keywords == ["a", "b"]

    def test_init_api_key(self):
        with patch.dict("os.environ", {"SEMANTIC_SCHOLAR_API_KEY": "testkey"}):
            s = SemanticScholarScraper()
            assert s._headers.get("x-api-key") == "testkey"

    def test_parse_paper(self, scraper):
        parsed = scraper._parse_paper(SAMPLE_RESPONSE["data"][0])
        assert parsed["source"] == "semantic_scholar"
        assert parsed["id"] == "abc123"
        assert parsed["title"] == "A Large-Scale Dataset for LLM Training"
        assert "Alice" in parsed["authors"]
        assert parsed["citation_count"] == 42
        assert parsed["arxiv_id"] == "2602.12345"
        assert parsed["doi"] == "10.1234/test"

    def test_parse_paper_no_url(self, scraper):
        parsed = scraper._parse_paper(SAMPLE_RESPONSE["data"][1])
        assert "semanticscholar.org" in parsed["url"]

    def test_parse_paper_empty(self, scraper):
        assert scraper._parse_paper({}) is None

    @pytest.mark.asyncio
    async def test_fetch_disabled(self):
        s = SemanticScholarScraper(
            config={"sources": {"semantic_scholar": {"enabled": False}}},
        )
        result = await s.fetch()
        assert result == []

    @pytest.mark.asyncio
    async def test_fetch_success(self, scraper):
        mock_http = AsyncMock()
        mock_http.get_json = AsyncMock(return_value=SAMPLE_RESPONSE)
        scraper._http = mock_http

        result = await scraper.fetch(days=365)
        assert len(result) == 2
        assert result[0]["source"] == "semantic_scholar"
        assert result[1]["id"] == "def456"

    @pytest.mark.asyncio
    async def test_fetch_deduplicates(self, scraper):
        mock_http = AsyncMock()
        mock_http.get_json = AsyncMock(return_value=SAMPLE_RESPONSE)
        scraper._http = mock_http
        scraper.keywords = ["query1", "query2"]

        result = await scraper.fetch(days=365)
        ids = [p["id"] for p in result]
        assert len(ids) == len(set(ids))

    @pytest.mark.asyncio
    async def test_fetch_handles_error(self, scraper):
        mock_http = AsyncMock()
        mock_http.get_json = AsyncMock(side_effect=Exception("API error"))
        scraper._http = mock_http

        result = await scraper.fetch(days=7)
        assert result == []

    def test_scraper_registered(self):
        from scrapers.registry import get_scraper

        scraper = get_scraper("semantic_scholar")
        assert scraper is not None
        assert scraper.name == "semantic_scholar"
        assert scraper.source_type == "paper"
