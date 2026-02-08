"""Tests for HuggingFace Papers scraper."""

import sys
from pathlib import Path
from unittest.mock import patch, AsyncMock

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from scrapers.hf_papers import HFPapersScraper


class TestHFPapersScraper:
    """Tests for HFPapersScraper class."""

    @pytest.fixture
    def scraper(self):
        """Create a scraper instance."""
        return HFPapersScraper(limit=10, days=7)

    def test_init(self, scraper):
        """Test scraper initialization."""
        assert scraper.limit == 10
        assert scraper.days == 7

    def test_parse_api_paper(self, scraper):
        """Test parsing a paper from API response."""
        raw = {
            "paper": {
                "id": "2401.12345",
                "title": "A New Dataset for NLP",
                "summary": "We present a new dataset...",
                "authors": [
                    {"name": "John Doe"},
                    {"name": "Jane Smith"},
                ],
                "publishedAt": "2024-01-15T10:00:00.000Z",
                "upvotes": 42,
            },
            "numComments": 5,
        }

        result = scraper._parse_api_paper(raw)

        assert result is not None
        assert result["source"] == "hf_papers"
        assert result["arxiv_id"] == "2401.12345"
        assert result["title"] == "A New Dataset for NLP"
        assert len(result["authors"]) == 2
        assert result["upvotes"] == 42
        assert "huggingface.co/papers" in result["url"]

    def test_is_dataset_related_by_title(self, scraper):
        """Test dataset detection by title."""
        paper = {
            "title": "Introducing a New Benchmark for LLM Evaluation",
            "summary": "",
        }
        assert scraper._is_dataset_related(paper) is True

    def test_is_dataset_related_by_summary(self, scraper):
        """Test dataset detection by summary."""
        paper = {
            "title": "Improving Language Models",
            "summary": "We collect a new instruction dataset with 100k examples.",
        }
        assert scraper._is_dataset_related(paper) is True

    def test_is_dataset_related_rlhf(self, scraper):
        """Test detection of RLHF-related papers."""
        paper = {
            "title": "Training with Human Feedback",
            "summary": "We use RLHF to improve model alignment.",
        }
        assert scraper._is_dataset_related(paper) is True

    def test_is_not_dataset_related(self, scraper):
        """Test non-dataset paper detection."""
        paper = {
            "title": "Efficient Attention Mechanisms",
            "summary": "We propose a new attention mechanism for transformers.",
        }
        assert scraper._is_dataset_related(paper) is False

    async def test_fetch_from_api(self, scraper):
        """Test fetching papers from API."""
        scraper._http.get_json = AsyncMock(
            return_value=[
                {
                    "paper": {
                        "id": "2401.12345",
                        "title": "New Dataset Paper",
                        "summary": "A dataset for testing",
                        "authors": [{"name": "Author"}],
                        "publishedAt": "2024-01-15T10:00:00.000Z",
                        "upvotes": 10,
                    },
                    "numComments": 2,
                }
            ]
        )

        papers = await scraper._fetch_from_api()

        assert len(papers) == 1
        assert papers[0]["arxiv_id"] == "2401.12345"

    async def test_fetch_from_api_error(self, scraper):
        """Test handling API errors."""
        scraper._http.get_json = AsyncMock(return_value=None)

        papers = await scraper._fetch_from_api()

        assert papers == []

    @patch.object(HFPapersScraper, "_fetch_from_page", new_callable=AsyncMock)
    @patch.object(HFPapersScraper, "_fetch_from_api", new_callable=AsyncMock)
    async def test_fetch_marks_dataset_papers(self, mock_api, mock_page, scraper):
        """Test that fetch marks dataset-related papers."""
        mock_api.return_value = [
            {
                "source": "hf_papers",
                "id": "2401.11111",
                "arxiv_id": "2401.11111",
                "title": "New Dataset for NLP",
                "summary": "We present a benchmark...",
                "authors": [],
                "upvotes": 5,
                "comments": 0,
                "published_at": None,
                "url": "https://huggingface.co/papers/2401.11111",
                "arxiv_url": "https://arxiv.org/abs/2401.11111",
            },
            {
                "source": "hf_papers",
                "id": "2401.22222",
                "arxiv_id": "2401.22222",
                "title": "Better Attention Mechanism",
                "summary": "We improve transformers...",
                "authors": [],
                "upvotes": 3,
                "comments": 0,
                "published_at": None,
                "url": "https://huggingface.co/papers/2401.22222",
                "arxiv_url": "https://arxiv.org/abs/2401.22222",
            },
        ]
        mock_page.return_value = []

        papers = await scraper.fetch()

        # First paper should be marked as dataset-related
        dataset_paper = next(p for p in papers if p["arxiv_id"] == "2401.11111")
        assert dataset_paper["is_dataset_paper"] is True

        # Second paper should not be marked
        other_paper = next(p for p in papers if p["arxiv_id"] == "2401.22222")
        assert other_paper["is_dataset_paper"] is False

    async def test_get_dataset_papers(self, scraper):
        """Test getting only dataset-related papers."""
        mock_fetch = AsyncMock(
            return_value=[
                {"arxiv_id": "1", "is_dataset_paper": True},
                {"arxiv_id": "2", "is_dataset_paper": False},
                {"arxiv_id": "3", "is_dataset_paper": True},
            ]
        )

        with patch.object(scraper, "fetch", new=mock_fetch):
            dataset_papers = await scraper.get_dataset_papers()

            assert len(dataset_papers) == 2
            assert all(p["is_dataset_paper"] for p in dataset_papers)

    def test_parse_page_article(self, scraper):
        """Test parsing paper from HTML article."""
        from bs4 import BeautifulSoup

        html = """
        <article>
            <h3><a href="/papers/2401.12345">A Dataset Paper Title</a></h3>
            <p>This paper presents a new dataset...</p>
        </article>
        """
        soup = BeautifulSoup(html, "html.parser")
        article = soup.select_one("article")

        result = scraper._parse_page_article(article)

        assert result is not None
        assert result["arxiv_id"] == "2401.12345"
        assert "Dataset Paper" in result["title"]


class TestHFPapersScraperIntegration:
    """Integration tests (require network)."""

    @pytest.mark.skip(reason="Requires network access")
    async def test_fetch_real_data(self):
        """Test fetching real data from HuggingFace."""
        scraper = HFPapersScraper(limit=5)
        papers = await scraper.fetch()

        assert isinstance(papers, list)
        # Should find some papers
        assert len(papers) >= 0
