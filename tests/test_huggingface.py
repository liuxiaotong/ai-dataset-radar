"""Tests for HuggingFace scraper."""

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from scrapers.huggingface import HuggingFaceScraper


class TestHuggingFaceScraper:
    """Tests for HuggingFaceScraper class."""

    @pytest.fixture
    def scraper(self):
        """Create a scraper instance with a mocked async HTTP client."""
        mock_http = MagicMock()
        return HuggingFaceScraper(limit=10, http_client=mock_http)

    def test_init(self, scraper):
        """Test scraper initialization."""
        assert scraper.limit == 10
        assert scraper.DATASETS_URL == "https://huggingface.co/api/datasets"
        assert scraper.MODELS_URL == "https://huggingface.co/api/models"

    def test_parse_dataset(self, scraper):
        """Test parsing a dataset from API response."""
        raw = {
            "id": "test-user/test-dataset",
            "author": "test-user",
            "downloads": 1000,
            "likes": 50,
            "tags": ["nlp", "english"],
            "description": "A test dataset",
            "createdAt": "2024-01-15T10:00:00.000Z",
        }

        result = scraper._parse_dataset(raw)

        assert result["source"] == "huggingface"
        assert result["id"] == "test-user/test-dataset"
        assert result["name"] == "test-dataset"
        assert result["author"] == "test-user"
        assert result["downloads"] == 1000
        assert result["url"] == "https://huggingface.co/datasets/test-user/test-dataset"

    def test_parse_model(self, scraper):
        """Test parsing a model from API response."""
        raw = {
            "id": "test-user/test-model",
            "author": "test-user",
            "downloads": 50000,
            "likes": 100,
            "pipeline_tag": "text-classification",
            "tags": ["pytorch", "bert"],
            "createdAt": "2024-01-15T10:00:00.000Z",
        }

        result = scraper._parse_model(raw)

        assert result["source"] == "huggingface"
        assert result["id"] == "test-user/test-model"
        assert result["name"] == "test-model"
        assert result["pipeline_tag"] == "text-classification"
        assert result["url"] == "https://huggingface.co/test-user/test-model"

    def test_extract_datasets_from_model_card_data(self, scraper):
        """Test extracting datasets from model card_data."""
        model_data = {
            "card_data": {"datasets": ["squad", "glue", "test-user/custom-dataset"]},
            "tags": [],
            "readme": "",
        }

        datasets = scraper.extract_datasets_from_model(model_data)

        assert "squad" in datasets
        assert "glue" in datasets
        assert "test-user/custom-dataset" in datasets

    def test_extract_datasets_from_tags(self, scraper):
        """Test extracting datasets from model tags."""
        model_data = {
            "card_data": {},
            "tags": ["dataset:squad", "dataset:mnli", "pytorch"],
            "readme": "",
        }

        datasets = scraper.extract_datasets_from_model(model_data)

        assert "squad" in datasets
        assert "mnli" in datasets
        assert len(datasets) == 2  # pytorch is not a dataset tag

    def test_extract_datasets_from_readme(self, scraper):
        """Test extracting datasets from README content."""
        model_data = {
            "card_data": {},
            "tags": [],
            "readme": """
# Model Card

This model was trained on the [SQuAD dataset](https://huggingface.co/datasets/squad).

We also used data from huggingface.co/datasets/glue and
[MNLI](https://huggingface.co/datasets/multi_nli).

Trained on the imdb dataset for sentiment analysis.
""",
        }

        datasets = scraper.extract_datasets_from_model(model_data)

        assert "squad" in datasets
        assert "glue" in datasets
        assert "multi_nli" in datasets
        assert "imdb" in datasets

    def test_extract_datasets_combined(self, scraper):
        """Test extracting datasets from multiple sources."""
        model_data = {
            "card_data": {"datasets": ["squad"]},
            "tags": ["dataset:glue"],
            "readme": "Trained on the imdb dataset.",
        }

        datasets = scraper.extract_datasets_from_model(model_data)

        assert "squad" in datasets
        assert "glue" in datasets
        assert "imdb" in datasets
        assert len(datasets) == 3

    async def test_fetch_trending_models(self, scraper):
        """Test fetching trending models."""
        scraper._http.get_json = AsyncMock(return_value=[
            {
                "id": "model/a",
                "author": "test",
                "downloads": 50000,
                "likes": 100,
                "pipeline_tag": "text-classification",
                "tags": [],
                "createdAt": "2024-01-15T10:00:00.000Z",
            },
            {
                "id": "model/b",
                "author": "test",
                "downloads": 500,  # Below threshold
                "likes": 10,
                "pipeline_tag": "fill-mask",
                "tags": [],
                "createdAt": "2024-01-15T10:00:00.000Z",
            },
        ])

        models = await scraper.fetch_trending_models(limit=10, min_downloads=1000)

        assert len(models) == 1
        assert models[0]["id"] == "model/a"
        assert models[0]["downloads"] == 50000

    async def test_fetch_trending_models_error(self, scraper):
        """Test handling API errors when fetching models."""
        scraper._http.get_json = AsyncMock(return_value=None)

        models = await scraper.fetch_trending_models()

        assert models == []

    async def test_fetch_model_card(self, scraper):
        """Test fetching a model card."""
        scraper._http.get_json = AsyncMock(return_value={
            "id": "test/model",
            "author": "test",
            "downloads": 10000,
            "likes": 50,
            "pipeline_tag": "text-classification",
            "tags": [],
            "cardData": {"datasets": ["squad"]},
            "createdAt": "2024-01-15T10:00:00.000Z",
        })
        scraper._http.get_text = AsyncMock(
            return_value="# Model README\n\nTrained on glue."
        )

        result = await scraper.fetch_model_card("test/model")

        assert result is not None
        assert result["id"] == "test/model"
        assert "Model README" in result["readme"]
        assert result["card_data"]["datasets"] == ["squad"]

    async def test_fetch_dataset_info(self, scraper):
        """Test fetching dataset info."""
        scraper._http.get_json = AsyncMock(return_value={
            "id": "squad",
            "author": "rajpurkar",
            "downloads": 500000,
            "likes": 1000,
            "tags": ["question-answering"],
            "description": "Stanford Question Answering Dataset",
            "createdAt": "2020-01-01T00:00:00.000Z",
        })

        result = await scraper.fetch_dataset_info("squad")

        assert result is not None
        assert result["id"] == "squad"
        assert result["downloads"] == 500000

    async def test_fetch_dataset_info_not_found(self, scraper):
        """Test handling 404 when fetching dataset info."""
        scraper._http.get_json = AsyncMock(return_value=None)

        result = await scraper.fetch_dataset_info("nonexistent")

        assert result is None

    @pytest.mark.asyncio
    async def test_fetch_respects_min_timestamp(self, scraper):
        """Ensure fetch stops when reaching older datasets."""
        newer = {
            "id": "user/new-ds",
            "author": "user",
            "downloads": 10,
            "likes": 1,
            "lastModified": "2024-06-10T00:00:00.000Z",
        }
        older = {
            "id": "user/old-ds",
            "author": "user",
            "downloads": 5,
            "likes": 0,
            "lastModified": "2024-06-01T00:00:00.000Z",
        }

        scraper._http.get_json = AsyncMock(return_value=[newer, older])

        results = await scraper.fetch(
            min_timestamp="2024-06-05T00:00:00", max_pages=2
        )

        assert len(results) == 1
        assert results[0]["id"] == "user/new-ds"
