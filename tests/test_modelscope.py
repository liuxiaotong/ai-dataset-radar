"""Tests for ModelScope scraper."""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from scrapers.modelscope import ModelScopeScraper


class TestModelScopeScraper:
    """Tests for ModelScopeScraper class."""

    @pytest.fixture
    def scraper(self):
        """Create a scraper instance."""
        return ModelScopeScraper(limit=10)

    def test_init(self, scraper):
        """Test scraper initialization."""
        assert scraper.limit == 10
        assert scraper.name == "modelscope"
        assert scraper.source_type == "dataset_registry"
        assert scraper.DATASETS_API == "https://modelscope.cn/api/v1/datasets"

    def test_parse_dataset(self, scraper):
        """Test parsing a dataset from API response."""
        raw = {
            "Name": "test-dataset",
            "Owner": "qwen",
            "Downloads": 5000,
            "Likes": 100,
            "Tags": ["nlp", "chinese"],
            "Description": "A test dataset",
            "ChineseDescription": "测试数据集",
            "License": "apache-2.0",
            "GmtCreate": "2024-06-15T10:00:00.000Z",
            "GmtModified": "2024-06-20T10:00:00.000Z",
        }

        result = scraper._parse_dataset(raw)

        assert result is not None
        assert result["source"] == "modelscope"
        assert result["id"] == "qwen/test-dataset"
        assert result["name"] == "test-dataset"
        assert result["author"] == "qwen"
        assert result["downloads"] == 5000
        assert result["description"] == "测试数据集"
        assert result["url"] == "https://modelscope.cn/datasets/qwen/test-dataset"

    def test_parse_dataset_lowercase_keys(self, scraper):
        """Test parsing with lowercase API keys (alternative response format)."""
        raw = {
            "name": "alt-dataset",
            "owner": "deepseek-ai",
            "downloads": 3000,
            "likes": 50,
            "tags": ["code"],
            "description": "An alternative format dataset",
            "license": "mit",
            "gmt_create": "2024-07-01T08:00:00.000Z",
        }

        result = scraper._parse_dataset(raw)

        assert result is not None
        assert result["source"] == "modelscope"
        assert result["id"] == "deepseek-ai/alt-dataset"
        assert result["name"] == "alt-dataset"
        assert result["author"] == "deepseek-ai"

    def test_parse_dataset_missing_owner(self, scraper):
        """Test parsing when owner is missing."""
        raw = {
            "Name": "orphan-dataset",
            "Downloads": 100,
        }

        result = scraper._parse_dataset(raw)

        assert result is not None
        assert result["id"] == "orphan-dataset"
        assert result["author"] == ""

    def test_parse_dataset_with_org_override(self, scraper):
        """Test parsing with org parameter override."""
        raw = {
            "Name": "special-dataset",
            "Downloads": 200,
        }

        result = scraper._parse_dataset(raw, org="BAAI")

        assert result is not None
        assert result["id"] == "BAAI/special-dataset"
        assert result["author"] == "BAAI"

    @patch("scrapers.modelscope.requests.get")
    def test_fetch_recent(self, mock_get, scraper):
        """Test fetching recent datasets."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "Data": {
                "Datasets": [
                    {
                        "Name": "dataset-a",
                        "Owner": "damo",
                        "Downloads": 1000,
                        "Likes": 10,
                        "Tags": [],
                        "Description": "Dataset A",
                        "GmtCreate": "2024-08-01T00:00:00.000Z",
                    },
                    {
                        "Name": "dataset-b",
                        "Owner": "qwen",
                        "Downloads": 2000,
                        "Likes": 20,
                        "Tags": ["nlp"],
                        "Description": "Dataset B",
                        "GmtCreate": "2024-08-02T00:00:00.000Z",
                    },
                ]
            }
        }
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        results = scraper._fetch_recent()

        assert len(results) == 2
        assert results[0]["id"] == "damo/dataset-a"
        assert results[1]["id"] == "qwen/dataset-b"

    @patch("scrapers.modelscope.requests.get")
    def test_fetch_recent_error(self, mock_get, scraper):
        """Test handling API errors."""
        import requests as req
        mock_get.side_effect = req.RequestException("Connection error")

        results = scraper._fetch_recent()

        assert results == []

    @patch("scrapers.modelscope.requests.get")
    def test_fetch_org_datasets(self, mock_get, scraper):
        """Test fetching datasets from specific organizations."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "Data": {
                "Datasets": [
                    {
                        "Name": "org-dataset",
                        "Owner": "qwen",
                        "Downloads": 500,
                        "Tags": [],
                        "Description": "Org dataset",
                        "GmtCreate": "2024-08-01T00:00:00.000Z",
                    },
                ]
            }
        }
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        results = scraper._fetch_org_datasets(["qwen"])

        assert len(results) == 1
        assert results[0]["author"] == "qwen"

    @patch("scrapers.modelscope.requests.get")
    def test_scrape_targeted(self, mock_get, scraper):
        """Test scrape with targeted org config."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "Data": {"Datasets": []}
        }
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        results = scraper.scrape(config={"watch_orgs": ["qwen"]})

        assert results == []
        mock_get.assert_called_once()

    def test_scraper_registered(self):
        """Test that the scraper is properly registered."""
        from scrapers.registry import get_scraper
        scraper = get_scraper("modelscope")
        assert scraper is not None
        assert scraper.name == "modelscope"
        assert scraper.source_type == "dataset_registry"
