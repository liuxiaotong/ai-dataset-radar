"""Tests for filtering and relevance scoring."""

import sys
from pathlib import Path

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from scrapers.github import GitHubScraper
from scrapers.github_org import GitHubOrgScraper


class TestGitHubRelevance:
    """Tests for GitHub relevance calculation."""

    @pytest.fixture
    def scraper(self):
        return GitHubScraper()

    def test_high_relevance_by_name(self, scraper):
        """Test high relevance when name contains multiple keywords."""
        repo = {
            "name": "awesome-dataset-benchmark",
            "description": "A collection",
            "topics": [],
        }
        relevance = scraper._calculate_relevance(repo)
        assert relevance == "high"

    def test_high_relevance_by_description(self, scraper):
        """Test high relevance when description contains keywords."""
        repo = {
            "name": "my-repo",
            "description": "RLHF dataset for preference training and evaluation",
            "topics": [],
        }
        relevance = scraper._calculate_relevance(repo)
        assert relevance == "high"

    def test_high_relevance_by_topics(self, scraper):
        """Test high relevance when topics contain dataset-related terms."""
        repo = {
            "name": "my-repo",
            "description": "Some code",
            "topics": ["dataset", "machine-learning"],
        }
        relevance = scraper._calculate_relevance(repo)
        assert relevance == "high"

    def test_high_relevance_benchmark_topic(self, scraper):
        """Test high relevance for benchmark topic."""
        repo = {
            "name": "my-repo",
            "description": "Testing things",
            "topics": ["benchmark"],
        }
        relevance = scraper._calculate_relevance(repo)
        assert relevance == "high"

    def test_low_relevance_no_keywords(self, scraper):
        """Test low relevance when no keywords match."""
        repo = {
            "name": "web-app",
            "description": "A simple web application",
            "topics": ["javascript", "react"],
        }
        relevance = scraper._calculate_relevance(repo)
        assert relevance == "low"

    def test_low_relevance_single_keyword(self, scraper):
        """Test low relevance with only one keyword match."""
        repo = {
            "name": "my-repo",
            "description": "Contains dataset word once",
            "topics": [],
        }
        relevance = scraper._calculate_relevance(repo)
        assert relevance == "low"

    def test_custom_keywords(self, scraper):
        """Test relevance with custom keywords."""
        repo = {
            "name": "custom-project",
            "description": "Uses transformers for training",
            "topics": [],
        }
        relevance = scraper._calculate_relevance(repo, keywords=["transformers", "training", "nlp"])
        assert relevance == "high"


class TestGitHubOrgRelevance:
    """Tests for GitHubOrgScraper relevance calculation."""

    @pytest.fixture
    def scraper(self):
        return GitHubOrgScraper()

    def test_high_relevance_dataset_topic(self, scraper):
        """Test high relevance for dataset topic."""
        repo = {
            "name": "project",
            "description": "Something",
            "topics": ["datasets"],
        }
        relevance = scraper._calculate_relevance(repo)
        assert relevance == "high"

    def test_high_relevance_training_data_topic(self, scraper):
        """Test high relevance for training-data topic."""
        repo = {
            "name": "project",
            "description": "Something",
            "topics": ["training-data"],
        }
        relevance = scraper._calculate_relevance(repo)
        assert relevance == "high"

    def test_high_relevance_multiple_keyword_matches(self, scraper):
        """Test high relevance with multiple keyword matches."""
        repo = {
            "name": "annotation-tool",
            "description": "Benchmark evaluation system",
            "topics": [],
        }
        relevance = scraper._calculate_relevance(repo)
        assert relevance == "high"

    def test_low_relevance_unrelated(self, scraper):
        """Test low relevance for unrelated repo."""
        repo = {
            "name": "docs",
            "description": "Documentation website",
            "topics": ["documentation"],
        }
        relevance = scraper._calculate_relevance(repo)
        assert relevance == "low"


class TestGitHubScraperIntegration:
    """Integration tests for GitHubScraper.scrape() with relevance."""

    @pytest.fixture
    def scraper(self):
        return GitHubScraper(limit=5)

    def test_scrape_adds_relevance_field(self, scraper):
        """Test that scrape() adds relevance field to repos."""
        # Create mock fetch method
        original_fetch = scraper.fetch
        scraper.fetch = lambda: [
            {
                "source": "github",
                "id": "test/repo",
                "name": "dataset-project",
                "description": "AI benchmark suite",
                "topics": ["machine-learning"],
            }
        ]

        try:
            result = scraper.scrape()
            assert len(result) == 1
            assert "relevance" in result[0]
            assert result[0]["relevance"] in ["high", "low"]
        finally:
            scraper.fetch = original_fetch

    def test_scrape_preserves_existing_relevance(self, scraper):
        """Test that scrape() doesn't override existing relevance."""
        original_fetch = scraper.fetch
        scraper.fetch = lambda: [
            {
                "source": "github",
                "id": "test/repo",
                "name": "test",
                "description": "test",
                "topics": [],
                "relevance": "manual",
            }
        ]

        try:
            result = scraper.scrape()
            assert result[0]["relevance"] == "manual"
        finally:
            scraper.fetch = original_fetch


class TestKeywordMatching:
    """Tests for keyword matching in different fields."""

    @pytest.fixture
    def scraper(self):
        return GitHubScraper()

    def test_case_insensitive_matching(self, scraper):
        """Test that keyword matching is case-insensitive."""
        repo = {
            "name": "My-DATASET-Project",
            "description": "RLHF training",
            "topics": [],
        }
        relevance = scraper._calculate_relevance(repo)
        assert relevance == "high"

    def test_partial_word_matching(self, scraper):
        """Test that partial words match keywords."""
        repo = {
            "name": "datasets-collection",
            "description": "benchmarking tools",
            "topics": [],
        }
        relevance = scraper._calculate_relevance(repo)
        assert relevance == "high"

    def test_topic_exact_matching(self, scraper):
        """Test that topics require exact matching."""
        repo = {
            "name": "project",
            "description": "project",
            "topics": ["dataset", "benchmark"],
        }
        relevance = scraper._calculate_relevance(repo)
        assert relevance == "high"

    def test_empty_fields_dont_crash(self, scraper):
        """Test handling of empty or None fields."""
        repo = {
            "name": "",
            "description": None,
            "topics": [],
        }
        relevance = scraper._calculate_relevance(repo)
        assert relevance == "low"

    def test_missing_fields_dont_crash(self, scraper):
        """Test handling of missing fields."""
        repo = {}
        relevance = scraper._calculate_relevance(repo)
        assert relevance == "low"
