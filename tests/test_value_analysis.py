"""Tests for v3 value analysis features."""

import sys
from pathlib import Path

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from analyzers.value_scorer import ValueScorer, ValueAggregator
from analyzers.model_card_analyzer import ModelCardAnalyzer
from scrapers.semantic_scholar import SemanticScholarScraper
from scrapers.pwc_sota import PwCSOTAScraper


class TestValueScorer:
    """Tests for ValueScorer class."""

    @pytest.fixture
    def scorer(self):
        """Create a ValueScorer instance."""
        return ValueScorer()

    def test_init(self, scorer):
        """Test initialization."""
        assert scorer.citation_growth_threshold == 10.0
        assert scorer.model_usage_threshold == 3
        assert len(scorer.TOP_INSTITUTIONS) > 0

    def test_score_dataset_sota_usage(self, scorer):
        """Test scoring for SOTA usage."""
        result = scorer.score_dataset(
            dataset_name="test-dataset",
            sota_model_count=3,
        )
        assert result["score_breakdown"]["sota_usage"] == 30
        assert result["total_score"] >= 30

    def test_score_dataset_citation_growth(self, scorer):
        """Test scoring for citation growth."""
        result = scorer.score_dataset(
            dataset_name="test-dataset",
            citation_monthly_growth=15.0,
        )
        assert result["score_breakdown"]["citation_growth"] == 20

    def test_score_dataset_citation_growth_partial(self, scorer):
        """Test partial scoring for citation growth."""
        result = scorer.score_dataset(
            dataset_name="test-dataset",
            citation_monthly_growth=5.0,  # Half the threshold
        )
        assert result["score_breakdown"]["citation_growth"] == 10

    def test_score_dataset_model_usage(self, scorer):
        """Test scoring for model usage."""
        result = scorer.score_dataset(
            dataset_name="test-dataset",
            model_usage_count=5,
        )
        assert result["score_breakdown"]["model_usage"] == 20

    def test_score_dataset_top_institution(self, scorer):
        """Test scoring for top institution."""
        result = scorer.score_dataset(
            dataset_name="test-dataset",
            institution="Google Research",
        )
        assert result["score_breakdown"]["top_institution"] == 15
        assert result["is_top_institution"] is True

    def test_score_dataset_top_institution_from_authors(self, scorer):
        """Test institution detection from author list."""
        result = scorer.score_dataset(
            dataset_name="test-dataset",
            authors=["John Doe (Stanford)", "Jane Smith (MIT)"],
        )
        assert result["is_top_institution"] is True

    def test_score_dataset_paper_and_code(self, scorer):
        """Test scoring for paper and code."""
        result = scorer.score_dataset(
            dataset_name="test-dataset",
            has_paper=True,
            has_code=True,
        )
        assert result["score_breakdown"]["paper_and_code"] == 10

    def test_score_dataset_paper_only(self, scorer):
        """Test scoring for paper only."""
        result = scorer.score_dataset(
            dataset_name="test-dataset",
            has_paper=True,
            has_code=False,
        )
        assert result["score_breakdown"]["paper_and_code"] == 5

    def test_score_dataset_large_scale(self, scorer):
        """Test scoring for large scale."""
        result = scorer.score_dataset(
            dataset_name="test-dataset",
            size_gb=15.0,
        )
        assert result["score_breakdown"]["large_scale"] == 5

    def test_score_dataset_combined(self, scorer):
        """Test combined scoring."""
        result = scorer.score_dataset(
            dataset_name="test-dataset",
            sota_model_count=2,
            citation_monthly_growth=12.0,
            model_usage_count=4,
            institution="DeepMind",
            has_paper=True,
            has_code=True,
            size_gb=20.0,
        )
        # Should get points from all categories
        assert result["total_score"] >= 80

    def test_batch_score(self, scorer):
        """Test batch scoring."""
        datasets = [
            {"name": "high-value", "sota_model_count": 5, "model_usage_count": 10},
            {"name": "medium-value", "sota_model_count": 1},
            {"name": "low-value"},
        ]
        results = scorer.batch_score(datasets)

        # Should be sorted by score descending
        assert results[0]["dataset_name"] == "high-value"
        assert results[0]["total_score"] > results[1]["total_score"]

    def test_filter_by_score(self, scorer):
        """Test filtering by score."""
        scored = [
            {"total_score": 80},
            {"total_score": 50},
            {"total_score": 30},
        ]
        filtered = scorer.filter_by_score(scored, min_score=40)
        assert len(filtered) == 2


class TestValueAggregator:
    """Tests for ValueAggregator class."""

    @pytest.fixture
    def aggregator(self):
        """Create a ValueAggregator instance."""
        return ValueAggregator()

    def test_init(self, aggregator):
        """Test initialization."""
        assert aggregator.scorer is not None
        assert aggregator._datasets == {}

    def test_add_semantic_scholar_data(self, aggregator):
        """Test adding Semantic Scholar data."""
        papers = [
            {
                "dataset_name": "test dataset",  # Use space instead of hyphen
                "citation_count": 100,
                "citation_monthly_growth": 15.0,
                "url": "https://example.com",
                "authors": ["Author 1"],
            }
        ]
        aggregator.add_semantic_scholar_data(papers)

        assert len(aggregator._datasets) == 1
        key = "test_dataset"
        assert aggregator._datasets[key]["citation_count"] == 100

    def test_add_model_card_data(self, aggregator):
        """Test adding model card data."""
        model_results = {
            "valuable_datasets": [
                {
                    "name": "test dataset",  # Use space instead of hyphen
                    "usage_count": 5,
                    "total_model_downloads": 10000,
                }
            ]
        }
        aggregator.add_model_card_data(model_results)

        assert len(aggregator._datasets) == 1
        key = "test_dataset"
        assert aggregator._datasets[key]["model_usage_count"] == 5

    def test_add_sota_data(self, aggregator):
        """Test adding SOTA data."""
        sota_results = {
            "ranked_datasets": [
                {
                    "name": "test dataset",  # Use space instead of hyphen
                    "sota_model_count": 3,
                    "areas": ["nlp"],
                    "url": "https://example.com",
                }
            ]
        }
        aggregator.add_sota_data(sota_results)

        assert len(aggregator._datasets) == 1
        key = "test_dataset"
        assert aggregator._datasets[key]["sota_model_count"] == 3

    def test_add_huggingface_data(self, aggregator):
        """Test adding HuggingFace data."""
        datasets = [
            {
                "name": "test dataset",  # Use space instead of hyphen
                "downloads": 5000,
                "likes": 100,
                "author": "google",
            }
        ]
        aggregator.add_huggingface_data(datasets)

        assert len(aggregator._datasets) == 1
        key = "test_dataset"
        assert aggregator._datasets[key]["downloads"] == 5000

    def test_get_scored_datasets(self, aggregator):
        """Test getting scored datasets."""
        # Add some data
        aggregator._datasets = {
            "high_value": {
                "name": "high-value",
                "sota_model_count": 5,
                "model_usage_count": 10,
            },
            "low_value": {
                "name": "low-value",
            },
        }

        scored = aggregator.get_scored_datasets(min_score=0)
        assert len(scored) == 2
        assert scored[0]["dataset_name"] == "high-value"

    def test_normalize_name(self, aggregator):
        """Test name normalization."""
        # Hyphens and special chars are removed
        assert aggregator._normalize_name("Test-Dataset") == "testdataset"
        # Spaces become underscores
        assert aggregator._normalize_name("test dataset") == "test_dataset"
        # Exclamation marks are removed, underscores preserved
        assert aggregator._normalize_name("Test_Dataset!") == "test_dataset"


class TestModelCardAnalyzer:
    """Tests for ModelCardAnalyzer class."""

    @pytest.fixture
    def analyzer(self):
        """Create a ModelCardAnalyzer instance."""
        return ModelCardAnalyzer(
            min_model_downloads=100,
            model_limit=10,
            min_dataset_usage=2,
        )

    def test_init(self, analyzer):
        """Test initialization."""
        assert analyzer.min_model_downloads == 100
        assert analyzer.model_limit == 10
        assert analyzer.min_dataset_usage == 2

    def test_extract_datasets_from_card_yaml(self, analyzer):
        """Test extracting datasets from YAML metadata."""
        card_content = """---
datasets:
  - squad
  - glue
  - mmlu
---
# Model Card

This is a test model.
"""
        datasets = analyzer.extract_datasets_from_card("test-model", card_content)

        names = [d["name"] for d in datasets]
        assert "squad" in names
        assert "glue" in names

    def test_extract_datasets_from_card_text(self, analyzer):
        """Test extracting datasets from text mentions."""
        card_content = """
# Model Card

This model was trained on the Wikipedia dataset and fine-tuned on custom data.
"""
        datasets = analyzer.extract_datasets_from_card("test-model", card_content)

        names = [d["name"].lower() for d in datasets]
        assert "wikipedia" in names

    def test_extract_datasets_from_card_hf_links(self, analyzer):
        """Test extracting datasets from HF links."""
        card_content = """
# Model Card

Training data: https://huggingface.co/datasets/openai/webgpt_comparisons
"""
        datasets = analyzer.extract_datasets_from_card("test-model", card_content)

        names = [d["name"] for d in datasets]
        assert "openai/webgpt_comparisons" in names


class TestSemanticScholarScraper:
    """Tests for SemanticScholarScraper class."""

    @pytest.fixture
    def scraper(self):
        """Create a SemanticScholarScraper instance."""
        return SemanticScholarScraper(
            limit=10,
            months_back=6,
            min_citations=10,
            min_monthly_growth=5,
        )

    def test_init(self, scraper):
        """Test initialization."""
        assert scraper.limit == 10
        assert scraper.months_back == 6
        assert scraper.min_citations == 10

    def test_parse_paper(self, scraper):
        """Test paper parsing."""
        item = {
            "paperId": "abc123",
            "title": "Test Dataset Paper",
            "abstract": "We introduce a new dataset...",
            "authors": [{"name": "Author 1"}, {"name": "Author 2"}],
            "year": 2024,
            "citationCount": 50,
            "publicationDate": "2024-01-15",
            "externalIds": {"ArXiv": "2401.12345"},
            "url": "https://example.com",
            "venue": "NeurIPS",
            "fieldsOfStudy": ["Computer Science"],
        }

        paper = scraper._parse_paper(item)

        assert paper["id"] == "abc123"
        assert paper["title"] == "Test Dataset Paper"
        assert paper["citation_count"] == 50
        assert paper["arxiv_id"] == "2401.12345"
        assert len(paper["authors"]) == 2

    def test_filter_by_impact(self, scraper):
        """Test filtering by impact."""
        papers = [
            {"citation_count": 100, "citation_monthly_growth": 3},  # Pass by citations
            {"citation_count": 5, "citation_monthly_growth": 10},  # Pass by growth
            {"citation_count": 3, "citation_monthly_growth": 2},  # Fail both
        ]

        filtered = scraper._filter_by_impact(papers)
        assert len(filtered) == 2

    def test_extract_dataset_info(self, scraper):
        """Test dataset info extraction from paper."""
        paper = {
            "id": "abc123",
            "title": "MMLU: A Benchmark for Multitask Learning",
            "abstract": "We introduce MMLU, a new dataset for evaluating...",
            "citation_count": 500,
            "citation_monthly_growth": 25.0,
            "authors": ["Author 1"],
            "year": 2023,
        }

        dataset_info = scraper.extract_dataset_info(paper)

        assert dataset_info is not None
        assert dataset_info["dataset_name"] == "MMLU"
        assert dataset_info["citation_count"] == 500

    def test_extract_dataset_info_no_match(self, scraper):
        """Test dataset extraction when paper isn't about datasets."""
        paper = {
            "id": "abc123",
            "title": "A New Method for Image Classification",
            "abstract": "We propose a novel neural network architecture...",
        }

        dataset_info = scraper.extract_dataset_info(paper)
        assert dataset_info is None


class TestPwCSOTAScraper:
    """Tests for PwCSOTAScraper class."""

    @pytest.fixture
    def scraper(self):
        """Create a PwCSOTAScraper instance."""
        return PwCSOTAScraper(
            areas=["robotics", "question-answering"],
            top_n=5,
        )

    def test_init(self, scraper):
        """Test initialization."""
        assert "robotics" in scraper.areas
        assert "question-answering" in scraper.areas
        assert scraper.top_n == 5

    def test_generate_report(self, scraper):
        """Test report generation."""
        results = {
            "total_associations": 100,
            "unique_datasets": 50,
            "areas_covered": ["robotics", "nlp"],
            "ranked_datasets": [
                {
                    "name": "test-dataset",
                    "sota_model_count": 10,
                    "areas": ["robotics"],
                    "sota_models": [{"model_name": "TestModel"}],
                }
            ],
        }

        report = scraper.generate_report(results)

        assert "SOTA Dataset Analysis" in report
        assert "test-dataset" in report
        assert "10" in report


class TestIntegration:
    """Integration tests for v3 value analysis."""

    def test_full_scoring_pipeline(self):
        """Test the complete scoring pipeline."""
        aggregator = ValueAggregator()

        # Simulate data from multiple sources
        aggregator.add_semantic_scholar_data(
            [
                {
                    "dataset_name": "high-value-dataset",
                    "citation_count": 500,
                    "citation_monthly_growth": 20.0,
                    "url": "https://example.com",
                    "authors": ["Google Research"],
                }
            ]
        )

        aggregator.add_model_card_data(
            {
                "valuable_datasets": [
                    {
                        "name": "high-value-dataset",
                        "usage_count": 10,
                        "total_model_downloads": 100000,
                    }
                ]
            }
        )

        aggregator.add_sota_data(
            {
                "ranked_datasets": [
                    {
                        "name": "high-value-dataset",
                        "sota_model_count": 5,
                        "areas": ["nlp"],
                    }
                ]
            }
        )

        scored = aggregator.get_scored_datasets()

        assert len(scored) == 1
        assert scored[0]["total_score"] >= 60  # Should be high value
        assert scored[0]["is_top_institution"] is True

    def test_report_generation(self):
        """Test report generation with aggregated data."""
        aggregator = ValueAggregator()

        aggregator._datasets = {
            "dataset_1": {
                "name": "dataset-1",
                "sota_model_count": 5,
                "model_usage_count": 10,
                "citation_monthly_growth": 15.0,
                "institution": "Stanford",
            },
            "dataset_2": {
                "name": "dataset-2",
                "model_usage_count": 3,
            },
        }

        report = aggregator.generate_report(min_score=0)

        assert "High-Value Dataset Report" in report
        assert "dataset-1" in report
