"""Tests for business intelligence features."""

import sys
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from filters import DomainFilter, OrganizationFilter, PostTrainingFilter
from analyzers.opportunities import OpportunityAnalyzer


class TestDomainFilter:
    """Tests for DomainFilter class."""

    @pytest.fixture
    def focus_areas(self):
        """Sample focus areas configuration."""
        return {
            "robotics": {
                "enabled": True,
                "keywords": ["robotics", "manipulation", "embodied", "gripper"],
                "hf_tags": ["task_categories:robotics"],
            },
            "rlhf": {
                "enabled": True,
                "keywords": ["preference", "human feedback", "RLHF", "DPO"],
                "hf_tags": [],
            },
            "multimodal": {
                "enabled": True,
                "keywords": ["vision-language", "VLM", "multimodal"],
                "hf_tags": ["task_categories:visual-question-answering"],
            },
            "disabled_area": {
                "enabled": False,
                "keywords": ["should", "not", "match"],
                "hf_tags": [],
            },
        }

    @pytest.fixture
    def domain_filter(self, focus_areas):
        """Create a DomainFilter instance."""
        return DomainFilter(focus_areas)

    def test_init(self, domain_filter):
        """Test initialization."""
        assert "robotics" in domain_filter._compiled_areas
        assert "rlhf" in domain_filter._compiled_areas
        assert "disabled_area" not in domain_filter._compiled_areas

    def test_classify_item_by_keyword(self, domain_filter):
        """Test item classification by keyword."""
        item = {
            "name": "robot-manipulation-dataset",
            "description": "A dataset for robotic manipulation tasks",
        }
        domains = domain_filter.classify_item(item)
        assert "robotics" in domains

    def test_classify_item_by_description(self, domain_filter):
        """Test item classification by description."""
        item = {
            "name": "some-dataset",
            "summary": "Training data for RLHF with human feedback",
        }
        domains = domain_filter.classify_item(item)
        assert "rlhf" in domains

    def test_classify_item_by_tag(self, domain_filter):
        """Test item classification by HF tag."""
        item = {
            "name": "robot-data",
            "description": "Some data",
            "tags": ["task_categories:robotics"],
        }
        domains = domain_filter.classify_item(item)
        assert "robotics" in domains

    def test_classify_item_multiple_domains(self, domain_filter):
        """Test item matching multiple domains."""
        item = {
            "name": "multimodal-robot-dataset",
            "description": "VLM training for embodied agents",
        }
        domains = domain_filter.classify_item(item)
        assert "robotics" in domains
        assert "multimodal" in domains

    def test_classify_item_no_match(self, domain_filter):
        """Test item that doesn't match any domain."""
        item = {
            "name": "generic-dataset",
            "description": "A general purpose dataset",
        }
        domains = domain_filter.classify_item(item)
        assert domains == []

    def test_filter_by_domain(self, domain_filter):
        """Test filtering items by domain."""
        items = [
            {"name": "robot-data", "description": "manipulation dataset"},
            {"name": "text-data", "description": "text classification"},
            {"name": "gripper-data", "description": "gripper control"},
        ]
        filtered = domain_filter.filter_by_domain(items, "robotics")
        assert len(filtered) == 2

    def test_classify_all(self, domain_filter):
        """Test classifying all items."""
        items = [
            {"name": "robot-data", "description": "manipulation"},
            {"name": "rlhf-data", "description": "human feedback"},
            {"name": "generic", "description": "general data"},
        ]
        result = domain_filter.classify_all(items)
        assert len(result["robotics"]) == 1
        assert len(result["rlhf"]) == 1
        assert len(result["uncategorized"]) == 1

    def test_enrich_items(self, domain_filter):
        """Test enriching items with domain info."""
        items = [
            {"name": "robot-manipulation", "description": "gripper control"},
        ]
        enriched = domain_filter.enrich_items(items)
        assert "domains" in enriched[0]
        assert "robotics" in enriched[0]["domains"]


class TestOrganizationFilter:
    """Tests for OrganizationFilter class."""

    @pytest.fixture
    def tracked_orgs(self):
        """Sample tracked organizations."""
        return {
            "bytedance": ["ByteDance", "字节", "TikTok"],
            "openai": ["OpenAI"],
            "google": ["Google", "DeepMind"],
        }

    @pytest.fixture
    def org_filter(self, tracked_orgs):
        """Create an OrganizationFilter instance."""
        return OrganizationFilter(tracked_orgs)

    def test_detect_org_by_author(self, org_filter):
        """Test organization detection by author."""
        item = {
            "author": "bytedance-research",
            "name": "some-dataset",
        }
        org = org_filter.detect_org(item)
        assert org == "bytedance"

    def test_detect_org_by_description(self, org_filter):
        """Test organization detection in description."""
        item = {
            "author": "unknown",
            "description": "Dataset released by Google DeepMind",
        }
        org = org_filter.detect_org(item)
        assert org == "google"

    def test_detect_org_by_authors_list(self, org_filter):
        """Test organization detection in authors list."""
        item = {
            "authors": ["John Doe", "Jane Smith (OpenAI)"],
            "title": "Some paper",
        }
        org = org_filter.detect_org(item)
        assert org == "openai"

    def test_detect_org_no_match(self, org_filter):
        """Test when no organization is detected."""
        item = {
            "author": "independent-researcher",
            "description": "A personal project",
        }
        org = org_filter.detect_org(item)
        assert org is None

    def test_classify_all(self, org_filter):
        """Test classifying all items by organization."""
        items = [
            {"author": "bytedance", "name": "bd-data"},
            {"author": "openai", "name": "oai-data"},
            {"author": "random", "name": "other-data"},
        ]
        result = org_filter.classify_all(items)
        assert len(result["bytedance"]) == 1
        assert len(result["openai"]) == 1
        assert len(result["other"]) == 1

    def test_enrich_items(self, org_filter):
        """Test enriching items with org detection."""
        items = [
            {"author": "google-research", "name": "test"},
        ]
        enriched = org_filter.enrich_items(items)
        assert enriched[0]["detected_org"] == "google"


class TestOpportunityAnalyzer:
    """Tests for OpportunityAnalyzer class."""

    @pytest.fixture
    def mock_db(self):
        """Create a mock database."""
        return MagicMock()

    @pytest.fixture
    def config(self):
        """Sample configuration."""
        return {
            "opportunities": {
                "annotation_signals": [
                    "human annotation",
                    "crowdsourced",
                    "data collection",
                ],
                "data_factory": {
                    "min_datasets": 2,
                    "days": 7,
                },
            },
            "tracked_orgs": {
                "openai": ["OpenAI"],
                "google": ["Google"],
            },
        }

    @pytest.fixture
    def analyzer(self, mock_db, config):
        """Create an OpportunityAnalyzer instance."""
        return OpportunityAnalyzer(mock_db, config)

    def test_init(self, analyzer):
        """Test initialization."""
        assert analyzer.factory_min_datasets == 2
        assert analyzer.factory_days == 7
        assert "human annotation" in analyzer.annotation_signals

    def test_detect_data_factories(self, analyzer):
        """Test data factory detection."""
        now = datetime.now().isoformat()
        datasets = [
            {"author": "prolific-author", "name": "ds1", "created_at": now},
            {"author": "prolific-author", "name": "ds2", "created_at": now},
            {"author": "prolific-author", "name": "ds3", "created_at": now},
            {"author": "normal-author", "name": "ds4", "created_at": now},
        ]
        factories = analyzer.detect_data_factories(datasets)
        assert len(factories) == 1
        assert factories[0]["author"] == "prolific-author"
        assert factories[0]["dataset_count"] == 3

    def test_detect_data_factories_threshold(self, analyzer):
        """Test data factory detection respects threshold."""
        now = datetime.now().isoformat()
        datasets = [
            {"author": "author1", "name": "ds1", "created_at": now},
            {"author": "author1", "name": "ds2", "created_at": now},  # Only 2, below threshold of 3
        ]
        # Our config has min_datasets=2, so this should be detected
        factories = analyzer.detect_data_factories(datasets)
        assert len(factories) == 1

    def test_extract_annotation_signals(self, analyzer):
        """Test annotation signal extraction."""
        papers = [
            {
                "title": "New Dataset with Human Annotation",
                "summary": "We collected data via crowdsourced annotation",
            },
            {
                "title": "Model Training Paper",
                "summary": "We trained a model on existing data",
            },
        ]
        opportunities = analyzer.extract_annotation_signals(papers)
        assert len(opportunities) == 1
        assert "human annotation" in opportunities[0]["signals"]
        assert "crowdsourced" in opportunities[0]["signals"]

    def test_extract_annotation_signals_with_org(self, analyzer):
        """Test annotation signal extraction with org detection."""
        papers = [
            {
                "title": "OpenAI's New Data Collection Effort",
                "summary": "Data collection methodology",
                "authors": ["Researcher (OpenAI)"],
            },
        ]
        opportunities = analyzer.extract_annotation_signals(papers)
        assert len(opportunities) == 1
        assert opportunities[0]["detected_org"] == "openai"

    def test_track_organization_activity(self, analyzer):
        """Test organization activity tracking."""
        datasets = [
            {"author": "google-research", "name": "google-ds"},
        ]
        papers = [
            {"title": "Google Paper", "authors": ["Google AI"]},
        ]
        activity = analyzer.track_organization_activity(datasets, papers)
        assert len(activity["google"]["datasets"]) == 1
        assert len(activity["google"]["papers"]) == 1

    def test_analyze(self, analyzer):
        """Test full analysis."""
        now = datetime.now().isoformat()
        datasets = [
            {"author": "factory-author", "name": "ds1", "created_at": now},
            {"author": "factory-author", "name": "ds2", "created_at": now},
        ]
        papers = [
            {"title": "Human Annotation Study", "summary": "crowdsourced data"},
        ]
        results = analyzer.analyze(datasets, papers)

        assert "data_factories" in results
        assert "annotation_opportunities" in results
        assert "org_activity" in results
        assert "summary" in results

    def test_generate_report(self, analyzer):
        """Test report generation."""
        results = {
            "data_factories": [
                {
                    "author": "test-author",
                    "dataset_count": 5,
                    "datasets": [{"name": "ds1"}, {"name": "ds2"}],
                    "possible_org": None,
                    "period_days": 7,
                }
            ],
            "annotation_opportunities": [
                {
                    "title": "Test Paper",
                    "signals": ["human annotation"],
                    "detected_org": "google",
                    "arxiv_id": "2024.12345",
                }
            ],
            "org_activity": {},
            "summary": {
                "data_factory_count": 1,
                "annotation_opportunity_count": 1,
                "active_org_count": 0,
            },
        }
        report = analyzer.generate_report(results)
        assert "Business Opportunity Analysis" in report
        assert "test-author" in report
        assert "Test Paper" in report


class TestPostTrainingFilter:
    """Tests for PostTrainingFilter class."""

    @pytest.fixture
    def pt_filter(self):
        """Create a PostTrainingFilter instance."""
        return PostTrainingFilter()

    def test_classify_sft_dataset(self, pt_filter):
        """Test SFT dataset classification."""
        item = {
            "name": "alpaca-instruct-52k",
            "description": "A instruction tuning dataset based on ShareGPT conversations",
            "tags": ["instruction", "chat"],
        }
        result = pt_filter.classify_item(item)
        assert "sft" in result
        assert result["sft"] > 0.5

    def test_classify_preference_dataset(self, pt_filter):
        """Test preference dataset classification."""
        item = {
            "name": "UltraFeedback-binarized",
            "description": "Preference dataset for DPO training with chosen and rejected pairs",
            "tags": ["preference", "rlhf"],
        }
        result = pt_filter.classify_item(item)
        assert "preference" in result
        assert result["preference"] > 0.5

    def test_classify_agent_dataset(self, pt_filter):
        """Test agent dataset classification."""
        item = {
            "name": "WebArena-trajectories",
            "description": "Web navigation trajectory data for agent training with function calling",
            "tags": ["agent", "tool-use"],
        }
        result = pt_filter.classify_item(item)
        assert "agent" in result
        assert result["agent"] > 0.5

    def test_classify_evaluation_dataset(self, pt_filter):
        """Test evaluation dataset classification."""
        item = {
            "name": "MMLU-pro-test",
            "description": "Benchmark dataset for evaluation with GPQA questions",
            "tags": ["benchmark", "evaluation"],
        }
        result = pt_filter.classify_item(item)
        assert "evaluation" in result
        assert result["evaluation"] > 0.5

    def test_classify_no_match(self, pt_filter):
        """Test item that doesn't match any category."""
        item = {
            "name": "generic-image-dataset",
            "description": "Random images from the internet",
            "tags": ["images"],
        }
        result = pt_filter.classify_item(item)
        assert result == {}

    def test_get_primary_category(self, pt_filter):
        """Test getting primary category."""
        item = {
            "name": "dpo-preference-data",
            "description": "DPO training with human feedback and reward model",
        }
        primary = pt_filter.get_primary_category(item)
        assert primary is not None
        assert primary[0] == "preference"
        assert primary[1] > 0

    def test_get_primary_category_no_match(self, pt_filter):
        """Test primary category when no match."""
        item = {"name": "generic", "description": "nothing special"}
        primary = pt_filter.get_primary_category(item)
        assert primary is None

    def test_filter_by_category(self, pt_filter):
        """Test filtering by category."""
        items = [
            {"name": "sft-data", "description": "instruction tuning dataset"},
            {"name": "generic", "description": "random data"},
            {"name": "alpaca-instruct", "description": "ShareGPT conversations"},
        ]
        filtered = pt_filter.filter_by_category(items, "sft", min_confidence=0.1)
        assert len(filtered) == 2

    def test_filter_by_category_with_confidence(self, pt_filter):
        """Test filtering by category with high confidence threshold."""
        items = [
            {"name": "dpo-data", "description": "DPO preference with chosen rejected"},
            {"name": "maybe-pref", "description": "some preference signals"},
        ]
        filtered = pt_filter.filter_by_category(items, "preference", min_confidence=0.5)
        # Only strong matches should pass
        assert len(filtered) >= 1
        assert all(item["pt_confidence"] >= 0.5 for item in filtered)

    def test_enrich_items(self, pt_filter):
        """Test enriching items with PT classifications."""
        items = [
            {"name": "instruction-data", "description": "instruction tuning"},
        ]
        enriched = pt_filter.enrich_items(items)
        assert "pt_categories" in enriched[0]
        assert "pt_primary" in enriched[0]

    def test_summarize(self, pt_filter):
        """Test summarizing post-training dataset distribution."""
        items = [
            {"name": "sft", "description": "instruction tuning"},
            {"name": "pref", "description": "DPO preference chosen rejected"},
            {"name": "agent", "description": "function calling tool use"},
            {"name": "eval", "description": "MMLU benchmark evaluation"},
            {"name": "generic", "description": "nothing special"},
        ]
        summary = pt_filter.summarize(items)
        assert summary["sft"]["count"] >= 1
        assert summary["preference"]["count"] >= 1
        assert summary["agent"]["count"] >= 1
        assert summary["evaluation"]["count"] >= 1
        assert summary["uncategorized"]["count"] >= 1

    def test_multiple_categories(self, pt_filter):
        """Test item matching multiple categories."""
        item = {
            "name": "agent-preference-benchmark",
            "description": "Agent trajectory data with preference annotation for tool use evaluation benchmark",
        }
        result = pt_filter.classify_item(item)
        # Should match multiple categories
        assert len(result) >= 2


class TestIntegration:
    """Integration tests for business intelligence features."""

    def test_domain_and_org_filter_together(self):
        """Test using domain and org filters together."""
        focus_areas = {
            "robotics": {
                "enabled": True,
                "keywords": ["robot", "manipulation"],
                "hf_tags": [],
            },
        }
        tracked_orgs = {
            "google": ["Google"],
        }

        domain_filter = DomainFilter(focus_areas)
        org_filter = OrganizationFilter(tracked_orgs)

        items = [
            {"author": "google", "name": "robot-dataset", "description": "manipulation"},
        ]

        # Enrich with both
        domain_filter.enrich_items(items)
        org_filter.enrich_items(items)

        assert items[0]["domains"] == ["robotics"]
        assert items[0]["detected_org"] == "google"
