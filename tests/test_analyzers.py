"""Tests for analyzer modules."""

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from db import RadarDatabase
from analyzers.model_dataset import ModelDatasetAnalyzer
from analyzers.trend import TrendAnalyzer


class TestModelDatasetAnalyzer:
    """Tests for ModelDatasetAnalyzer class."""

    @pytest.fixture
    def db(self):
        """Create a temporary database."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        db = RadarDatabase(db_path)
        yield db

        db.close()
        if os.path.exists(db_path):
            os.unlink(db_path)

    @pytest.fixture
    def analyzer(self, db):
        """Create an analyzer instance."""
        return ModelDatasetAnalyzer(db, {})

    def test_init(self, analyzer):
        """Test analyzer initialization."""
        assert analyzer.db is not None
        assert analyzer.config == {}
        assert analyzer.scraper is not None

    def test_rank_datasets(self, analyzer):
        """Test ranking datasets by usage."""
        dataset_usage = {
            "dataset-a": [
                {"model_id": "m1", "downloads": 1000},
                {"model_id": "m2", "downloads": 2000},
            ],
            "dataset-b": [
                {"model_id": "m3", "downloads": 500},
            ],
            "dataset-c": [
                {"model_id": "m4", "downloads": 10000},
                {"model_id": "m5", "downloads": 5000},
                {"model_id": "m6", "downloads": 3000},
            ],
        }

        ranked = analyzer._rank_datasets(dataset_usage)

        # dataset-c has most models (3)
        assert ranked[0]["dataset_id"] == "dataset-c"
        assert ranked[0]["model_count"] == 3
        assert ranked[0]["total_model_downloads"] == 18000

        # dataset-a has 2 models
        assert ranked[1]["dataset_id"] == "dataset-a"
        assert ranked[1]["model_count"] == 2

        # dataset-b has 1 model
        assert ranked[2]["dataset_id"] == "dataset-b"
        assert ranked[2]["model_count"] == 1

    def test_generate_report(self, analyzer):
        """Test report generation."""
        results = {
            "models_analyzed": 10,
            "total_links": 25,
            "unique_datasets": 15,
            "top_datasets": [
                {
                    "dataset_id": "squad",
                    "model_count": 5,
                    "total_model_downloads": 100000,
                    "models": ["model-a", "model-b", "model-c"],
                },
            ],
        }

        report = analyzer.generate_report(results)

        assert "Model-Dataset Relationship Analysis" in report
        assert "Models analyzed: 10" in report
        assert "Total links found: 25" in report
        assert "squad" in report
        assert "Used by 5 models" in report

    @patch.object(ModelDatasetAnalyzer, "analyze")
    def test_get_high_value_datasets(self, mock_analyze, db):
        """Test getting high value datasets from database."""
        # Add some test data
        ds1 = db.upsert_dataset("hf", "ds1", "ds1")
        ds2 = db.upsert_dataset("hf", "ds2", "ds2")
        m1 = db.upsert_model("m1", "m1", downloads=1000)
        m2 = db.upsert_model("m2", "m2", downloads=2000)

        db.add_model_dataset_link(m1, ds1)
        db.add_model_dataset_link(m2, ds1)
        db.add_model_dataset_link(m1, ds2)

        analyzer = ModelDatasetAnalyzer(db, {})
        high_value = analyzer.get_high_value_datasets()

        assert len(high_value) == 2
        assert high_value[0]["dataset_id"] == "ds1"  # Used by more models


class TestTrendAnalyzer:
    """Tests for TrendAnalyzer class."""

    @pytest.fixture
    def db(self):
        """Create a temporary database."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        db = RadarDatabase(db_path)
        yield db

        db.close()
        if os.path.exists(db_path):
            os.unlink(db_path)

    @pytest.fixture
    def analyzer(self, db):
        """Create an analyzer instance."""
        return TrendAnalyzer(db, {})

    def test_init(self, analyzer):
        """Test analyzer initialization."""
        assert analyzer.db is not None
        assert analyzer.config == {}

    def test_record_daily_stats(self, analyzer):
        """Test recording daily statistics."""
        datasets = [
            {
                "source": "huggingface",
                "id": "test/dataset1",
                "name": "dataset1",
                "downloads": 1000,
                "likes": 50,
            },
            {
                "source": "huggingface",
                "id": "test/dataset2",
                "name": "dataset2",
                "downloads": 2000,
                "likes": 100,
            },
        ]

        recorded = analyzer.record_daily_stats(datasets)

        assert recorded == 2

        # Verify in database
        all_datasets = analyzer.db.get_all_datasets()
        assert len(all_datasets) == 2

    def test_record_daily_stats_empty(self, analyzer):
        """Test recording empty dataset list."""
        recorded = analyzer.record_daily_stats([])
        assert recorded == 0

    def test_get_dataset_trend(self, analyzer):
        """Test getting trend for a specific dataset."""
        from datetime import datetime, timedelta

        # Create dataset with history (use recent dates)
        ds_id = analyzer.db.upsert_dataset("huggingface", "test/ds", "ds")
        today = datetime.now()
        day1 = (today - timedelta(days=7)).strftime("%Y-%m-%d")
        day2 = today.strftime("%Y-%m-%d")

        analyzer.db.record_daily_stats(ds_id, downloads=100, date=day1)
        analyzer.db.record_daily_stats(ds_id, downloads=150, date=day2)

        trend = analyzer.get_dataset_trend("test/ds")

        assert trend is not None
        assert trend["dataset"]["dataset_id"] == "test/ds"
        assert len(trend["history"]) == 2
        assert trend["growth_7d"] == 0.5  # 50% growth

    def test_get_dataset_trend_not_found(self, analyzer):
        """Test getting trend for non-existent dataset."""
        trend = analyzer.get_dataset_trend("nonexistent")
        assert trend is None

    def test_calculate_trends(self, analyzer, db):
        """Test calculating trends for all datasets."""
        from datetime import datetime, timedelta

        # Create some test data with recent dates
        ds1 = db.upsert_dataset("huggingface", "ds1", "ds1")
        ds2 = db.upsert_dataset("huggingface", "ds2", "ds2")

        today = datetime.now()
        day1 = (today - timedelta(days=7)).strftime("%Y-%m-%d")
        day2 = today.strftime("%Y-%m-%d")

        db.record_daily_stats(ds1, downloads=100, date=day1)
        db.record_daily_stats(ds1, downloads=150, date=day2)
        db.record_daily_stats(ds2, downloads=200, date=day1)
        db.record_daily_stats(ds2, downloads=250, date=day2)

        summary = analyzer.calculate_trends(days=[7])

        assert summary["total_datasets"] == 2
        assert summary["trends_calculated"] == 2

    def test_get_rising_datasets(self, analyzer, db):
        """Test getting rising datasets."""
        ds1 = db.upsert_dataset("huggingface", "ds1", "ds1")
        ds2 = db.upsert_dataset("huggingface", "ds2", "ds2")

        # ds1 has 100% growth
        db.record_trend(ds1, downloads_7d_growth=1.0)
        # ds2 has 20% growth (below threshold)
        db.record_trend(ds2, downloads_7d_growth=0.2)

        rising = analyzer.get_rising_datasets(min_growth=0.5, days=7)

        assert len(rising) == 1
        assert rising[0]["growth"] == 1.0

    def test_analyze(self, analyzer):
        """Test full analysis pipeline."""
        datasets = [
            {
                "source": "huggingface",
                "id": "test/ds",
                "name": "ds",
                "downloads": 1000,
                "likes": 50,
            },
        ]

        results = analyzer.analyze(datasets)

        assert "recorded_count" in results
        assert results["recorded_count"] == 1
        assert "trend_summary" in results
        assert "rising_7d" in results
        assert "rising_30d" in results

    def test_generate_report(self, analyzer):
        """Test report generation."""
        results = {
            "recorded_count": 50,
            "trend_summary": {
                "total_datasets": 50,
                "trends_calculated": 25,
                "datasets_with_growth": 10,
            },
            "rising_7d": [
                {
                    "name": "popular-dataset",
                    "growth": 0.75,
                    "url": "https://example.com",
                },
            ],
            "rising_30d": [],
        }

        report = analyzer.generate_report(results)

        assert "Dataset Trend Analysis" in report
        assert "Total datasets tracked: 50" in report
        assert "Trends calculated: 25" in report
        assert "popular-dataset" in report
        assert "75.0%" in report

    def test_generate_report_no_rising(self, analyzer):
        """Test report generation with no rising datasets."""
        results = {
            "recorded_count": 10,
            "trend_summary": {
                "total_datasets": 10,
                "trends_calculated": 0,
                "datasets_with_growth": 0,
            },
            "rising_7d": [],
            "rising_30d": [],
        }

        report = analyzer.generate_report(results)

        assert "No rising datasets found yet" in report
