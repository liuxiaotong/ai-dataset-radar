"""Tests for database module."""

import os
import sys
import tempfile
import pytest
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from db import RadarDatabase, get_database


class TestRadarDatabase:
    """Tests for RadarDatabase class."""

    @pytest.fixture
    def db(self):
        """Create a temporary database for testing."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        db = RadarDatabase(db_path)
        yield db

        # Cleanup
        db.close()
        if os.path.exists(db_path):
            os.unlink(db_path)

    def test_init_creates_tables(self, db):
        """Test that database initialization creates all required tables."""
        import sqlite3

        conn = sqlite3.connect(db.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cursor.fetchall()}
        conn.close()

        expected_tables = {"datasets", "daily_stats", "models", "model_datasets", "trends"}
        assert expected_tables.issubset(tables)

    def test_upsert_dataset(self, db):
        """Test inserting and updating a dataset."""
        # Insert
        db_id = db.upsert_dataset(
            source="huggingface",
            dataset_id="test/dataset",
            name="dataset",
            author="test",
            url="https://example.com",
        )
        assert db_id > 0

        # Verify
        dataset = db.get_dataset("huggingface", "test/dataset")
        assert dataset is not None
        assert dataset["name"] == "dataset"
        assert dataset["author"] == "test"

        # Update (upsert same record)
        db_id2 = db.upsert_dataset(
            source="huggingface",
            dataset_id="test/dataset",
            name="dataset-updated",
            author="test-updated",
        )
        assert db_id2 == db_id  # Same record

        # Verify update
        dataset = db.get_dataset("huggingface", "test/dataset")
        assert dataset["name"] == "dataset-updated"

    def test_get_dataset_not_found(self, db):
        """Test getting a non-existent dataset returns None."""
        result = db.get_dataset("huggingface", "nonexistent")
        assert result is None

    def test_get_all_datasets(self, db):
        """Test getting all datasets with optional source filter."""
        db.upsert_dataset("huggingface", "hf/ds1", "ds1")
        db.upsert_dataset("huggingface", "hf/ds2", "ds2")
        db.upsert_dataset("paperswithcode", "pwc/ds1", "ds1")

        all_datasets = db.get_all_datasets()
        assert len(all_datasets) == 3

        hf_datasets = db.get_all_datasets(source="huggingface")
        assert len(hf_datasets) == 2

    def test_record_daily_stats(self, db):
        """Test recording daily statistics."""
        from datetime import datetime

        db_id = db.upsert_dataset("huggingface", "test/ds", "ds")
        today = datetime.now().strftime("%Y-%m-%d")

        db.record_daily_stats(
            dataset_db_id=db_id,
            downloads=1000,
            likes=50,
            date=today,
        )

        history = db.get_stats_history(db_id, days=30)
        assert len(history) == 1
        assert history[0]["downloads"] == 1000
        assert history[0]["likes"] == 50

    def test_upsert_model(self, db):
        """Test inserting and updating a model."""
        db_id = db.upsert_model(
            model_id="test/model",
            name="model",
            author="test",
            downloads=10000,
            likes=100,
            pipeline_tag="text-classification",
        )
        assert db_id > 0

        model = db.get_model("test/model")
        assert model is not None
        assert model["downloads"] == 10000
        assert model["pipeline_tag"] == "text-classification"

    def test_get_top_models(self, db):
        """Test getting top models by downloads."""
        db.upsert_model("model/a", "a", downloads=100)
        db.upsert_model("model/b", "b", downloads=1000)
        db.upsert_model("model/c", "c", downloads=500)

        top = db.get_top_models(limit=2)
        assert len(top) == 2
        assert top[0]["downloads"] == 1000
        assert top[1]["downloads"] == 500

    def test_model_dataset_link(self, db):
        """Test linking models to datasets."""
        model_id = db.upsert_model("test/model", "model", downloads=1000)
        dataset_id = db.upsert_dataset("huggingface", "test/dataset", "dataset")

        db.add_model_dataset_link(model_id, dataset_id, "training")

        # Get datasets for model
        datasets = db.get_datasets_for_model(model_id)
        assert len(datasets) == 1
        assert datasets[0]["dataset_id"] == "test/dataset"

        # Get models for dataset
        models = db.get_models_using_dataset(dataset_id)
        assert len(models) == 1
        assert models[0]["model_id"] == "test/model"

    def test_get_most_used_datasets(self, db):
        """Test getting datasets ranked by model usage."""
        # Create datasets
        ds1 = db.upsert_dataset("hf", "ds1", "ds1")
        ds2 = db.upsert_dataset("hf", "ds2", "ds2")

        # Create models
        m1 = db.upsert_model("m1", "m1", downloads=1000)
        m2 = db.upsert_model("m2", "m2", downloads=2000)
        m3 = db.upsert_model("m3", "m3", downloads=500)

        # Link: ds1 used by 3 models, ds2 used by 1 model
        db.add_model_dataset_link(m1, ds1)
        db.add_model_dataset_link(m2, ds1)
        db.add_model_dataset_link(m3, ds1)
        db.add_model_dataset_link(m1, ds2)

        most_used = db.get_most_used_datasets(limit=10)
        assert len(most_used) == 2
        assert most_used[0]["dataset_id"] == "ds1"
        assert most_used[0]["model_count"] == 3

    def test_record_and_get_trends(self, db):
        """Test recording and retrieving trend data."""
        from datetime import datetime

        ds_id = db.upsert_dataset("hf", "ds", "ds")
        today = datetime.now().strftime("%Y-%m-%d")

        db.record_trend(
            dataset_db_id=ds_id,
            downloads_7d_growth=0.5,
            downloads_30d_growth=1.2,
            date=today,
        )

        # Get rising datasets
        rising = db.get_rising_datasets(min_growth=0.4, days=7)
        assert len(rising) == 1
        assert rising[0]["growth"] == 0.5

    def test_calculate_growth_rate(self, db):
        """Test growth rate calculation."""
        from datetime import datetime, timedelta

        ds_id = db.upsert_dataset("hf", "ds", "ds")

        # Record stats over multiple days (use recent dates)
        today = datetime.now()
        day1 = (today - timedelta(days=7)).strftime("%Y-%m-%d")
        day2 = today.strftime("%Y-%m-%d")

        db.record_daily_stats(ds_id, downloads=100, date=day1)
        db.record_daily_stats(ds_id, downloads=150, date=day2)

        growth = db.calculate_growth_rate(ds_id, days=7)
        assert growth == 0.5  # 50% growth

    def test_calculate_growth_rate_insufficient_data(self, db):
        """Test growth rate with insufficient data returns None."""
        ds_id = db.upsert_dataset("hf", "ds", "ds")
        db.record_daily_stats(ds_id, downloads=100, date="2024-01-01")

        growth = db.calculate_growth_rate(ds_id, days=7)
        assert growth is None


class TestGetDatabase:
    """Tests for get_database helper function."""

    def test_get_database_from_config(self):
        """Test creating database from config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            config = {"database": {"path": db_path}}

            db = get_database(config)
            assert os.path.exists(db_path)
            db.close()

    def test_get_database_default_path(self):
        """Test creating database with default path."""
        config = {}
        db = get_database(config)
        assert db.db_path == "data/radar.db"
        db.close()
