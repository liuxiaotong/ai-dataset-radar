"""Tests for JSON output formatting."""

import json
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from output_formatter import DualOutputFormatter


class TestDualOutputFormatter:
    """Tests for DualOutputFormatter."""

    @pytest.fixture
    def formatter(self):
        """Create formatter with temp directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield DualOutputFormatter(output_dir=tmpdir)

    @pytest.fixture
    def sample_data(self):
        """Sample data for testing."""
        return {
            "period": {
                "days": 7,
                "start": None,
                "end": datetime.now().isoformat(),
            },
            "datasets": [
                {"id": "ds1", "name": "Dataset 1", "source": "huggingface"},
                {"id": "ds2", "name": "Dataset 2", "source": "huggingface"},
            ],
            "github_activity": [
                {"id": "org/repo1", "name": "Repo 1"},
            ],
            "papers": [
                {"id": "paper1", "title": "Paper 1"},
                {"id": "paper2", "title": "Paper 2"},
            ],
            "blog_posts": [
                {
                    "source": "Blog A",
                    "articles": [
                        {"title": "Article 1"},
                        {"title": "Article 2"},
                    ],
                }
            ],
            "labs_activity": {},
            "vendor_activity": {},
        }

    def test_creates_output_directory(self):
        """Test that output directory is created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            subdir = os.path.join(tmpdir, "nested", "reports")
            formatter = DualOutputFormatter(output_dir=subdir)
            assert os.path.exists(subdir)

    def test_save_reports_creates_both_files(self, sample_data):
        """Test that both MD and JSON files are created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            formatter = DualOutputFormatter(output_dir=tmpdir)
            md_path, json_path = formatter.save_reports(
                markdown_content="# Test Report",
                data=sample_data,
            )

            assert os.path.exists(md_path)
            assert os.path.exists(json_path)
            assert md_path.endswith(".md")
            assert json_path.endswith(".json")

    def test_markdown_content_saved(self, sample_data):
        """Test that markdown content is saved correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            formatter = DualOutputFormatter(output_dir=tmpdir)
            content = "# Test Report\n\nThis is a test."
            md_path, _ = formatter.save_reports(
                markdown_content=content,
                data=sample_data,
            )

            with open(md_path, "r", encoding="utf-8") as f:
                saved = f.read()
            assert saved == content

    def test_json_is_valid(self, sample_data):
        """Test that JSON output is valid JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            formatter = DualOutputFormatter(output_dir=tmpdir)
            _, json_path = formatter.save_reports(
                markdown_content="# Test",
                data=sample_data,
            )

            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            assert isinstance(data, dict)


class TestJSONSchema:
    """Tests for JSON output schema validation."""

    @pytest.fixture
    def formatter(self):
        return DualOutputFormatter(output_dir="/tmp")

    def test_has_generated_at(self, formatter):
        """Test that output has generated_at timestamp."""
        data = {"datasets": [], "papers": [], "github_activity": [], "blog_posts": []}
        output = formatter._format_json_output(data)
        assert "generated_at" in output
        # Should be valid ISO format
        datetime.fromisoformat(output["generated_at"])

    def test_has_summary(self, formatter):
        """Test that output has summary section."""
        data = {"datasets": [], "papers": [], "github_activity": [], "blog_posts": []}
        output = formatter._format_json_output(data)
        assert "summary" in output
        assert isinstance(output["summary"], dict)

    def test_summary_has_required_fields(self, formatter):
        """Test that summary has all required count fields."""
        data = {"datasets": [], "papers": [], "github_activity": [], "blog_posts": []}
        output = formatter._format_json_output(data)

        summary = output["summary"]
        assert "total_datasets" in summary
        assert "total_github_orgs" in summary
        assert "total_github_repos" in summary
        assert "total_github_repos_high_relevance" in summary
        assert "total_papers" in summary
        assert "total_blog_posts" in summary

    def test_summary_counts_are_integers(self, formatter):
        """Test that all counts are integers."""
        data = {
            "datasets": [{"id": "1"}, {"id": "2"}],
            "papers": [{"id": "p1"}],
            "github_activity": [{"id": "r1"}, {"id": "r2"}, {"id": "r3"}],
            "blog_posts": [{"articles": [{"t": 1}, {"t": 2}]}],
        }
        output = formatter._format_json_output(data)

        for key, value in output["summary"].items():
            assert isinstance(value, int), f"{key} should be int, got {type(value)}"

    def test_datasets_count_correct(self, formatter):
        """Test that dataset count is correct."""
        data = {
            "datasets": [{"id": "1"}, {"id": "2"}, {"id": "3"}],
            "papers": [],
            "github_activity": [],
            "blog_posts": [],
        }
        output = formatter._format_json_output(data)
        assert output["summary"]["total_datasets"] == 3

    def test_papers_count_correct(self, formatter):
        """Test that paper count is correct."""
        data = {
            "datasets": [],
            "papers": [{"id": "1"}, {"id": "2"}],
            "github_activity": [],
            "blog_posts": [],
        }
        output = formatter._format_json_output(data)
        assert output["summary"]["total_papers"] == 2

    def test_repos_count_correct(self, formatter):
        """Test that repo count is correct."""
        data = {
            "datasets": [],
            "papers": [],
            "github_activity": [
                {"org": "org1", "repos_updated": [{"name": "r1"}, {"name": "r2"}]},
                {"org": "org2", "repos_updated": [{"name": "r3", "relevance": "high"}]},
            ],
            "blog_posts": [],
        }
        output = formatter._format_json_output(data)
        assert output["summary"]["total_github_orgs"] == 2
        assert output["summary"]["total_github_repos"] == 3
        assert output["summary"]["total_github_repos_high_relevance"] == 1

    def test_blog_posts_count_correct(self, formatter):
        """Test that blog post count aggregates articles."""
        data = {
            "datasets": [],
            "papers": [],
            "github_activity": [],
            "blog_posts": [
                {"source": "A", "articles": [{"t": 1}, {"t": 2}]},
                {"source": "B", "articles": [{"t": 3}]},
            ],
        }
        output = formatter._format_json_output(data)
        assert output["summary"]["total_blog_posts"] == 3

    def test_has_period(self, formatter):
        """Test that output has period section with calculated start."""
        data = {
            "period": {"days": 7, "start": None, "end": "2024-01-01"},
            "datasets": [],
            "papers": [],
            "github_activity": [],
            "blog_posts": [],
        }
        output = formatter._format_json_output(data)
        assert "period" in output
        assert output["period"]["days"] == 7
        # start should be calculated, not null
        assert output["period"]["start"] is not None
        assert output["period"]["end"] is not None

    def test_preserves_allowed_data(self, formatter):
        """Test that allowed fields are preserved in output."""
        data = {
            "datasets": [{"id": "ds1", "author": "test_org", "description": "A test dataset"}],
            "papers": [{"id": "p1", "title": "Test"}],
            "github_activity": [{"id": "r1", "stars": 100}],
            "blog_posts": [],
        }
        output = formatter._format_json_output(data)

        # Datasets: only allowed fields are preserved (id, author, description are allowed)
        assert output["datasets"][0]["id"] == "ds1"
        assert output["datasets"][0]["author"] == "test_org"
        assert output["datasets"][0]["description"] == "A test dataset"
        # Papers and github_activity are passed through as-is
        assert output["papers"][0]["title"] == "Test"
        assert output["github_activity"][0]["stars"] == 100


class TestFormatSummary:
    """Tests for format_summary method."""

    @pytest.fixture
    def formatter(self):
        return DualOutputFormatter(output_dir="/tmp")

    def test_format_summary_output(self, formatter):
        """Test that format_summary returns readable string."""
        data = {
            "datasets": [{"id": "1"}, {"id": "2"}],
            "papers": [{"id": "1"}],
            "github_activity": [
                {"org": "org1", "repos_updated": [{"name": "r1", "relevance": "high"}]},
                {"org": "org2", "repos_updated": [{"name": "r2"}, {"name": "r3"}]},
            ],
            "blog_posts": [{"articles": [{"t": 1}]}],
        }
        summary = formatter.format_summary(data)

        assert "2 datasets" in summary
        assert "3 repos" in summary
        assert "1 high relevance" in summary
        assert "1 papers" in summary
        assert "1 blog posts" in summary

    def test_format_summary_empty_data(self, formatter):
        """Test format_summary with empty data."""
        data = {
            "datasets": [],
            "papers": [],
            "github_activity": [],
            "blog_posts": [],
        }
        summary = formatter.format_summary(data)

        assert "0 datasets" in summary
        assert "0 repos" in summary
        assert "0 papers" in summary
        assert "0 blog posts" in summary
