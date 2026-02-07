"""Shared pytest fixtures for AI Dataset Radar tests."""

import json
import sys
from pathlib import Path

import pytest

# Add project paths
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "mcp_server"))


@pytest.fixture
def project_root():
    """Return project root path."""
    return PROJECT_ROOT


@pytest.fixture
def sample_report():
    """Create a minimal report fixture for reuse across tests."""
    return {
        "generated_at": "2024-01-15T10:00:00",
        "period": {"days": 7, "start": "2024-01-08", "end": "2024-01-15"},
        "summary": {
            "total_datasets": 5,
            "total_github_orgs": 3,
            "total_github_repos": 20,
            "total_github_repos_high_relevance": 2,
            "total_papers": 8,
            "total_blog_posts": 4,
        },
        "datasets": [
            {"id": "test/dataset1", "category": "synthetic", "downloads": 100, "description": "Test synthetic data"},
            {"id": "test/dataset2", "category": "sft", "downloads": 200, "description": "Test SFT data"},
        ],
        "datasets_by_type": {"synthetic": ["test/dataset1"], "sft": ["test/dataset2"]},
        "github_activity": [
            {
                "org": "test-org",
                "repos_count": 1,
                "repos_updated": [
                    {
                        "name": "repo1",
                        "full_name": "test-org/repo1",
                        "description": "Test dataset repo",
                        "stars": 100,
                        "relevance": "high",
                        "relevance_signals": ["dataset"],
                        "signals": ["dataset"],
                        "topics": [],
                        "url": "https://github.com/test-org/repo1",
                    },
                ],
                "has_activity": True,
            }
        ],
        "papers": [
            {"title": "Test Paper", "url": "https://arxiv.org/abs/1234", "abstract": "Test abstract", "source": "arxiv", "categories": ["cs.CL"]},
        ],
        "blog_posts": [
            {
                "source": "Test Blog",
                "articles": [
                    {"title": "Test Article", "url": "https://example.com/article", "date": "2024-01-15", "signals": ["rlhf"], "snippet": "Test content"},
                ],
            }
        ],
        "x_activity": {
            "accounts": [],
            "search_results": [],
        },
    }


@pytest.fixture
def reports_dir(tmp_path, sample_report):
    """Create a temporary reports directory with a sample report."""
    reports = tmp_path / "data" / "reports"
    reports.mkdir(parents=True)
    (reports / "intel_report_2024-01-15.json").write_text(
        json.dumps(sample_report)
    )
    return reports
