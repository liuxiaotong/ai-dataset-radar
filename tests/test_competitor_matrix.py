"""Tests for CompetitorMatrix analyzer."""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from analyzers.competitor_matrix import CompetitorMatrix


@pytest.fixture
def matrix():
    return CompetitorMatrix()


# ─── Basic structure ────────────────────────────────────────────────────────

class TestBuild:
    def test_empty_inputs(self, matrix):
        result = matrix.build()
        assert result["matrix"] == {}
        assert result["rankings"] == {}
        assert result["top_orgs"] == []
        assert result["org_details"] == {}

    def test_returns_all_keys(self, matrix):
        result = matrix.build(datasets=[])
        assert "matrix" in result
        assert "rankings" in result
        assert "top_orgs" in result
        assert "org_details" in result

    def test_datasets_counted(self, matrix):
        datasets = [
            {"id": "openai/gsm8k", "author": "openai", "description": "Math dataset", "tags": []},
            {"id": "openai/summarize", "author": "openai", "description": "Summarization", "tags": []},
        ]
        result = matrix.build(datasets=datasets)
        assert "openai" in result["matrix"]
        assert result["org_details"]["openai"]["datasets"] == 2

    def test_github_repos_counted(self, matrix):
        github = [
            {
                "org": "openai",
                "repos_updated": [
                    {"name": "evals", "url": "https://github.com/openai/evals"},
                ],
            }
        ]
        result = matrix.build(github_activity=github)
        assert result["org_details"]["openai"]["repos"] == 1

    def test_multiple_orgs(self, matrix):
        datasets = [
            {"id": "openai/ds1", "author": "openai", "description": "test", "tags": []},
            {"id": "meta-llama/ds1", "author": "meta-llama", "description": "test", "tags": []},
        ]
        result = matrix.build(datasets=datasets)
        assert len(result["top_orgs"]) >= 2


# ─── Rankings ───────────────────────────────────────────────────────────────

class TestRankings:
    def test_rankings_sorted_desc(self, matrix):
        datasets = [
            {"id": "openai/a", "author": "openai", "description": "test", "tags": ["code"]},
            {"id": "openai/b", "author": "openai", "description": "test", "tags": ["code"]},
            {"id": "meta-llama/a", "author": "meta-llama", "description": "test", "tags": ["code"]},
        ]
        result = matrix.build(datasets=datasets)
        for cat, ranked in result["rankings"].items():
            scores = [r[1] for r in ranked]
            assert scores == sorted(scores, reverse=True)

    def test_top_orgs_sorted_desc(self, matrix):
        datasets = [
            {"id": "openai/a", "author": "openai", "description": "test", "tags": []},
            {"id": "openai/b", "author": "openai", "description": "test", "tags": []},
            {"id": "meta-llama/a", "author": "meta-llama", "description": "test", "tags": []},
        ]
        result = matrix.build(datasets=datasets)
        totals = [t[1] for t in result["top_orgs"]]
        assert totals == sorted(totals, reverse=True)


# ─── Edge cases ─────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_unknown_org_skipped(self, matrix):
        datasets = [
            {"id": "random_user/test", "author": "random_user", "description": "no org", "tags": []},
        ]
        result = matrix.build(datasets=datasets)
        assert "random_user" not in result["matrix"]

    def test_none_inputs(self, matrix):
        result = matrix.build(datasets=None, github_activity=None, papers=None, blog_posts=None)
        assert result["matrix"] == {}
