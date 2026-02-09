"""Tests for daily change tracker."""

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from analyzers.change_tracker import (
    compare_reports,
    find_previous_report,
    format_changes_markdown,
    generate_change_summary,
)


def _make_report(
    datasets=None, github_activity=None, papers=None,
    blog_posts=None, x_activity=None, datasets_by_type=None,
    summary_overrides=None,
):
    """Build a minimal JSON report structure for testing."""
    datasets = datasets or []
    github_activity = github_activity or []
    papers = papers or []
    blog_posts = blog_posts or []
    x_activity = x_activity or {}
    datasets_by_type = datasets_by_type or {}

    total_repos = sum(
        len(org.get("repos_updated", []))
        for org in github_activity if isinstance(org, dict)
    )
    total_high = sum(
        1 for org in github_activity if isinstance(org, dict)
        for r in org.get("repos_updated", [])
        if r.get("relevance") == "high"
    )
    total_blogs = sum(
        len(b.get("articles", []))
        for b in blog_posts if isinstance(b, dict)
    )
    total_tweets = sum(
        len(a.get("relevant_tweets", []))
        for a in (x_activity.get("accounts", []) if isinstance(x_activity, dict) else x_activity)
        if isinstance(a, dict)
    )

    summary = {
        "total_datasets": len(datasets),
        "total_github_repos": total_repos,
        "total_github_repos_high_relevance": total_high,
        "total_papers": len(papers),
        "total_blog_posts": total_blogs,
        "total_x_tweets": total_tweets,
    }
    if summary_overrides:
        summary.update(summary_overrides)

    return {
        "summary": summary,
        "datasets": datasets,
        "datasets_by_type": datasets_by_type,
        "github_activity": github_activity,
        "papers": papers,
        "blog_posts": blog_posts,
        "x_activity": x_activity,
    }


def _write_report(base_dir: Path, date_str: str, report: dict):
    """Write a report JSON to the expected directory structure."""
    date_dir = base_dir / date_str
    date_dir.mkdir(parents=True, exist_ok=True)
    path = date_dir / f"intel_report_{date_str}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f)
    return path


class TestFindPreviousReport:
    def test_finds_most_recent(self, tmp_path):
        _write_report(tmp_path, "2026-02-07", _make_report())
        _write_report(tmp_path, "2026-02-08", _make_report())
        _write_report(tmp_path, "2026-02-09", _make_report())

        result = find_previous_report(tmp_path, "2026-02-09")
        assert result is not None
        assert "2026-02-08" in str(result)

    def test_first_run_returns_none(self, tmp_path):
        _write_report(tmp_path, "2026-02-09", _make_report())
        result = find_previous_report(tmp_path, "2026-02-09")
        assert result is None

    def test_skips_current_date(self, tmp_path):
        _write_report(tmp_path, "2026-02-07", _make_report())
        _write_report(tmp_path, "2026-02-09", _make_report())
        result = find_previous_report(tmp_path, "2026-02-09")
        assert result is not None
        assert "2026-02-07" in str(result)

    def test_nonexistent_dir(self, tmp_path):
        result = find_previous_report(tmp_path / "nonexistent", "2026-02-09")
        assert result is None


class TestCompareReports:
    def test_new_and_removed_datasets(self):
        prev = _make_report(datasets=[
            {"id": "org/ds-a", "downloads": 100, "likes": 5, "category": "code"},
            {"id": "org/ds-b", "downloads": 200, "likes": 10, "category": "sft"},
        ])
        curr = _make_report(datasets=[
            {"id": "org/ds-b", "downloads": 250, "likes": 12, "category": "sft"},
            {"id": "org/ds-c", "downloads": 50, "likes": 1, "category": "multimodal"},
        ])

        changes = compare_reports(prev, curr, "2026-02-08", "2026-02-09")

        assert len(changes["new_datasets"]) == 1
        assert changes["new_datasets"][0]["id"] == "org/ds-c"

        assert len(changes["removed_datasets"]) == 1
        assert changes["removed_datasets"][0]["id"] == "org/ds-a"

    def test_download_movers(self):
        prev = _make_report(datasets=[
            {"id": "org/ds-a", "downloads": 100, "likes": 5, "category": "code"},
            {"id": "org/ds-b", "downloads": 1000, "likes": 50, "category": "sft"},
        ])
        curr = _make_report(datasets=[
            {"id": "org/ds-a", "downloads": 500, "likes": 8, "category": "code"},
            {"id": "org/ds-b", "downloads": 1050, "likes": 52, "category": "sft"},
        ])

        changes = compare_reports(prev, curr, "2026-02-08", "2026-02-09")
        movers = changes["download_movers"]

        assert len(movers) == 2
        # ds-a has larger absolute delta (400) vs ds-b (50)
        assert movers[0]["id"] == "org/ds-a"
        assert movers[0]["delta"] == 400
        assert movers[0]["growth_pct"] == pytest.approx(400.0)

    def test_github_new_and_gone(self):
        prev = _make_report(github_activity=[
            {"org": "openai", "repos_updated": [
                {"full_name": "openai/gpt", "stars": 1000, "relevance": "high"},
            ]},
        ])
        curr = _make_report(github_activity=[
            {"org": "openai", "repos_updated": [
                {"full_name": "openai/codex", "stars": 500, "relevance": "high"},
            ]},
        ])

        changes = compare_reports(prev, curr, "2026-02-08", "2026-02-09")
        assert len(changes["new_repos"]) == 1
        assert changes["new_repos"][0]["full_name"] == "openai/codex"
        assert len(changes["gone_repos"]) == 1
        assert changes["gone_repos"][0]["full_name"] == "openai/gpt"

    def test_stars_movers(self):
        prev = _make_report(github_activity=[
            {"org": "org", "repos_updated": [
                {"full_name": "org/repo-a", "stars": 100, "relevance": "high"},
                {"full_name": "org/repo-b", "stars": 5000, "relevance": "high"},
            ]},
        ])
        curr = _make_report(github_activity=[
            {"org": "org", "repos_updated": [
                {"full_name": "org/repo-a", "stars": 300, "relevance": "high"},
                {"full_name": "org/repo-b", "stars": 5010, "relevance": "high"},
            ]},
        ])

        changes = compare_reports(prev, curr, "2026-02-08", "2026-02-09")
        movers = changes["stars_movers"]
        assert movers[0]["full_name"] == "org/repo-a"
        assert movers[0]["delta"] == 200

    def test_new_papers(self):
        prev = _make_report(papers=[
            {"title": "Paper A", "source": "arxiv"},
        ])
        curr = _make_report(papers=[
            {"title": "Paper A", "source": "arxiv"},
            {"title": "Paper B", "source": "hf_papers"},
        ])

        changes = compare_reports(prev, curr, "2026-02-08", "2026-02-09")
        assert len(changes["new_papers"]) == 1
        assert changes["new_papers"][0]["title"] == "Paper B"

    def test_summary_deltas(self):
        prev = _make_report(
            datasets=[{"id": "a"}, {"id": "b"}],
            papers=[{"title": "P1"}],
        )
        curr = _make_report(
            datasets=[{"id": "a"}, {"id": "b"}, {"id": "c"}],
            papers=[{"title": "P1"}, {"title": "P2"}],
        )

        changes = compare_reports(prev, curr, "2026-02-08", "2026-02-09")
        deltas = {d["label"]: d["delta"] for d in changes["summary_deltas"]}
        assert deltas["Datasets"] == 1
        assert deltas["Papers"] == 1

    def test_category_changes(self):
        prev = _make_report(
            datasets_by_type={"code": ["a", "b"], "sft": ["c"]},
        )
        curr = _make_report(
            datasets_by_type={"code": ["a"], "sft": ["c", "d"], "rl_environment": ["e"]},
        )

        changes = compare_reports(prev, curr, "2026-02-08", "2026-02-09")
        cat_changes = {c["category"]: c["delta"] for c in changes["category_changes"]}
        assert cat_changes["code"] == -1
        assert cat_changes["sft"] == 1
        assert cat_changes["rl_environment"] == 1


class TestFormatChangesMarkdown:
    def test_basic_structure(self):
        changes = compare_reports(
            _make_report(datasets=[{"id": "a", "downloads": 100}]),
            _make_report(datasets=[{"id": "a", "downloads": 200}, {"id": "b", "downloads": 50, "category": "code", "likes": 3}]),
            "2026-02-08", "2026-02-09",
        )
        md = format_changes_markdown(changes)

        assert "# Daily Changes: 2026-02-09" in md
        assert "2026-02-08" in md
        assert "## Summary" in md
        assert "## New Datasets (1)" in md
        assert "## Download Movers" in md

    def test_empty_sections_omitted(self):
        # Same datasets, no changes except downloads
        changes = compare_reports(
            _make_report(datasets=[{"id": "a", "downloads": 100}]),
            _make_report(datasets=[{"id": "a", "downloads": 100}]),
            "2026-02-08", "2026-02-09",
        )
        md = format_changes_markdown(changes)

        assert "## New Datasets" not in md
        assert "## Removed Datasets" not in md
        assert "## Download Movers" not in md
        assert "## New GitHub Repos" not in md
        assert "## New Papers" not in md


class TestGenerateChangeSummary:
    def test_end_to_end(self, tmp_path):
        prev = _make_report(
            datasets=[
                {"id": "org/old", "downloads": 100, "likes": 5, "category": "code"},
            ],
            papers=[{"title": "Old Paper", "source": "arxiv"}],
        )
        curr = _make_report(
            datasets=[
                {"id": "org/old", "downloads": 150, "likes": 7, "category": "code"},
                {"id": "org/new", "downloads": 30, "likes": 1, "category": "sft"},
            ],
            papers=[
                {"title": "Old Paper", "source": "arxiv"},
                {"title": "New Paper", "source": "hf_papers"},
            ],
        )

        _write_report(tmp_path, "2026-02-08", prev)
        _write_report(tmp_path, "2026-02-09", curr)

        result = generate_change_summary(tmp_path, "2026-02-09")
        assert result is not None

        changes_path = Path(result)
        assert changes_path.exists()
        assert "2026-02-09_changes.md" in changes_path.name

        content = changes_path.read_text(encoding="utf-8")
        assert "## New Datasets (1)" in content
        assert "org/new" in content
        assert "## New Papers (1)" in content
        assert "New Paper" in content
        assert "## Download Movers" in content

    def test_first_run_returns_none(self, tmp_path):
        _write_report(tmp_path, "2026-02-09", _make_report())
        result = generate_change_summary(tmp_path, "2026-02-09")
        assert result is None
