"""Tests for MCP Server radar_search, radar_diff, and parameter extensions."""

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "mcp_server"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))


@pytest.fixture
def rich_report():
    """Create a detailed mock report with all data sources."""
    return {
        "generated_at": "2024-02-01T10:00:00",
        "period": {"days": 7, "start": "2024-01-25", "end": "2024-02-01"},
        "summary": {
            "total_datasets": 5,
            "total_github_orgs": 3,
            "total_github_repos": 20,
            "total_github_repos_high_relevance": 4,
            "total_papers": 8,
            "total_blog_posts": 6,
        },
        "datasets": [
            {"id": "openai/gsm8k", "category": "evaluation", "description": "Grade school math", "downloads": 50000, "all_categories": ["evaluation", "math"]},
            {"id": "meta/llama-sft", "category": "sft", "description": "SFT training data for Llama", "downloads": 30000, "all_categories": ["sft"]},
            {"id": "nvidia/helpsteer2", "category": "preference", "description": "RLHF preference dataset", "downloads": 10000, "all_categories": ["preference", "rlhf"]},
            {"id": "openai/webgpt", "category": "preference", "description": "WebGPT comparison data", "downloads": 5000, "all_categories": ["preference"]},
            {"id": "bigcode/starcoderdata", "category": "code", "description": "Code pretraining data", "downloads": 80000, "all_categories": ["code"]},
        ],
        "github_activity": [
            {
                "org": "argilla-io",
                "repos_count": 3,
                "repos_updated": [
                    {"name": "argilla", "full_name": "argilla-io/argilla", "description": "Collaboration tool for AI datasets", "stars": 4800, "relevance": "high", "relevance_signals": ["dataset", "rlhf"], "topics": ["annotation", "llm"], "signals": ["dataset", "rlhf", "annotation"]},
                    {"name": "distilabel", "full_name": "argilla-io/distilabel", "description": "Synthetic data framework", "stars": 3000, "relevance": "high", "relevance_signals": ["dataset"], "topics": ["synthetic-data"], "signals": ["synthetic-data"]},
                ],
                "has_activity": True,
            },
            {
                "org": "scaleapi",
                "repos_count": 2,
                "repos_updated": [
                    {"name": "llm-engine", "full_name": "scaleapi/llm-engine", "description": "LLM fine-tuning engine", "stars": 500, "relevance": "medium", "relevance_signals": ["fine-tuning"], "topics": [], "signals": ["fine-tuning"]},
                ],
                "has_activity": True,
            },
        ],
        "papers": [
            {"title": "RLHF at Scale", "url": "https://arxiv.org/abs/2401.001", "abstract": "Scaling RLHF training", "source": "arxiv", "categories": ["cs.CL"]},
            {"title": "Synthetic Data Generation", "url": "https://arxiv.org/abs/2401.002", "abstract": "Methods for synthetic data", "source": "arxiv", "categories": ["cs.AI"]},
            {"title": "Dataset Deduplication", "url": "https://hf.co/papers/123", "abstract": "Deduplication techniques", "source": "huggingface", "categories": []},
        ],
        "blog_posts": [
            {
                "source": "OpenAI Blog",
                "articles": [
                    {"title": "GPT-5 Training Data", "url": "https://openai.com/blog/gpt5-data", "date": "2024-01-30", "signals": ["dataset", "training data"], "snippet": "We describe our data pipeline"},
                    {"title": "Safety Updates", "url": "https://openai.com/blog/safety", "date": "2024-01-28", "signals": ["alignment"]},
                ],
            },
            {
                "source": "Microsoft Research",
                "articles": [
                    {"title": "Phi-4 Synthetic Data", "url": "https://msft.com/phi4", "date": "2024-01-29", "signals": ["synthetic data"], "snippet": "Using synthetic data for Phi-4"},
                ],
            },
        ],
        "x_activity": {
            "accounts": [
                {
                    "username": "OpenAI",
                    "total_tweets": 5,
                    "relevant_tweets": [
                        {"username": "OpenAI", "text": "We're releasing a new dataset for RLHF research", "url": "https://x.com/OpenAI/status/123", "date": "2024-01-30"},
                    ],
                    "has_activity": True,
                },
            ],
            "search_results": [],
        },
    }


@pytest.fixture
def older_report():
    """Create an older mock report for diff testing."""
    return {
        "generated_at": "2024-01-25T10:00:00",
        "summary": {
            "total_datasets": 3,
            "total_github_orgs": 2,
            "total_github_repos": 15,
            "total_github_repos_high_relevance": 2,
            "total_papers": 5,
            "total_blog_posts": 4,
        },
        "datasets": [
            {"id": "openai/gsm8k"},
            {"id": "meta/llama-sft"},
            {"id": "removed/old-dataset"},
        ],
        "github_activity": [
            {
                "org": "argilla-io",
                "repos_updated": [
                    {"full_name": "argilla-io/argilla"},
                ],
            },
        ],
        "papers": [
            {"title": "RLHF at Scale"},
            {"title": "Old Paper About Transformers"},
        ],
        "blog_posts": [
            {
                "source": "OpenAI Blog",
                "articles": [
                    {"url": "https://openai.com/blog/old-post"},
                ],
            },
        ],
    }


def _write_report(tmp_path, date_str, report):
    """Helper to write a report to tmp_path."""
    reports_dir = tmp_path / "data" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / f"intel_report_{date_str}.json").write_text(json.dumps(report))


# ─── radar_search tests ──────────────────────────────────────────────────────

class TestSearchInReport:
    """Test the search_in_report helper function."""

    def test_search_datasets_by_keyword(self, rich_report):
        from server import search_in_report
        results = search_in_report(rich_report, "rlhf", ["datasets"], 10)
        assert "datasets" in results
        ids = [d["id"] for d in results["datasets"]]
        assert "nvidia/helpsteer2" in ids

    def test_search_github_by_keyword(self, rich_report):
        from server import search_in_report
        results = search_in_report(rich_report, "synthetic", ["github"], 10)
        assert "github" in results
        names = [r["name"] for r in results["github"]]
        assert "argilla-io/distilabel" in names

    def test_search_papers_by_keyword(self, rich_report):
        from server import search_in_report
        results = search_in_report(rich_report, "deduplication", ["papers"], 10)
        assert "papers" in results
        assert results["papers"][0]["title"] == "Dataset Deduplication"

    def test_search_blogs_by_keyword(self, rich_report):
        from server import search_in_report
        results = search_in_report(rich_report, "synthetic", ["blogs"], 10)
        assert "blogs" in results
        assert results["blogs"][0]["source"] == "Microsoft Research"

    def test_search_x_by_keyword(self, rich_report):
        from server import search_in_report
        results = search_in_report(rich_report, "RLHF", ["x"], 10)
        assert "x" in results
        assert results["x"][0]["username"] == "OpenAI"

    def test_search_all_sources(self, rich_report):
        from server import search_in_report
        results = search_in_report(rich_report, "dataset", [], 10)
        # Should find matches in datasets, github, papers, blogs, x
        assert len(results) >= 3

    def test_search_no_match(self, rich_report):
        from server import search_in_report
        results = search_in_report(rich_report, "xyznonexistent", [], 10)
        assert results == {}

    def test_search_regex_pattern(self, rich_report):
        from server import search_in_report
        results = search_in_report(rich_report, r"llama|Llama", ["datasets"], 10)
        assert "datasets" in results
        ids = [d["id"] for d in results["datasets"]]
        assert "meta/llama-sft" in ids

    def test_search_invalid_regex_fallback(self, rich_report):
        from server import search_in_report
        # Invalid regex should fall back to literal search
        results = search_in_report(rich_report, "[invalid(regex", [], 10)
        assert isinstance(results, dict)

    def test_search_limit_respected(self, rich_report):
        from server import search_in_report
        results = search_in_report(rich_report, "data", ["datasets"], 2)
        if "datasets" in results:
            assert len(results["datasets"]) <= 2

    def test_search_case_insensitive(self, rich_report):
        from server import search_in_report
        results = search_in_report(rich_report, "OPENAI", ["datasets"], 10)
        assert "datasets" in results


# ─── radar_diff tests ────────────────────────────────────────────────────────

class TestDiffReports:
    """Test the diff_reports helper function."""

    def test_diff_summary_changes(self, older_report, rich_report):
        from server import diff_reports
        diff = diff_reports(older_report, rich_report)

        assert diff["summary_changes"]["total_datasets"]["before"] == 3
        assert diff["summary_changes"]["total_datasets"]["after"] == 5
        assert diff["summary_changes"]["total_datasets"]["delta"] == 2

    def test_diff_new_datasets(self, older_report, rich_report):
        from server import diff_reports
        diff = diff_reports(older_report, rich_report)

        new_ds = diff["new_items"].get("datasets", [])
        assert "nvidia/helpsteer2" in new_ds
        assert "openai/webgpt" in new_ds
        assert "bigcode/starcoderdata" in new_ds

    def test_diff_removed_datasets(self, older_report, rich_report):
        from server import diff_reports
        diff = diff_reports(older_report, rich_report)

        removed_ds = diff["removed_items"].get("datasets", [])
        assert "removed/old-dataset" in removed_ds

    def test_diff_new_repos(self, older_report, rich_report):
        from server import diff_reports
        diff = diff_reports(older_report, rich_report)

        new_repos = diff["new_items"].get("github_repos", [])
        assert "argilla-io/distilabel" in new_repos
        assert "scaleapi/llm-engine" in new_repos

    def test_diff_new_papers(self, older_report, rich_report):
        from server import diff_reports
        diff = diff_reports(older_report, rich_report)

        new_papers = diff["new_items"].get("papers", [])
        assert "Synthetic Data Generation" in new_papers
        assert "Dataset Deduplication" in new_papers

    def test_diff_new_blog_articles(self, older_report, rich_report):
        from server import diff_reports
        diff = diff_reports(older_report, rich_report)

        new_blogs = diff["new_items"].get("blog_articles", [])
        assert "https://openai.com/blog/gpt5-data" in new_blogs
        assert "https://openai.com/blog/safety" in new_blogs

    def test_diff_identical_reports(self, rich_report):
        from server import diff_reports
        diff = diff_reports(rich_report, rich_report)

        assert diff["summary_changes"] == {}
        assert diff["new_items"] == {}
        assert diff["removed_items"] == {}

    def test_diff_period_labels(self, older_report, rich_report):
        from server import diff_reports
        diff = diff_reports(older_report, rich_report)

        assert diff["period_a"] == "2024-01-25"
        assert diff["period_b"] == "2024-02-01"


# ─── MCP tool execution tests ────────────────────────────────────────────────

class TestRadarSearchTool:
    """Test radar_search tool execution via call_tool."""

    @pytest.mark.asyncio
    async def test_search_returns_results(self, rich_report, tmp_path):
        from server import call_tool
        _write_report(tmp_path, "2024-02-01", rich_report)

        with patch("server.PROJECT_ROOT", tmp_path):
            result = await call_tool("radar_search", {"query": "rlhf"})
            text = result[0].text
            assert "搜索结果" in text
            assert "rlhf" in text.lower() or "RLHF" in text

    @pytest.mark.asyncio
    async def test_search_no_results(self, rich_report, tmp_path):
        from server import call_tool
        _write_report(tmp_path, "2024-02-01", rich_report)

        with patch("server.PROJECT_ROOT", tmp_path):
            result = await call_tool("radar_search", {"query": "xyznonexistent"})
            text = result[0].text
            assert "未找到" in text

    @pytest.mark.asyncio
    async def test_search_with_source_filter(self, rich_report, tmp_path):
        from server import call_tool
        _write_report(tmp_path, "2024-02-01", rich_report)

        with patch("server.PROJECT_ROOT", tmp_path):
            result = await call_tool("radar_search", {"query": "data", "sources": ["github"]})
            text = result[0].text
            assert "GitHub" in text

    @pytest.mark.asyncio
    async def test_search_no_report(self, tmp_path):
        from server import call_tool

        with patch("server.PROJECT_ROOT", tmp_path):
            result = await call_tool("radar_search", {"query": "test"})
            text = result[0].text
            assert "没有找到报告" in text

    @pytest.mark.asyncio
    async def test_search_empty_query(self, rich_report, tmp_path):
        from server import call_tool
        _write_report(tmp_path, "2024-02-01", rich_report)

        with patch("server.PROJECT_ROOT", tmp_path):
            result = await call_tool("radar_search", {"query": ""})
            text = result[0].text
            assert "请提供" in text


class TestRadarDiffTool:
    """Test radar_diff tool execution via call_tool."""

    @pytest.mark.asyncio
    async def test_diff_two_reports(self, older_report, rich_report, tmp_path):
        from server import call_tool
        _write_report(tmp_path, "2024-01-25", older_report)
        _write_report(tmp_path, "2024-02-01", rich_report)

        with patch("server.PROJECT_ROOT", tmp_path):
            result = await call_tool("radar_diff", {})
            text = result[0].text
            assert "报告对比" in text
            assert "新增" in text

    @pytest.mark.asyncio
    async def test_diff_by_date(self, older_report, rich_report, tmp_path):
        from server import call_tool
        _write_report(tmp_path, "2024-01-25", older_report)
        _write_report(tmp_path, "2024-02-01", rich_report)

        with patch("server.PROJECT_ROOT", tmp_path):
            result = await call_tool("radar_diff", {"date_a": "2024-01-25", "date_b": "2024-02-01"})
            text = result[0].text
            assert "2024-01-25" in text
            assert "2024-02-01" in text

    @pytest.mark.asyncio
    async def test_diff_missing_date(self, older_report, rich_report, tmp_path):
        from server import call_tool
        _write_report(tmp_path, "2024-01-25", older_report)
        _write_report(tmp_path, "2024-02-01", rich_report)

        with patch("server.PROJECT_ROOT", tmp_path):
            result = await call_tool("radar_diff", {"date_a": "2099-01-01"})
            text = result[0].text
            assert "未找到" in text

    @pytest.mark.asyncio
    async def test_diff_insufficient_reports(self, rich_report, tmp_path):
        from server import call_tool
        _write_report(tmp_path, "2024-02-01", rich_report)

        with patch("server.PROJECT_ROOT", tmp_path):
            result = await call_tool("radar_diff", {})
            text = result[0].text
            assert "至少两份" in text


# ─── Parameter extension tests ───────────────────────────────────────────────

class TestParameterExtensions:
    """Test new parameters on existing tools."""

    @pytest.mark.asyncio
    async def test_radar_datasets_org_filter(self, rich_report, tmp_path):
        from server import call_tool
        _write_report(tmp_path, "2024-02-01", rich_report)

        with patch("server.PROJECT_ROOT", tmp_path):
            result = await call_tool("radar_datasets", {"org": "openai"})
            text = result[0].text
            assert "gsm8k" in text
            assert "webgpt" in text
            assert "starcoderdata" not in text

    @pytest.mark.asyncio
    async def test_radar_datasets_org_and_category(self, rich_report, tmp_path):
        from server import call_tool
        _write_report(tmp_path, "2024-02-01", rich_report)

        with patch("server.PROJECT_ROOT", tmp_path):
            result = await call_tool("radar_datasets", {"org": "openai", "category": "preference"})
            text = result[0].text
            assert "webgpt" in text
            assert "gsm8k" not in text

    @pytest.mark.asyncio
    async def test_radar_github_org_filter(self, rich_report, tmp_path):
        from server import call_tool
        _write_report(tmp_path, "2024-02-01", rich_report)

        with patch("server.PROJECT_ROOT", tmp_path):
            result = await call_tool("radar_github", {"org": "argilla", "relevance": "high"})
            text = result[0].text
            assert "argilla" in text
            assert "scaleapi" not in text

    def test_radar_scan_has_sources_param(self):
        """Test radar_scan tool definition includes sources parameter."""
        from server import list_tools
        import asyncio

        tools = asyncio.run(list_tools())
        scan_tool = next(t for t in tools if t.name == "radar_scan")
        assert "sources" in scan_tool.inputSchema["properties"]

    def test_radar_search_tool_registered(self):
        """Test radar_search tool is registered."""
        from server import list_tools
        import asyncio

        tools = asyncio.run(list_tools())
        tool_names = {t.name for t in tools}
        assert "radar_search" in tool_names
        assert "radar_diff" in tool_names

    def test_all_11_tools_registered(self):
        """Test all 11 tools are registered."""
        from server import list_tools
        import asyncio

        tools = asyncio.run(list_tools())
        tool_names = {t.name for t in tools}

        expected = {
            "radar_scan", "radar_summary", "radar_datasets",
            "radar_github", "radar_papers", "radar_blogs",
            "radar_config", "radar_search", "radar_diff",
            "radar_trend", "radar_history",
        }
        assert expected == tool_names


# ─── Helper function tests ───────────────────────────────────────────────────

class TestHelperFunctions:
    """Test new helper functions."""

    def test_get_report_by_date(self, tmp_path):
        from server import get_report_by_date
        reports_dir = tmp_path / "data" / "reports"
        reports_dir.mkdir(parents=True)
        (reports_dir / "intel_report_2024-02-01.json").write_text('{"test": true}')

        with patch("server.PROJECT_ROOT", tmp_path):
            result = get_report_by_date("2024-02-01")
            assert result == {"test": True}

    def test_get_report_by_date_not_found(self, tmp_path):
        from server import get_report_by_date

        with patch("server.PROJECT_ROOT", tmp_path):
            result = get_report_by_date("2099-01-01")
            assert result is None

    def test_get_all_reports_sorted(self, tmp_path):
        from server import get_all_reports_sorted
        reports_dir = tmp_path / "data" / "reports"
        reports_dir.mkdir(parents=True)
        (reports_dir / "intel_report_2024-01-01.json").write_text("{}")
        (reports_dir / "intel_report_2024-02-01.json").write_text("{}")
        (reports_dir / "intel_report_2024-01-15.json").write_text("{}")

        with patch("server.PROJECT_ROOT", tmp_path):
            paths = get_all_reports_sorted()
            assert len(paths) == 3
            assert "2024-02-01" in str(paths[0])  # Most recent first

    def test_get_all_reports_empty(self, tmp_path):
        from server import get_all_reports_sorted

        with patch("server.PROJECT_ROOT", tmp_path):
            paths = get_all_reports_sorted()
            assert paths == []
