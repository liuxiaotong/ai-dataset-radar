"""Tests for MCP Server tools."""

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

pytest.importorskip("mcp", reason="mcp package not installed")

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "mcp_server"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))


class TestMCPServerHelpers:
    """Test helper functions in MCP server."""

    def test_get_latest_report_no_reports(self, tmp_path):
        """Test get_latest_report when no reports exist."""
        from server import get_latest_report

        # Temporarily override PROJECT_ROOT
        with patch("server.PROJECT_ROOT", tmp_path):
            result = get_latest_report()
            assert result is None

    def test_get_latest_report_with_reports(self, tmp_path):
        """Test get_latest_report returns latest report."""
        from server import get_latest_report

        reports_dir = tmp_path / "data" / "reports"
        reports_dir.mkdir(parents=True)

        # Create test reports
        report1 = {"summary": {"total_datasets": 5}}
        report2 = {"summary": {"total_datasets": 10}}

        (reports_dir / "intel_report_2024-01-01.json").write_text(json.dumps(report1))
        (reports_dir / "intel_report_2024-01-02.json").write_text(json.dumps(report2))

        with patch("server.PROJECT_ROOT", tmp_path):
            result = get_latest_report()
            assert result is not None
            assert result["summary"]["total_datasets"] == 10

    def test_get_latest_report_path(self, tmp_path):
        """Test get_latest_report_path returns correct path."""
        from server import get_latest_report_path

        reports_dir = tmp_path / "data" / "reports"
        reports_dir.mkdir(parents=True)

        (reports_dir / "intel_report_2024-01-01.json").write_text("{}")
        (reports_dir / "intel_report_2024-01-02.json").write_text("{}")

        with patch("server.PROJECT_ROOT", tmp_path):
            result = get_latest_report_path()
            assert result is not None
            assert "2024-01-02" in str(result)


class TestMCPToolInputValidation:
    """Test input validation for MCP tools."""

    def test_radar_scan_default_days(self):
        """Test radar_scan uses default days=7."""
        # This tests the tool schema, not the actual execution
        from server import list_tools
        import asyncio

        tools = asyncio.run(list_tools())
        scan_tool = next(t for t in tools if t.name == "radar_scan")

        assert scan_tool.inputSchema["properties"]["days"]["default"] == 7

    def test_radar_datasets_categories(self):
        """Test radar_datasets has correct category filter."""
        from server import list_tools
        import asyncio

        tools = asyncio.run(list_tools())
        datasets_tool = next(t for t in tools if t.name == "radar_datasets")

        desc = datasets_tool.inputSchema["properties"]["category"]["description"]
        assert "synthetic" in desc
        assert "sft" in desc
        assert "preference" in desc

    def test_radar_blogs_tool_exists(self):
        """Test radar_blogs tool is registered."""
        from server import list_tools
        import asyncio

        tools = asyncio.run(list_tools())
        tool_names = [t.name for t in tools]

        assert "radar_blogs" in tool_names

    def test_all_tools_have_descriptions(self):
        """Test all tools have non-empty descriptions."""
        from server import list_tools
        import asyncio

        tools = asyncio.run(list_tools())

        for tool in tools:
            assert tool.description, f"Tool {tool.name} has no description"
            # Chinese descriptions are shorter in char count but equally meaningful
            assert len(tool.description) >= 5, f"Tool {tool.name} has short description"


class TestMCPToolExecution:
    """Test tool execution with mocked data."""

    @pytest.fixture
    def mock_report(self):
        """Create a mock report for testing."""
        return {
            "generated_at": "2024-01-15T10:00:00",
            "period": {"days": 7, "start": "2024-01-08", "end": "2024-01-15"},
            "summary": {
                "total_datasets": 15,
                "total_github_orgs": 5,
                "total_github_repos": 50,
                "total_github_repos_high_relevance": 3,
                "total_papers": 20,
                "total_blog_posts": 10,
            },
            "datasets": [
                {"id": "test/dataset1", "category": "synthetic", "downloads": 100},
                {"id": "test/dataset2", "category": "sft", "downloads": 200},
            ],
            "datasets_by_type": {"synthetic": ["test/dataset1"], "sft": ["test/dataset2"]},
            "github_activity": [
                {
                    "org": "test-org",
                    "repos_updated": [
                        {
                            "name": "repo1",
                            "stars": 100,
                            "relevance": "high",
                            "relevance_signals": ["dataset"],
                        },
                    ],
                }
            ],
            "papers": [
                {
                    "title": "Test Paper",
                    "url": "https://arxiv.org/abs/1234",
                    "abstract": "Test abstract",
                },
            ],
            "blog_posts": [
                {
                    "source": "Test Blog",
                    "articles": [
                        {
                            "title": "Test Article",
                            "url": "https://example.com/article",
                            "date": "2024-01-15",
                            "signals": ["rlhf"],
                        },
                    ],
                }
            ],
        }

    @pytest.mark.asyncio
    async def test_radar_summary_with_report(self, mock_report, tmp_path):
        """Test radar_summary returns correct summary."""
        from server import call_tool

        reports_dir = tmp_path / "data" / "reports"
        reports_dir.mkdir(parents=True)
        (reports_dir / "intel_report_2024-01-15.json").write_text(json.dumps(mock_report))

        with patch("server.PROJECT_ROOT", tmp_path):
            result = await call_tool("radar_summary", {})
            assert len(result) == 1
            text = result[0].text
            assert "15" in text  # total_datasets
            assert "20" in text  # total_papers

    @pytest.mark.asyncio
    async def test_radar_datasets_filter_by_category(self, mock_report, tmp_path):
        """Test radar_datasets filters by category."""
        from server import call_tool

        reports_dir = tmp_path / "data" / "reports"
        reports_dir.mkdir(parents=True)
        (reports_dir / "intel_report_2024-01-15.json").write_text(json.dumps(mock_report))

        with patch("server.PROJECT_ROOT", tmp_path):
            result = await call_tool("radar_datasets", {"category": "synthetic"})
            text = result[0].text
            assert "dataset1" in text
            assert "dataset2" not in text

    @pytest.mark.asyncio
    async def test_radar_blogs_returns_articles(self, mock_report, tmp_path):
        """Test radar_blogs returns blog articles."""
        from server import call_tool

        reports_dir = tmp_path / "data" / "reports"
        reports_dir.mkdir(parents=True)
        (reports_dir / "intel_report_2024-01-15.json").write_text(json.dumps(mock_report))

        with patch("server.PROJECT_ROOT", tmp_path):
            result = await call_tool("radar_blogs", {})
            text = result[0].text
            assert "Test Blog" in text
            assert "Test Article" in text

    @pytest.mark.asyncio
    async def test_radar_github_filters_relevance(self, mock_report, tmp_path):
        """Test radar_github filters by relevance."""
        from server import call_tool

        reports_dir = tmp_path / "data" / "reports"
        reports_dir.mkdir(parents=True)
        (reports_dir / "intel_report_2024-01-15.json").write_text(json.dumps(mock_report))

        with patch("server.PROJECT_ROOT", tmp_path):
            result = await call_tool("radar_github", {"relevance": "high"})
            text = result[0].text
            assert "repo1" in text
            assert "test-org" in text

    @pytest.mark.asyncio
    async def test_radar_papers_returns_papers(self, mock_report, tmp_path):
        """Test radar_papers returns paper list."""
        from server import call_tool

        reports_dir = tmp_path / "data" / "reports"
        reports_dir.mkdir(parents=True)
        (reports_dir / "intel_report_2024-01-15.json").write_text(json.dumps(mock_report))

        with patch("server.PROJECT_ROOT", tmp_path):
            result = await call_tool("radar_papers", {"limit": 5})
            text = result[0].text
            assert "Test Paper" in text

    @pytest.mark.asyncio
    async def test_unknown_tool_returns_error(self):
        """Test unknown tool returns error message."""
        from server import call_tool

        result = await call_tool("unknown_tool", {})
        text = result[0].text
        assert "未知工具" in text


class TestMCPServerIntegration:
    """Integration tests for MCP server."""

    def test_server_initialization(self):
        """Test server initializes correctly."""
        from server import server

        assert server.name == "ai-dataset-radar"

    def test_tools_list_complete(self):
        """Test all expected tools are registered."""
        from server import list_tools
        import asyncio

        tools = asyncio.run(list_tools())
        tool_names = {t.name for t in tools}

        expected_tools = {
            "radar_scan",
            "radar_summary",
            "radar_datasets",
            "radar_github",
            "radar_papers",
            "radar_blogs",
            "radar_config",
            "radar_search",
            "radar_diff",
            "radar_trend",
            "radar_history",
        }

        assert expected_tools == tool_names
