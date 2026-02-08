"""Tests for MCP Server radar_trend, radar_history tools and _fmt_growth helper."""

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


class TestFmtGrowth:
    """Test _fmt_growth helper function."""

    def test_positive_growth(self):
        from server import _fmt_growth

        assert _fmt_growth(0.5) == "+50.0%"

    def test_negative_growth(self):
        from server import _fmt_growth

        assert _fmt_growth(-0.2) == "-20.0%"

    def test_zero_growth(self):
        from server import _fmt_growth

        assert _fmt_growth(0.0) == "0.0%"

    def test_none_growth(self):
        from server import _fmt_growth

        assert _fmt_growth(None) == "N/A"

    def test_infinity_growth(self):
        from server import _fmt_growth

        result = _fmt_growth(float("inf"))
        assert "∞" in result or "inf" in result.lower()

    def test_small_growth(self):
        from server import _fmt_growth

        assert _fmt_growth(0.005) == "+0.5%"

    def test_large_growth(self):
        from server import _fmt_growth

        assert _fmt_growth(10.0) == "+1000.0%"


class TestRadarTrendTool:
    """Test radar_trend tool definitions."""

    def test_radar_trend_tool_exists(self):
        """Test radar_trend tool is registered."""
        from server import list_tools
        import asyncio

        tools = asyncio.run(list_tools())
        tool_names = [t.name for t in tools]
        assert "radar_trend" in tool_names

    def test_radar_trend_modes(self):
        """Test radar_trend has correct mode enum."""
        from server import list_tools
        import asyncio

        tools = asyncio.run(list_tools())
        trend_tool = next(t for t in tools if t.name == "radar_trend")

        modes = trend_tool.inputSchema["properties"]["mode"]["enum"]
        assert "top_growing" in modes
        assert "rising" in modes
        assert "breakthroughs" in modes
        assert "dataset" in modes

    def test_radar_trend_default_mode(self):
        """Test radar_trend default mode is top_growing."""
        from server import list_tools
        import asyncio

        tools = asyncio.run(list_tools())
        trend_tool = next(t for t in tools if t.name == "radar_trend")

        assert trend_tool.inputSchema["properties"]["mode"]["default"] == "top_growing"

    def test_radar_trend_has_dataset_id(self):
        """Test radar_trend has dataset_id parameter."""
        from server import list_tools
        import asyncio

        tools = asyncio.run(list_tools())
        trend_tool = next(t for t in tools if t.name == "radar_trend")

        assert "dataset_id" in trend_tool.inputSchema["properties"]


class TestRadarHistoryTool:
    """Test radar_history tool definitions."""

    def test_radar_history_tool_exists(self):
        """Test radar_history tool is registered."""
        from server import list_tools
        import asyncio

        tools = asyncio.run(list_tools())
        tool_names = [t.name for t in tools]
        assert "radar_history" in tool_names

    def test_radar_history_default_limit(self):
        """Test radar_history default limit is 10."""
        from server import list_tools
        import asyncio

        tools = asyncio.run(list_tools())
        history_tool = next(t for t in tools if t.name == "radar_history")

        assert history_tool.inputSchema["properties"]["limit"]["default"] == 10


class TestRadarTrendExecution:
    """Test radar_trend tool execution with mocked database."""

    @pytest.mark.asyncio
    async def test_trend_no_database(self, tmp_path):
        """Test radar_trend when no database exists."""
        from server import call_tool

        with patch("server.PROJECT_ROOT", tmp_path):
            result = await call_tool("radar_trend", {})
            text = result[0].text
            assert "数据库不存在" in text

    @pytest.mark.asyncio
    async def test_trend_top_growing_empty(self, tmp_path):
        """Test radar_trend top_growing with empty database."""
        from server import call_tool

        # Create a real but empty database
        db_path = tmp_path / "data" / "radar.db"
        db_path.parent.mkdir(parents=True)

        from db import RadarDatabase

        db = RadarDatabase(str(db_path))
        db.close()

        with patch("server.PROJECT_ROOT", tmp_path):
            result = await call_tool("radar_trend", {"mode": "top_growing"})
            text = result[0].text
            assert "增长最快" in text
            assert "暂无" in text

    @pytest.mark.asyncio
    async def test_trend_rising_empty(self, tmp_path):
        """Test radar_trend rising mode with empty database."""
        from server import call_tool

        db_path = tmp_path / "data" / "radar.db"
        db_path.parent.mkdir(parents=True)

        from db import RadarDatabase

        db = RadarDatabase(str(db_path))
        db.close()

        with patch("server.PROJECT_ROOT", tmp_path):
            result = await call_tool("radar_trend", {"mode": "rising"})
            text = result[0].text
            assert "上升中" in text

    @pytest.mark.asyncio
    async def test_trend_breakthroughs_empty(self, tmp_path):
        """Test radar_trend breakthroughs mode with empty database."""
        from server import call_tool

        db_path = tmp_path / "data" / "radar.db"
        db_path.parent.mkdir(parents=True)

        from db import RadarDatabase

        db = RadarDatabase(str(db_path))
        db.close()

        with patch("server.PROJECT_ROOT", tmp_path):
            result = await call_tool("radar_trend", {"mode": "breakthroughs"})
            text = result[0].text
            assert "突破" in text

    @pytest.mark.asyncio
    async def test_trend_dataset_mode_not_found(self, tmp_path):
        """Test radar_trend dataset mode when dataset doesn't exist."""
        from server import call_tool

        db_path = tmp_path / "data" / "radar.db"
        db_path.parent.mkdir(parents=True)

        from db import RadarDatabase

        db = RadarDatabase(str(db_path))
        db.close()

        with patch("server.PROJECT_ROOT", tmp_path):
            result = await call_tool(
                "radar_trend", {"mode": "dataset", "dataset_id": "nonexistent/ds"}
            )
            text = result[0].text
            assert "未找到" in text

    @pytest.mark.asyncio
    async def test_trend_dataset_mode_with_data(self, tmp_path):
        """Test radar_trend dataset mode with actual data."""
        from server import call_tool
        from db import RadarDatabase
        from datetime import datetime

        db_path = tmp_path / "data" / "radar.db"
        db_path.parent.mkdir(parents=True)

        db = RadarDatabase(str(db_path))
        today = datetime.now().strftime("%Y-%m-%d")

        # Insert a dataset and daily stats
        db_id = db.upsert_dataset(
            source="huggingface",
            dataset_id="test-org/my-dataset",
            name="my-dataset",
            author="test-org",
        )
        db.record_daily_stats(db_id, downloads=1000, likes=50, date=today)
        db._get_connection().commit()
        db.close()

        with patch("server.PROJECT_ROOT", tmp_path):
            result = await call_tool(
                "radar_trend", {"mode": "dataset", "dataset_id": "test-org/my-dataset"}
            )
            text = result[0].text
            assert "趋势" in text or "test-org/my-dataset" in text

    @pytest.mark.asyncio
    async def test_trend_unknown_mode(self, tmp_path):
        """Test radar_trend with unknown mode."""
        from server import call_tool

        db_path = tmp_path / "data" / "radar.db"
        db_path.parent.mkdir(parents=True)

        from db import RadarDatabase

        db = RadarDatabase(str(db_path))
        db.close()

        with patch("server.PROJECT_ROOT", tmp_path):
            result = await call_tool("radar_trend", {"mode": "invalid_mode"})
            text = result[0].text
            assert "未知模式" in text

    @pytest.mark.asyncio
    async def test_trend_top_growing_with_data(self, tmp_path):
        """Test radar_trend top_growing mode with seeded database."""
        from server import call_tool
        from db import RadarDatabase
        from datetime import datetime, timedelta

        db_path = tmp_path / "data" / "radar.db"
        db_path.parent.mkdir(parents=True)

        db = RadarDatabase(str(db_path))
        today = datetime.now().strftime("%Y-%m-%d")
        week_ago = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")

        # Insert dataset with growth
        db_id = db.upsert_dataset(
            source="huggingface",
            dataset_id="fast-grower/data",
            name="data",
            author="fast-grower",
        )
        db.record_daily_stats(db_id, downloads=100, date=week_ago)
        db.record_daily_stats(db_id, downloads=500, date=today)
        # Calculate growth rate and record trend
        growth = db.calculate_growth_rate(db_id, days=7)
        db.record_trend(db_id, downloads_7d_growth=growth, date=today)
        db._get_connection().commit()
        db.close()

        with patch("server.PROJECT_ROOT", tmp_path):
            result = await call_tool("radar_trend", {"mode": "top_growing", "days": 7})
            text = result[0].text
            assert "增长最快" in text
            assert "fast-grower" in text or "data" in text


class TestRadarHistoryExecution:
    """Test radar_history tool execution."""

    @pytest.mark.asyncio
    async def test_history_no_reports(self, tmp_path):
        """Test radar_history when no reports exist."""
        from server import call_tool

        with patch("server.PROJECT_ROOT", tmp_path):
            result = await call_tool("radar_history", {})
            text = result[0].text
            assert "没有找到" in text

    @pytest.mark.asyncio
    async def test_history_single_report(self, tmp_path):
        """Test radar_history with one report."""
        from server import call_tool

        reports_dir = tmp_path / "data" / "reports"
        reports_dir.mkdir(parents=True)

        report = {
            "generated_at": "2024-03-01T10:00:00",
            "summary": {
                "total_datasets": 10,
                "total_github_repos": 30,
                "total_github_repos_high_relevance": 5,
                "total_papers": 15,
                "total_blog_posts": 8,
            },
        }
        (reports_dir / "intel_report_2024-03-01.json").write_text(json.dumps(report))

        with patch("server.PROJECT_ROOT", tmp_path):
            result = await call_tool("radar_history", {})
            text = result[0].text
            assert "2024-03-01" in text
            assert "10" in text  # total_datasets
            assert "历史" in text

    @pytest.mark.asyncio
    async def test_history_multiple_reports_shows_trend(self, tmp_path):
        """Test radar_history shows trend line with multiple reports."""
        from server import call_tool

        reports_dir = tmp_path / "data" / "reports"
        reports_dir.mkdir(parents=True)

        report1 = {
            "generated_at": "2024-01-01T10:00:00",
            "summary": {
                "total_datasets": 5,
                "total_github_repos": 20,
                "total_github_repos_high_relevance": 2,
                "total_papers": 10,
                "total_blog_posts": 4,
            },
        }
        report2 = {
            "generated_at": "2024-02-01T10:00:00",
            "summary": {
                "total_datasets": 12,
                "total_github_repos": 35,
                "total_github_repos_high_relevance": 5,
                "total_papers": 18,
                "total_blog_posts": 9,
            },
        }
        (reports_dir / "intel_report_2024-01-01.json").write_text(json.dumps(report1))
        (reports_dir / "intel_report_2024-02-01.json").write_text(json.dumps(report2))

        with patch("server.PROJECT_ROOT", tmp_path):
            result = await call_tool("radar_history", {})
            text = result[0].text
            assert "趋势" in text
            assert "2024-01-01" in text
            assert "2024-02-01" in text
            # Should show delta
            assert "+7" in text  # datasets: 5 → 12

    @pytest.mark.asyncio
    async def test_history_respects_limit(self, tmp_path):
        """Test radar_history respects limit parameter."""
        from server import call_tool

        reports_dir = tmp_path / "data" / "reports"
        reports_dir.mkdir(parents=True)

        for i in range(5):
            report = {
                "generated_at": f"2024-0{i + 1}-01T10:00:00",
                "summary": {
                    "total_datasets": i * 5,
                    "total_github_repos": i * 10,
                    "total_github_repos_high_relevance": i,
                    "total_papers": i * 3,
                    "total_blog_posts": i * 2,
                },
            }
            (reports_dir / f"intel_report_2024-0{i + 1}-01.json").write_text(json.dumps(report))

        with patch("server.PROJECT_ROOT", tmp_path):
            result = await call_tool("radar_history", {"limit": 2})
            text = result[0].text
            assert "最近 2 期" in text

    @pytest.mark.asyncio
    async def test_history_table_format(self, tmp_path):
        """Test radar_history outputs a markdown table."""
        from server import call_tool

        reports_dir = tmp_path / "data" / "reports"
        reports_dir.mkdir(parents=True)

        report = {
            "generated_at": "2024-06-15T10:00:00",
            "summary": {
                "total_datasets": 20,
                "total_github_repos": 50,
                "total_github_repos_high_relevance": 8,
                "total_papers": 25,
                "total_blog_posts": 12,
            },
        }
        (reports_dir / "intel_report_2024-06-15.json").write_text(json.dumps(report))

        with patch("server.PROJECT_ROOT", tmp_path):
            result = await call_tool("radar_history", {})
            text = result[0].text
            # Should contain markdown table separators
            assert "|" in text
            assert "---" in text

    @pytest.mark.asyncio
    async def test_history_handles_corrupt_report(self, tmp_path):
        """Test radar_history skips corrupt JSON files gracefully."""
        from server import call_tool

        reports_dir = tmp_path / "data" / "reports"
        reports_dir.mkdir(parents=True)

        # Write a valid report and a corrupt one
        valid = {
            "generated_at": "2024-02-01T10:00:00",
            "summary": {
                "total_datasets": 10,
                "total_github_repos": 20,
                "total_github_repos_high_relevance": 3,
                "total_papers": 5,
                "total_blog_posts": 2,
            },
        }
        (reports_dir / "intel_report_2024-02-01.json").write_text(json.dumps(valid))
        (reports_dir / "intel_report_2024-01-01.json").write_text("{corrupt json!!!")

        with patch("server.PROJECT_ROOT", tmp_path):
            result = await call_tool("radar_history", {})
            text = result[0].text
            # Should still show the valid report
            assert "2024-02-01" in text


class TestMainIntelTrendIntegration:
    """Test that main_intel.py imports trend components correctly."""

    def test_main_intel_imports_trend_analyzer(self):
        """Test main_intel imports TrendAnalyzer."""
        from analyzers.trend import TrendAnalyzer

        assert TrendAnalyzer is not None

    def test_main_intel_imports_radar_database(self):
        """Test main_intel imports RadarDatabase."""
        from db import RadarDatabase

        assert RadarDatabase is not None
