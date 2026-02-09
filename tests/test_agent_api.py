"""Tests for the FastAPI REST server (agent/api.py).

Covers authentication, rate limiting, all REST endpoints, input validation,
error responses, and helper functions.
"""

import json
import sys
import time
from pathlib import Path
from types import ModuleType
from unittest.mock import AsyncMock, mock_open, patch

import pytest

# Add project paths for imports (matches convention used by other test files)
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

pytest.importorskip("fastapi", reason="fastapi package not installed")
pytest.importorskip("httpx", reason="httpx package not installed (needed by TestClient)")

from fastapi.testclient import TestClient

import agent.api as api_module

# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------

SAMPLE_REPORT = {
    "generated_at": "2025-05-01T12:00:00Z",
    "period": {"start": "2025-04-24", "end": "2025-05-01", "days": 7},
    "summary": {
        "total_datasets": 42,
        "total_github_orgs": 1,
        "total_github_repos": 3,
        "total_github_repos_high_relevance": 2,
        "total_papers": 8,
        "total_blog_posts": 20,
        "total_x_tweets": 0,
    },
    "datasets": [
        {"id": "openai/alpha-sft-v1", "category": "sft_instruction", "downloads": 5000, "org": "openai"},
        {"id": "anthropic/beta-preference-v2", "category": "rlhf_preference", "downloads": 3000, "org": "anthropic"},
        {"id": "meta/gamma-code-v3", "category": "code", "downloads": 100, "org": "meta"},
    ],
    "github_activity": [
        {
            "org": "test-org",
            "repos_count": 3,
            "repos_updated": [
                {"name": "repo-a", "full_name": "test-org/repo-a", "stars": 1200, "relevance": "high"},
                {"name": "repo-b", "full_name": "test-org/repo-b", "stars": 300, "relevance": "low"},
                {"name": "repo-c", "full_name": "test-org/repo-c", "stars": 900, "relevance": "high"},
            ],
            "has_activity": True,
        },
    ],
    "papers": [
        {"title": "Paper A", "source": "arxiv", "is_dataset_paper": True},
        {"title": "Paper B", "source": "huggingface", "is_dataset_paper": False},
        {"title": "Paper C", "source": "arxiv", "is_dataset_paper": False},
    ],
    "blog_posts": [
        {
            "source": "OpenAI Blog",
            "category": "us_frontier",
            "articles": [
                {"title": "New dataset release", "url": "https://openai.com/blog/1"},
                {"title": "Model update", "url": "https://openai.com/blog/2"},
            ],
        },
        {
            "source": "Qwen Blog",
            "category": "china",
            "articles": [
                {"title": "Qwen dataset", "url": "https://qwen.ai/blog/1"},
            ],
        },
    ],
}

SAMPLE_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "AI Dataset Radar Report",
    "type": "object",
}

SAMPLE_TOOLS = {
    "name": "ai-dataset-radar",
    "tools": [{"name": "radar_scan", "description": "Run a scan"}],
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_rate_limits():
    """Clear per-IP rate-limit store before and after every test."""
    api_module._rate_limit_store.clear()
    yield
    api_module._rate_limit_store.clear()


@pytest.fixture()
def client():
    """TestClient with authentication *disabled* (API_KEY is empty)."""
    original = api_module.API_KEY
    api_module.API_KEY = ""
    try:
        yield TestClient(api_module.app)
    finally:
        api_module.API_KEY = original


@pytest.fixture()
def auth_client():
    """TestClient with authentication *enabled* (API_KEY set)."""
    original = api_module.API_KEY
    api_module.API_KEY = "test-secret-key"
    try:
        yield TestClient(api_module.app)
    finally:
        api_module.API_KEY = original


def _mock_main_intel():
    """Return a context manager that injects a fake ``main_intel`` module.

    The ``/scan`` endpoint does ``from main_intel import run_intel_scan``
    inside the handler function.  We inject a mock module into
    ``sys.modules`` so the local import resolves without touching disk.
    """
    fake_module = ModuleType("main_intel")
    fake_module.run_intel_scan = AsyncMock(return_value=None)
    return patch.dict("sys.modules", {"main_intel": fake_module}), fake_module


# ===================================================================
# 1. Root endpoint
# ===================================================================


class TestRootEndpoint:
    def test_root_returns_api_info(self, client):
        """GET / returns API metadata and endpoint listing."""
        resp = client.get("/")
        assert resp.status_code == 200
        body = resp.json()
        assert body["name"] == "AI Dataset Radar API"
        assert "/scan" in body["endpoints"]
        assert "/summary" in body["endpoints"]
        assert "/datasets" in body["endpoints"]
        assert "/tools" in body["endpoints"]
        assert "docs" in body


# ===================================================================
# 2. API-key authentication
# ===================================================================


class TestAuthentication:
    def test_auth_disabled_when_no_key(self, client):
        """When API_KEY is empty any request passes without credentials."""
        with patch("agent.api.get_latest_report", return_value=SAMPLE_REPORT):
            resp = client.get("/summary")
        assert resp.status_code == 200

    def test_scan_requires_auth_when_key_set(self, auth_client):
        """POST /scan returns 401 when no API key is provided."""
        resp = auth_client.post("/scan", json={"days": 7})
        assert resp.status_code == 401
        assert "Invalid or missing API key" in resp.json()["detail"]

    def test_scan_auth_via_header(self, auth_client):
        """POST /scan succeeds with the correct X-API-Key header."""
        ctx, fake_mod = _mock_main_intel()
        with ctx, \
             patch("agent.api.get_latest_report", return_value=SAMPLE_REPORT), \
             patch("agent.api.get_latest_report_path", return_value=Path("/tmp/report.json")):
            resp = auth_client.post(
                "/scan",
                json={"days": 7},
                headers={"X-API-Key": "test-secret-key"},
            )
        assert resp.status_code == 200

    def test_scan_auth_via_query_param(self, auth_client):
        """POST /scan accepts api_key as a query parameter."""
        ctx, _ = _mock_main_intel()
        with ctx, \
             patch("agent.api.get_latest_report", return_value=SAMPLE_REPORT), \
             patch("agent.api.get_latest_report_path", return_value=Path("/tmp/report.json")):
            resp = auth_client.post(
                "/scan?api_key=test-secret-key",
                json={"days": 7},
            )
        assert resp.status_code == 200

    def test_scan_auth_wrong_key(self, auth_client):
        """POST /scan with an incorrect key returns 401."""
        resp = auth_client.post(
            "/scan",
            json={"days": 7},
            headers={"X-API-Key": "wrong-key"},
        )
        assert resp.status_code == 401


# ===================================================================
# 3. Rate-limiting middleware
# ===================================================================


class TestRateLimiting:
    def test_requests_within_limit_succeed(self, client):
        """Normal requests within the rate limit return 200."""
        resp = client.get("/")
        assert resp.status_code == 200

    def test_rate_limit_exceeded_returns_429(self, client):
        """Exceeding the rate limit returns 429 with Retry-After header."""
        original = api_module.RATE_LIMIT_REQUESTS
        api_module.RATE_LIMIT_REQUESTS = 3
        try:
            for _ in range(3):
                resp = client.get("/")
                assert resp.status_code == 200

            resp = client.get("/")
            assert resp.status_code == 429
            assert "Rate limit exceeded" in resp.json()["detail"]
            assert "Retry-After" in resp.headers
        finally:
            api_module.RATE_LIMIT_REQUESTS = original

    def test_old_entries_are_cleaned_up(self, client):
        """Timestamps older than the window are pruned, freeing capacity."""
        original = api_module.RATE_LIMIT_REQUESTS
        api_module.RATE_LIMIT_REQUESTS = 2
        # Inject an old timestamp that should be discarded
        api_module._rate_limit_store["testclient"] = [time.time() - 120.0]
        try:
            resp = client.get("/")
            assert resp.status_code == 200
        finally:
            api_module.RATE_LIMIT_REQUESTS = original


# ===================================================================
# 4. POST /scan
# ===================================================================


class TestScanEndpoint:
    def test_scan_success(self, client):
        """POST /scan runs the scanner and returns a summary."""
        ctx, fake_mod = _mock_main_intel()
        with ctx, \
             patch("agent.api.get_latest_report", return_value=SAMPLE_REPORT), \
             patch("agent.api.get_latest_report_path", return_value=Path("/tmp/report.json")):
            resp = client.post("/scan", json={"days": 7})
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert body["report_path"] is not None
        assert body["summary"]["total_datasets"] == 42

    def test_scan_default_days(self, client):
        """POST /scan without explicit days defaults to 7."""
        ctx, fake_mod = _mock_main_intel()
        with ctx, \
             patch("agent.api.get_latest_report", return_value=SAMPLE_REPORT), \
             patch("agent.api.get_latest_report_path", return_value=Path("/tmp/report.json")):
            resp = client.post("/scan", json={})
        assert resp.status_code == 200
        fake_mod.run_intel_scan.assert_called_once_with(days=7)

    def test_scan_invalid_days_too_low(self, client):
        """POST /scan with days=0 returns 422 (must be >= 1)."""
        resp = client.post("/scan", json={"days": 0})
        assert resp.status_code == 422

    def test_scan_invalid_days_too_high(self, client):
        """POST /scan with days=91 returns 422 (must be <= 90)."""
        resp = client.post("/scan", json={"days": 91})
        assert resp.status_code == 422

    def test_scan_returns_error_on_exception(self, client):
        """POST /scan returns success=False when the scanner raises."""
        ctx, fake_mod = _mock_main_intel()
        fake_mod.run_intel_scan.side_effect = RuntimeError("scan failed")
        with ctx:
            resp = client.post("/scan", json={"days": 7})
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is False
        assert "scan failed" in body["message"]


# ===================================================================
# 5. GET /summary
# ===================================================================


class TestSummaryEndpoint:
    def test_summary_returns_report_data(self, client):
        """GET /summary returns generated_at, period, and summary."""
        with patch("agent.api.get_latest_report", return_value=SAMPLE_REPORT):
            resp = client.get("/summary")
        assert resp.status_code == 200
        body = resp.json()
        assert body["generated_at"] == "2025-05-01T12:00:00Z"
        assert body["period"]["days"] == 7
        assert body["summary"]["total_datasets"] == 42

    def test_summary_no_report(self, client):
        """GET /summary returns 404 when no report exists."""
        with patch("agent.api.get_latest_report", return_value=None):
            resp = client.get("/summary")
        assert resp.status_code == 404
        assert "No report found" in resp.json()["detail"]


# ===================================================================
# 6. GET /datasets
# ===================================================================


class TestDatasetsEndpoint:
    def test_datasets_all(self, client):
        """GET /datasets returns all datasets by default."""
        with patch("agent.api.get_latest_report", return_value=SAMPLE_REPORT):
            resp = client.get("/datasets")
        assert resp.status_code == 200
        assert resp.json()["count"] == 3

    def test_datasets_filter_by_category(self, client):
        """GET /datasets?category=sft_instruction filters correctly (exact match)."""
        with patch("agent.api.get_latest_report", return_value=SAMPLE_REPORT):
            resp = client.get("/datasets?category=sft_instruction")
        body = resp.json()
        assert body["count"] == 1
        assert body["datasets"][0]["id"] == "openai/alpha-sft-v1"

    def test_datasets_filter_by_min_downloads(self, client):
        """GET /datasets?min_downloads=1000 excludes low-download datasets."""
        with patch("agent.api.get_latest_report", return_value=SAMPLE_REPORT):
            resp = client.get("/datasets?min_downloads=1000")
        body = resp.json()
        assert body["count"] == 2
        assert body["datasets"][0]["downloads"] == 5000

    def test_datasets_limit(self, client):
        """GET /datasets?limit=1 caps the number of results."""
        with patch("agent.api.get_latest_report", return_value=SAMPLE_REPORT):
            resp = client.get("/datasets?limit=1")
        assert resp.json()["count"] == 1

    def test_datasets_invalid_limit_too_low(self, client):
        """limit=0 is below the minimum (1) and returns 422."""
        resp = client.get("/datasets?limit=0")
        assert resp.status_code == 422

    def test_datasets_invalid_limit_too_high(self, client):
        """limit=501 exceeds the maximum (500) and returns 422."""
        resp = client.get("/datasets?limit=501")
        assert resp.status_code == 422

    def test_datasets_no_report(self, client):
        """GET /datasets returns 404 when no report exists."""
        with patch("agent.api.get_latest_report", return_value=None):
            resp = client.get("/datasets")
        assert resp.status_code == 404

    def test_datasets_sorted_by_downloads_desc(self, client):
        """Datasets are returned sorted by downloads descending."""
        with patch("agent.api.get_latest_report", return_value=SAMPLE_REPORT):
            resp = client.get("/datasets")
        downloads = [d["downloads"] for d in resp.json()["datasets"]]
        assert downloads == sorted(downloads, reverse=True)


# ===================================================================
# 7. GET /github
# ===================================================================


class TestGithubEndpoint:
    def test_github_default_high_relevance(self, client):
        """GET /github defaults to relevance=high."""
        with patch("agent.api.get_latest_report", return_value=SAMPLE_REPORT):
            resp = client.get("/github")
        body = resp.json()
        assert body["count"] == 2
        for repo in body["repos"]:
            assert repo["relevance"] == "high"

    def test_github_all_relevance(self, client):
        """GET /github?relevance=all returns every repo."""
        with patch("agent.api.get_latest_report", return_value=SAMPLE_REPORT):
            resp = client.get("/github?relevance=all")
        assert resp.json()["count"] == 3

    def test_github_low_relevance(self, client):
        """GET /github?relevance=low returns only low-relevance repos."""
        with patch("agent.api.get_latest_report", return_value=SAMPLE_REPORT):
            resp = client.get("/github?relevance=low")
        body = resp.json()
        assert body["count"] == 1
        assert body["repos"][0]["name"] == "repo-b"

    def test_github_sorted_by_stars_desc(self, client):
        """Repos are sorted by stars descending."""
        with patch("agent.api.get_latest_report", return_value=SAMPLE_REPORT):
            resp = client.get("/github?relevance=all")
        stars = [r["stars"] for r in resp.json()["repos"]]
        assert stars == sorted(stars, reverse=True)

    def test_github_limit(self, client):
        """GET /github?limit=1 caps results."""
        with patch("agent.api.get_latest_report", return_value=SAMPLE_REPORT):
            resp = client.get("/github?relevance=all&limit=1")
        assert resp.json()["count"] == 1

    def test_github_no_report(self, client):
        """GET /github returns 404 when no report exists."""
        with patch("agent.api.get_latest_report", return_value=None):
            resp = client.get("/github")
        assert resp.status_code == 404


# ===================================================================
# 8. GET /papers
# ===================================================================


class TestPapersEndpoint:
    def test_papers_all(self, client):
        """GET /papers returns all papers by default."""
        with patch("agent.api.get_latest_report", return_value=SAMPLE_REPORT):
            resp = client.get("/papers")
        assert resp.json()["count"] == 3

    def test_papers_filter_by_source(self, client):
        """GET /papers?source=arxiv filters by source."""
        with patch("agent.api.get_latest_report", return_value=SAMPLE_REPORT):
            resp = client.get("/papers?source=arxiv")
        body = resp.json()
        assert body["count"] == 2
        for p in body["papers"]:
            assert p["source"] == "arxiv"

    def test_papers_dataset_only(self, client):
        """GET /papers?dataset_only=true returns only dataset papers."""
        with patch("agent.api.get_latest_report", return_value=SAMPLE_REPORT):
            resp = client.get("/papers?dataset_only=true")
        body = resp.json()
        assert body["count"] == 1
        assert body["papers"][0]["title"] == "Paper A"

    def test_papers_limit(self, client):
        """GET /papers?limit=1 caps results."""
        with patch("agent.api.get_latest_report", return_value=SAMPLE_REPORT):
            resp = client.get("/papers?limit=1")
        assert resp.json()["count"] == 1

    def test_papers_no_report(self, client):
        """GET /papers returns 404 when no report exists."""
        with patch("agent.api.get_latest_report", return_value=None):
            resp = client.get("/papers")
        assert resp.status_code == 404


# ===================================================================
# 9. GET /blogs
# ===================================================================


class TestBlogsEndpoint:
    def test_blogs_all(self, client):
        """GET /blogs returns flattened articles from all sources."""
        with patch("agent.api.get_latest_report", return_value=SAMPLE_REPORT):
            resp = client.get("/blogs")
        body = resp.json()
        assert body["count"] == 3  # 2 OpenAI + 1 Qwen
        assert body["sources"] == 2

    def test_blogs_filter_by_source(self, client):
        """GET /blogs?source=OpenAI filters by source name (case-insensitive)."""
        with patch("agent.api.get_latest_report", return_value=SAMPLE_REPORT):
            resp = client.get("/blogs?source=OpenAI")
        body = resp.json()
        assert body["count"] == 2
        for a in body["articles"]:
            assert "OpenAI" in a["source"]

    def test_blogs_filter_by_category(self, client):
        """GET /blogs?category=china returns only matching category."""
        with patch("agent.api.get_latest_report", return_value=SAMPLE_REPORT):
            resp = client.get("/blogs?category=china")
        body = resp.json()
        assert body["count"] == 1
        assert body["articles"][0]["source"] == "Qwen Blog"

    def test_blogs_limit(self, client):
        """GET /blogs?limit=1 caps the number of articles."""
        with patch("agent.api.get_latest_report", return_value=SAMPLE_REPORT):
            resp = client.get("/blogs?limit=1")
        assert resp.json()["count"] == 1

    def test_blogs_no_report(self, client):
        """GET /blogs returns 404 when no report exists."""
        with patch("agent.api.get_latest_report", return_value=None):
            resp = client.get("/blogs")
        assert resp.status_code == 404

    def test_blogs_articles_include_source_and_category(self, client):
        """Each flattened article inherits source and category from its parent."""
        with patch("agent.api.get_latest_report", return_value=SAMPLE_REPORT):
            resp = client.get("/blogs")
        for article in resp.json()["articles"]:
            assert "source" in article
            assert "category" in article


# ===================================================================
# 10. GET /config
# ===================================================================


class TestConfigEndpoint:
    def test_config_success(self, client, tmp_path):
        """GET /config parses and returns the YAML config file."""
        import yaml

        config_data = {"orgs": ["openai", "anthropic"]}
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump(config_data))

        # Patch the config_path constructed inside the endpoint
        with patch.object(Path, "exists", return_value=True), \
             patch("builtins.open", mock_open(read_data=yaml.dump(config_data))):
            resp = client.get("/config")
        assert resp.status_code == 200
        assert resp.json()["orgs"] == ["openai", "anthropic"]

    def test_config_file_not_found(self, client):
        """GET /config returns 404 when config.yaml does not exist."""
        with patch.object(Path, "exists", return_value=False):
            resp = client.get("/config")
        assert resp.status_code == 404
        assert "Config file not found" in resp.json()["detail"]


# ===================================================================
# 11. GET /schema
# ===================================================================


class TestSchemaEndpoint:
    def test_schema_success(self, client):
        """GET /schema returns the JSON schema definition."""
        with patch.object(Path, "exists", return_value=True), \
             patch("builtins.open", mock_open(read_data=json.dumps(SAMPLE_SCHEMA))):
            resp = client.get("/schema")
        assert resp.status_code == 200
        assert resp.json()["title"] == "AI Dataset Radar Report"

    def test_schema_file_not_found(self, client):
        """GET /schema returns 404 when schema.json is missing."""
        with patch.object(Path, "exists", return_value=False):
            resp = client.get("/schema")
        assert resp.status_code == 404
        assert "Schema file not found" in resp.json()["detail"]


# ===================================================================
# 12. GET /tools
# ===================================================================


class TestToolsEndpoint:
    def test_tools_success(self, client):
        """GET /tools returns tool definitions for function calling."""
        with patch.object(Path, "exists", return_value=True), \
             patch("builtins.open", mock_open(read_data=json.dumps(SAMPLE_TOOLS))):
            resp = client.get("/tools")
        assert resp.status_code == 200
        body = resp.json()
        assert body["name"] == "ai-dataset-radar"
        assert len(body["tools"]) == 1

    def test_tools_file_not_found(self, client):
        """GET /tools returns 404 when tools.json is missing."""
        with patch.object(Path, "exists", return_value=False):
            resp = client.get("/tools")
        assert resp.status_code == 404
        assert "Tools file not found" in resp.json()["detail"]


# ===================================================================
# 13. Helper functions
# ===================================================================


class TestHelperFunctions:
    def test_get_reports_dir_path(self):
        """get_reports_dir points to <project>/data/reports."""
        result = api_module.get_reports_dir()
        assert result.name == "reports"
        assert result.parent.name == "data"

    def test_get_latest_report_no_directory(self, tmp_path):
        """Returns None when reports directory does not exist."""
        with patch.object(api_module, "get_reports_dir", return_value=tmp_path / "missing"):
            assert api_module.get_latest_report() is None

    def test_get_latest_report_empty_directory(self, tmp_path):
        """Returns None when reports directory contains no matching files."""
        reports_dir = tmp_path / "reports"
        reports_dir.mkdir()
        with patch.object(api_module, "get_reports_dir", return_value=reports_dir):
            assert api_module.get_latest_report() is None

    def test_get_latest_report_picks_latest(self, tmp_path):
        """Returns the most recent report based on filename sorting."""
        reports_dir = tmp_path / "reports"
        reports_dir.mkdir()
        (reports_dir / "intel_report_2025-04-01.json").write_text(
            json.dumps({"summary": {"total_datasets": 1}})
        )
        (reports_dir / "intel_report_2025-05-01.json").write_text(
            json.dumps({"summary": {"total_datasets": 99}})
        )
        with patch.object(api_module, "get_reports_dir", return_value=reports_dir):
            result = api_module.get_latest_report()
        assert result["summary"]["total_datasets"] == 99

    def test_get_latest_report_path_returns_path(self, tmp_path):
        """Returns the Path to the latest report file."""
        reports_dir = tmp_path / "reports"
        reports_dir.mkdir()
        (reports_dir / "intel_report_2025-05-01.json").write_text("{}")
        with patch.object(api_module, "get_reports_dir", return_value=reports_dir):
            result = api_module.get_latest_report_path()
        assert result is not None
        assert result.name == "intel_report_2025-05-01.json"

    def test_get_latest_report_path_none_when_empty(self, tmp_path):
        """Returns None when no report files are present."""
        reports_dir = tmp_path / "reports"
        reports_dir.mkdir()
        with patch.object(api_module, "get_reports_dir", return_value=reports_dir):
            assert api_module.get_latest_report_path() is None


# ===================================================================
# 14. API_KEY env-var interaction
# ===================================================================


class TestApiKeyEnvVar:
    def test_module_api_key_controls_auth(self):
        """Changing API_KEY at runtime enables/disables auth."""
        original = api_module.API_KEY
        api_module.API_KEY = "runtime-key"
        tc = TestClient(api_module.app)
        try:
            # Without key -> 401
            resp = tc.post("/scan", json={"days": 7})
            assert resp.status_code == 401

            # With correct key -> passes auth
            ctx, _ = _mock_main_intel()
            with ctx, \
                 patch("agent.api.get_latest_report", return_value=SAMPLE_REPORT), \
                 patch("agent.api.get_latest_report_path", return_value=Path("/tmp/r.json")):
                resp = tc.post(
                    "/scan",
                    json={"days": 7},
                    headers={"X-API-Key": "runtime-key"},
                )
            assert resp.status_code == 200
        finally:
            api_module.API_KEY = original


# ===================================================================
# 15. Edge cases & combined filters
# ===================================================================


class TestEdgeCases:
    def test_datasets_category_case_insensitive(self, client):
        """Category filter is case-insensitive (SFT_INSTRUCTION matches sft_instruction)."""
        with patch("agent.api.get_latest_report", return_value=SAMPLE_REPORT):
            resp = client.get("/datasets?category=SFT_INSTRUCTION")
        assert resp.json()["count"] == 1

    def test_datasets_combined_filters(self, client):
        """Multiple filters are applied together (category + min_downloads)."""
        with patch("agent.api.get_latest_report", return_value=SAMPLE_REPORT):
            resp = client.get("/datasets?category=sft_instruction&min_downloads=10000")
        assert resp.json()["count"] == 0

    def test_blogs_source_case_insensitive(self, client):
        """Blog source filter is case-insensitive."""
        with patch("agent.api.get_latest_report", return_value=SAMPLE_REPORT):
            resp = client.get("/blogs?source=openai")
        assert resp.json()["count"] == 2

    def test_github_limit_validation_zero(self, client):
        """GET /github?limit=0 returns 422."""
        resp = client.get("/github?limit=0")
        assert resp.status_code == 422

    def test_papers_limit_validation_over_max(self, client):
        """GET /papers?limit=501 returns 422."""
        resp = client.get("/papers?limit=501")
        assert resp.status_code == 422

    def test_nonexistent_endpoint_returns_404(self, client):
        """Unknown paths return 404."""
        resp = client.get("/nonexistent")
        assert resp.status_code == 404

    def test_scan_boundary_days_1(self, client):
        """POST /scan with days=1 (lower boundary) succeeds."""
        ctx, _ = _mock_main_intel()
        with ctx, \
             patch("agent.api.get_latest_report", return_value=SAMPLE_REPORT), \
             patch("agent.api.get_latest_report_path", return_value=Path("/tmp/r.json")):
            resp = client.post("/scan", json={"days": 1})
        assert resp.status_code == 200

    def test_scan_boundary_days_90(self, client):
        """POST /scan with days=90 (upper boundary) succeeds."""
        ctx, _ = _mock_main_intel()
        with ctx, \
             patch("agent.api.get_latest_report", return_value=SAMPLE_REPORT), \
             patch("agent.api.get_latest_report_path", return_value=Path("/tmp/r.json")):
            resp = client.post("/scan", json={"days": 90})
        assert resp.status_code == 200

    def test_datasets_empty_report_no_key(self, client):
        """GET /datasets returns 0 results when report has no 'datasets' key."""
        empty_report = {"generated_at": "2025-05-01T00:00:00Z", "summary": {}}
        with patch("agent.api.get_latest_report", return_value=empty_report):
            resp = client.get("/datasets")
        assert resp.status_code == 200
        assert resp.json()["count"] == 0

    def test_github_empty_repos(self, client):
        """GET /github returns 0 results when report has no github_activity."""
        report = {**SAMPLE_REPORT, "github_activity": []}
        with patch("agent.api.get_latest_report", return_value=report):
            resp = client.get("/github?relevance=all")
        assert resp.status_code == 200
        assert resp.json()["count"] == 0

    def test_papers_combined_source_and_dataset_only(self, client):
        """Combining source and dataset_only filters together."""
        with patch("agent.api.get_latest_report", return_value=SAMPLE_REPORT):
            resp = client.get("/papers?source=arxiv&dataset_only=true")
        body = resp.json()
        assert body["count"] == 1
        assert body["papers"][0]["title"] == "Paper A"
        assert body["papers"][0]["source"] == "arxiv"
