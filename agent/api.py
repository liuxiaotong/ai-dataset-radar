"""
AI Dataset Radar - HTTP API for AI Agents

A lightweight FastAPI server that exposes radar capabilities as REST endpoints.
Any AI agent can call these endpoints to access dataset intelligence.

Usage:
    uvicorn agent.api:app --host 0.0.0.0 --port 8080

Or run directly:
    python -m agent.api

Security:
    Set RADAR_API_KEY env var to enable API key authentication.
    Clients must pass X-API-Key header or ?api_key= query parameter.
    If RADAR_API_KEY is not set, authentication is disabled (open access).
"""

import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from fastapi import FastAPI, HTTPException, Query, Request, Security
    from fastapi.responses import JSONResponse, RedirectResponse
    from fastapi.staticfiles import StaticFiles
    from fastapi.security import APIKeyHeader, APIKeyQuery
    from pydantic import BaseModel, Field
    from starlette.middleware.base import BaseHTTPMiddleware
except ImportError:
    print("FastAPI not installed. Run: pip install fastapi uvicorn")
    sys.exit(1)


# ============================================================
# Security: API Key Authentication
# ============================================================

API_KEY = os.environ.get("RADAR_API_KEY", "")

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
api_key_query = APIKeyQuery(name="api_key", auto_error=False)


async def verify_api_key(
    header_key: Optional[str] = Security(api_key_header),
    query_key: Optional[str] = Security(api_key_query),
) -> Optional[str]:
    """Verify API key from header or query parameter."""
    if not API_KEY:
        return None  # Auth disabled
    key = header_key or query_key
    if not key or key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return key


# ============================================================
# Security: Rate Limiting Middleware
# ============================================================

# Per-IP request tracking: {ip: [(timestamp, ...),]}
_rate_limit_store: dict[str, list[float]] = defaultdict(list)
RATE_LIMIT_REQUESTS = int(os.environ.get("RADAR_RATE_LIMIT", "60"))
RATE_LIMIT_WINDOW = 60  # seconds


class RateLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        client_ip = request.client.host if request.client else "unknown"
        now = time.time()

        # Clean old entries
        _rate_limit_store[client_ip] = [
            t for t in _rate_limit_store[client_ip] if now - t < RATE_LIMIT_WINDOW
        ]

        if len(_rate_limit_store[client_ip]) >= RATE_LIMIT_REQUESTS:
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded. Try again later."},
                headers={"Retry-After": str(RATE_LIMIT_WINDOW)},
            )

        _rate_limit_store[client_ip].append(now)
        response = await call_next(request)
        return response


app = FastAPI(
    title="AI Dataset Radar API",
    description="REST API for AI agents to access dataset intelligence",
    version="1.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(RateLimitMiddleware)

if not API_KEY:
    import logging

    logging.getLogger("uvicorn").warning(
        "RADAR_API_KEY not set — API is open without authentication. "
        "Set RADAR_API_KEY env var to enable API key auth."
    )


# ============================================================
# Models
# ============================================================


class ScanRequest(BaseModel):
    days: int = Field(7, ge=1, le=90, description="Look-back period in days (1-90)")


class ScanResponse(BaseModel):
    success: bool
    report_path: Optional[str] = None
    summary: dict
    message: Optional[str] = None


class DatasetFilter(BaseModel):
    category: Optional[str] = None
    min_downloads: Optional[int] = None
    limit: int = 50


# ============================================================
# Helpers
# ============================================================


def get_reports_dir() -> Path:
    """Get the reports directory path."""
    return Path(__file__).parent.parent / "data" / "reports"


def get_latest_report() -> Optional[dict]:
    """Load the latest JSON report."""
    reports_dir = get_reports_dir()
    if not reports_dir.exists():
        return None

    json_files = sorted(reports_dir.glob("intel_report_*.json"), reverse=True)
    if not json_files:
        return None

    with open(json_files[0]) as f:
        return json.load(f)


def get_latest_report_path() -> Optional[Path]:
    """Get path to latest report."""
    reports_dir = get_reports_dir()
    if not reports_dir.exists():
        return None

    json_files = sorted(reports_dir.glob("intel_report_*.json"), reverse=True)
    return json_files[0] if json_files else None


# ============================================================
# Endpoints
# ============================================================


@app.get("/ui")
async def redirect_to_dashboard():
    """Redirect to the web dashboard."""
    return RedirectResponse(url="/dashboard")


@app.get("/health")
async def health():
    """Health check endpoint for monitoring and container orchestration."""
    report = get_latest_report()
    return {
        "status": "ok",
        "auth_enabled": bool(API_KEY),
        "has_report": report is not None,
        "report_generated_at": report.get("generated_at") if report else None,
    }


@app.get("/")
async def root():
    """API info and available endpoints."""
    return {
        "name": "AI Dataset Radar API",
        "version": "1.1.0",
        "description": "REST API for AI agents to access dataset intelligence",
        "endpoints": {
            "/dashboard": "GET - Web visualization dashboard",
            "/health": "GET - Health check",
            "/scan": "POST - Run a new intelligence scan",
            "/summary": "GET - Get latest report summary",
            "/datasets": "GET - List datasets with optional filters",
            "/github": "GET - Get GitHub repository activity",
            "/papers": "GET - Get recent papers",
            "/blogs": "GET - Get blog articles",
            "/config": "GET - Get monitoring configuration (secrets redacted)",
            "/schema": "GET - Get JSON schema for report format",
            "/tools": "GET - Get tool definitions for function calling",
        },
        "docs": "/docs",
    }


@app.post("/scan", response_model=ScanResponse, dependencies=[Security(verify_api_key)])
async def run_scan(request: ScanRequest):
    """
    Run a full intelligence scan.

    This collects datasets, repos, papers, and blog posts from 30+ AI organizations.
    Results are saved to data/reports/ and returned as summary.
    """
    try:
        # Import and run the scanner
        from main_intel import run_intel_scan

        scan_result = await run_intel_scan(days=request.days)

        report = get_latest_report()
        summary = report.get("summary", {}) if report else scan_result

        return ScanResponse(
            success=True,
            report_path=str(get_latest_report_path()),
            summary=summary,
        )
    except Exception as e:
        return ScanResponse(
            success=False,
            summary={},
            message=str(e),
        )


@app.get("/summary")
async def get_summary():
    """
    Get summary of the latest intelligence report.

    Fast way to check recent findings without running a new scan.
    """
    report = get_latest_report()
    if not report:
        raise HTTPException(status_code=404, detail="No report found. Run /scan first.")

    return {
        "generated_at": report.get("generated_at"),
        "period": report.get("period"),
        "summary": report.get("summary"),
    }


@app.get("/datasets")
async def list_datasets(
    category: Optional[str] = Query(
        None,
        description="Filter by category: sft, preference, synthetic, agent, multimodal, code, evaluation",
    ),
    min_downloads: Optional[int] = Query(None, ge=0, description="Minimum download count"),
    limit: int = Query(50, ge=1, le=500, description="Maximum results to return (1-500)"),
):
    """
    List datasets from the latest report with optional filters.

    Categories:
    - sft: Instruction-following datasets
    - preference: RLHF/DPO training data
    - synthetic: AI-generated datasets
    - agent: Tool use and agent training
    - multimodal: Image/audio/video
    - code: Programming datasets
    - evaluation: Benchmarks
    """
    report = get_latest_report()
    if not report:
        raise HTTPException(status_code=404, detail="No report found. Run /scan first.")

    datasets = report.get("datasets", [])

    # Apply filters
    if category:
        datasets = [d for d in datasets if category.lower() in d.get("category", "").lower()]

    if min_downloads:
        datasets = [d for d in datasets if d.get("downloads", 0) >= min_downloads]

    # Sort by downloads
    datasets = sorted(datasets, key=lambda x: x.get("downloads", 0), reverse=True)

    return {
        "count": len(datasets[:limit]),
        "filters": {"category": category, "min_downloads": min_downloads},
        "datasets": datasets[:limit],
    }


@app.get("/github")
async def list_github_repos(
    relevance: str = Query("high", description="Filter: high, low, or all"),
    limit: int = Query(50, ge=1, le=500, description="Maximum results (1-500)"),
):
    """
    Get GitHub repository activity from watched organizations.

    Relevance levels:
    - high: Repos with dataset-related keywords in name/description/topics
    - low: Other repos from watched orgs
    - all: Everything
    """
    report = get_latest_report()
    if not report:
        raise HTTPException(status_code=404, detail="No report found. Run /scan first.")

    # Flatten repos from github_activity[].repos_updated
    github_activity = report.get("github_activity", [])
    repos = []
    for org_data in github_activity:
        org_name = org_data.get("org", "")
        for repo in org_data.get("repos_updated", []):
            if "org" not in repo:
                repo["org"] = org_name
            repos.append(repo)

    if relevance != "all":
        repos = [r for r in repos if r.get("relevance") == relevance]

    # Sort by stars
    repos = sorted(repos, key=lambda x: x.get("stars", 0), reverse=True)

    return {
        "count": len(repos[:limit]),
        "filter": {"relevance": relevance},
        "orgs": len(github_activity),
        "repos": repos[:limit],
    }


@app.get("/papers")
async def list_papers(
    source: str = Query("all", description="Filter: arxiv, huggingface, or all"),
    dataset_only: bool = Query(False, description="Only papers introducing datasets"),
    limit: int = Query(50, ge=1, le=500, description="Maximum results (1-500)"),
):
    """
    Get recent papers from arXiv and HuggingFace Daily Papers.

    Use dataset_only=true to find papers that introduce new datasets.
    """
    report = get_latest_report()
    if not report:
        raise HTTPException(status_code=404, detail="No report found. Run /scan first.")

    papers = report.get("papers", [])

    if source != "all":
        papers = [p for p in papers if p.get("source") == source]

    if dataset_only:
        papers = [p for p in papers if p.get("is_dataset_paper")]

    return {
        "count": len(papers[:limit]),
        "filters": {"source": source, "dataset_only": dataset_only},
        "papers": papers[:limit],
    }


@app.get("/blogs")
async def list_blogs(
    source: Optional[str] = Query(None, description="Filter by blog name"),
    category: str = Query(
        "all", description="Filter: us_frontier, us_emerging, china, research, data_vendor, all"
    ),
    limit: int = Query(50, ge=1, le=500, description="Maximum articles (1-500)"),
):
    """
    Get blog articles from 17 monitored sources.

    Sources include: OpenAI, Anthropic, Google AI, DeepMind, Meta AI,
    Mistral, Scale AI, Qwen, Tencent, Zhipu, and more.
    """
    report = get_latest_report()
    if not report:
        raise HTTPException(status_code=404, detail="No report found. Run /scan first.")

    blog_posts = report.get("blog_posts", [])

    if source:
        blog_posts = [b for b in blog_posts if source.lower() in b.get("source", "").lower()]

    if category != "all":
        blog_posts = [b for b in blog_posts if b.get("category") == category]

    # Flatten and limit articles
    all_articles = []
    for blog in blog_posts:
        for article in blog.get("articles", [])[:limit]:
            all_articles.append(
                {
                    "source": blog.get("source"),
                    "category": blog.get("category"),
                    **article,
                }
            )

    return {
        "count": len(all_articles[:limit]),
        "sources": len(blog_posts),
        "articles": all_articles[:limit],
    }


def _redact_secrets(obj, sensitive_keys=("token", "api_key", "secret", "password", "credential")):
    """Recursively redact sensitive values from config."""
    if isinstance(obj, dict):
        return {
            k: ("***" if any(s in k.lower() for s in sensitive_keys) and v else _redact_secrets(v, sensitive_keys))
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [_redact_secrets(item, sensitive_keys) for item in obj]
    return obj


@app.get("/config")
async def get_config():
    """
    Get current monitoring configuration (sensitive values redacted).

    Shows watched organizations, blog sources, and classification keywords.
    """
    config_path = Path(__file__).parent.parent / "config.yaml"

    if not config_path.exists():
        raise HTTPException(status_code=404, detail="Config file not found")

    try:
        import yaml

        with open(config_path) as f:
            config = yaml.safe_load(f)
        return _redact_secrets(config)
    except ImportError:
        raise HTTPException(status_code=500, detail="YAML parser not available")


@app.get("/schema")
async def get_schema():
    """
    Get JSON schema for the report format.

    Use this to understand the structure of radar outputs.
    """
    schema_path = Path(__file__).parent / "schema.json"

    if not schema_path.exists():
        raise HTTPException(status_code=404, detail="Schema file not found")

    with open(schema_path) as f:
        return json.load(f)


@app.get("/tools")
async def get_tools():
    """
    Get tool definitions for function calling.

    Use these definitions with OpenAI/Anthropic function calling
    or LangChain tools.
    """
    tools_path = Path(__file__).parent / "tools.json"

    if not tools_path.exists():
        raise HTTPException(status_code=404, detail="Tools file not found")

    with open(tools_path) as f:
        return json.load(f)


# ============================================================
# Static Dashboard
# ============================================================

_static_dir = Path(__file__).parent / "static"
if _static_dir.is_dir():
    app.mount("/dashboard", StaticFiles(directory=str(_static_dir), html=True), name="dashboard")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    import threading
    import webbrowser

    import uvicorn

    port = int(os.environ.get("PORT", "8080"))
    threading.Timer(1.0, lambda: webbrowser.open(f"http://localhost:{port}/dashboard")).start()
    print(f"\n  Dashboard: http://localhost:{port}/dashboard\n")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")
