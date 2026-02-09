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

        # Prune empty IP keys to prevent unbounded memory growth
        if not _rate_limit_store[client_ip]:
            del _rate_limit_store[client_ip]
            # Still allow this request through (no history = not rate limited)
            _rate_limit_store[client_ip].append(now)
            response = await call_next(request)
            return response

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
    api_insights: bool = Field(False, description="Use LLM API to generate insights (default: off)")


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


def _find_all_report_jsons(reports_dir: Path) -> list:
    """Find all intel_report JSON files (date-subdir + flat layouts)."""
    files = list(reports_dir.glob("*/intel_report_*.json"))
    files += list(reports_dir.glob("intel_report_*.json"))
    return sorted(set(files), reverse=True)


def get_latest_report() -> Optional[dict]:
    """Load the latest JSON report."""
    reports_dir = get_reports_dir()
    if not reports_dir.exists():
        return None

    json_files = _find_all_report_jsons(reports_dir)
    if not json_files:
        return None

    try:
        with open(json_files[0], encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def get_latest_report_path() -> Optional[Path]:
    """Get path to latest report."""
    reports_dir = get_reports_dir()
    if not reports_dir.exists():
        return None

    json_files = _find_all_report_jsons(reports_dir)
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

        scan_result = await run_intel_scan(days=request.days, api_insights=request.api_insights)

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


@app.get("/summary", dependencies=[Security(verify_api_key)])
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


@app.get("/datasets", dependencies=[Security(verify_api_key)])
async def list_datasets(
    category: Optional[str] = Query(
        None,
        description="Filter by category: sft_instruction, rlhf_preference, reward_model, "
        "synthetic, agent_tool, multimodal, multilingual, rl_environment, code, evaluation, other",
    ),
    min_downloads: Optional[int] = Query(None, ge=0, description="Minimum download count"),
    limit: int = Query(50, ge=1, le=500, description="Maximum results to return (1-500)"),
):
    """
    List datasets from the latest report with optional filters.

    Categories:
    - sft_instruction: Instruction-following / SFT datasets
    - rlhf_preference: RLHF / DPO / preference data
    - reward_model: Reward model training data
    - synthetic: AI-generated datasets
    - agent_tool: Agent / tool use / function calling
    - multimodal: Image / audio / video
    - multilingual: Multi-language datasets
    - rl_environment: RL / embodied / robotics
    - code: Programming datasets
    - evaluation: Benchmarks
    - other: Uncategorized
    """
    report = get_latest_report()
    if not report:
        raise HTTPException(status_code=404, detail="No report found. Run /scan first.")

    datasets = report.get("datasets", [])

    # Apply filters
    if category:
        datasets = [d for d in datasets if d.get("category", "").lower() == category.lower()]

    if min_downloads is not None:
        datasets = [d for d in datasets if d.get("downloads", 0) >= min_downloads]

    # Sort by downloads
    datasets = sorted(datasets, key=lambda x: x.get("downloads", 0), reverse=True)

    return {
        "count": len(datasets[:limit]),
        "filters": {"category": category, "min_downloads": min_downloads},
        "datasets": datasets[:limit],
    }


@app.get("/github", dependencies=[Security(verify_api_key)])
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
            repo_copy = dict(repo)
            if "org" not in repo_copy:
                repo_copy["org"] = org_name
            repos.append(repo_copy)

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


@app.get("/papers", dependencies=[Security(verify_api_key)])
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


@app.get("/blogs", dependencies=[Security(verify_api_key)])
async def list_blogs(
    source: Optional[str] = Query(None, description="Filter by blog name"),
    category: str = Query(
        "all", description="Filter: us_frontier, us_emerging, china, research, data_vendor, all"
    ),
    limit: int = Query(50, ge=1, le=500, description="Maximum articles (1-500)"),
):
    """
    Get blog articles from 62+ monitored sources.

    Categories: us_frontier, us_emerging, china, research, data_vendor.
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


@app.get("/reddit", dependencies=[Security(verify_api_key)])
async def list_reddit(
    subreddit: Optional[str] = Query(None, description="Filter by subreddit name"),
    min_score: Optional[int] = Query(None, description="Minimum post score"),
    limit: int = Query(50, ge=1, le=500, description="Maximum posts (1-500)"),
):
    """
    Get Reddit posts from monitored AI/ML subreddits.

    Monitors: r/MachineLearning, r/LocalLLaMA, r/LanguageTechnology, etc.
    Posts are filtered by AI dataset signal keywords and minimum score.
    """
    report = get_latest_report()
    if not report:
        raise HTTPException(status_code=404, detail="No report found. Run /scan first.")

    reddit = report.get("reddit_activity", {})
    posts = reddit.get("posts", [])

    if subreddit:
        posts = [p for p in posts if subreddit.lower() == p.get("subreddit", "").lower()]

    if min_score is not None:
        posts = [p for p in posts if p.get("score", 0) >= min_score]

    return {
        "count": len(posts[:limit]),
        "metadata": reddit.get("metadata", {}),
        "posts": posts[:limit],
    }


def _redact_secrets(obj, sensitive_keys=("token", "api_key", "secret", "password", "credential")):
    """Recursively redact sensitive values from config."""
    if isinstance(obj, dict):
        return {
            k: ("***" if any(s in k.lower() for s in sensitive_keys) else _redact_secrets(v, sensitive_keys))
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [_redact_secrets(item, sensitive_keys) for item in obj]
    return obj


@app.get("/config", dependencies=[Security(verify_api_key)])
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

        with open(config_path, encoding="utf-8") as f:
            config = yaml.safe_load(f)
        return _redact_secrets(config)
    except ImportError:
        raise HTTPException(status_code=500, detail="YAML parser not available")


@app.get("/schema", dependencies=[Security(verify_api_key)])
async def get_schema():
    """
    Get JSON schema for the report format.

    Use this to understand the structure of radar outputs.
    """
    schema_path = Path(__file__).parent / "schema.json"

    if not schema_path.exists():
        raise HTTPException(status_code=404, detail="Schema file not found")

    with open(schema_path, encoding="utf-8") as f:
        return json.load(f)


@app.get("/search", dependencies=[Security(verify_api_key)])
async def search(
    q: str = Query(..., min_length=1, description="Search keyword"),
    sources: Optional[str] = Query(None, description="Comma-separated: datasets,github,papers,blogs,reddit"),
    limit: int = Query(10, ge=1, le=100, description="Max results per source"),
):
    """
    Full-text search across all data sources.

    Searches datasets, GitHub repos, papers, blogs, and Reddit posts.
    """
    report = get_latest_report()
    if not report:
        raise HTTPException(status_code=404, detail="No report found. Run /scan first.")

    q_lower = q.lower()
    allowed_sources = set(sources.split(",")) if sources else {"datasets", "github", "papers", "blogs", "reddit"}

    results = {}

    if "datasets" in allowed_sources:
        datasets = report.get("datasets", [])
        matched = [
            d for d in datasets
            if q_lower in (d.get("id", "") + " " + d.get("description", "") + " " + d.get("author", "")).lower()
        ]
        results["datasets"] = matched[:limit]

    if "github" in allowed_sources:
        repos = []
        for org_data in report.get("github_activity", []):
            for repo in org_data.get("repos_updated", []):
                text = (repo.get("full_name", "") + " " + repo.get("description", "") + " " + " ".join(repo.get("topics", []))).lower()
                if q_lower in text:
                    repos.append(repo)
        results["github"] = repos[:limit]

    if "papers" in allowed_sources:
        papers = report.get("papers", [])
        matched = [
            p for p in papers
            if q_lower in (p.get("title", "") + " " + p.get("abstract", "") + " " + " ".join(p.get("authors", []))).lower()
        ]
        results["papers"] = matched[:limit]

    if "blogs" in allowed_sources:
        blog_results = []
        for blog in report.get("blog_posts", []):
            for article in blog.get("articles", []):
                text = (article.get("title", "") + " " + article.get("summary", "")).lower()
                if q_lower in text:
                    blog_results.append({**article, "source": blog.get("source")})
        results["blogs"] = blog_results[:limit]

    if "reddit" in allowed_sources:
        reddit = report.get("reddit_activity", {})
        posts = reddit.get("posts", [])
        matched = [
            p for p in posts
            if q_lower in (p.get("title", "") + " " + p.get("selftext", "")).lower()
        ]
        results["reddit"] = matched[:limit]

    total = sum(len(v) for v in results.values())
    return {"query": q, "total": total, "results": results}


@app.get("/trends", dependencies=[Security(verify_api_key)])
async def get_trends(
    limit: int = Query(30, ge=1, le=90, description="Number of recent reports to include"),
):
    """
    Get historical trend data across reports.

    Reads summary stats from each dated report to produce time-series data
    for charting dataset counts, repos, papers, etc. over time.
    """
    reports_dir = get_reports_dir()
    if not reports_dir.exists():
        raise HTTPException(status_code=404, detail="No reports directory found.")

    # Scan report directories sorted by date
    date_dirs = sorted(
        [d for d in reports_dir.iterdir() if d.is_dir() and len(d.name) == 10],
        key=lambda d: d.name,
        reverse=True,
    )[:limit]

    timeline = []
    for date_dir in reversed(date_dirs):
        json_files = sorted(date_dir.glob("intel_report_*.json"), reverse=True)
        if not json_files:
            continue
        try:
            with open(json_files[0], encoding="utf-8") as f:
                data = json.load(f)
            summary = data.get("summary", {})
            timeline.append({
                "date": date_dir.name,
                "datasets": summary.get("total_datasets", 0),
                "repos": summary.get("total_github_repos", 0),
                "papers": summary.get("total_papers", 0),
                "blogs": summary.get("total_blog_posts", 0),
                "reddit": summary.get("total_reddit_posts", 0),
            })
        except (json.JSONDecodeError, OSError):
            continue

    # Build per-org activity over time (from datasets_by_type)
    org_activity = {}
    for entry in timeline:
        date = entry["date"]
        date_dir = reports_dir / date
        json_files = sorted(date_dir.glob("intel_report_*.json"), reverse=True)
        if not json_files:
            continue
        try:
            with open(json_files[0], encoding="utf-8") as f:
                data = json.load(f)
            datasets_by_type = data.get("datasets_by_type", {})
            for category, ds_list in datasets_by_type.items():
                if category not in org_activity:
                    org_activity[category] = {}
                org_activity[category][date] = len(ds_list) if isinstance(ds_list, list) else 0
        except (json.JSONDecodeError, OSError):
            continue

    return {
        "count": len(timeline),
        "timeline": timeline,
        "datasets_by_category": org_activity,
    }


@app.get("/matrix", dependencies=[Security(verify_api_key)])
async def get_matrix():
    """
    Get competitor matrix — organizations × data types cross-reference.

    Shows dataset counts, repo/paper/blog activity per org with rankings.
    """
    report = get_latest_report()
    if not report:
        raise HTTPException(status_code=404, detail="No report found. Run /scan first.")
    matrix = report.get("competitor_matrix", {})
    if not matrix:
        return {"matrix": {}, "rankings": {}, "top_orgs": [], "org_details": {}}
    return matrix


@app.get("/lineage", dependencies=[Security(verify_api_key)])
async def get_lineage():
    """
    Get dataset lineage — derivation, version chains, and fork relationships.
    """
    report = get_latest_report()
    if not report:
        raise HTTPException(status_code=404, detail="No report found. Run /scan first.")
    lineage = report.get("dataset_lineage", {})
    if not lineage:
        return {"edges": [], "root_datasets": [], "version_chains": {}, "fork_trees": {}, "stats": {}}
    # Convert edge tuples to dicts for JSON
    edges = lineage.get("edges", [])
    if edges and isinstance(edges[0], (list, tuple)):
        edges = [{"child": e[0], "parent": e[1], "type": e[2]} for e in edges]
    return {**lineage, "edges": edges}


@app.get("/org-graph", dependencies=[Security(verify_api_key)])
async def get_org_graph():
    """
    Get organization relationship graph — nodes, edges, clusters, centrality.
    """
    report = get_latest_report()
    if not report:
        raise HTTPException(status_code=404, detail="No report found. Run /scan first.")
    graph = report.get("org_graph", {})
    if not graph:
        return {"nodes": [], "edges": [], "clusters": [], "centrality": {}}
    return graph


@app.get("/alerts", dependencies=[Security(verify_api_key)])
async def get_alerts(limit: int = 50):
    """
    Get recent alert history.

    Returns alerts from the most recent alert logs, sorted by timestamp descending.
    """
    reports_dir = get_reports_dir()
    if not reports_dir.exists():
        return {"alerts": [], "total": 0}

    all_alerts = []
    # Scan date-specific alert files
    for date_dir in sorted(reports_dir.iterdir(), reverse=True):
        if not date_dir.is_dir() or not date_dir.name[:4].isdigit():
            continue
        alert_file = date_dir / "alerts.json"
        if alert_file.exists():
            try:
                with open(alert_file, encoding="utf-8") as f:
                    alerts = json.load(f)
                all_alerts.extend(alerts)
            except (json.JSONDecodeError, ValueError):
                continue
        if len(all_alerts) >= limit:
            break

    # Sort by timestamp descending, limit
    all_alerts.sort(key=lambda a: a.get("timestamp", ""), reverse=True)
    all_alerts = all_alerts[:limit]

    return {"alerts": all_alerts, "total": len(all_alerts)}


@app.get("/tools", dependencies=[Security(verify_api_key)])
async def get_tools():
    """
    Get tool definitions for function calling.

    Use these definitions with OpenAI/Anthropic function calling
    or LangChain tools.
    """
    tools_path = Path(__file__).parent / "tools.json"

    if not tools_path.exists():
        raise HTTPException(status_code=404, detail="Tools file not found")

    with open(tools_path, encoding="utf-8") as f:
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
