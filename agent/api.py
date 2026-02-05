"""
AI Dataset Radar - HTTP API for AI Agents

A lightweight FastAPI server that exposes radar capabilities as REST endpoints.
Any AI agent can call these endpoints to access dataset intelligence.

Usage:
    uvicorn agent.api:app --host 0.0.0.0 --port 8080

Or run directly:
    python -m agent.api
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from fastapi import FastAPI, HTTPException, Query
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel
except ImportError:
    print("FastAPI not installed. Run: pip install fastapi uvicorn")
    sys.exit(1)


app = FastAPI(
    title="AI Dataset Radar API",
    description="REST API for AI agents to access dataset intelligence",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)


# ============================================================
# Models
# ============================================================

class ScanRequest(BaseModel):
    days: int = 7


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

@app.get("/")
async def root():
    """API info and available endpoints."""
    return {
        "name": "AI Dataset Radar API",
        "version": "1.0.0",
        "description": "REST API for AI agents to access dataset intelligence",
        "endpoints": {
            "/scan": "POST - Run a new intelligence scan",
            "/summary": "GET - Get latest report summary",
            "/datasets": "GET - List datasets with optional filters",
            "/github": "GET - Get GitHub repository activity",
            "/papers": "GET - Get recent papers",
            "/blogs": "GET - Get blog articles",
            "/config": "GET - Get monitoring configuration",
            "/schema": "GET - Get JSON schema for report format",
            "/tools": "GET - Get tool definitions for function calling",
        },
        "docs": "/docs",
    }


@app.post("/scan", response_model=ScanResponse)
async def run_scan(request: ScanRequest):
    """
    Run a full intelligence scan.

    This collects datasets, repos, papers, and blog posts from 30+ AI organizations.
    Results are saved to data/reports/ and returned as summary.
    """
    try:
        # Import and run the scanner
        from main_intel import run_intel_scan

        result = run_intel_scan(days=request.days)

        report = get_latest_report()
        summary = report.get("summary", {}) if report else {}

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
    category: Optional[str] = Query(None, description="Filter by category: sft, preference, synthetic, agent, multimodal, code, evaluation"),
    min_downloads: Optional[int] = Query(None, description="Minimum download count"),
    limit: int = Query(50, description="Maximum results to return"),
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
    limit: int = Query(50, description="Maximum results"),
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

    repos = report.get("github_repos", [])

    if relevance != "all":
        repos = [r for r in repos if r.get("relevance") == relevance]

    # Sort by stars
    repos = sorted(repos, key=lambda x: x.get("stars", 0), reverse=True)

    return {
        "count": len(repos[:limit]),
        "filter": {"relevance": relevance},
        "repos": repos[:limit],
    }


@app.get("/papers")
async def list_papers(
    source: str = Query("all", description="Filter: arxiv, huggingface, or all"),
    dataset_only: bool = Query(False, description="Only papers introducing datasets"),
    limit: int = Query(50, description="Maximum results"),
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
    category: str = Query("all", description="Filter: us_frontier, us_emerging, china, research, data_vendor, all"),
    limit: int = Query(50, description="Maximum articles"),
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
            all_articles.append({
                "source": blog.get("source"),
                "category": blog.get("category"),
                **article,
            })

    return {
        "count": len(all_articles[:limit]),
        "sources": len(blog_posts),
        "articles": all_articles[:limit],
    }


@app.get("/config")
async def get_config():
    """
    Get current monitoring configuration.

    Shows watched organizations, blog sources, and classification keywords.
    """
    config_path = Path(__file__).parent.parent / "config.yaml"

    if not config_path.exists():
        raise HTTPException(status_code=404, detail="Config file not found")

    try:
        import yaml
        with open(config_path) as f:
            config = yaml.safe_load(f)
        return config
    except ImportError:
        # Return raw text if yaml not available
        return {"raw": config_path.read_text()}


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
# Main
# ============================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
