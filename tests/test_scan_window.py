"""Historical scan windows: bounds, filtering, and source gap notes."""

from datetime import date

from output_formatter import DualOutputFormatter
from scan_window import (
    INCOMPLETE_SOURCES,
    ScanWindow,
    arxiv_submitted_clause,
    render_window_gaps,
    restrict_to_window,
    semantic_scholar_range,
)


WINDOW = ScanWindow(date(2026, 10, 10), date(2026, 10, 17))


def test_window_rejects_empty_span():
    try:
        ScanWindow(date(2026, 10, 17), date(2026, 10, 17))
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_lookback_reaches_start_from_later_today():
    assert WINDOW.lookback_days(date(2026, 11, 4)) == 26
    assert WINDOW.days == 7
    assert WINDOW.contains("2026-10-10")
    assert WINDOW.contains("2026-10-16T23:00:00Z")
    assert not WINDOW.contains("2026-10-17")
    assert not WINDOW.contains("2026-10-09")
    assert not WINDOW.contains("")


def test_restrict_keeps_only_in_window_items():
    filtered = restrict_to_window(
        WINDOW,
        lab_activity={
            "labs": {
                "frontier_labs": {
                    "OpenAI": {
                        "datasets": [
                            {"id": "in", "last_modified": "2026-10-12T00:00:00Z"},
                            {"id": "after", "last_modified": "2026-10-20T00:00:00Z"},
                            {"id": "undated"},
                        ],
                        "models": [],
                    }
                }
            }
        },
        vendor_activity={"vendors": {}},
        github_activity=[
            {
                "org": "openai",
                "repos_updated": [
                    {"name": "old", "updated_at": "2026-10-11"},
                    {"name": "new", "updated_at": "2026-11-01"},
                ],
            }
        ],
        blog_activity=[{"source": "blog", "articles": [{"title": "a", "date": "2026-10-15"}]}],
        x_activity={"accounts": [{"username": "a", "relevant_tweets": [{"date": "2026-10-01"}]}], "search_results": []},
        reddit_activity={"posts": [{"date": "2026-10-13"}], "metadata": {}},
        hn_activity={"stories": [{"date": "2026-10-14"}], "metadata": {}},
        kaggle_datasets=[{"id": "k", "last_modified": "2026-10-11"}],
        semantic_scholar_papers=[{"publication_date": "2026-10-12"}],
        gh_trending={"repos": [{"name": "live", "date": "2026-11-04"}], "metadata": {}},
        producthunt={"products": [{"title": "now", "date": "2026-11-01"}], "metadata": {}},
        papers=[{"title": "p", "created_at": "2026-10-16T01:00:00"}],
    )

    datasets = filtered["lab_activity"]["labs"]["frontier_labs"]["OpenAI"]["datasets"]
    assert [item["id"] for item in datasets] == ["in"]
    assert [repo["name"] for repo in filtered["github_activity"][0]["repos_updated"]] == ["old"]
    assert filtered["x_activity"]["accounts"] == []
    assert filtered["gh_trending"]["repos"] == []
    assert filtered["producthunt"]["products"] == []
    assert len(filtered["papers"]) == 1
    assert len(filtered["blog_activity"][0]["articles"]) == 1


def test_insight_call_backs_off_on_gateway_overload():
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import analyze_insights

    class Overloaded(Exception):
        def __init__(self):
            super().__init__("Server error '502 Bad Gateway'")
            self.response = type("Resp", (), {"status_code": 502})()

    assert analyze_insights._insight_retry_wait(Overloaded(), 0, 6) == 30
    assert analyze_insights._insight_retry_wait(Overloaded(), 1, 6) == 60
    assert analyze_insights._insight_retry_wait(Overloaded(), 2, 6) == 120
    assert analyze_insights._insight_retry_wait(Overloaded(), 5, 6) is None
    assert analyze_insights._insight_retry_wait(RuntimeError("parse error"), 0, 4) == 0
    assert analyze_insights._insight_retry_wait(RuntimeError("parse error"), 1, 4) is None


def test_anthropic_fallback_hides_codex_key(monkeypatch):
    import os
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import analyze_insights

    for name in (
        "INSIGHTS_OPENAI_API_KEY",
        "SENTINEL_CODEX_API_KEY",
        "ANTHROPIC_API_KEY",
        "INSIGHTS_MODEL",
        "INSIGHTS_API_KEY",
        "INSIGHTS_BASE_URL",
        "OPENAI_MODEL",
        "INSIGHTS_OPENAI_MODEL",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("SENTINEL_CODEX_API_KEY", "kly-proxy-test")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setattr(analyze_insights, "HAS_CREW", False)

    cfg = analyze_insights._anthropic_fallback_config()

    assert cfg["provider"] == "anthropic"
    assert cfg["api_key"] == "sk-ant-test"
    assert os.environ["SENTINEL_CODEX_API_KEY"] == "kly-proxy-test"


def test_backfill_report_filename_uses_window_end(tmp_path):
    formatter = DualOutputFormatter(output_dir=str(tmp_path))
    md_path, json_path = formatter.save_reports(
        "# week",
        {
            "period": {
                "days": 7,
                "start": "2026-10-10",
                "end": "2026-10-17",
                "backfill": True,
            },
            "datasets": [],
            "papers": [],
            "blog_posts": [],
            "x_activity": {},
            "labs_activity": {},
            "vendor_activity": {},
            "datasets_by_type": {},
            "trend_data": {},
            "github_activity": [],
        },
    )
    assert md_path.endswith("intel_report_2026-10-17.md")
    assert json_path.endswith("intel_report_2026-10-17.json")


def test_gap_note_names_incomplete_sources_and_historical_queries():
    text = render_window_gaps(WINDOW)
    assert "Backfilled retroactively" in text
    assert "2026-10-10" in text and "2026-10-17" in text
    for source in INCOMPLETE_SOURCES:
        assert source["id"] in text
    assert "submittedDate" in text
    assert arxiv_submitted_clause(WINDOW.start, WINDOW.end) == "submittedDate:[202610100000 TO 202610162359]"
    assert semantic_scholar_range(WINDOW.start, WINDOW.end) == "2026-10-10:2026-10-16"
