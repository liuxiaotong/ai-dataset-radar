"""Unit tests for src/insights_pr_watch.py.

Covers the two judgement questions the Feishu card is built from:
  - how long has this PR been open ("挂了多久")
  - why is it stuck ("卡在哪": missing required status / CI failure /
    conflict / change-requested / fallback "其他")

All fixtures are fake PR JSON shaped like `gh pr view --json ...` output;
no network access.
"""
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

from insights_pr_watch import (
    build_card,
    build_run_failure_card,
    determine_blockers,
    format_created_beijing,
    hours_open,
    is_stale,
    parse_week,
)

NOW = datetime(2026, 9, 26, 2, 0, 0, tzinfo=timezone.utc)


def make_pr(**overrides):
    pr = {
        "number": 135,
        "title": "前沿洞察 W29：NVIDIA数据栈领跑具身智能｜人类判断决定安全落地",
        "url": "https://github.com/liuxiaotong/knowlyr-website/pull/135",
        "createdAt": "2026-09-19T02:17:30Z",
        "updatedAt": "2026-09-19T02:18:18Z",
        "state": "OPEN",
        "isDraft": False,
        "mergeable": "MERGEABLE",
        "mergeStateStatus": "BEHIND",
        "reviewDecision": "",
        "statusCheckRollup": [
            {
                "name": "check",
                "workflowName": "CI",
                "status": "COMPLETED",
                "conclusion": "SUCCESS",
            },
            {
                "name": "control",
                "workflowName": "Merge Controller",
                "status": "COMPLETED",
                "conclusion": "SUCCESS",
            },
        ],
    }
    pr.update(overrides)
    return pr


# ── parse_week ──

def test_parse_week_extracts_code():
    assert parse_week("前沿洞察 W29：NVIDIA数据栈领跑具身智能") == "W29"


def test_parse_week_handles_ascii_colon():
    assert parse_week("前沿洞察 W31: something") == "W31"


def test_parse_week_returns_none_for_other_titles():
    assert parse_week("chore: bump deps") is None


# ── hours_open / is_stale ──

def test_hours_open_matches_pr_135_real_gap():
    pr = make_pr()
    # Created 2026-09-19T02:17:30Z, "now" is 2026-09-26T02:00:00Z -> ~six days, 23h43m short of 7 full days
    hours = hours_open(pr, now=NOW)
    assert 166 < hours < 168


def test_is_stale_true_when_older_than_threshold():
    pr = make_pr()
    assert is_stale(pr, min_hours=24, now=NOW) is True


def test_is_stale_false_when_fresh():
    pr = make_pr(createdAt="2026-09-26T01:00:00Z")
    assert is_stale(pr, min_hours=24, now=NOW) is False


def test_is_stale_false_for_draft():
    pr = make_pr(isDraft=True)
    assert is_stale(pr, min_hours=24, now=NOW) is False


def test_is_stale_false_when_closed():
    pr = make_pr(state="CLOSED")
    assert is_stale(pr, min_hours=24, now=NOW) is False


# ── determine_blockers ──

def test_missing_required_status_is_reported():
    # This is the actual W29 / PR #135 situation: no linrui-review context
    # ever showed up in the rollup at all.
    pr = make_pr()
    reasons = determine_blockers(pr)
    assert any("林锐" in r and "还没审" in r for r in reasons)


def test_required_status_failed_is_reported():
    pr = make_pr(
        statusCheckRollup=[
            {"name": "linrui-review", "status": "COMPLETED", "conclusion": "FAILURE"},
        ]
    )
    reasons = determine_blockers(pr)
    assert any("林锐" in r and "没通过" in r for r in reasons)


def test_required_status_pending_is_reported():
    pr = make_pr(
        statusCheckRollup=[
            {"name": "linrui-review", "status": "IN_PROGRESS", "conclusion": ""},
        ]
    )
    reasons = determine_blockers(pr)
    assert any("林锐" in r and "审查中" in r for r in reasons)


def test_required_status_success_is_not_reported_as_blocker():
    pr = make_pr(
        statusCheckRollup=[
            {"name": "linrui-review", "status": "COMPLETED", "conclusion": "SUCCESS"},
            {"name": "check", "workflowName": "CI", "status": "COMPLETED", "conclusion": "SUCCESS"},
        ]
    )
    reasons = determine_blockers(pr)
    assert not any("林锐" in r for r in reasons)


def test_ci_failure_is_reported():
    pr = make_pr(
        statusCheckRollup=[
            {"name": "linrui-review", "status": "COMPLETED", "conclusion": "SUCCESS"},
            {"name": "check", "workflowName": "CI", "status": "COMPLETED", "conclusion": "FAILURE"},
        ]
    )
    reasons = determine_blockers(pr)
    assert any("CI 检查没过" in r and "CI/check" in r for r in reasons)


def test_conflict_via_mergeable_is_reported():
    pr = make_pr(mergeable="CONFLICTING")
    reasons = determine_blockers(pr)
    assert any("冲突" in r for r in reasons)


def test_conflict_via_merge_state_dirty_is_reported():
    pr = make_pr(mergeStateStatus="DIRTY")
    reasons = determine_blockers(pr)
    assert any("冲突" in r for r in reasons)


def test_change_requested_is_reported():
    pr = make_pr(
        statusCheckRollup=[
            {"name": "linrui-review", "status": "COMPLETED", "conclusion": "SUCCESS"},
            {"name": "check", "workflowName": "CI", "status": "COMPLETED", "conclusion": "SUCCESS"},
        ],
        reviewDecision="CHANGES_REQUESTED",
    )
    reasons = determine_blockers(pr)
    assert any("request changes" in r for r in reasons)


def test_format_created_beijing_converts_utc_to_cst():
    # 2026-09-19T02:17:30Z is 2026-09-19 10:17:30 in Beijing time (UTC+8).
    assert format_created_beijing("2026-09-19T02:17:30Z") == "09-19 10:17 创建"


def test_fallback_reason_when_all_green():
    pr = make_pr(
        statusCheckRollup=[
            {"name": "linrui-review", "status": "COMPLETED", "conclusion": "SUCCESS"},
            {"name": "check", "workflowName": "CI", "status": "COMPLETED", "conclusion": "SUCCESS"},
        ],
        mergeable="MERGEABLE",
        mergeStateStatus="CLEAN",
        reviewDecision="",
    )
    reasons = determine_blockers(pr)
    assert any("其他" in r for r in reasons)


# ── build_card ──

def test_build_card_single_pr_header_and_button():
    pr = make_pr()
    reasons = determine_blockers(pr)
    card = build_card([pr], {pr["number"]: reasons}, now=NOW)
    assert card["header"]["title"]["content"] == "情报周刊 W29 还没发布"
    assert card["header"]["template"] == "red"  # ~167h >= 72h threshold

    action = next(e for e in card["elements"] if e["tag"] == "action")
    assert action["actions"][0]["url"] == pr["url"]


def test_build_card_orange_below_threshold():
    pr = make_pr(createdAt="2026-09-25T00:00:00Z")  # ~26h before NOW
    reasons = determine_blockers(pr)
    card = build_card([pr], {pr["number"]: reasons}, now=NOW)
    assert card["header"]["template"] == "orange"


def test_build_card_multiple_prs_lists_both_weeks():
    pr1 = make_pr()
    pr2 = make_pr(number=140, title="前沿洞察 W30：xyz", url="https://github.com/liuxiaotong/knowlyr-website/pull/140")
    blockers = {
        pr1["number"]: determine_blockers(pr1),
        pr2["number"]: determine_blockers(pr2),
    }
    card = build_card([pr1, pr2], blockers, now=NOW)
    assert "W29" in card["header"]["title"]["content"]
    assert "W30" in card["header"]["title"]["content"]
    actions = [e for e in card["elements"] if e["tag"] == "action"]
    assert len(actions) == 2


def test_build_card_dup_note_included():
    pr = make_pr()
    reasons = determine_blockers(pr)
    card = build_card([pr], {pr["number"]: reasons}, now=NOW, dup_notes={pr["number"]: "本周 2026-09-26 的新内容因此未生成"})
    div = next(e for e in card["elements"] if e["tag"] == "div")
    assert "本周 2026-09-26 的新内容因此未生成" in div["text"]["content"]


def test_build_card_requires_at_least_one_pr():
    with pytest.raises(ValueError):
        build_card([], {})


def test_build_run_failure_card_points_at_the_actions_run():
    run_url = "https://github.com/liuxiaotong/ai-dataset-radar/actions/runs/41158067179"
    card = build_run_failure_card(
        run_url=run_url,
        run_id="41158067179",
        workflow_name="Generate Weekly Insights",
    )
    assert card["header"]["title"]["content"] == "情报周刊生成失败"
    assert card["header"]["template"] == "red"
    div = next(e for e in card["elements"] if e["tag"] == "div")
    content = div["text"]["content"]
    assert "Generate Weekly Insights" in content
    assert "#41158067179" in content
    assert "`" not in content
    assert "\n- " not in content
    action = next(e for e in card["elements"] if e["tag"] == "action")
    assert action["actions"][0]["url"] == run_url
    assert action["actions"][0]["text"]["content"] == "打开失败日志"


def test_build_run_failure_card_omits_button_without_url():
    card = build_run_failure_card(run_url="", run_id="")
    assert all(e["tag"] != "action" for e in card["elements"])
    div = next(e for e in card["elements"] if e["tag"] == "div")
    assert "本次运行" in div["text"]["content"]


def test_build_card_content_has_no_backticks_or_dash_bullets():
    # Feishu's lark_md renderer showed literal "`" and "- " to Kai instead of
    # rendering them, so the card must avoid both: use plain text and a
    # "·" bullet instead.
    pr = make_pr()
    reasons = determine_blockers(pr)
    card = build_card([pr], {pr["number"]: reasons}, now=NOW)
    div = next(e for e in card["elements"] if e["tag"] == "div")
    content = div["text"]["content"]
    assert "`" not in content
    assert "\n- " not in content
    assert "· " in content
    # And the time is human Beijing time, not a raw UTC ISO stamp.
    assert "UTC" not in content
    assert "09-19 10:17" in content


def test_failure_cli_dry_run_prints_card_and_does_not_send():
    root = Path(__file__).resolve().parents[1]
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/insights_pr_watch_cli.py",
            "--dry-run",
            "failure",
            "--run-url",
            "https://github.com/liuxiaotong/ai-dataset-radar/actions/runs/41158067179",
            "--run-id",
            "41158067179",
            "--workflow-name",
            "Generate Weekly Insights",
        ],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert "情报周刊生成失败" in proc.stdout
    assert "41158067179" in proc.stdout
    assert "未发送" in proc.stdout
    assert "Authorization" not in proc.stdout
    assert "kly-proxy" not in proc.stdout
