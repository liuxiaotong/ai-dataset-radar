"""Detect stale "前沿洞察" (Frontier Insights) PRs on knowlyr-website and
build a Feishu interactive card that reports why each one is stuck.

Pure judgement logic (hours-open + blocker detection + card building) lives
here so it can be unit tested with fake PR JSON fixtures, without touching
the network. Network/CLI I/O (gh api, gh pr list, sending to Feishu) lives
in scripts/insights_pr_watch_cli.py and scripts/notify_feishu_card.sh.
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

WEEK_TITLE_RE = re.compile(r"^前沿洞察\s*(W\d+)\s*[：:]")

REQUIRED_CONTEXT = "linrui-review"

# Feishu private-DM audience only cares about people, not CI context slugs.
REQUIRED_CONTEXT_HUMAN_NAMES = {"linrui-review": "林锐"}

BEIJING_OFFSET = timedelta(hours=8)

# Rollup "conclusion" values that count as a completed, non-success outcome.
FAILING_CONCLUSIONS = {"FAILURE", "CANCELLED", "TIMED_OUT", "ACTION_REQUIRED", "STARTUP_FAILURE"}
PENDING_STATUSES = {"IN_PROGRESS", "QUEUED", "PENDING", "WAITING", "REQUESTED"}


def parse_week(title: str) -> Optional[str]:
    """Extract the "Wxx" week code from a PR title like "前沿洞察 W29：...".

    Returns None if the title does not match the expected pattern.
    """
    m = WEEK_TITLE_RE.match(title or "")
    return m.group(1) if m else None


def parse_iso(ts: str) -> datetime:
    """Parse a GitHub ISO-8601 timestamp (e.g. 2026-09-19T02:17:30Z)."""
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def hours_open(pr: Dict[str, Any], now: Optional[datetime] = None) -> float:
    """Hours since the PR's createdAt, using `now` (default: current UTC time)."""
    now = now or datetime.now(timezone.utc)
    created = parse_iso(pr["createdAt"])
    return (now - created).total_seconds() / 3600.0


def is_stale(pr: Dict[str, Any], min_hours: float = 24.0, now: Optional[datetime] = None) -> bool:
    """True if an open, non-draft PR has been open for at least `min_hours`."""
    if pr.get("state") != "OPEN":
        return False
    if pr.get("isDraft"):
        return False
    return hours_open(pr, now=now) >= min_hours


def find_check(rollup: List[Dict[str, Any]], name: str) -> Optional[Dict[str, Any]]:
    for check in rollup or []:
        if check.get("name") == name:
            return check
    return None


def determine_blockers(pr: Dict[str, Any], required_context: str = REQUIRED_CONTEXT) -> List[str]:
    """Return a list of human-readable (Chinese) reasons this PR is stuck.

    Looks at:
      - the required status check (missing / failed / pending)
      - other CI checks that failed
      - merge conflicts (mergeable == CONFLICTING, or mergeStateStatus == DIRTY)
      - reviewDecision == CHANGES_REQUESTED
    Falls back to a generic "其他" reason if none of the above apply but the
    PR is still open past the staleness threshold (caller decides that).
    """
    reasons: List[str] = []
    rollup = pr.get("statusCheckRollup") or []
    human = REQUIRED_CONTEXT_HUMAN_NAMES.get(required_context, required_context)

    required = find_check(rollup, required_context)
    if required is None:
        reasons.append(f"{human}还没审")
    else:
        conclusion = (required.get("conclusion") or "").upper()
        status = (required.get("status") or "").upper()
        if conclusion in FAILING_CONCLUSIONS:
            reasons.append(f"{human}审查没通过")
        elif status in PENDING_STATUSES or (status == "COMPLETED" and not conclusion):
            reasons.append(f"{human}审查中，还没出结果")
        # else: required check present and SUCCESS -> not a blocker

    for check in rollup:
        name = check.get("name")
        if name == required_context:
            continue
        conclusion = (check.get("conclusion") or "").upper()
        if conclusion in FAILING_CONCLUSIONS:
            workflow = check.get("workflowName") or ""
            label = f"{workflow}/{name}" if workflow else name
            reasons.append(f"CI 检查没过：{label}（{conclusion}）")

    mergeable = (pr.get("mergeable") or "").upper()
    merge_state = (pr.get("mergeStateStatus") or "").upper()
    if mergeable == "CONFLICTING" or merge_state == "DIRTY":
        reasons.append("有合并冲突，需要 rebase/解决冲突")

    if (pr.get("reviewDecision") or "").upper() == "CHANGES_REQUESTED":
        reasons.append("有 reviewer 提了修改意见（request changes），还没处理")

    if not reasons:
        reasons.append("其他：没发现失败检查/冲突/修改意见，但一直没合，可能是没指定 reviewer 或人工卡住了")

    return reasons


def format_hours(hours: float) -> str:
    if hours < 48:
        return f"{hours:.0f} 小时"
    days = hours / 24.0
    return f"{hours:.0f} 小时（约 {days:.1f} 天）"


def _severity_template(hours: float) -> str:
    return "red" if hours >= 72 else "orange"


def format_created_beijing(iso_ts: str) -> str:
    """Format createdAt in Beijing time the way people actually read it,
    e.g. "09-19 10:17 创建" instead of a raw UTC ISO timestamp."""
    dt = parse_iso(iso_ts) + BEIJING_OFFSET
    return dt.strftime("%m-%d %H:%M 创建")


def build_pr_block(
    pr: Dict[str, Any],
    reasons: List[str],
    now: Optional[datetime] = None,
    dup_note: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Build the div + hr + action card elements for a single stale PR."""
    week = parse_week(pr.get("title", "")) or "?"
    hours = hours_open(pr, now=now)
    created = format_created_beijing(pr["createdAt"])
    reason_lines = "\n".join(f"· {r}" for r in reasons)

    lines = [
        f"**{pr.get('title', '(untitled)')}**",
        f"已挂：{format_hours(hours)}（{created}）",
        "卡在：",
        reason_lines,
        "影响：上一期不合，下一期不会生成新 PR",
    ]
    if dup_note:
        lines.append(dup_note)

    content = "\n".join(lines)

    elements: List[Dict[str, Any]] = [
        {"tag": "div", "text": {"tag": "lark_md", "content": content}},
        {
            "tag": "action",
            "actions": [
                {
                    "tag": "button",
                    "text": {"tag": "plain_text", "content": f"打开 PR {week}"},
                    "type": "primary",
                    "url": pr.get("url", ""),
                }
            ],
        },
        {"tag": "hr"},
    ]
    return elements


def build_card(
    stale_prs: List[Dict[str, Any]],
    blockers_by_pr: Dict[int, List[str]],
    now: Optional[datetime] = None,
    dup_notes: Optional[Dict[int, str]] = None,
) -> Dict[str, Any]:
    """Build the full Feishu interactive card for one or more stale PRs.

    `blockers_by_pr` / `dup_notes` are keyed by PR number.
    """
    if not stale_prs:
        raise ValueError("build_card requires at least one stale PR")

    dup_notes = dup_notes or {}
    weeks = [parse_week(pr.get("title", "")) or "?" for pr in stale_prs]
    week_label = "、".join(dict.fromkeys(weeks))  # de-dup, keep order
    max_hours = max(hours_open(pr, now=now) for pr in stale_prs)

    elements: List[Dict[str, Any]] = []
    for pr in stale_prs:
        reasons = blockers_by_pr.get(pr["number"], ["未知"])
        dup_note = dup_notes.get(pr["number"])
        elements.extend(build_pr_block(pr, reasons, now=now, dup_note=dup_note))

    # Drop the trailing hr for a cleaner card end.
    if elements and elements[-1].get("tag") == "hr":
        elements.pop()

    card = {
        "config": {"wide_screen_mode": True},
        "header": {
            "title": {"tag": "plain_text", "content": f"情报周刊 {week_label} 还没发布"},
            "template": _severity_template(max_hours),
        },
        "elements": elements,
    }
    return card
