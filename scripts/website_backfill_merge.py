#!/usr/bin/env python3
"""Resolve stacked weekly-issue JSON conflicts, then check Saturday's window."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path


def show(stage: str, path: str):
    try:
        raw = subprocess.check_output(["git", "show", f":{stage}:{path}"])
    except subprocess.CalledProcessError:
        return None
    return json.loads(raw)


def week_number(week: str) -> int:
    text = str(week)
    digits = text[1:] if text[:1] in "Ww" else text
    try:
        return int(digits)
    except ValueError:
        return 10**9


def merge_issues(ours, theirs):
    merged = {}
    for item in list(theirs) + list(ours):
        if isinstance(item, dict) and item.get("week"):
            merged[item["week"]] = item
    return [merged[week] for week in sorted(merged, key=week_number)]


def merge_value(ours, theirs, path: str):
    if isinstance(ours, list) and isinstance(theirs, list):
        if any(isinstance(item, dict) and item.get("week") for item in ours + theirs):
            return merge_issues(ours, theirs)
        seen = set()
        out = []
        for item in list(theirs) + list(ours):
            blob = json.dumps(item, sort_keys=True, ensure_ascii=False)
            if blob not in seen:
                seen.add(blob)
                out.append(item)
        return out
    if isinstance(ours, dict) and isinstance(theirs, dict):
        for key in ("issues", "items"):
            if isinstance(ours.get(key), list) and isinstance(theirs.get(key), list):
                combined = dict(theirs)
                combined.update(ours)
                combined[key] = merge_issues(ours[key], theirs[key])
                return combined
        combined = dict(theirs)
        combined.update(ours)
        return combined
    raise SystemExit(f"cannot auto-merge {path}")


def resolve() -> None:
    conflicts = subprocess.check_output(
        ["git", "diff", "--name-only", "--diff-filter=U"], text=True
    ).split()
    print("conflicts:", conflicts)
    for path in conflicts:
        ours = show("2", path)
        theirs = show("3", path)
        if ours is None or theirs is None:
            raise SystemExit(f"unhandled non-json conflict: {path}")
        merged = merge_value(ours, theirs, path)
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(merged, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        subprocess.check_call(["git", "add", "--", path])
    leftover = subprocess.check_output(
        ["git", "diff", "--name-only", "--diff-filter=U"], text=True
    ).split()
    if leftover:
        raise SystemExit(f"still conflicted: {leftover}")


def load_issues() -> list:
    payload = json.loads(Path("insights/.issues.json").read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("issues") or payload.get("items") or []
    return payload


def verify() -> None:
    from weekly_window import WeeklyWindow, find_overlaps

    issues = load_issues()
    print("ISSUE_COUNT", len(issues))
    for item in issues[-6:]:
        print(
            "ISSUE",
            item.get("week"),
            str(item.get("date_start", item.get("start", "")))[:10],
            str(item.get("date_end", item.get("end", "")))[:10],
        )
    by_week = {item.get("week"): item for item in issues}
    for week, start, end in (
        ("W33", "2026-10-10", "2026-10-17"),
        ("W34", "2026-10-17", "2026-10-24"),
        ("W35", "2026-10-24", "2026-10-31"),
    ):
        item = by_week.get(week)
        if not item:
            raise SystemExit(f"missing {week}")
        got = (
            str(item.get("date_start", item.get("start", "")))[:10],
            str(item.get("date_end", item.get("end", "")))[:10],
        )
        print("WINDOW", week, got[0], got[1])
        if got != (start, end):
            raise SystemExit(f"{week} window {got} != {(start, end)}")
    numbers = [week_number(item.get("week", "")) for item in issues if item.get("week")]
    nxt = max(numbers) + 1
    print("NEXT_WEEK", f"W{nxt}")
    if nxt != 36:
        raise SystemExit(f"expected next week W36, got W{nxt}")
    end = datetime(2026, 11, 7, 2, 0, 0)
    start = end - timedelta(days=7)
    candidate = WeeklyWindow(start.date(), end.date())
    conflicts = find_overlaps(candidate, issues)
    print("SATURDAY_WINDOW", candidate.as_dict(), "CONFLICTS", json.dumps(conflicts, ensure_ascii=False))
    if conflicts:
        raise SystemExit("Saturday window overlaps an existing issue")
    if candidate.as_dict() != {"start": "2026-10-31", "end": "2026-11-07"}:
        raise SystemExit(f"unexpected Saturday window {candidate.as_dict()}")
    print("SATURDAY_OK")


def main() -> None:
    command = sys.argv[1] if len(sys.argv) > 1 else ""
    if command == "resolve":
        resolve()
    elif command == "verify":
        verify()
    else:
        raise SystemExit("usage: website_backfill_merge.py resolve|verify")


if __name__ == "__main__":
    main()
