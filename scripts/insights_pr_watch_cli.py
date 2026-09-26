#!/usr/bin/env python3
"""CLI wrapper around src/insights_pr_watch.py.

Two modes:

  watch   Lists every open knowlyr-website PR whose title starts with
          "前沿洞察 ", finds the ones open for >= --min-hours, and sends
          (or, with --dry-run, prints) one Feishu card covering all of them.
          Used by .github/workflows/insights-pr-watch.yml.

  dup     Reports on a single, already-known PR (the one run_weekly.sh just
          found instead of creating a new one) and appends a note about the
          new content that was skipped. Used from run_weekly.sh's
          publish_via_pr() dedup branch.

Sending is delegated to scripts/notify_feishu_card.sh so that all
tenant_access_token / HTTP concerns stay in one place. A send failure never
raises here in `dup` mode's caller (run_weekly.sh) beyond a non-fatal exit
code + warning, by design (see run_weekly.sh).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from insights_pr_watch import (  # noqa: E402
    REQUIRED_CONTEXT,
    build_card,
    determine_blockers,
    is_stale,
)

DEFAULT_SENDER = str(SCRIPT_DIR / "notify_feishu_card.sh")
DEFAULT_TITLE_PREFIX = "前沿洞察 "


def gh_json(args: list) -> object:
    result = subprocess.run(
        ["gh", *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def fetch_open_prs(repo: str) -> list:
    return gh_json(
        [
            "pr",
            "list",
            "--repo",
            repo,
            "--state",
            "open",
            "--limit",
            "100",
            "--json",
            "number,title,url,createdAt,updatedAt,state,isDraft,mergeable,"
            "mergeStateStatus,reviewDecision,statusCheckRollup",
        ]
    )


def fetch_pr(repo: str, number: int) -> dict:
    return gh_json(
        [
            "pr",
            "view",
            str(number),
            "--repo",
            repo,
            "--json",
            "number,title,url,createdAt,updatedAt,state,isDraft,mergeable,"
            "mergeStateStatus,reviewDecision,statusCheckRollup",
        ]
    )


def pr_number_from_url(url: str) -> int:
    return int(url.rstrip("/").rsplit("/", 1)[-1])


def send_card(card: dict, sender: str, dry_run: bool) -> int:
    args = [sender]
    if dry_run:
        args.append("--dry-run")
    proc = subprocess.run(args, input=json.dumps(card, ensure_ascii=False), text=True)
    return proc.returncode


def cmd_watch(args: argparse.Namespace) -> int:
    prs = fetch_open_prs(args.repo)
    title_prs = [p for p in prs if p.get("title", "").startswith(args.title_prefix)]
    stale = [p for p in title_prs if is_stale(p, min_hours=args.min_hours)]

    if not stale:
        print(f"没有超过 {args.min_hours:.0f} 小时的「{args.title_prefix}」PR，无需通知。")
        return 0

    blockers_by_pr = {p["number"]: determine_blockers(p, args.required_context) for p in stale}
    card = build_card(stale, blockers_by_pr)

    rc = send_card(card, args.sender, dry_run=args.dry_run)
    if rc != 0 and not args.dry_run:
        print(f"::warning::insights-pr-watch: 飞书卡片发送失败 (exit={rc})", file=sys.stderr)
    return rc


def cmd_dup(args: argparse.Namespace) -> int:
    if args.pr_url:
        number = pr_number_from_url(args.pr_url)
        pr = fetch_pr(args.repo, number)
    else:
        pr = fetch_pr(args.repo, args.pr_number)

    reasons = determine_blockers(pr, args.required_context)
    dup_note = f"本周 {args.date} 的新内容因此未生成"
    card = build_card([pr], {pr["number"]: reasons}, dup_notes={pr["number"]: dup_note})

    rc = send_card(card, args.sender, dry_run=args.dry_run)
    if rc != 0 and not args.dry_run:
        print(f"::warning::insights-pr-watch(dup): 飞书卡片发送失败 (exit={rc})", file=sys.stderr)
    return rc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=os.environ.get("WEBSITE_REPO", "liuxiaotong/knowlyr-website"))
    parser.add_argument("--title-prefix", default=DEFAULT_TITLE_PREFIX)
    parser.add_argument("--required-context", default=REQUIRED_CONTEXT)
    parser.add_argument("--sender", default=DEFAULT_SENDER, help="Path to the Feishu send script")
    parser.add_argument("--dry-run", action="store_true", help="Print card JSON; do not send")

    sub = parser.add_subparsers(dest="mode", required=True)

    watch = sub.add_parser("watch", help="Scan all open insights PRs for staleness")
    watch.add_argument("--min-hours", type=float, default=24.0)
    watch.set_defaults(func=cmd_watch)

    dup = sub.add_parser("dup", help="Report on one known PR (dedup path from run_weekly.sh)")
    group = dup.add_mutually_exclusive_group(required=True)
    group.add_argument("--pr-url")
    group.add_argument("--pr-number", type=int)
    dup.add_argument("--date", required=True, help="Today's date (YYYY-MM-DD), for the dup note")
    dup.set_defaults(func=cmd_dup)

    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
