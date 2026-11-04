# Historical weekly windows

A live Saturday run scans the seven days ending now and writes `data/reports/<today>/`.

A backfill passes an explicit half-open calendar window. The scanner collects what it can for that range, drops items outside it, and writes `data/reports/<end>/`. `analyze_insights.py --date <end>` and the rest of `run_weekly.sh` then use that folder. The overlap check in `run_weekly.sh` still rejects a window that shares any day with an issue already recorded in `knowlyr-website` `insights/.issues.json`.

```bash
bash run_weekly.sh \
  --window-start=2026-10-10 \
  --window-end=2026-10-17 \
  --week-offset=0 \
  --no-auto-merge
```

`--window-end` is exclusive. `--week-offset` is added to the next week number taken from `.issues.json`, so several open backfill PRs do not all claim the same week. The scheduled workflow does not set a window, so Saturday stays a live seven-day scan.

Watermarks are ignored and not updated for a historical window, so a backfill does not move the cursor used by a later incremental scan.

## Sources that are incomplete for a past window

These feeds have no historical query. Rows are kept only when a publish, created, or last-updated date falls inside the window. That filter does not bring back items the source no longer returns.

| Source | Why a past week is incomplete |
| --- | --- |
| GitHub Trending | Live ranking. No date-range API. Each row is stamped with the scan time, so it is dropped from a past window. |
| Product Hunt | Current RSS feed. Older launches age out. |
| Reddit | Current `/new` listing. Older posts age out. |
| Hugging Face Daily Papers | Current daily list, not an archive. |
| X / Twitter | Recent search is about seven days. User timelines and RSSHub feeds keep a short recent page. |
| Kaggle | First page of a current keyword search, sorted by update time. |
| Hugging Face org datasets and models | Latest page by `lastModified` (100 datasets / 50 models). Later edits push the week off the page, and a newer `lastModified` hides earlier activity. |
| GitHub org repos | First 30 repositories by `updated_at`. A later push hides the repository from the earlier week. |
| Blogs | Posts still present in the RSS feed or HTML page. Feeds drop older entries. |
| Download-growth trends | Computed from the local radar database at scan time. A past window does not rebuild them. |

## Sources queried for the window

| Source | Query |
| --- | --- |
| arXiv | `submittedDate` from the inclusive start through the day before the exclusive end. |
| Semantic Scholar | `publicationDateOrYear` over that same inclusive range, capped per keyword. |
| Hacker News | Algolia `created_at_i` lower and upper bounds, 50 hits per keyword. |

Other sources are fetched with a lookback long enough to reach the window start, then filtered by date.
