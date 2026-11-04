"""Half-open historical scan windows for weekly insight backfills.

A window is ``[start, end)`` on calendar dates. Reports are written to
``data/reports/<end>/``. Sources that cannot be queried for a past range are
still fetched with a lookback that reaches ``start``, then items are kept only
when their publish, created, or last-updated date falls inside the window.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Any, Iterable, Mapping, Optional


# Sources that cannot reconstruct a past week. The website backfill PR quotes this list.
INCOMPLETE_SOURCES: tuple[dict[str, str], ...] = (
    {
        "id": "github_trending",
        "summary": "Live ranking with no date-range API. Rows are stamped with the scan time and dropped.",
    },
    {
        "id": "product_hunt",
        "summary": "Current RSS feed. Older launches age out.",
    },
    {
        "id": "reddit",
        "summary": "Current /new listing. Older posts age out.",
    },
    {
        "id": "hf_papers",
        "summary": "Hugging Face Daily Papers is the current list, not a date-range archive.",
    },
    {
        "id": "x",
        "summary": "Recent search is about seven days. Timelines and RSSHub feeds keep a short page.",
    },
    {
        "id": "kaggle",
        "summary": "First page of a current keyword search, sorted by update time.",
    },
    {
        "id": "huggingface_orgs",
        "summary": (
            "Latest HF page by lastModified (100 datasets / 50 models). "
            "Later edits push the week off the page."
        ),
    },
    {
        "id": "github_orgs",
        "summary": "First 30 GitHub repos by updated_at. A later push hides that repo.",
    },
    {
        "id": "blogs",
        "summary": "Only posts still listed in the RSS feed or HTML page.",
    },
    {
        "id": "trends",
        "summary": "Growth stats come from the local radar database, not a rebuilt past week.",
    },
)


def parse_iso_date(value: str) -> date:
    text = str(value or "").strip()
    if not text:
        raise ValueError("empty date")
    return date.fromisoformat(text[:10])


class ScanWindow:
    """Inclusive start, exclusive end, both calendar dates."""

    def __init__(self, start: date, end: date):
        if end <= start:
            raise ValueError(f"window end {end.isoformat()} must be after start {start.isoformat()}")
        self.start = start
        self.end = end

    @classmethod
    def parse(cls, start: str, end: str) -> "ScanWindow":
        return cls(parse_iso_date(start), parse_iso_date(end))

    @property
    def days(self) -> int:
        return (self.end - self.start).days

    def lookback_days(self, today: Optional[date] = None) -> int:
        """Days from today back to start, so now-based cutoffs still reach the window."""
        today = today or datetime.now(timezone.utc).date()
        return max(self.days, (today - self.start).days + 1)

    def contains(self, value: Any) -> bool:
        parsed = coerce_date(value)
        if parsed is None:
            return False
        return self.start <= parsed < self.end

    def as_period(self) -> dict[str, Any]:
        return {
            "days": self.days,
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "backfill": True,
        }


def coerce_date(value: Any) -> Optional[date]:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value).strip()
    if not text or text.startswith("http"):
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text).date()
    except ValueError:
        pass
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        return None


def first_date(item: Mapping[str, Any], keys: Iterable[str]) -> Optional[date]:
    for key in keys:
        if item.get(key):
            parsed = coerce_date(item.get(key))
            if parsed is not None:
                return parsed
    return None


def keep_item(item: Mapping[str, Any], window: ScanWindow, keys: Iterable[str]) -> bool:
    parsed = first_date(item, keys)
    return parsed is not None and window.start <= parsed < window.end


_ACTIVITY_KEYS = ("last_modified", "lastModified", "updated_at", "pushed_at", "created_at", "createdAt")
_PUBLISHED_KEYS = ("published_at", "publication_date", "published", "created_at", "date")
_POST_KEYS = ("date", "created_at", "published_at", "created_utc")


def _filter_list(items: list, window: ScanWindow, keys: Iterable[str]) -> list:
    return [item for item in items or [] if isinstance(item, Mapping) and keep_item(item, window, keys)]


def _filter_org_tree(tree: dict, window: ScanWindow, keys: Iterable[str], fields: tuple[str, ...]) -> dict:
    filtered: dict = {}
    for group, orgs in (tree or {}).items():
        if not isinstance(orgs, dict):
            filtered[group] = orgs
            continue
        kept_orgs = {}
        for name, data in orgs.items():
            if not isinstance(data, dict):
                continue
            copied = dict(data)
            has_items = False
            for field in fields:
                if isinstance(copied.get(field), list):
                    copied[field] = _filter_list(copied[field], window, keys)
                    if copied[field]:
                        has_items = True
            if has_items:
                kept_orgs[name] = copied
        filtered[group] = kept_orgs
    return filtered


def restrict_to_window(
    window: ScanWindow,
    *,
    lab_activity: dict,
    vendor_activity: dict,
    github_activity: list,
    blog_activity: list,
    x_activity: dict,
    reddit_activity: dict,
    hn_activity: dict,
    kaggle_datasets: list,
    semantic_scholar_papers: list,
    gh_trending: dict,
    producthunt: dict,
    papers: list,
) -> dict[str, Any]:
    """Drop items whose dates fall outside ``window``. Undated items are dropped."""
    labs = _filter_org_tree(
        (lab_activity or {}).get("labs") or {},
        window,
        _ACTIVITY_KEYS,
        ("datasets", "models"),
    )
    vendors = _filter_org_tree(
        (vendor_activity or {}).get("vendors") or {},
        window,
        _ACTIVITY_KEYS,
        ("datasets", "models", "repos"),
    )

    github_kept = []
    for org in github_activity or []:
        if not isinstance(org, dict):
            continue
        repos = _filter_list(org.get("repos_updated") or [], window, ("updated_at", "pushed_at"))
        if repos:
            copied = dict(org)
            copied["repos_updated"] = repos
            github_kept.append(copied)

    blogs_kept = []
    for source in blog_activity or []:
        if not isinstance(source, dict):
            continue
        articles = _filter_list(source.get("articles") or [], window, _POST_KEYS)
        if articles:
            copied = dict(source)
            copied["articles"] = articles
            blogs_kept.append(copied)

    x_accounts = []
    for account in (x_activity or {}).get("accounts") or []:
        if not isinstance(account, dict):
            continue
        tweets = _filter_list(account.get("relevant_tweets") or [], window, _POST_KEYS)
        if tweets:
            copied = dict(account)
            copied["relevant_tweets"] = tweets
            x_accounts.append(copied)
    x_search = _filter_list((x_activity or {}).get("search_results") or [], window, _POST_KEYS)

    return {
        "lab_activity": {"labs": labs},
        "vendor_activity": {"vendors": vendors},
        "github_activity": github_kept,
        "blog_activity": blogs_kept,
        "x_activity": {"accounts": x_accounts, "search_results": x_search},
        "reddit_activity": {
            "posts": _filter_list((reddit_activity or {}).get("posts") or [], window, _POST_KEYS),
            "metadata": (reddit_activity or {}).get("metadata") or {},
        },
        "hn_activity": {
            "stories": _filter_list((hn_activity or {}).get("stories") or [], window, _POST_KEYS),
            "metadata": (hn_activity or {}).get("metadata") or {},
        },
        "kaggle_datasets": _filter_list(kaggle_datasets, window, ("last_modified", "created_at")),
        "semantic_scholar_papers": _filter_list(
            semantic_scholar_papers, window, ("publication_date", "published_at", "created_at")
        ),
        "gh_trending": {
            "repos": _filter_list((gh_trending or {}).get("repos") or [], window, ("date",)),
            "metadata": (gh_trending or {}).get("metadata") or {},
        },
        "producthunt": {
            "products": _filter_list((producthunt or {}).get("products") or [], window, _POST_KEYS),
            "metadata": (producthunt or {}).get("metadata") or {},
        },
        "papers": _filter_list(papers, window, _PUBLISHED_KEYS),
    }


def render_window_gaps(window: ScanWindow) -> str:
    """Markdown note embedded in the website backfill PR."""
    lines = [
        "Backfilled retroactively for "
        f"{window.start.isoformat()} inclusive through {window.end.isoformat()} exclusive.",
        "This was not the live Saturday run.",
        "",
        "These sources are incomplete for a past window.",
        "Dated rows are kept only when that date falls inside the window.",
        "The filter does not recover rows the source no longer returns:",
        "",
    ]
    for source in INCOMPLETE_SOURCES:
        lines.append(f"- {source['id']}: {source['summary']}")
    lines.extend(
        [
            "",
            "These sources are queried with an explicit historical range and then filtered to the window:",
            "- arxiv: submittedDate range",
            "- semantic_scholar: publicationDateOrYear, capped per keyword",
            "- hacker_news: Algolia created_at_i bounds, 50 hits per keyword",
        ]
    )
    return "\n".join(lines) + "\n"


def arxiv_submitted_clause(start: date, end: date) -> str:
    """arXiv submittedDate clause. ``end`` is exclusive."""
    last = end - timedelta(days=1)
    return f"submittedDate:[{start:%Y%m%d}0000 TO {last:%Y%m%d}2359]"


def semantic_scholar_range(start: date, end: date) -> str:
    """Semantic Scholar publicationDateOrYear value. ``end`` is exclusive."""
    last = end - timedelta(days=1)
    return f"{start.isoformat()}:{last.isoformat()}"
