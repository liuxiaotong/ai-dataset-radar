"""Daily change tracker — compares consecutive scan reports."""

import json
import logging
from pathlib import Path

logger = logging.getLogger("change_tracker")


def generate_change_summary(
    reports_base_dir: Path,
    current_date: str,
) -> str | None:
    """Compare today's report with the previous and write a change summary.

    Args:
        reports_base_dir: Path to data/reports/ directory.
        current_date: Today's date string (YYYY-MM-DD).

    Returns:
        Path string to the generated changes file, or None if no previous report.
    """
    reports_base_dir = Path(reports_base_dir)
    prev_path = find_previous_report(reports_base_dir, current_date)
    if prev_path is None:
        return None

    curr_dir = reports_base_dir / current_date
    curr_candidates = sorted(curr_dir.glob("intel_report_*.json"))
    # Pick the main report (not _changes or other suffixes)
    curr_path = None
    for c in curr_candidates:
        if "_changes" not in c.name and "_anomalies" not in c.name:
            curr_path = c
            break
    if curr_path is None:
        return None

    with open(prev_path, encoding="utf-8") as f:
        prev_data = json.load(f)
    with open(curr_path, encoding="utf-8") as f:
        curr_data = json.load(f)

    prev_date = prev_path.parent.name
    changes = compare_reports(prev_data, curr_data, prev_date, current_date)
    markdown = format_changes_markdown(changes)

    out_path = curr_dir / f"intel_report_{current_date}_changes.md"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(markdown)

    return str(out_path)


def find_previous_report(reports_base_dir: Path, current_date: str) -> Path | None:
    """Find the most recent intel_report JSON before the given date."""
    reports_base_dir = Path(reports_base_dir)
    if not reports_base_dir.exists():
        return None

    date_dirs = sorted(
        [
            d
            for d in reports_base_dir.iterdir()
            if d.is_dir() and d.name != current_date and d.name[:4].isdigit()
        ],
        reverse=True,
    )
    for date_dir in date_dirs:
        for jp in sorted(date_dir.glob("intel_report_*.json"), reverse=True):
            if "_changes" not in jp.name and "_anomalies" not in jp.name:
                return jp
    return None


def compare_reports(
    prev: dict, curr: dict, prev_date: str, curr_date: str
) -> dict:
    """Compare two JSON reports and return structured change data."""
    changes = {"prev_date": prev_date, "curr_date": curr_date}

    # 1. Summary deltas
    prev_summary = prev.get("summary", {})
    curr_summary = curr.get("summary", {})
    summary_fields = [
        ("total_datasets", "Datasets"),
        ("total_github_repos", "GitHub Repos"),
        ("total_github_repos_high_relevance", "High Relevance Repos"),
        ("total_papers", "Papers"),
        ("total_blog_posts", "Blog Posts"),
        ("total_x_tweets", "X Tweets"),
    ]
    deltas = []
    for field, label in summary_fields:
        pv = prev_summary.get(field, 0) or 0
        cv = curr_summary.get(field, 0) or 0
        deltas.append({"label": label, "prev": pv, "curr": cv, "delta": cv - pv})
    changes["summary_deltas"] = deltas

    # 2. Dataset changes
    changes.update(_compare_datasets(prev.get("datasets", []), curr.get("datasets", [])))

    # 2.5 Category distribution changes
    changes["category_changes"] = _compare_categories(
        prev.get("datasets_by_type", {}), curr.get("datasets_by_type", {})
    )

    # 3. GitHub repo changes
    changes.update(_compare_github(prev.get("github_activity", []), curr.get("github_activity", [])))

    # 4. Paper changes
    changes["new_papers"] = _compare_papers(prev.get("papers", []), curr.get("papers", []))

    # 5. Blog & X counts
    changes["new_blog_count"] = _count_blog_articles(curr.get("blog_posts", [])) - _count_blog_articles(prev.get("blog_posts", []))
    changes["new_tweet_count"] = _count_tweets(curr.get("x_activity", {})) - _count_tweets(prev.get("x_activity", {}))

    return changes


def format_changes_markdown(changes: dict) -> str:
    """Format changes dict into Markdown."""
    lines = []
    lines.append(f"# Daily Changes: {changes['curr_date']}")
    lines.append("")
    lines.append(f"> Compared against: {changes['prev_date']}")
    lines.append("")

    # Summary table
    lines.append("## Summary")
    lines.append("")
    lines.append("| Metric | Previous | Current | Change |")
    lines.append("|--------|----------|---------|--------|")
    for d in changes["summary_deltas"]:
        delta_str = f"+{d['delta']}" if d["delta"] > 0 else str(d["delta"]) if d["delta"] != 0 else "--"
        lines.append(f"| {d['label']} | {d['prev']:,} | {d['curr']:,} | {delta_str} |")
    lines.append("")

    # Category distribution
    cat_changes = changes.get("category_changes", [])
    if cat_changes:
        lines.append("## Category Distribution Changes")
        lines.append("")
        lines.append("| Category | Previous | Current | Change |")
        lines.append("|----------|----------|---------|--------|")
        for c in cat_changes:
            delta_str = f"+{c['delta']}" if c["delta"] > 0 else str(c["delta"])
            lines.append(
                f"| {c['category']} | {c['prev']} | {c['curr']} | {delta_str} |"
            )
        lines.append("")

    # New datasets
    new_ds = changes.get("new_datasets", [])
    if new_ds:
        lines.append(f"## New Datasets ({len(new_ds)})")
        lines.append("")
        lines.append("| Dataset | Category | Downloads | Likes |")
        lines.append("|---------|----------|-----------|-------|")
        for ds in new_ds:
            lines.append(
                f"| {ds.get('id', '')} | {ds.get('category', '')} "
                f"| {ds.get('downloads', 0):,} | {ds.get('likes', 0):,} |"
            )
        lines.append("")

    # Removed datasets
    removed_ds = changes.get("removed_datasets", [])
    if removed_ds:
        lines.append(f"## Removed Datasets ({len(removed_ds)})")
        lines.append("")
        lines.append("| Dataset | Category |")
        lines.append("|---------|----------|")
        for ds in removed_ds:
            lines.append(f"| {ds.get('id', '')} | {ds.get('category', '')} |")
        lines.append("")

    # Download movers
    movers = changes.get("download_movers", [])
    if movers:
        lines.append(f"## Download Movers (Top {len(movers)})")
        lines.append("")
        lines.append("| Dataset | Previous | Current | Change | Growth |")
        lines.append("|---------|----------|---------|--------|--------|")
        for m in movers:
            growth = f"{m['growth_pct']:+.1f}%" if m["growth_pct"] != float("inf") else "new"
            delta_str = f"+{m['delta']:,}" if m["delta"] > 0 else f"{m['delta']:,}"
            lines.append(
                f"| {m['id']} | {m['prev_downloads']:,} | {m['curr_downloads']:,} "
                f"| {delta_str} | {growth} |"
            )
        lines.append("")

    # New GitHub repos
    new_repos = changes.get("new_repos", [])
    if new_repos:
        lines.append(f"## New GitHub Repos ({len(new_repos)})")
        lines.append("")
        lines.append("| Repo | Stars | Relevance |")
        lines.append("|------|-------|-----------|")
        for r in new_repos[:15]:
            lines.append(
                f"| {r.get('full_name', '')} | {r.get('stars', 0):,} "
                f"| {r.get('relevance', '')} |"
            )
        if len(new_repos) > 15:
            lines.append(f"| ... and {len(new_repos) - 15} more | | |")
        lines.append("")

    # Gone repos
    gone_repos = changes.get("gone_repos", [])
    if gone_repos:
        lines.append(f"## Removed GitHub Repos ({len(gone_repos)})")
        lines.append("")
        lines.append("| Repo | Stars |")
        lines.append("|------|-------|")
        for r in gone_repos[:10]:
            lines.append(f"| {r.get('full_name', '')} | {r.get('stars', 0):,} |")
        lines.append("")

    # Stars movers
    star_movers = changes.get("stars_movers", [])
    if star_movers:
        lines.append(f"## Stars Movers (Top {len(star_movers)})")
        lines.append("")
        lines.append("| Repo | Previous | Current | Change |")
        lines.append("|------|----------|---------|--------|")
        for m in star_movers:
            delta_str = f"+{m['delta']:,}" if m["delta"] > 0 else f"{m['delta']:,}"
            lines.append(
                f"| {m['full_name']} | {m['prev_stars']:,} | {m['curr_stars']:,} "
                f"| {delta_str} |"
            )
        lines.append("")

    # New papers
    new_papers = changes.get("new_papers", [])
    if new_papers:
        lines.append(f"## New Papers ({len(new_papers)})")
        lines.append("")
        lines.append("| Title | Source |")
        lines.append("|-------|--------|")
        for p in new_papers:
            title = p.get("title", "")[:80]
            lines.append(f"| {title} | {p.get('source', '')} |")
        lines.append("")

    # Blog & X
    blog_delta = changes.get("new_blog_count", 0)
    tweet_delta = changes.get("new_tweet_count", 0)
    if blog_delta != 0 or tweet_delta != 0:
        lines.append("## Blog & X Activity")
        lines.append("")
        if blog_delta != 0:
            sign = "+" if blog_delta > 0 else ""
            lines.append(f"- **Blog articles**: {sign}{blog_delta}")
        if tweet_delta != 0:
            sign = "+" if tweet_delta > 0 else ""
            lines.append(f"- **X tweets**: {sign}{tweet_delta}")
        lines.append("")

    lines.append("---")
    lines.append("*Auto-generated by AI Dataset Radar change tracker*")
    lines.append("")

    return "\n".join(lines)


# --- Internal helpers ---


def _compare_categories(prev_by_type: dict, curr_by_type: dict) -> list[dict]:
    """Compare dataset category distribution."""
    all_cats = sorted(set(prev_by_type.keys()) | set(curr_by_type.keys()))
    changes = []
    for cat in all_cats:
        prev_list = prev_by_type.get(cat, [])
        curr_list = curr_by_type.get(cat, [])
        prev_count = len(prev_list) if isinstance(prev_list, list) else 0
        curr_count = len(curr_list) if isinstance(curr_list, list) else 0
        if prev_count != curr_count:
            changes.append({
                "category": cat,
                "prev": prev_count,
                "curr": curr_count,
                "delta": curr_count - prev_count,
            })
    return changes


def _compare_datasets(prev_datasets: list, curr_datasets: list) -> dict:
    prev_by_id = {ds.get("id", ds.get("name", "")): ds for ds in prev_datasets}
    curr_by_id = {ds.get("id", ds.get("name", "")): ds for ds in curr_datasets}

    prev_ids = set(prev_by_id.keys())
    curr_ids = set(curr_by_id.keys())

    new_datasets = [curr_by_id[did] for did in sorted(curr_ids - prev_ids)]
    removed_datasets = [prev_by_id[did] for did in sorted(prev_ids - curr_ids)]

    movers = []
    for did in prev_ids & curr_ids:
        prev_dl = prev_by_id[did].get("downloads", 0) or 0
        curr_dl = curr_by_id[did].get("downloads", 0) or 0
        delta = curr_dl - prev_dl
        if delta != 0:
            growth = (delta / prev_dl * 100) if prev_dl > 0 else float("inf")
            movers.append({
                "id": did,
                "prev_downloads": prev_dl,
                "curr_downloads": curr_dl,
                "delta": delta,
                "growth_pct": growth,
            })
    movers.sort(key=lambda x: abs(x["delta"]), reverse=True)

    return {
        "new_datasets": new_datasets,
        "removed_datasets": removed_datasets,
        "download_movers": movers[:5],
    }


def _flatten_repos(github_activity: list) -> dict:
    repos = {}
    for org in github_activity:
        if not isinstance(org, dict):
            continue
        for repo in org.get("repos_updated", []):
            fn = repo.get("full_name")
            if fn:
                repos[fn] = repo
    return repos


def _compare_github(prev_activity: list, curr_activity: list) -> dict:
    prev_repos = _flatten_repos(prev_activity)
    curr_repos = _flatten_repos(curr_activity)

    prev_names = set(prev_repos.keys())
    curr_names = set(curr_repos.keys())

    new_repos = [curr_repos[n] for n in sorted(curr_names - prev_names)]
    gone_repos = [prev_repos[n] for n in sorted(prev_names - curr_names)]

    star_movers = []
    for name in prev_names & curr_names:
        prev_stars = prev_repos[name].get("stars", 0) or 0
        curr_stars = curr_repos[name].get("stars", 0) or 0
        delta = curr_stars - prev_stars
        if delta != 0:
            star_movers.append({
                "full_name": name,
                "prev_stars": prev_stars,
                "curr_stars": curr_stars,
                "delta": delta,
            })
    star_movers.sort(key=lambda x: abs(x["delta"]), reverse=True)

    return {
        "new_repos": new_repos,
        "gone_repos": gone_repos,
        "stars_movers": star_movers[:5],
    }


def _compare_papers(prev_papers: list, curr_papers: list) -> list:
    prev_titles = {p.get("title", "").lower().strip() for p in prev_papers}
    new = [p for p in curr_papers if p.get("title", "").lower().strip() not in prev_titles]
    return new


def _count_blog_articles(blog_posts: list) -> int:
    count = 0
    for b in blog_posts:
        if isinstance(b, dict):
            articles = b.get("articles", [])
            if isinstance(articles, list):
                count += len(articles)
    return count


def _count_tweets(x_activity) -> int:
    if isinstance(x_activity, dict):
        accounts = x_activity.get("accounts", [])
    elif isinstance(x_activity, list):
        accounts = x_activity
    else:
        return 0
    count = 0
    for a in accounts:
        if isinstance(a, dict):
            tweets = a.get("relevant_tweets", [])
            if isinstance(tweets, list):
                count += len(tweets)
    return count
