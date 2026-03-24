"""
Additions for src/alerting.py  (P1 alert rules: sudden_burst + quality_drop)

HOW TO APPLY
============
1. In DEFAULT_RULES dict, add two entries:
       "sudden_burst": 20,    # alert if one org publishes >= N datasets in one scan
       "quality_drop": 0.3,   # alert if avg quality score drops by this fraction

2. In AlertManager.evaluate(), after the line:
       alerts.extend(self._check_changes(report, prev_report))
   add:
       alerts.extend(self._check_sudden_burst(report))
       alerts.extend(self._check_quality_drop(report, prev_report))

3. Paste the two methods below into AlertManager (before _deduplicate).
"""

from typing import Optional
# (Alert is already imported from alerting.py)


# ── Rule: sudden burst ─────────────────────────────────────────────────────────
def _check_sudden_burst(self, report: dict) -> list:
    """Alert when a single org publishes an unusually large number of datasets."""
    threshold = self.rules.get("sudden_burst")
    if not threshold:
        return []
    alerts = []
    datasets = report.get("datasets", [])
    org_counts: dict[str, int] = {}
    for ds in datasets:
        org = ds.get("author") or ds.get("org") or "unknown"
        org_counts[org] = org_counts.get(org, 0) + 1
    for org, count in org_counts.items():
        if count >= threshold:
            alerts.append(Alert(
                rule="sudden_burst",
                severity="warning",
                title=f"Sudden burst: {org} published {count} datasets",
                detail=(
                    f"Org '{org}' published {count} datasets in this scan window, "
                    f"exceeding threshold of {threshold}. "
                    "May indicate bulk import noise or a major data release."
                ),
            ))
    return alerts


# ── Rule: quality drop ─────────────────────────────────────────────────────────
def _check_quality_drop(self, report: dict, prev_report: Optional[dict]) -> list:
    """Alert when average dataset quality score drops significantly."""
    threshold = self.rules.get("quality_drop")
    if not threshold or prev_report is None:
        return []

    def _avg_quality(rep: dict) -> Optional[float]:
        datasets = rep.get("datasets", [])
        scores = [
            d.get("quality_score") or d.get("quality", {}).get("total")
            for d in datasets
        ]
        scores = [s for s in scores if s is not None]
        return sum(scores) / len(scores) if scores else None

    prev_avg = _avg_quality(prev_report)
    curr_avg = _avg_quality(report)
    if prev_avg is None or curr_avg is None or prev_avg == 0:
        return []

    drop_ratio = (prev_avg - curr_avg) / prev_avg
    if drop_ratio >= threshold:
        return [Alert(
            rule="quality_drop",
            severity="warning",
            title=f"Quality score dropped {drop_ratio:.0%} (prev={prev_avg:.2f}, curr={curr_avg:.2f})",
            detail=(
                f"Average dataset quality score fell from {prev_avg:.2f} to {curr_avg:.2f} "
                f"({drop_ratio:.0%} drop), exceeding threshold of {threshold:.0%}. "
                "Check if low-quality sources were added or scoring logic changed."
            ),
        )]
    return []
