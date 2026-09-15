"""Date-window validation for the weekly insights publisher.

The scanner's ``--days 7`` window is a half-open interval: ``start`` is the
earliest timestamp and ``end`` is exactly seven days later.  The website
registry stores those boundaries as date-only values.  We therefore compare
calendar boundaries using ``[start, end)`` semantics so a rerun cannot publish
a second issue covering the same days, while a window starting exactly at the
previous window's end is allowed.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class WeeklyWindow:
    """A half-open calendar-date window ``[start, end)``."""

    start: date
    end: date

    def __post_init__(self) -> None:
        if self.end <= self.start:
            raise ValueError(f"window end {self.end} must be after start {self.start}")

    @property
    def days(self) -> int:
        return (self.end - self.start).days

    def overlaps(self, other: "WeeklyWindow") -> bool:
        return self.start < other.end and other.start < self.end

    def overlap_days(self, other: "WeeklyWindow") -> int:
        if not self.overlaps(other):
            return 0
        start = max(self.start, other.start)
        end = min(self.end, other.end)
        return (end - start).days

    def is_adjacent_to(self, other: "WeeklyWindow") -> bool:
        return self.end == other.start or other.end == self.start

    def as_dict(self) -> dict[str, str]:
        return {"start": self.start.isoformat(), "end": self.end.isoformat()}


def parse_calendar_date(value: Any) -> date:
    """Parse an ISO date or datetime into a calendar date."""

    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value or "").strip()
    if not text:
        raise ValueError("empty date value")
    # ISO timestamps may end in Z; fromisoformat accepts the equivalent offset.
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text).date()
    except ValueError:
        return date.fromisoformat(text[:10])


def window_from_mapping(mapping: Mapping[str, Any], *, period_key: str | None = None) -> WeeklyWindow:
    """Build a window from a report or issue mapping."""

    source: Mapping[str, Any] = mapping
    if period_key:
        nested = mapping.get(period_key)
        if not isinstance(nested, Mapping):
            raise ValueError(f"missing {period_key} mapping")
        source = nested
    start_value = source.get("date_start", source.get("start"))
    end_value = source.get("date_end", source.get("end"))
    if start_value in (None, "") or end_value in (None, ""):
        raise ValueError("window requires both start and end")
    return WeeklyWindow(parse_calendar_date(start_value), parse_calendar_date(end_value))


def window_from_report(report: Mapping[str, Any]) -> WeeklyWindow:
    """Build a window from the scanner JSON report."""

    return window_from_mapping(report, period_key="period")


def find_overlaps(candidate: WeeklyWindow, issues: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return existing issue entries whose calendar windows overlap candidate."""

    matches: list[dict[str, Any]] = []
    for issue in issues:
        try:
            existing = window_from_mapping(issue)
        except (TypeError, ValueError):
            # Legacy/non-weekly registry entries do not participate in the gate.
            continue
        overlap_days = candidate.overlap_days(existing)
        if overlap_days:
            matches.append(
                {
                    "week": issue.get("week", "unknown"),
                    "window": existing.as_dict(),
                    "overlap_days": overlap_days,
                    "overlap_ratio": round(overlap_days / min(candidate.days, existing.days), 4),
                }
            )
    return matches


def validate_no_overlap(candidate: WeeklyWindow, issues: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return conflicts; an empty list means the candidate is publishable.

    We reject every calendar-boundary overlap, not only exact duplicates.  This
    is intentionally stricter than a percentage threshold: the only permitted
    repeat-run boundary is an adjacent half-open window (for example Sep 1–8
    followed by Sep 8–15).
    """

    return find_overlaps(candidate, issues)
