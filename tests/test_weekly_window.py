"""Tests for weekly publication window deduplication."""

from datetime import date

from weekly_window import WeeklyWindow, find_overlaps, window_from_report


def test_exact_window_is_rejected() -> None:
    candidate = WeeklyWindow(date(2026, 8, 29), date(2026, 9, 5))
    issues = [{"week": "W26", "date_start": "2026-08-29", "date_end": "2026-09-05"}]

    matches = find_overlaps(candidate, issues)

    assert matches[0]["week"] == "W26"
    assert matches[0]["overlap_days"] == 7
    assert matches[0]["overlap_ratio"] == 1.0


def test_large_overlap_is_rejected() -> None:
    candidate = WeeklyWindow(date(2026, 8, 30), date(2026, 9, 6))
    issues = [{"week": "W26", "date_start": "2026-08-29", "date_end": "2026-09-05"}]

    matches = find_overlaps(candidate, issues)

    assert matches[0]["overlap_days"] == 6
    assert matches[0]["overlap_ratio"] == 0.8571


def test_adjacent_windows_are_allowed() -> None:
    candidate = WeeklyWindow(date(2026, 9, 5), date(2026, 9, 12))
    issues = [{"week": "W26", "date_start": "2026-08-29", "date_end": "2026-09-05"}]

    assert find_overlaps(candidate, issues) == []


def test_report_period_accepts_iso_timestamps() -> None:
    report = {
        "period": {
            "start": "2026-08-30T10:00:00+08:00",
            "end": "2026-09-06T10:00:00+08:00",
        }
    }

    assert window_from_report(report).as_dict() == {"start": "2026-08-30", "end": "2026-09-06"}


def test_malformed_legacy_issue_is_ignored() -> None:
    candidate = WeeklyWindow(date(2026, 9, 5), date(2026, 9, 12))

    assert find_overlaps(candidate, [{"week": "legacy"}, {"week": "W26", "date_start": "2026-08-29", "date_end": "2026-09-05"}]) == []
