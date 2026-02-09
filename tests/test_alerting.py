"""Tests for the alerting module."""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from alerting import Alert, AlertManager, _format_email_body, _hours_between


# ── Fixtures ─────────────────────────────────────────────────

@pytest.fixture
def base_config():
    return {
        "alerting": {
            "enabled": True,
            "channels": [],
            "dedup_hours": 24,
            "rules": {
                "zero_data": True,
                "threshold_dataset_drop": 0.3,
                "trend_breakthrough": True,
                "trend_rapid_growth": 2.0,
                "change_new_org": True,
                "change_dataset_removed": True,
            },
        },
        "notifications": {
            "markdown": {"output_dir": "data"},
        },
    }


@pytest.fixture
def sample_report():
    return {
        "datasets": [
            {"id": "org1/ds1", "author": "org1", "downloads": 1000},
            {"id": "org2/ds2", "author": "org2", "downloads": 500},
        ],
        "github_activity": [
            {"org": "org1", "repos_updated": 3},
        ],
        "papers": [{"title": "Paper A"}],
        "blog_posts": [{"title": "Post A"}],
        "reddit_activity": {"posts": [{"title": "Thread A"}]},
        "trend_data": {},
    }


@pytest.fixture
def prev_report():
    return {
        "datasets": [
            {"id": "org1/ds1", "author": "org1", "downloads": 800},
            {"id": "org2/ds2", "author": "org2", "downloads": 400},
            {"id": "org3/ds3", "author": "org3", "downloads": 200},
        ],
        "github_activity": [
            {"org": "org1", "repos_updated": 5},
            {"org": "org3", "repos_updated": 2},
        ],
        "papers": [{"title": "Paper A"}],
        "blog_posts": [{"title": "Post A"}],
        "reddit_activity": {"posts": [{"title": "Thread A"}]},
        "trend_data": {},
    }


# ── Alert dataclass ──────────────────────────────────────────

class TestAlert:
    def test_create_alert(self):
        a = Alert(rule="test", severity="info", title="Test", detail="Detail")
        assert a.rule == "test"
        assert a.severity == "info"
        assert a.fingerprint  # Auto-generated
        assert a.timestamp  # Auto-generated

    def test_fingerprint_deterministic(self):
        a1 = Alert(rule="test", severity="info", title="Same", detail="D1")
        a2 = Alert(rule="test", severity="warning", title="Same", detail="D2")
        assert a1.fingerprint == a2.fingerprint  # Same rule+title

    def test_fingerprint_differs(self):
        a1 = Alert(rule="test", severity="info", title="Alpha", detail="D")
        a2 = Alert(rule="test", severity="info", title="Beta", detail="D")
        assert a1.fingerprint != a2.fingerprint

    def test_to_dict(self):
        a = Alert(rule="r", severity="s", title="t", detail="d")
        d = a.to_dict()
        assert d["rule"] == "r"
        assert d["severity"] == "s"
        assert "fingerprint" in d
        assert "timestamp" in d

    def test_custom_fingerprint(self):
        a = Alert(rule="r", severity="s", title="t", detail="d", fingerprint="custom")
        assert a.fingerprint == "custom"


# ── Zero data rules ──────────────────────────────────────────

class TestZeroData:
    def test_empty_datasets(self, base_config):
        mgr = AlertManager(base_config)
        report = {
            "datasets": [],
            "github_activity": [{"org": "x"}],
            "papers": [{"t": "p"}],
            "blog_posts": [{"t": "b"}],
            "reddit_activity": {"posts": [{"t": "r"}]},
            "trend_data": {},
        }
        alerts = mgr._check_zero_data(report)
        assert len(alerts) == 1
        assert alerts[0].rule == "zero_data_datasets"
        assert alerts[0].severity == "critical"

    def test_empty_github(self, base_config):
        mgr = AlertManager(base_config)
        report = {
            "datasets": [{"id": "x"}],
            "github_activity": [],
            "papers": [{"t": "p"}],
            "blog_posts": [{"t": "b"}],
            "reddit_activity": {"posts": [{"t": "r"}]},
            "trend_data": {},
        }
        alerts = mgr._check_zero_data(report)
        assert len(alerts) == 1
        assert alerts[0].rule == "zero_data_github"

    def test_empty_reddit(self, base_config):
        mgr = AlertManager(base_config)
        report = {
            "datasets": [{"id": "x"}],
            "github_activity": [{"org": "x"}],
            "papers": [{"t": "p"}],
            "blog_posts": [{"t": "b"}],
            "reddit_activity": {"posts": []},
            "trend_data": {},
        }
        alerts = mgr._check_zero_data(report)
        assert len(alerts) == 1
        assert alerts[0].rule == "zero_data_reddit"

    def test_multiple_empty(self, base_config):
        mgr = AlertManager(base_config)
        report = {
            "datasets": [],
            "github_activity": [],
            "papers": [],
            "blog_posts": [],
            "reddit_activity": {"posts": []},
            "trend_data": {},
        }
        alerts = mgr._check_zero_data(report)
        assert len(alerts) == 5

    def test_all_ok(self, base_config, sample_report):
        mgr = AlertManager(base_config)
        alerts = mgr._check_zero_data(sample_report)
        assert len(alerts) == 0

    def test_disabled_rule(self, base_config, sample_report):
        base_config["alerting"]["rules"]["zero_data"] = False
        mgr = AlertManager(base_config)
        report = {"datasets": [], "github_activity": [], "papers": [],
                  "blog_posts": [], "reddit_activity": {"posts": []}, "trend_data": {}}
        assert mgr._check_zero_data(report) == []


# ── Threshold rules ──────────────────────────────────────────

class TestThresholds:
    def test_dataset_drop_triggers(self, base_config):
        mgr = AlertManager(base_config)
        prev = {"datasets": [{"id": f"d{i}", "author": "a"} for i in range(10)],
                "github_activity": []}
        curr = {"datasets": [{"id": f"d{i}", "author": "a"} for i in range(5)],
                "github_activity": []}
        alerts = mgr._check_thresholds(curr, prev)
        drop_alerts = [a for a in alerts if a.rule == "threshold_dataset_drop"]
        assert len(drop_alerts) == 1
        assert "50%" in drop_alerts[0].title

    def test_small_drop_ignored(self, base_config):
        mgr = AlertManager(base_config)
        prev = {"datasets": [{"id": f"d{i}", "author": "a"} for i in range(10)],
                "github_activity": []}
        curr = {"datasets": [{"id": f"d{i}", "author": "a"} for i in range(8)],
                "github_activity": []}
        alerts = mgr._check_thresholds(curr, prev)
        drop_alerts = [a for a in alerts if a.rule == "threshold_dataset_drop"]
        assert len(drop_alerts) == 0

    def test_org_silence(self, base_config):
        mgr = AlertManager(base_config)
        prev = {"datasets": [{"id": "d1", "author": "orgA"}],
                "github_activity": [{"org": "orgB"}]}
        curr = {"datasets": [{"id": "d2", "author": "orgA"}],
                "github_activity": []}
        alerts = mgr._check_thresholds(curr, prev)
        silent = [a for a in alerts if a.rule == "threshold_org_silent"]
        assert len(silent) == 1
        assert "orgB" in silent[0].title

    def test_no_prev_report(self, base_config, sample_report):
        mgr = AlertManager(base_config)
        alerts = mgr._check_thresholds(sample_report, None)
        assert len(alerts) == 0


# ── Trend rules ──────────────────────────────────────────────

class TestTrends:
    def test_breakthrough_detected(self, base_config):
        mgr = AlertManager(base_config)
        report = {
            "trend_data": {
                "breakthroughs": [
                    {"name": "hot-dataset", "old_downloads": 50, "current_downloads": 2000},
                ],
                "rising_7d": [],
            }
        }
        alerts = mgr._check_trends(report)
        assert len(alerts) == 1
        assert alerts[0].rule == "trend_breakthrough"
        assert "hot-dataset" in alerts[0].title

    def test_rapid_growth(self, base_config):
        mgr = AlertManager(base_config)
        report = {
            "trend_data": {
                "breakthroughs": [],
                "rising_7d": [
                    {"name": "growing-ds", "growth": 3.5},
                    {"name": "slow-ds", "growth": 0.5},
                ],
            }
        }
        alerts = mgr._check_trends(report)
        assert len(alerts) == 1
        assert alerts[0].rule == "trend_rapid_growth"
        assert "growing-ds" in alerts[0].title

    def test_no_trends(self, base_config, sample_report):
        mgr = AlertManager(base_config)
        alerts = mgr._check_trends(sample_report)
        assert len(alerts) == 0

    def test_disabled_breakthrough(self, base_config):
        base_config["alerting"]["rules"]["trend_breakthrough"] = False
        mgr = AlertManager(base_config)
        report = {"trend_data": {
            "breakthroughs": [{"name": "x", "old_downloads": 0, "current_downloads": 5000}],
            "rising_7d": [],
        }}
        alerts = mgr._check_trends(report)
        assert len(alerts) == 0


# ── Change rules ─────────────────────────────────────────────

class TestChanges:
    def test_new_org(self, base_config):
        mgr = AlertManager(base_config)
        prev = {"datasets": [{"id": "d1", "author": "oldOrg"}], "github_activity": []}
        curr = {"datasets": [{"id": "d1", "author": "oldOrg"},
                             {"id": "d2", "author": "newOrg"}], "github_activity": []}
        alerts = mgr._check_changes(curr, prev)
        new_org_alerts = [a for a in alerts if a.rule == "change_new_org"]
        assert len(new_org_alerts) == 1
        assert "newOrg" in new_org_alerts[0].title

    def test_dataset_removed(self, base_config):
        mgr = AlertManager(base_config)
        prev = {"datasets": [{"id": "ds1"}, {"id": "ds2"}], "github_activity": []}
        curr = {"datasets": [{"id": "ds1"}], "github_activity": []}
        alerts = mgr._check_changes(curr, prev)
        removed = [a for a in alerts if a.rule == "change_dataset_removed"]
        assert len(removed) == 1
        assert "ds2" in removed[0].title

    def test_no_prev(self, base_config, sample_report):
        mgr = AlertManager(base_config)
        assert mgr._check_changes(sample_report, None) == []

    def test_no_changes(self, base_config, sample_report):
        mgr = AlertManager(base_config)
        alerts = mgr._check_changes(sample_report, sample_report)
        assert len(alerts) == 0


# ── Deduplication ────────────────────────────────────────────

class TestDedup:
    def test_no_history(self, base_config, tmp_path):
        base_config["notifications"] = {"markdown": {"output_dir": str(tmp_path)}}
        mgr = AlertManager(base_config)
        alerts = [Alert(rule="r", severity="info", title="t", detail="d")]
        result = mgr._deduplicate(alerts)
        assert len(result) == 1

    def test_recent_duplicate_filtered(self, base_config, tmp_path):
        base_config["notifications"] = {"markdown": {"output_dir": str(tmp_path)}}
        mgr = AlertManager(base_config)

        a = Alert(rule="r", severity="info", title="t", detail="d")
        # Write fake history with same fingerprint
        log_path = tmp_path / "alerts.json"
        now_ts = datetime.now(timezone.utc).replace(tzinfo=None).isoformat()
        history = [{"fingerprint": a.fingerprint, "timestamp": now_ts}]
        log_path.write_text(json.dumps(history))

        result = mgr._deduplicate([a])
        assert len(result) == 0

    def test_old_duplicate_passes(self, base_config, tmp_path):
        base_config["notifications"] = {"markdown": {"output_dir": str(tmp_path)}}
        mgr = AlertManager(base_config)

        a = Alert(rule="r", severity="info", title="t", detail="d")
        # Write history with timestamp >24h ago
        old_ts = "2020-01-01T00:00:00"
        log_path = tmp_path / "alerts.json"
        history = [{"fingerprint": a.fingerprint, "timestamp": old_ts}]
        log_path.write_text(json.dumps(history))

        result = mgr._deduplicate([a])
        assert len(result) == 1

    def test_corrupt_history(self, base_config, tmp_path):
        base_config["notifications"] = {"markdown": {"output_dir": str(tmp_path)}}
        mgr = AlertManager(base_config)

        log_path = tmp_path / "alerts.json"
        log_path.write_text("not json")

        alerts = [Alert(rule="r", severity="info", title="t", detail="d")]
        result = mgr._deduplicate(alerts)
        assert len(result) == 1  # Gracefully passes through


# ── Dispatch ─────────────────────────────────────────────────

class TestDispatch:
    @patch("alerting.requests.post")
    def test_webhook_dispatch(self, mock_post, base_config, tmp_path):
        base_config["notifications"] = {
            "markdown": {"output_dir": str(tmp_path)},
            "webhook": {"url": "https://hooks.example.com/test"},
        }
        base_config["alerting"]["channels"] = ["webhook"]
        mock_post.return_value = MagicMock(status_code=200)
        mock_post.return_value.raise_for_status = MagicMock()

        mgr = AlertManager(base_config)
        alerts = [Alert(rule="r", severity="info", title="t", detail="d")]
        mgr._dispatch(alerts)

        mock_post.assert_called_once()
        payload = mock_post.call_args[1]["json"]
        assert payload["event"] == "radar_alert"
        assert payload["alert_count"] == 1

    @patch("alerting.smtplib.SMTP")
    def test_email_dispatch(self, mock_smtp, base_config, tmp_path):
        base_config["notifications"] = {
            "markdown": {"output_dir": str(tmp_path)},
            "email": {
                "smtp_server": "smtp.test.com",
                "smtp_port": 587,
                "username": "user@test.com",
                "password": "pass",
                "from_addr": "from@test.com",
                "to_addrs": ["to@test.com"],
            },
        }
        base_config["alerting"]["channels"] = ["email"]

        mock_server = MagicMock()
        mock_smtp.return_value.__enter__ = MagicMock(return_value=mock_server)
        mock_smtp.return_value.__exit__ = MagicMock(return_value=False)

        mgr = AlertManager(base_config)
        alerts = [Alert(rule="r", severity="critical", title="Critical!", detail="d")]
        mgr._dispatch(alerts)

        mock_smtp.assert_called_once_with("smtp.test.com", 587)
        mock_server.send_message.assert_called_once()

    def test_no_channels(self, base_config, tmp_path):
        base_config["notifications"] = {"markdown": {"output_dir": str(tmp_path)}}
        base_config["alerting"]["channels"] = []
        mgr = AlertManager(base_config)
        # Should not raise
        mgr._dispatch([Alert(rule="r", severity="info", title="t", detail="d")])


# ── Save log ─────────────────────────────────────────────────

class TestSaveLog:
    def test_creates_files(self, base_config, tmp_path):
        base_config["notifications"] = {"markdown": {"output_dir": str(tmp_path)}}
        mgr = AlertManager(base_config)
        alerts = [Alert(rule="r", severity="info", title="t", detail="d")]
        mgr._save_log(alerts)

        # Global log
        global_log = tmp_path / "alerts.json"
        assert global_log.exists()
        data = json.loads(global_log.read_text())
        assert len(data) == 1
        assert data[0]["rule"] == "r"

    def test_appends_to_existing(self, base_config, tmp_path):
        base_config["notifications"] = {"markdown": {"output_dir": str(tmp_path)}}
        mgr = AlertManager(base_config)

        # First batch
        mgr._save_log([Alert(rule="a", severity="info", title="A", detail="d")])
        # Second batch
        mgr._save_log([Alert(rule="b", severity="info", title="B", detail="d")])

        data = json.loads((tmp_path / "alerts.json").read_text())
        assert len(data) == 2


# ── Full workflow ────────────────────────────────────────────

class TestFullWorkflow:
    def test_evaluate_disabled(self, base_config, sample_report):
        base_config["alerting"]["enabled"] = False
        mgr = AlertManager(base_config)
        alerts = mgr.evaluate(sample_report)
        assert alerts == []

    def test_evaluate_no_issues(self, base_config, sample_report, tmp_path):
        base_config["notifications"] = {"markdown": {"output_dir": str(tmp_path)}}
        mgr = AlertManager(base_config)
        alerts = mgr.evaluate(sample_report)
        assert len(alerts) == 0

    def test_evaluate_with_issues(self, base_config, tmp_path):
        base_config["notifications"] = {"markdown": {"output_dir": str(tmp_path)}}
        mgr = AlertManager(base_config)
        report = {
            "datasets": [],
            "github_activity": [],
            "papers": [],
            "blog_posts": [],
            "reddit_activity": {"posts": []},
            "trend_data": {},
        }
        alerts = mgr.evaluate(report)
        assert len(alerts) == 5  # 5 zero_data alerts
        # Verify saved
        assert (tmp_path / "alerts.json").exists()

    def test_evaluate_with_prev_report(self, base_config, sample_report, prev_report, tmp_path):
        base_config["notifications"] = {"markdown": {"output_dir": str(tmp_path)}}
        mgr = AlertManager(base_config)
        alerts = mgr.evaluate(sample_report, prev_report)
        # org3 went silent, ds3 removed
        rules = {a.rule for a in alerts}
        assert "threshold_org_silent" in rules
        assert "change_dataset_removed" in rules

    @patch("alerting.requests.post")
    def test_end_to_end_dispatch(self, mock_post, base_config, tmp_path):
        base_config["notifications"] = {
            "markdown": {"output_dir": str(tmp_path)},
            "webhook": {"url": "https://hooks.example.com"},
        }
        base_config["alerting"]["channels"] = ["webhook"]
        mock_post.return_value = MagicMock(status_code=200)
        mock_post.return_value.raise_for_status = MagicMock()

        mgr = AlertManager(base_config)
        report = {
            "datasets": [],
            "github_activity": [{"org": "x"}],
            "papers": [{"t": "p"}],
            "blog_posts": [{"t": "b"}],
            "reddit_activity": {"posts": [{"t": "r"}]},
            "trend_data": {},
        }
        alerts = mgr.evaluate(report)
        assert len(alerts) == 1
        mock_post.assert_called_once()


# ── Helpers ──────────────────────────────────────────────────

class TestHelpers:
    def test_hours_between(self):
        assert _hours_between("2026-01-01T00:00:00", "2026-01-01T01:00:00") == 1.0
        assert _hours_between("2026-01-01T00:00:00", "2026-01-02T00:00:00") == 24.0

    def test_hours_between_invalid(self):
        assert _hours_between("bad", "data") == float("inf")

    def test_format_email_body(self):
        alerts = [
            Alert(rule="r1", severity="critical", title="Critical!", detail="Fix now"),
            Alert(rule="r2", severity="info", title="FYI", detail="Note"),
        ]
        body = _format_email_body(alerts)
        assert "CRITICAL" in body
        assert "INFO" in body
        assert "Critical!" in body
        assert "FYI" in body
