"""Anomaly detection and alerting for AI Dataset Radar.

Evaluates scan results against configurable rules and dispatches
alerts via Email/Webhook when thresholds are exceeded.
"""

import hashlib
import json
import logging
import os
import smtplib
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from email.mime.text import MIMEText
from pathlib import Path
from typing import Optional

import requests

logger = logging.getLogger(__name__)

DEFAULT_RULES = {
    "zero_data": True,
    "threshold_dataset_drop": 0.3,
    "trend_breakthrough": True,
    "trend_rapid_growth": 2.0,
    "change_new_org": True,
    "change_dataset_removed": True,
}


@dataclass
class Alert:
    rule: str
    severity: str  # "critical", "warning", "info"
    title: str
    detail: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).replace(tzinfo=None).isoformat())
    fingerprint: str = ""

    def __post_init__(self):
        if not self.fingerprint:
            raw = f"{self.rule}:{self.title}"
            self.fingerprint = hashlib.sha256(raw.encode()).hexdigest()[:16]

    def to_dict(self) -> dict:
        return asdict(self)


class AlertManager:
    """Evaluates scan results and dispatches alerts."""

    def __init__(self, config: dict):
        self.config = config
        alert_cfg = config.get("alerting", {})
        self.enabled = alert_cfg.get("enabled", False)
        self.channels = alert_cfg.get("channels", [])
        self.dedup_hours = alert_cfg.get("dedup_hours", 24)
        self.rules = {**DEFAULT_RULES, **alert_cfg.get("rules", {})}
        self.output_dir = Path(config.get("notifications", {}).get(
            "markdown", {}
        ).get("output_dir", "data"))

    def evaluate(
        self,
        report: dict,
        prev_report: Optional[dict] = None,
    ) -> list[Alert]:
        """Run all rules, deduplicate, dispatch, and save alerts."""
        if not self.enabled:
            return []

        alerts = []
        alerts.extend(self._check_zero_data(report))
        alerts.extend(self._check_thresholds(report, prev_report))
        alerts.extend(self._check_trends(report))
        alerts.extend(self._check_changes(report, prev_report))

        alerts = self._deduplicate(alerts)

        if alerts:
            self._save_log(alerts)
            self._dispatch(alerts)

        return alerts

    # ── Rule: zero data ──────────────────────────────────────

    def _check_zero_data(self, report: dict) -> list[Alert]:
        if not self.rules.get("zero_data"):
            return []

        alerts = []
        checks = [
            ("datasets", report.get("datasets", []), "Datasets: 0 from all tracked orgs"),
            ("github", report.get("github_activity", []), "GitHub: 0 active orgs"),
            ("papers", report.get("papers", []), "Papers: 0 results"),
            ("blogs", report.get("blog_posts", []), "Blogs: 0 active sources"),
            ("reddit", report.get("reddit_activity", {}).get("posts", []), "Reddit: 0 posts"),
        ]
        for source, data, msg in checks:
            if len(data) == 0:
                alerts.append(Alert(
                    rule=f"zero_data_{source}",
                    severity="critical",
                    title=msg,
                    detail=f"Data source '{source}' returned 0 results. Check connectivity and API keys.",
                ))
        return alerts

    # ── Rule: thresholds ─────────────────────────────────────

    def _check_thresholds(
        self,
        report: dict,
        prev_report: Optional[dict],
    ) -> list[Alert]:
        if prev_report is None:
            return []

        alerts = []
        drop_pct = self.rules.get("threshold_dataset_drop", 0.3)

        curr_count = len(report.get("datasets", []))
        prev_count = len(prev_report.get("datasets", []))

        if prev_count > 0 and curr_count < prev_count:
            change = (prev_count - curr_count) / prev_count
            if change >= drop_pct:
                alerts.append(Alert(
                    rule="threshold_dataset_drop",
                    severity="warning",
                    title=f"Dataset count dropped {change:.0%} ({prev_count} → {curr_count})",
                    detail=(
                        f"Total datasets dropped from {prev_count} to {curr_count}"
                        f" ({change:.0%} decrease). Threshold: {drop_pct:.0%}."
                    ),
                ))

        # Org silence: active org in prev becomes completely absent in curr
        prev_orgs = _extract_active_orgs(prev_report)
        curr_orgs = _extract_active_orgs(report)
        silent_orgs = prev_orgs - curr_orgs
        for org in sorted(silent_orgs):
            alerts.append(Alert(
                rule="threshold_org_silent",
                severity="warning",
                title=f"Org '{org}' went silent",
                detail=f"Organization '{org}' was active in the previous report but has zero activity now.",
            ))

        return alerts

    # ── Rule: trends ─────────────────────────────────────────

    def _check_trends(self, report: dict) -> list[Alert]:
        alerts = []
        trend_data = report.get("trend_data", {})

        if self.rules.get("trend_breakthrough"):
            for ds in trend_data.get("breakthroughs", []):
                name = ds.get("name", ds.get("id", "unknown"))
                old_dl = ds.get("old_downloads", 0)
                new_dl = ds.get("current_downloads", 0)
                alerts.append(Alert(
                    rule="trend_breakthrough",
                    severity="info",
                    title=f"Breakthrough: {name} ({old_dl:,} → {new_dl:,} downloads)",
                    detail=f"Dataset '{name}' crossed the breakthrough threshold ({old_dl:,} → {new_dl:,}).",
                ))

        min_growth = self.rules.get("trend_rapid_growth", 2.0)
        if min_growth:
            for ds in trend_data.get("rising_7d", []):
                growth = ds.get("growth", 0)
                if growth >= min_growth:
                    name = ds.get("name", ds.get("id", "unknown"))
                    alerts.append(Alert(
                        rule="trend_rapid_growth",
                        severity="info",
                        title=f"Rapid growth: {name} (+{growth:.0%} in 7d)",
                        detail=f"Dataset '{name}' grew {growth:.0%} in 7 days (threshold: {min_growth:.0%}).",
                    ))

        return alerts

    # ── Rule: changes ────────────────────────────────────────

    def _check_changes(
        self,
        report: dict,
        prev_report: Optional[dict],
    ) -> list[Alert]:
        if prev_report is None:
            return []

        alerts = []

        if self.rules.get("change_new_org"):
            prev_orgs = _extract_all_orgs(prev_report)
            curr_orgs = _extract_all_orgs(report)
            new_orgs = curr_orgs - prev_orgs
            for org in sorted(new_orgs):
                alerts.append(Alert(
                    rule="change_new_org",
                    severity="info",
                    title=f"New org detected: {org}",
                    detail=f"Organization '{org}' appeared for the first time in this scan.",
                ))

        if self.rules.get("change_dataset_removed"):
            prev_ids = {d.get("id", d.get("name", "")) for d in prev_report.get("datasets", [])}
            curr_ids = {d.get("id", d.get("name", "")) for d in report.get("datasets", [])}
            removed = prev_ids - curr_ids - {""}
            for ds_id in sorted(removed):
                alerts.append(Alert(
                    rule="change_dataset_removed",
                    severity="warning",
                    title=f"Dataset removed: {ds_id}",
                    detail=f"Dataset '{ds_id}' was present in the previous report but is now missing.",
                ))

        return alerts

    # ── Deduplication ────────────────────────────────────────

    def _deduplicate(self, alerts: list[Alert]) -> list[Alert]:
        """Filter out alerts that were already fired within dedup_hours."""
        log_path = self.output_dir / "alerts.json"
        recent_fps = set()

        if log_path.exists():
            try:
                with open(log_path, "r", encoding="utf-8") as f:
                    history = json.load(f)
                cutoff = datetime.now(timezone.utc).replace(tzinfo=None).isoformat()
                # Walk backwards through history
                for entry in reversed(history):
                    ts = entry.get("timestamp", "")
                    if ts and _hours_between(ts, cutoff) <= self.dedup_hours:
                        recent_fps.add(entry.get("fingerprint", ""))
                    elif ts:
                        break  # History is chronological, stop when outside window
            except (json.JSONDecodeError, KeyError):
                pass

        return [a for a in alerts if a.fingerprint not in recent_fps]

    # ── Dispatch ─────────────────────────────────────────────

    def _dispatch(self, alerts: list[Alert]) -> None:
        """Send alerts via configured channels."""
        notif_cfg = self.config.get("notifications", {})

        if "webhook" in self.channels:
            self._send_webhook(alerts, notif_cfg.get("webhook", {}))

        if "email" in self.channels:
            self._send_email(alerts, notif_cfg.get("email", {}))

    def _send_webhook(self, alerts: list[Alert], cfg: dict) -> None:
        url = _expand_env(cfg.get("url", ""))
        if not url:
            url = os.environ.get("WEBHOOK_URL", "")
        if not url:
            logger.warning("Webhook URL not configured, skipping webhook alerts")
            return

        payload = {
            "event": "radar_alert",
            "timestamp": datetime.now(timezone.utc).replace(tzinfo=None).isoformat(),
            "alert_count": len(alerts),
            "alerts": [a.to_dict() for a in alerts],
        }
        try:
            resp = requests.post(url, json=payload, timeout=10)
            resp.raise_for_status()
            logger.info("Webhook alerts sent (%d alerts)", len(alerts))
        except Exception as e:
            logger.warning("Webhook alert failed: %s", e)

    def _send_email(self, alerts: list[Alert], cfg: dict) -> None:
        smtp_server = _expand_env(cfg.get("smtp_server", "")) or os.environ.get("SMTP_SERVER", "")
        username = _expand_env(cfg.get("username", "")) or os.environ.get("SMTP_USERNAME", "")
        password = _expand_env(cfg.get("password", "")) or os.environ.get("SMTP_PASSWORD", "")
        from_addr = _expand_env(cfg.get("from_addr", "")) or os.environ.get("EMAIL_FROM", "")
        to_addrs = cfg.get("to_addrs", [])
        if not to_addrs:
            to_env = os.environ.get("EMAIL_TO", "")
            if to_env:
                to_addrs = [a.strip() for a in to_env.split(",")]

        if not all([smtp_server, username, from_addr, to_addrs]):
            logger.warning("Email not fully configured, skipping email alerts")
            return

        body = _format_email_body(alerts)
        msg = MIMEText(body, "plain", "utf-8")
        date_str = datetime.now(timezone.utc).replace(tzinfo=None).strftime("%Y-%m-%d")
        msg["Subject"] = f"[Radar Alert] {len(alerts)} alert(s) - {date_str}"
        msg["From"] = from_addr
        msg["To"] = ", ".join(to_addrs)

        try:
            port = cfg.get("smtp_port", 587)
            with smtplib.SMTP(smtp_server, port) as server:
                server.starttls()
                server.login(username, password)
                server.send_message(msg)
            logger.info("Email alerts sent (%d alerts)", len(alerts))
        except Exception as e:
            logger.warning("Email alert failed: %s", e)

    # ── Save log ─────────────────────────────────────────────

    def _save_log(self, alerts: list[Alert]) -> Path:
        """Append alerts to global log and date-specific log."""
        # Global dedup log
        global_log = self.output_dir / "alerts.json"
        history = []
        if global_log.exists():
            try:
                with open(global_log, "r", encoding="utf-8") as f:
                    history = json.load(f)
            except (json.JSONDecodeError, ValueError):
                pass

        new_entries = [a.to_dict() for a in alerts]
        history.extend(new_entries)

        os.makedirs(global_log.parent, exist_ok=True)
        with open(global_log, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)

        # Date-specific log
        date_str = datetime.now(timezone.utc).replace(tzinfo=None).strftime("%Y-%m-%d")
        date_dir = self.output_dir / "reports" / date_str
        os.makedirs(date_dir, exist_ok=True)
        date_log = date_dir / "alerts.json"

        date_history = []
        if date_log.exists():
            try:
                with open(date_log, "r", encoding="utf-8") as f:
                    date_history = json.load(f)
            except (json.JSONDecodeError, ValueError):
                pass

        date_history.extend(new_entries)
        with open(date_log, "w", encoding="utf-8") as f:
            json.dump(date_history, f, ensure_ascii=False, indent=2)

        logger.info("Saved %d alerts to %s", len(alerts), date_log)
        return date_log


# ── Helpers ──────────────────────────────────────────────────


def _extract_active_orgs(report: dict) -> set[str]:
    """Extract orgs that have at least one dataset or active repo."""
    orgs = set()
    for ds in report.get("datasets", []):
        author = ds.get("author", "")
        if author:
            orgs.add(author)
    for repo in report.get("github_activity", []):
        org = repo.get("org", "")
        if org:
            orgs.add(org)
    return orgs


def _extract_all_orgs(report: dict) -> set[str]:
    """Extract all mentioned org names from datasets + github."""
    return _extract_active_orgs(report)


def _hours_between(ts1: str, ts2: str) -> float:
    """Calculate hours between two ISO timestamps."""
    try:
        t1 = datetime.fromisoformat(ts1)
        t2 = datetime.fromisoformat(ts2)
        return abs((t2 - t1).total_seconds()) / 3600
    except (ValueError, TypeError):
        return float("inf")


def _expand_env(value: str) -> str:
    """Expand ${VAR} or $VAR patterns in a string."""
    if not value:
        return value
    return os.path.expandvars(value)


def _format_email_body(alerts: list[Alert]) -> str:
    """Format alerts into a plain-text email body."""
    lines = ["AI Dataset Radar - Alert Summary", "=" * 40, ""]

    by_severity = {"critical": [], "warning": [], "info": []}
    for a in alerts:
        by_severity.get(a.severity, by_severity["info"]).append(a)

    for severity in ("critical", "warning", "info"):
        group = by_severity[severity]
        if not group:
            continue
        icon = {"critical": "!!!", "warning": "!!", "info": "i"}[severity]
        lines.append(f"[{icon}] {severity.upper()} ({len(group)})")
        lines.append("-" * 30)
        for a in group:
            lines.append(f"  {a.title}")
            lines.append(f"    {a.detail}")
            lines.append("")

    lines.append("---")
    lines.append("Generated by AI Dataset Radar")
    return "\n".join(lines)
