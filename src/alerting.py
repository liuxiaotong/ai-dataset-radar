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
    "sudden_burst": True,
    "sudden_burst_threshold": 20,
    "quality_drop": True,
    "quality_drop_threshold": 0.3,
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
            "marketing_dir", "output/notifications"
        ))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._alert_history_path = self.output_dir / "alert_history.json"
        self._alert_history = self._load_alert_history()

    def _load_alert_history(self) -> dict:
        if self._alert_history_path.exists():
            try:
                return json.loads(self._alert_history_path.read_text())
            except Exception:
                return {}
        return {}

    def _save_alert_history(self):
        self._alert_history_path.write_text(json.dumps(self._alert_history, indent=2))

    def _is_duplicate(self, alert: Alert) -> bool:
        key = alert.fingerprint
        if key not in self._alert_history:
            return False
        last_sent = datetime.fromisoformat(self._alert_history[key])
        elapsed = (datetime.now() - last_sent).total_seconds() / 3600
        return elapsed < self.dedup_hours

    def evaluate(self, current: dict, previous: dict) -> list[Alert]:
        """Run all rules and return fired alerts."""
        alerts = []
        alerts.extend(self._check_thresholds(current, previous))
        alerts.extend(self._check_trends(current, previous))
        alerts.extend(self._check_changes(current, previous))
        alerts.extend(self._check_sudden_burst(current, previous))
        alerts.extend(self._check_quality_drop(current, previous))
        return alerts

    def dispatch(self, alerts: list[Alert]) -> None:
        """Filter duplicates and send alerts via configured channels."""
        if not self.enabled:
            logger.info("Alerting disabled, skipping dispatch")
            return
        new_alerts = [a for a in alerts if not self._is_duplicate(a)]
        if not new_alerts:
            logger.info("All alerts are duplicates, nothing to send")
            return
        for channel in self.channels:
            channel_type = channel.get("type", "")
            if channel_type == "email":
                self._send_email(channel, new_alerts)
            elif channel_type == "webhook":
                self._send_webhook(channel, new_alerts)
            else:
                logger.warning(f"Unknown channel type: {channel_type}")
        for a in new_alerts:
            self._alert_history[a.fingerprint] = a.timestamp
        self._save_alert_history()

    def _send_email(self, channel: dict, alerts: list[Alert]) -> None:
        smtp_host = channel.get("smtp_host", "")
        smtp_port = int(channel.get("smtp_port", 587))
        username = channel.get("username", "") or os.environ.get("SMTPUSER", "")
        password = channel.get("password", "") or os.environ.get("SMTPPASS", "")
        recipients = channel.get("recipients", [])
        if not all ([smtp_host, username, password, recipients]):
            logger.error("Email channel missing required config")
            return
        body = self._format_email_body(alerts)
        msg = MIMEText(body)
        msg["Subject"] = f"[AI Dataset Radar] {len(alerts)} Alert(s) Detected"
        msg["From"] = username
        msg["To"] = ", ".join(recipients)
        try:
            with smtplib.SMTP(smtp_host, smtp_port) as server:
                server.starttls()
                server.login(username, password)
                server.sendmessage(msg)
            logger.info(f"Email sent to {recipients}")
        except Exception as e:
            logger.error(f"Failed to send email: {e}")

    def _send_webhook(self, channel: dict, alerts: list[Alert]) -> None:
        url = channel.get("url", "")
        if not url:
            logger.error("Webhook channel missing url")
            return
        payload = {
            "alerts": [a.to_dict() for a in alerts],
            "total": len(alerts),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        try:
            resp = requests.post(url, json=payload, timeout=10)
            resp.raise_for_status()
            logger.info(f"Webhook sent to {url}")
        except Exception as e:
            logger.error(f"Failed to send webhook: {e}")

    def _check_thresholds(self, current: dict, previous: dict) -> list[Alert]:
        alerts = []
        current_count = len(current.get("datasets", []))
        prev_count = len(previous.get("datasets", []))
        if self.rules.get("zero_data") and current_count == 0:
            alerts.append(Alert(
                rule="zero_data",
                severity="critical",
                title="Zero datasets detected",
                detail="Current scan returned 0 datasets. Possible source outage or config error.",
            ))
        if prev_count > 0:
            drop_ratio = (prev_count - current_count) / prev_count
            threshold = self.rules.get("threshold_dataset_drop", 0.3)
            if drop_ratio >= threshold:
                alerts.append(Alert(
                    rule="threshold_dataset_drop",
                    severity="critical",
                    title=f"Dataset count dropped {drop_ratio:.1%}",
                    detail=f"Count fell from {prev_count} to {current_count} ({drop_ratio:.1%} drop, threshold={threshold:.0%}).",
                ))
        return alerts

    def _check_trends(self, current: dict, previous: dict) -> list[Alert]:
        alerts = []
        current_trends = current.get("trending_datasets", [])
        prev_trends = previous.get("trending_datasets", [])
        prev_ids = {s["id"] for s in prev_trends if "id" in s}
        if self.rules.get("trend_breakthrough"):
            for dataset in current_trends:
                if dataset.get("id") not in prev_ids:
                    alerts.append(Alert(
                        rule="trend_breakthrough",
                        severity="info",
                        title=f"New trending dataset: {dataset.get('name', 'unknown')}",
                        detail=f"Dataset '{dataset.get('name', 'unknown')}' appeared in trending for the first time.",
                    ))
        growth_threshold = self.rules.get("trend_rapid_growth", 2.0)
        prev_downloads = {d["id"]: d.get("downloads", 0) for d in prev_trends if "id" in d}
        for dataset in current_trends:
            ds_id = dataset.get("id")
            if not ds_id or ds_id not in prev_downloads:
                continue
            prev_dl = prev_downloads[ds_id]
            curr_dl = dataset.get("downloads", 0)
            if prev_dl > 0 and curr_dl / prev_dl >= growth_threshold:
                alerts.append(Alert(
                    rule="trend_rapid_growth",
                    severity="warning",
                    title=f"Rapid growth: {dataset.get('name', 'unknown')}",
                    detail=f"Downloads grew from {prev_dl} to {curr_dl} ({curr_dl/prev_dl:.1f}x, threshold={growth_threshold:.1f}x).",
                ))
        return alerts

    def _check_changes(self, current: dict, previous: dict) -> list[Alert]:
        alerts = []
        current_orgs = {ds.get("author") or ds.get("org") for ds in current.get("datasets", [])}
        prev_orgs = {ds.get("author") or ds.get("org") for ds in previous.get("datasets", [])}
        if self.rules.get("change_new_org"):
            for org in current_orgs - prev_orgs:
                if org:
                    alerts.append(Alert(
                        rule="change_new_org",
                        severity="info",
                        title=f"New org detected: {org}",
                        detail=f"Organization '{org}' appeared in the dataset landscape for the first time.",
                    ))
        current_ids = {ds.get("id") for ds in current.get("datasets", [])}
        prev_ids = {ds.get("id") for ds in previous.get("datasets", [])}
        if self.rules.get("change_dataset_removed"):
            removed = prev_ids - current_ids - {None}
            if removed:
                alerts.append(Alert(
                    rule="change_dataset_removed",
                    severity="warning",
                    title=f"{len(removed)} dataset(s) removed",
                    detail=f"Removed IDs: {', '.join(str(i) for i in list(removed)[:5])}{' ...' if len(removed) > 5 else ''}.",
                ))
        return alerts

    def _save_alerts_to_file(self, alerts: list[Alert], scan_id: str) -> Path:
        output_path = self.output_dir / f"alerts_{scan_id}.json"
        data = {
            "scan_id": scan_id,
            "total": len(alerts),
            "alerts": [a.to_dict() for a in alerts],
        }
        output_path.write_text(json.dumps(data, indent=2))
        return output_path

    def _format_email_body(self, alerts: list[Alert]) -> str:
        lines = ["AI Dataset Radar Alert Report", "=" * 40, ""]
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

    def _check_sudden_burst(self, current: dict, previous: dict) -> list:
        """单中亏 org 作次‌scan 新增数损集计分 warningÌ�"""
        alerts = []
        if not self.rules.get("sudden_burst", True):
            return alerts
        threshold = self.rules.get("sudden_burst_threshold", 20)
        current_by_org = {}
        for ds in current.get("datasets", []):
            org = ds.get("author") or ds.get("org") or "unknown"
            current_by_org[org] = current_by_org.get(org, 0) + 1
        prev_by_org = {}
        for ds in previous.get("datasets", []):
            org = ds.get("author") or ds.get("org") or "unknown"
            prev_by_org[org] = prev_by_org.get(org, 0) + 1
        for org, count in current_by_org.items():
            prev_count = prev_by_org.get(org, 0)
            delta = count - prev_count
            if delta >= threshold:
                alerts.append(Alert(
                    rule="sudden_burst",
                    severity="warning",
                    title=f"Sudden burst: {org} +{delta} datasets",
                    detail=f"Org '{org}' added {delta} datasets in one scan (threshold={threshold}).",
                ))
        return alerts

    def _check_quality_drop(self, current: dict, previous: dict) -> list:
        """平型责重分 为 30% 触号 warningÌ�"""
        alerts = []
        if not self.rules.get("quality_drop", True):
            return alerts
        drop_threshold = self.rules.get("quality_drop_threshold", 0.3)
        def avg_quality(scan):
            scores = [ds.get("quality_score", 0) for ds in scan.get("datasets", []) if ds.get("quality_score") is not None]
            return sum(scores) / len(scores) if scores else None
        curr_q = avg_quality(current)
        prev_q = avg_quality(previous)
        if curr_q is None or prev_q is None or prev_q == 0:
            return alerts
        drop_ratio = (prev_q - curr_q) / prev_q
        if drop_ratio >= drop_threshold:
            alerts.append(Alert(
                rule="quality_drop",
                severity="warning",
                title=f"Quality drop: {drop_ratio:.1%} decrease",
                detail=f"Avg quality score dropped from {prev_q:.2f} to {curr_q:.2f} ({drop_ratio:.1%} drop, threshold={drop_threshold:.0%}).",
            ))
        return alerts
