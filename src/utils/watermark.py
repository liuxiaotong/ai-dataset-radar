"""Watermark store for incremental scanning.

Persists per-source timestamps so subsequent scans only process new data.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


class WatermarkStore:
    """Persist per-source scan watermarks (latest timestamp seen).

    Storage format (JSON):
        {
          "labs": "2026-02-09T15:30:45",
          "vendors": "2026-02-09T15:30:45",
          "github": "2026-02-09T12:00:00",
          ...
          "_meta": {"last_updated": "...", "version": 1}
        }
    """

    def __init__(self, path: Path | str = "data/watermarks.json"):
        self.path = Path(path)
        self._data: dict = self._load()

    def get(self, source: str) -> str | None:
        """Get watermark timestamp for a source, or None if not set."""
        return self._data.get(source)

    def set(self, source: str, timestamp: str) -> None:
        """Set watermark timestamp for a source and persist."""
        self._data[source] = timestamp
        self._data["_meta"] = {
            "last_updated": datetime.now(timezone.utc).replace(tzinfo=None).isoformat(),
            "version": 1,
        }
        self._save()

    def get_all(self) -> dict[str, str]:
        """Get all watermarks (excluding _meta)."""
        return {k: v for k, v in self._data.items() if not k.startswith("_")}

    def reset(self, source: str | None = None) -> None:
        """Reset watermark(s). If source is None, reset all."""
        if source:
            self._data.pop(source, None)
        else:
            self._data = {}
        self._save()

    def _load(self) -> dict:
        """Load watermarks from disk. Returns empty dict on missing/corrupt file."""
        if not self.path.exists():
            return {}
        try:
            text = self.path.read_text(encoding="utf-8")
            data = json.loads(text)
            if not isinstance(data, dict):
                logger.warning("Watermark file corrupt (not a dict), resetting")
                return {}
            return data
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to load watermarks from %s: %s", self.path, e)
            return {}

    def _save(self) -> None:
        """Persist watermarks to disk."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(self._data, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
