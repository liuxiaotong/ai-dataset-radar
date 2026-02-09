"""Tests for WatermarkStore (incremental scanning)."""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from utils.watermark import WatermarkStore


# ---------------------------------------------------------------------------
# Basic get/set
# ---------------------------------------------------------------------------

class TestGetSet:
    def test_get_missing_returns_none(self, tmp_path):
        store = WatermarkStore(tmp_path / "wm.json")
        assert store.get("labs") is None

    def test_set_and_get(self, tmp_path):
        store = WatermarkStore(tmp_path / "wm.json")
        store.set("labs", "2026-02-09T15:30:45")
        assert store.get("labs") == "2026-02-09T15:30:45"

    def test_set_overwrites(self, tmp_path):
        store = WatermarkStore(tmp_path / "wm.json")
        store.set("labs", "2026-02-08T00:00:00")
        store.set("labs", "2026-02-09T12:00:00")
        assert store.get("labs") == "2026-02-09T12:00:00"

    def test_multiple_sources(self, tmp_path):
        store = WatermarkStore(tmp_path / "wm.json")
        store.set("labs", "2026-02-09T10:00:00")
        store.set("github", "2026-02-09T11:00:00")
        store.set("blogs", "2026-02-09T12:00:00")
        assert store.get("labs") == "2026-02-09T10:00:00"
        assert store.get("github") == "2026-02-09T11:00:00"
        assert store.get("blogs") == "2026-02-09T12:00:00"


# ---------------------------------------------------------------------------
# get_all
# ---------------------------------------------------------------------------

class TestGetAll:
    def test_empty(self, tmp_path):
        store = WatermarkStore(tmp_path / "wm.json")
        assert store.get_all() == {}

    def test_excludes_meta(self, tmp_path):
        store = WatermarkStore(tmp_path / "wm.json")
        store.set("labs", "2026-02-09T10:00:00")
        all_wm = store.get_all()
        assert "labs" in all_wm
        assert "_meta" not in all_wm


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset_single(self, tmp_path):
        store = WatermarkStore(tmp_path / "wm.json")
        store.set("labs", "2026-02-09T10:00:00")
        store.set("github", "2026-02-09T11:00:00")
        store.reset("labs")
        assert store.get("labs") is None
        assert store.get("github") == "2026-02-09T11:00:00"

    def test_reset_all(self, tmp_path):
        store = WatermarkStore(tmp_path / "wm.json")
        store.set("labs", "2026-02-09T10:00:00")
        store.set("github", "2026-02-09T11:00:00")
        store.reset()
        assert store.get_all() == {}

    def test_reset_nonexistent(self, tmp_path):
        store = WatermarkStore(tmp_path / "wm.json")
        store.reset("nonexistent")  # Should not raise


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_survives_reload(self, tmp_path):
        path = tmp_path / "wm.json"
        store1 = WatermarkStore(path)
        store1.set("labs", "2026-02-09T10:00:00")
        store1.set("github", "2026-02-09T11:00:00")

        store2 = WatermarkStore(path)
        assert store2.get("labs") == "2026-02-09T10:00:00"
        assert store2.get("github") == "2026-02-09T11:00:00"

    def test_file_created_on_set(self, tmp_path):
        path = tmp_path / "subdir" / "wm.json"
        assert not path.exists()
        store = WatermarkStore(path)
        store.set("labs", "2026-02-09T10:00:00")
        assert path.exists()

    def test_meta_has_last_updated(self, tmp_path):
        path = tmp_path / "wm.json"
        store = WatermarkStore(path)
        store.set("labs", "2026-02-09T10:00:00")
        raw = json.loads(path.read_text())
        assert "_meta" in raw
        assert "last_updated" in raw["_meta"]
        assert raw["_meta"]["version"] == 1


# ---------------------------------------------------------------------------
# Corrupt / missing file handling
# ---------------------------------------------------------------------------

class TestCorrupt:
    def test_missing_file(self, tmp_path):
        store = WatermarkStore(tmp_path / "does_not_exist.json")
        assert store.get("labs") is None
        assert store.get_all() == {}

    def test_corrupt_json(self, tmp_path):
        path = tmp_path / "wm.json"
        path.write_text("{invalid json", encoding="utf-8")
        store = WatermarkStore(path)
        assert store.get("labs") is None

    def test_non_dict_json(self, tmp_path):
        path = tmp_path / "wm.json"
        path.write_text("[1, 2, 3]", encoding="utf-8")
        store = WatermarkStore(path)
        assert store.get_all() == {}

    def test_recover_after_corrupt(self, tmp_path):
        path = tmp_path / "wm.json"
        path.write_text("not json", encoding="utf-8")
        store = WatermarkStore(path)
        store.set("labs", "2026-02-09T10:00:00")
        assert store.get("labs") == "2026-02-09T10:00:00"

        store2 = WatermarkStore(path)
        assert store2.get("labs") == "2026-02-09T10:00:00"
