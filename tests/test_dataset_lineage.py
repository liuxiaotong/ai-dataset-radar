"""Tests for DatasetLineageTracker analyzer."""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from analyzers.dataset_lineage import DatasetLineageTracker


@pytest.fixture
def tracker():
    return DatasetLineageTracker()


# ─── Derivation detection ──────────────────────────────────────────────────

class TestDerivation:
    def test_based_on(self, tracker):
        datasets = [
            {"id": "user/child", "description": "based on openai/gsm8k"},
        ]
        result = tracker.analyze(datasets)
        assert any(e[0] == "user/child" and e[1] == "openai/gsm8k" for e in result["edges"])

    def test_derived_from(self, tracker):
        datasets = [
            {"id": "user/ds", "description": "This dataset is derived from squad_v2"},
        ]
        result = tracker.analyze(datasets)
        assert any(e[1] == "squad_v2" and e[2] == "derived_from" for e in result["edges"])

    def test_fine_tuned_on(self, tracker):
        datasets = [
            {"id": "user/ft", "description": "fine-tuned on databricks/dolly-15k"},
        ]
        result = tracker.analyze(datasets)
        assert any(e[1] == "databricks/dolly-15k" for e in result["edges"])

    def test_subset_of(self, tracker):
        datasets = [
            {"id": "user/small", "description": "A filtered subset of openai/webgpt_comparisons"},
        ]
        result = tracker.analyze(datasets)
        assert any("openai/webgpt_comparisons" in e[1] for e in result["edges"])

    def test_no_derivation(self, tracker):
        datasets = [
            {"id": "user/original", "description": "A completely new dataset we created"},
        ]
        result = tracker.analyze(datasets)
        assert len(result["edges"]) == 0

    def test_case_insensitive(self, tracker):
        datasets = [
            {"id": "user/ds", "description": "Based On some_dataset for training"},
        ]
        result = tracker.analyze(datasets)
        assert any(e[2] == "derived_from" for e in result["edges"])


# ─── Version chain detection ───────────────────────────────────────────────

class TestVersionChains:
    def test_detects_version_chain(self, tracker):
        datasets = [
            {"id": "org/mydata-v1", "description": "Version 1"},
            {"id": "org/mydata-v2", "description": "Version 2"},
            {"id": "org/mydata-v3", "description": "Version 3"},
        ]
        result = tracker.analyze(datasets)
        assert "mydata" in result["version_chains"]
        chain = result["version_chains"]["mydata"]
        assert len(chain) == 3

    def test_version_edges_created(self, tracker):
        datasets = [
            {"id": "org/data-v1", "description": ""},
            {"id": "org/data-v2", "description": ""},
        ]
        result = tracker.analyze(datasets)
        assert any(e[2] == "version_of" for e in result["edges"])

    def test_no_chain_for_single(self, tracker):
        datasets = [
            {"id": "org/data-v1", "description": "Only one version"},
        ]
        result = tracker.analyze(datasets)
        assert len(result["version_chains"]) == 0


# ─── Fork detection ────────────────────────────────────────────────────────

class TestForkDetection:
    def test_detects_forks(self, tracker):
        datasets = [
            {"id": "alice/common_dataset", "description": "Alice's version"},
            {"id": "bob/common_dataset", "description": "Bob's version"},
        ]
        result = tracker.analyze(datasets)
        assert "common_dataset" in result["fork_trees"]
        assert len(result["fork_trees"]["common_dataset"]) == 2

    def test_no_fork_same_author(self, tracker):
        datasets = [
            {"id": "alice/ds1", "description": ""},
            {"id": "alice/ds2", "description": ""},
        ]
        result = tracker.analyze(datasets)
        assert len(result["fork_trees"]) == 0


# ─── Stats and roots ───────────────────────────────────────────────────────

class TestStats:
    def test_stats_keys(self, tracker):
        result = tracker.analyze([])
        stats = result["stats"]
        assert "total_datasets" in stats
        assert "total_edges" in stats
        assert "derivation_edges" in stats
        assert "version_chains" in stats
        assert "fork_groups" in stats

    def test_root_datasets(self, tracker):
        datasets = [
            {"id": "parent/ds", "description": "Original"},
            {"id": "child/ds", "description": "based on parent/ds"},
        ]
        result = tracker.analyze(datasets)
        assert "parent/ds" in result["root_datasets"]

    def test_empty_datasets(self, tracker):
        result = tracker.analyze([])
        assert result["edges"] == []
        assert result["root_datasets"] == []
        assert result["version_chains"] == {}
        assert result["fork_trees"] == {}
        assert result["stats"]["total_datasets"] == 0
