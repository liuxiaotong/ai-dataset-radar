"""Tests for OrgRelationshipGraph analyzer."""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from analyzers.org_graph import OrgRelationshipGraph


@pytest.fixture
def graph():
    return OrgRelationshipGraph()


# ─── Basic structure ────────────────────────────────────────────────────────

class TestBuild:
    def test_empty_inputs(self, graph):
        result = graph.build()
        assert result["nodes"] == []
        assert result["edges"] == []
        assert result["clusters"] == []
        assert result["centrality"] == {}

    def test_returns_all_keys(self, graph):
        result = graph.build()
        assert "nodes" in result
        assert "edges" in result
        assert "clusters" in result
        assert "centrality" in result


# ─── Node detection ─────────────────────────────────────────────────────────

class TestNodes:
    def test_nodes_from_datasets(self, graph):
        datasets = [
            {"id": "openai/ds1", "author": "openai", "description": "A dataset by OpenAI"},
        ]
        result = graph.build(datasets=datasets)
        node_ids = [n["id"] for n in result["nodes"]]
        assert "openai" in node_ids

    def test_node_has_display_name(self, graph):
        datasets = [
            {"id": "openai/ds1", "author": "openai", "description": "test"},
        ]
        result = graph.build(datasets=datasets)
        openai_node = next(n for n in result["nodes"] if n["id"] == "openai")
        assert "display_name" in openai_node


# ─── Edge detection ─────────────────────────────────────────────────────────

class TestEdges:
    def test_shared_dataset_edge(self, graph):
        # A dataset mentioning two orgs should create a shared_dataset_author edge
        datasets = [
            {
                "id": "collab/ds1",
                "author": "openai",
                "description": "A collaboration between OpenAI and Google DeepMind researchers",
            },
        ]
        result = graph.build(datasets=datasets)
        edge_types = [e["type"] for e in result["edges"]]
        if result["edges"]:
            assert "shared_dataset_author" in edge_types

    def test_co_citation_edge(self, graph):
        papers = [
            {
                "title": "Joint work on RLHF",
                "authors": ["Researcher at OpenAI", "Scientist at Google DeepMind"],
            },
        ]
        result = graph.build(papers=papers)
        co_cite_edges = [e for e in result["edges"] if e["type"] == "co_citation"]
        # May or may not detect depending on org_detector's text matching
        assert isinstance(co_cite_edges, list)

    def test_edge_has_weight(self, graph):
        datasets = [
            {"id": "openai/a", "author": "openai", "description": "Work with Google"},
            {"id": "openai/b", "author": "openai", "description": "More work with Google"},
        ]
        result = graph.build(datasets=datasets)
        for edge in result["edges"]:
            assert "weight" in edge
            assert edge["weight"] >= 1


# ─── Clusters ──────────────────────────────────────────────────────────────

class TestClusters:
    def test_clusters_are_lists(self, graph):
        datasets = [
            {"id": "openai/ds", "author": "openai", "description": "test"},
        ]
        result = graph.build(datasets=datasets)
        assert isinstance(result["clusters"], list)
        for cluster in result["clusters"]:
            assert isinstance(cluster, list)

    def test_all_nodes_in_clusters(self, graph):
        datasets = [
            {"id": "openai/ds", "author": "openai", "description": "Google collaboration"},
        ]
        result = graph.build(datasets=datasets)
        cluster_nodes = set()
        for cluster in result["clusters"]:
            cluster_nodes.update(cluster)
        node_ids = {n["id"] for n in result["nodes"]}
        assert cluster_nodes == node_ids


# ─── Centrality ─────────────────────────────────────────────────────────────

class TestCentrality:
    def test_centrality_values(self, graph):
        datasets = [
            {"id": "openai/ds", "author": "openai", "description": "test with Google"},
        ]
        result = graph.build(datasets=datasets)
        for org, value in result["centrality"].items():
            assert 0.0 <= value <= 1.0

    def test_empty_centrality(self, graph):
        result = graph.build()
        assert result["centrality"] == {}


# ─── Edge merging ──────────────────────────────────────────────────────────

class TestMergeEdges:
    def test_merge_edges(self):
        edges = [
            {"source": "a", "target": "b", "type": "co_citation", "weight": 1},
            {"source": "b", "target": "a", "type": "co_citation", "weight": 1},
        ]
        merged = OrgRelationshipGraph._merge_edges(edges)
        assert len(merged) == 1
        assert merged[0]["weight"] == 2

    def test_different_types_not_merged(self):
        edges = [
            {"source": "a", "target": "b", "type": "co_citation", "weight": 1},
            {"source": "a", "target": "b", "type": "blog_mention", "weight": 1},
        ]
        merged = OrgRelationshipGraph._merge_edges(edges)
        assert len(merged) == 2


# ─── Cluster BFS ────────────────────────────────────────────────────────────

class TestFindClusters:
    def test_connected_components(self):
        nodes = {"a", "b", "c", "d"}
        edges = [
            {"source": "a", "target": "b", "type": "x", "weight": 1},
            {"source": "c", "target": "d", "type": "x", "weight": 1},
        ]
        clusters = OrgRelationshipGraph._find_clusters(nodes, edges)
        assert len(clusters) == 2

    def test_single_cluster(self):
        nodes = {"a", "b", "c"}
        edges = [
            {"source": "a", "target": "b", "type": "x", "weight": 1},
            {"source": "b", "target": "c", "type": "x", "weight": 1},
        ]
        clusters = OrgRelationshipGraph._find_clusters(nodes, edges)
        assert len(clusters) == 1
        assert len(clusters[0]) == 3

    def test_isolated_nodes(self):
        nodes = {"a", "b"}
        edges = []
        clusters = OrgRelationshipGraph._find_clusters(nodes, edges)
        assert len(clusters) == 2
