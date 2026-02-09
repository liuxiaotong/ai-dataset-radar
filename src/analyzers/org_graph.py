"""Organization Relationship Graph — extract inter-org collaboration and competition edges."""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Optional

from .org_detector import OrgDetector

logger = logging.getLogger(__name__)

# Edge types
EDGE_SHARED_DATASET = "shared_dataset_author"
EDGE_GITHUB_FORK = "github_fork"
EDGE_CO_CITATION = "co_citation"
EDGE_BLOG_MENTION = "blog_mention"
EDGE_SHARED_TOPIC = "shared_topic"


class OrgRelationshipGraph:
    """Build a graph of relationships between AI organizations.

    Nodes are organizations; edges represent collaboration or competition signals.
    """

    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}
        self._org_detector = OrgDetector(config)

    def build(
        self,
        datasets: list[dict] | None = None,
        github_activity: list[dict] | None = None,
        papers: list[dict] | None = None,
        blog_posts: list[dict] | None = None,
    ) -> dict:
        """Build the org relationship graph.

        Returns
        -------
        dict with keys:
            nodes      – [{id, display_name, dataset_count, ...}, ...]
            edges      – [{source, target, type, weight}, ...]
            clusters   – [[org1, org2, ...], ...]  connected components
            centrality – {org: degree_centrality, ...}
        """
        datasets = datasets or []
        github_activity = github_activity or []
        papers = papers or []
        blog_posts = blog_posts or []

        edges: list[dict] = []
        node_stats: dict[str, dict] = defaultdict(lambda: {
            "dataset_count": 0, "repo_count": 0, "paper_count": 0, "blog_count": 0,
        })

        # --- Shared dataset authors ---
        # Datasets with multiple org signals in description/authors
        for ds in datasets:
            orgs = self._detect_all_orgs(ds)
            if len(orgs) >= 2:
                for i in range(len(orgs)):
                    for j in range(i + 1, len(orgs)):
                        edges.append({
                            "source": orgs[i], "target": orgs[j],
                            "type": EDGE_SHARED_DATASET,
                            "weight": 1,
                            "context": ds.get("id", ""),
                        })
            for org in orgs:
                node_stats[org]["dataset_count"] += 1

        # --- GitHub forks / shared topics ---
        org_topics: dict[str, set[str]] = defaultdict(set)
        for org_entry in github_activity:
            org_name = org_entry.get("org", "")
            org_key = self._org_detector.detect_from_author(org_name) or org_name.lower()
            repos = org_entry.get("repos_updated", [])
            node_stats[org_key]["repo_count"] += len(repos)
            for repo in repos:
                topics = repo.get("topics", [])
                for t in topics:
                    org_topics[org_key].add(t.lower())
                # Fork detection
                full_name = repo.get("full_name", "")
                if "/" in full_name:
                    fork_org = full_name.split("/")[0].lower()
                    detected = self._org_detector.detect_from_author(fork_org)
                    if detected and detected != org_key:
                        edges.append({
                            "source": org_key, "target": detected,
                            "type": EDGE_GITHUB_FORK,
                            "weight": 1,
                            "context": full_name,
                        })

        # Shared topics: connect orgs that share 3+ topics
        org_list = list(org_topics.keys())
        for i in range(len(org_list)):
            for j in range(i + 1, len(org_list)):
                shared = org_topics[org_list[i]] & org_topics[org_list[j]]
                if len(shared) >= 3:
                    edges.append({
                        "source": org_list[i], "target": org_list[j],
                        "type": EDGE_SHARED_TOPIC,
                        "weight": len(shared),
                        "context": ", ".join(sorted(shared)[:5]),
                    })

        # --- Paper co-citation ---
        for paper in papers:
            authors = paper.get("authors", [])
            paper_orgs = set()
            for author in authors[:10]:
                org = self._org_detector.detect_from_text(str(author))
                if org:
                    paper_orgs.add(org)
            org_from_title = self._org_detector.detect_from_text(paper.get("title", ""))
            if org_from_title:
                paper_orgs.add(org_from_title)

            for org in paper_orgs:
                node_stats[org]["paper_count"] += 1

            paper_orgs = sorted(paper_orgs)
            for i in range(len(paper_orgs)):
                for j in range(i + 1, len(paper_orgs)):
                    edges.append({
                        "source": paper_orgs[i], "target": paper_orgs[j],
                        "type": EDGE_CO_CITATION,
                        "weight": 1,
                        "context": paper.get("title", "")[:80],
                    })

        # --- Blog mentions ---
        for source_entry in blog_posts:
            source_name = source_entry.get("source", "")
            source_org = self._org_detector.detect_from_text(source_name)
            if not source_org:
                continue
            articles = source_entry.get("articles", [])
            node_stats[source_org]["blog_count"] += len(articles)
            for article in articles:
                text = article.get("title", "") + " " + article.get("summary", "")
                mentioned_orgs = self._detect_all_orgs_from_text(text)
                for mentioned in mentioned_orgs:
                    if mentioned != source_org:
                        edges.append({
                            "source": source_org, "target": mentioned,
                            "type": EDGE_BLOG_MENTION,
                            "weight": 1,
                            "context": article.get("title", "")[:80],
                        })

        # Deduplicate and merge edge weights
        edges = self._merge_edges(edges)

        # Build nodes list
        all_org_ids = set()
        for e in edges:
            all_org_ids.add(e["source"])
            all_org_ids.add(e["target"])
        all_org_ids.update(node_stats.keys())

        nodes = []
        for org_id in sorted(all_org_ids):
            stats = node_stats.get(org_id, {})
            nodes.append({
                "id": org_id,
                "display_name": self._org_detector.get_org_display_name(org_id),
                **stats,
            })

        # Connected components (BFS)
        clusters = self._find_clusters(all_org_ids, edges)

        # Degree centrality
        centrality = self._compute_centrality(all_org_ids, edges)

        return {
            "nodes": nodes,
            "edges": edges,
            "clusters": clusters,
            "centrality": centrality,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _detect_all_orgs(self, dataset: dict) -> list[str]:
        """Detect all organizations mentioned in a dataset."""
        orgs = set()
        author = dataset.get("author", "")
        org = self._org_detector.detect_from_author(author)
        if org:
            orgs.add(org)
        for field in ("description", "readme"):
            text = dataset.get(field, "")
            if text:
                detected = self._detect_all_orgs_from_text(text)
                orgs.update(detected)
        return sorted(orgs)

    def _detect_all_orgs_from_text(self, text: str) -> list[str]:
        """Detect all org references in a text block."""
        if not text:
            return []
        orgs = set()
        # OrgDetector.orgs is {org_key: [alias1, alias2, ...]}
        text_lower = text.lower()
        for org_key, aliases in self._org_detector.orgs.items():
            check_terms = [org_key] + list(aliases)
            for term in check_terms:
                if len(term) >= 3 and term in text_lower:
                    orgs.add(org_key)
                    break
        return sorted(orgs)

    @staticmethod
    def _merge_edges(edges: list[dict]) -> list[dict]:
        """Merge duplicate edges, summing weights."""
        merged: dict[tuple, dict] = {}
        for e in edges:
            key = (min(e["source"], e["target"]), max(e["source"], e["target"]), e["type"])
            if key in merged:
                merged[key]["weight"] += e.get("weight", 1)
            else:
                merged[key] = {
                    "source": key[0],
                    "target": key[1],
                    "type": key[2],
                    "weight": e.get("weight", 1),
                }
        return sorted(merged.values(), key=lambda x: x["weight"], reverse=True)

    @staticmethod
    def _find_clusters(nodes: set[str], edges: list[dict]) -> list[list[str]]:
        """Find connected components via BFS."""
        adj: dict[str, set[str]] = defaultdict(set)
        for e in edges:
            adj[e["source"]].add(e["target"])
            adj[e["target"]].add(e["source"])

        visited: set[str] = set()
        clusters: list[list[str]] = []
        for node in sorted(nodes):
            if node in visited:
                continue
            # BFS
            component: list[str] = []
            queue = [node]
            while queue:
                current = queue.pop(0)
                if current in visited:
                    continue
                visited.add(current)
                component.append(current)
                for neighbor in sorted(adj.get(current, set())):
                    if neighbor not in visited:
                        queue.append(neighbor)
            if component:
                clusters.append(sorted(component))

        clusters.sort(key=lambda c: len(c), reverse=True)
        return clusters

    @staticmethod
    def _compute_centrality(nodes: set[str], edges: list[dict]) -> dict[str, float]:
        """Compute simple degree centrality."""
        degree: dict[str, int] = defaultdict(int)
        for e in edges:
            degree[e["source"]] += 1
            degree[e["target"]] += 1

        n = len(nodes)
        if n <= 1:
            return {org: 0.0 for org in nodes}

        return {
            org: round(degree.get(org, 0) / (n - 1), 4)
            for org in sorted(nodes)
        }
