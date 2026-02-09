"""Competitor Matrix — cross-reference orgs × data types."""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Optional

from .org_detector import OrgDetector
from .data_type_classifier import DataTypeClassifier, DataType

logger = logging.getLogger(__name__)


class CompetitorMatrix:
    """Build a matrix of organizations vs dataset categories.

    Aggregates datasets, GitHub repos, papers, and blog posts per org
    to produce a competitive landscape view.
    """

    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}
        self._org_detector = OrgDetector(config)
        self._classifier = DataTypeClassifier(config)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(
        self,
        datasets: list[dict] | None = None,
        github_activity: list[dict] | None = None,
        papers: list[dict] | None = None,
        blog_posts: list[dict] | None = None,
    ) -> dict:
        """Build the competitor matrix from all available data sources.

        Returns
        -------
        dict with keys:
            matrix      – {org: {category: count, ...}, ...}
            rankings    – {category: [(org, count), ...], ...}
            top_orgs    – [(org, total), ...] sorted desc
            org_details – {org: {datasets: N, repos: N, papers: N, blogs: N}}
        """
        datasets = datasets or []
        github_activity = github_activity or []
        papers = papers or []
        blog_posts = blog_posts or []

        matrix: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        org_details: dict[str, dict[str, int]] = defaultdict(
            lambda: {"datasets": 0, "repos": 0, "papers": 0, "blogs": 0}
        )

        # --- Datasets ---
        for ds in datasets:
            org = self._detect_org(ds)
            if not org:
                continue
            types = self._classifier.classify(ds)
            primary = types[0].value if types else "other"
            matrix[org][primary] += 1
            org_details[org]["datasets"] += 1

        # --- GitHub repos ---
        for org_entry in github_activity:
            org_name = org_entry.get("org", "")
            org_key = self._org_detector.detect_from_author(org_name) or org_name.lower()
            repos = org_entry.get("repos_updated", [])
            if not repos:
                continue
            org_details[org_key]["repos"] += len(repos)
            for repo in repos:
                # Count repos as contributing to "code" category
                matrix[org_key]["code"] += 1

        # --- Papers ---
        for paper in papers:
            authors = paper.get("authors", [])
            org = None
            for author in authors[:5]:
                org = self._org_detector.detect_from_text(str(author))
                if org:
                    break
            if not org:
                org = self._org_detector.detect_from_text(paper.get("title", ""))
            if not org:
                continue
            org_details[org]["papers"] += 1
            cat = paper.get("category", "other")
            matrix[org][cat] += 1

        # --- Blog posts ---
        for source_entry in blog_posts:
            source_name = source_entry.get("source", "")
            org = self._org_detector.detect_from_text(source_name)
            articles = source_entry.get("articles", [])
            if not org or not articles:
                continue
            org_details[org]["blogs"] += len(articles)

        # Build rankings per category
        rankings: dict[str, list[tuple[str, int]]] = {}
        all_categories = set()
        for org_cats in matrix.values():
            all_categories.update(org_cats.keys())

        for cat in sorted(all_categories):
            ranked = [(org, counts[cat]) for org, counts in matrix.items() if cat in counts]
            ranked.sort(key=lambda x: x[1], reverse=True)
            rankings[cat] = ranked

        # Top orgs by total activity
        top_orgs = []
        for org in matrix:
            total = sum(matrix[org].values())
            top_orgs.append((org, total))
        top_orgs.sort(key=lambda x: x[1], reverse=True)

        return {
            "matrix": {org: dict(cats) for org, cats in matrix.items()},
            "rankings": rankings,
            "top_orgs": top_orgs,
            "org_details": dict(org_details),
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _detect_org(self, dataset: dict) -> str | None:
        """Detect organization from a dataset dict."""
        author = dataset.get("author", "")
        org = self._org_detector.detect_from_author(author)
        if org:
            return org
        desc = dataset.get("description", "")
        return self._org_detector.detect_from_text(desc)
