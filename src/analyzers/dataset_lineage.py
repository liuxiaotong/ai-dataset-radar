"""Dataset Lineage Tracker — discover derivation, versioning, and fork relationships."""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from typing import Optional

logger = logging.getLogger(__name__)

# Patterns that indicate derivation from another dataset
_DERIVATION_PATTERNS = [
    re.compile(r"(?:based on|derived from|built on|extends|extension of|fine[- ]?tuned on|trained on|subset of|filtered from|sampled from|translated from|distilled from)\s+[\"']?([A-Za-z0-9_/.-]{3,60})[\"']?", re.IGNORECASE),
    re.compile(r"(?:using|from)\s+(?:the\s+)?([A-Za-z0-9_/-]{3,60})\s+dataset", re.IGNORECASE),
]

# Pattern for version detection (dataset-v1, dataset_v2, etc.)
_VERSION_RE = re.compile(r"^(.+?)[-_]?v(\d+(?:\.\d+)*)$", re.IGNORECASE)


class DatasetLineageTracker:
    """Track derivation relationships between datasets.

    Identifies:
    - Direct derivations (dataset A is based on dataset B)
    - Version chains (dataset-v1 → dataset-v2 → dataset-v3)
    - Forks (same base name, different authors)
    """

    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}

    def analyze(self, datasets: list[dict]) -> dict:
        """Analyze lineage relationships among datasets.

        Returns
        -------
        dict with keys:
            edges          – [(child_id, parent_id, relation_type), ...]
            root_datasets  – [dataset_id, ...]   (datasets with no parents)
            version_chains – {base_name: [v1_id, v2_id, ...]}
            fork_trees     – {base_name: [author1/name, author2/name, ...]}
            stats          – summary counters
        """
        edges: list[tuple[str, str, str]] = []
        version_map: dict[str, list[tuple[str, str]]] = defaultdict(list)  # base -> [(version, id)]
        name_to_authors: dict[str, list[str]] = defaultdict(list)  # base_name -> [full_ids]

        dataset_ids = {ds.get("id", "") for ds in datasets}

        for ds in datasets:
            ds_id = ds.get("id", "")
            if not ds_id:
                continue

            # --- Derivation edges ---
            text = " ".join(filter(None, [
                ds.get("description", ""),
                ds.get("readme", ""),
                str(ds.get("card_data", "")),
            ]))
            for pattern in _DERIVATION_PATTERNS:
                for match in pattern.finditer(text):
                    parent = match.group(1).strip().rstrip(".,;)")
                    if parent and parent != ds_id and len(parent) > 3:
                        edges.append((ds_id, parent, "derived_from"))

            # --- Version detection ---
            name = ds_id.split("/")[-1] if "/" in ds_id else ds_id
            vm = _VERSION_RE.match(name)
            if vm:
                base = vm.group(1)
                version = vm.group(2)
                author = ds_id.split("/")[0] if "/" in ds_id else ""
                version_map[base].append((version, ds_id))
                # Also add an edge from newer to older if both exist
            else:
                base = name

            # --- Fork detection ---
            name_lower = name.lower()
            name_to_authors[name_lower].append(ds_id)

        # Build version chains (sorted by version string)
        version_chains: dict[str, list[str]] = {}
        for base, versions in version_map.items():
            if len(versions) >= 2:
                sorted_versions = sorted(versions, key=lambda x: x[0])
                chain = [v[1] for v in sorted_versions]
                version_chains[base] = chain
                # Add version edges
                for i in range(1, len(chain)):
                    edges.append((chain[i], chain[i - 1], "version_of"))

        # Build fork trees (same name, different authors)
        fork_trees: dict[str, list[str]] = {}
        for name, ids in name_to_authors.items():
            authors = {did.split("/")[0] for did in ids if "/" in did}
            if len(authors) >= 2:
                fork_trees[name] = sorted(ids)

        # Deduplicate edges
        edges = list(set(edges))

        # Find root datasets (no parents)
        children = {e[0] for e in edges}
        parents = {e[1] for e in edges}
        all_nodes = children | parents
        root_datasets = sorted(parents - children)

        return {
            "edges": edges,
            "root_datasets": root_datasets,
            "version_chains": version_chains,
            "fork_trees": fork_trees,
            "stats": {
                "total_datasets": len(datasets),
                "total_edges": len(edges),
                "derivation_edges": sum(1 for e in edges if e[2] == "derived_from"),
                "version_chains": len(version_chains),
                "fork_groups": len(fork_trees),
            },
        }
