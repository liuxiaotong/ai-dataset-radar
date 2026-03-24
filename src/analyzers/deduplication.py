"""Cross-source dataset deduplication for AI Dataset Radar.

Generates a fingerprint per dataset based on normalized name + author,
so the same dataset appearing on HuggingFace, ModelScope, etc. is counted once.
"""

import hashlib
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


def _normalize_name(name: str) -> str:
    """Lowercase, strip org prefix, collapse whitespace/punctuation."""
    name = name.lower().strip()
    if "/" in name:
        name = name.split("/", 1)[1]
    name = re.sub(r"[\s_\-\.]+", "-", name)
    name = re.sub(r+-v\d+(\.\d+)?$", "", name)
    return name


def _normalize_author(author: str) -> str:
    return (author or "").lower().strip()


def dataset_fingerprint(dataset: dict) -> str:
    """Return a 16-char hex fingerprint for a dataset dict."""
    name = _normalize_name(dataset.get("name") or dataset.get("id") or "")
    author = _normalize_author(
        dataset.get("author")
        or dataset.get("org")
        or dataset.get("owner")
        or dataset.get("organization")
        or ""
    )
    raw = f"{name}|{author}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


class CrossSourceDeduplicator:
    """Deduplicates dataset dicts that may come from multiple sources."""

    def __init__(self):
        self._seen: dict = {}
        self._duplicates: list = []

    def deduplicate(self, datasets: list) -> list:
        """Return deduplicated list. First occurrence wins."""
        self._seen.clear()
        self._duplicates.clear()

        unique = []
        for ds in datasets:
            fp = dataset_fingerprint(ds)
            if fp not in self._seen:
                self._seen[fp] = ds
                unique.append({**ds, "_fingerprint": fp})
            else:
                self._duplicates.append({**ds, "_fingerprint": fp})
                logger.debug(
                    "Duplicate dropped: %s (source=%s)",
                    ds.get("name") or ds.get("id"),
                    ds.get("source"),
                )

        logger.info(
            "Deduplication: %d input -> %d unique, %d removed",
            len(datasets), len(unique), len(self._duplicates),
        )
        return unique

    def stats(self) -> dict:
        return {
            "unique": len(self._seen),
            "duplicates_removed": len(self._duplicates),
            "duplicate_details": [
                {
                    "name": d.get("name") or d.get("id"),
                    "source": d.get("source"),
                    "fingerprint": d.get("_fingerprint"),
                }
                for d in self._duplicates
            ],
        }
