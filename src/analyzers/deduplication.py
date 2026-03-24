"""Cross-source dataset deduplication for AI Dataset Radar.

Generates a fingerprint per dataset based on normalized name + author,
then groups duplicates across sources (HuggingFace, ModelScope, etc.).
"""

import hashlib
import re
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


def _normalize(text: str) -> str:
    """Lowercase, strip org prefix, collapse whitespace and punctuation."""
    text = text.lower().strip()
    # Remove org prefix (e.g. "openai/gpt4-data" -> "gpt4-data")
    if "/" in text:
        text = text.split("/", 1)[-1]
    # Collapse non-alphanumeric to single space
    text = re.sub(r"[^a-z0-9]+", " ", text).strip()
    return text


def dataset_fingerprint(dataset: dict) -> str:
    """Return a 16-char SHA-256 fingerprint for cross-source dedup.

    Uses normalized name + normalized author as the key.
    """
    name = _normalize(dataset.get("name") or dataset.get("id") or "")
    author = _normalize(dataset.get("author") or dataset.get("org") or "")
    raw = f"{name}|{author}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


@dataclass
class DedupResult:
    total_input: int
    unique_count: int
    duplicate_groups: list[dict] = field(default_factory=list)
    deduped: list[dict] = field(default_factory=list)


def deduplicate_datasets(datasets: list[dict]) -> DedupResult:
    """Deduplicate a flat list of datasets from multiple sources.

    For each duplicate group, keeps the entry with the most fields populated.

    Args:
        datasets: Raw dataset list from combined scraper output.

    Returns:
        DedupResult with unique canonical list and duplicate group metadata.
    """
    groups: dict[str, list[dict]] = {}
    for ds in datasets:
        fp = dataset_fingerprint(ds)
        groups.setdefault(fp, []).append(ds)

    duplicate_groups = []
    deduped = []

    for fp, group in groups.items():
        sources = list({ds.get("source", "unknown") for ds in group})
        if len(group) > 1:
            duplicate_groups.append({
                "fingerprint": fp,
                "sources": sources,
                "count": len(group),
                "datasets": group,
            })
            logger.debug(
                "Dedup: fingerprint=%s appears in %d sources: %s",
                fp, len(group), sources,
            )

        # Pick canonical: prefer entry with most non-null fields
        canonical = max(group, key=lambda d: sum(1 for v in d.values() if v))
        canonical = {**canonical, "_sources": sources, "_dedup_fp": fp}
        deduped.append(canonical)

    result = DedupResult(
        total_input=len(datasets),
        unique_count=len(deduped),
        duplicate_groups=duplicate_groups,
        deduped=deduped,
    )
    logger.info(
        "Dedup complete: %d -> %d unique (%d duplicate groups)",
        result.total_input, result.unique_count, len(duplicate_groups),
    )
    return result
