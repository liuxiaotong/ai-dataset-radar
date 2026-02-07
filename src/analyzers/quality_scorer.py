"""Dataset quality scoring for filtering low-quality entries.

Scores each dataset on a 0-10 scale based on metadata completeness,
usage metrics, and other quality indicators.
"""

import re
from typing import Optional


class QualityScorer:
    """Score datasets by quality (0-10 scale).

    Scoring dimensions:
    - Has README/description > 100 chars: +2
    - Has downloads > 10: +1
    - Has downloads > 1000: +2
    - Has explicit license: +1
    - Has tags/tasks defined: +1
    - Has associated paper: +2
    - Author is known institution: +1
    """

    # Known institutions for bonus scoring
    KNOWN_INSTITUTIONS = [
        # Tech companies
        "alibaba",
        "bytedance",
        "baidu",
        "tencent",
        "huawei",
        "google",
        "meta",
        "microsoft",
        "openai",
        "anthropic",
        "nvidia",
        "apple",
        "amazon",
        "aws",
        # AI labs
        "deepseek",
        "yi",
        "zhipu",
        "moonshot",
        "01-ai",
        "minimax",
        "huggingface",
        "stability",
        "cohere",
        "mistral",
        # Research institutions
        "thudm",
        "fudan",
        "sjtu",
        "pku",
        "thu",
        "nju",
        "stanford",
        "berkeley",
        "mit",
        "cmu",
        "princeton",
        # Organizations
        "fair",
        "deepmind",
        "brain",
        "research",
    ]

    def __init__(self, config: Optional[dict] = None):
        """Initialize the quality scorer.

        Args:
            config: Optional configuration dict.
        """
        config = config or {}
        scorer_config = config.get("quality_filter", {}).get("scoring", {})

        # Scoring weights (can be customized via config)
        self.weights = {
            "description": scorer_config.get("description", 2),
            "downloads_low": scorer_config.get("downloads_low", 1),
            "downloads_high": scorer_config.get("downloads_high", 2),
            "license": scorer_config.get("license", 1),
            "tags": scorer_config.get("tags", 1),
            "paper": scorer_config.get("paper", 2),
            "institution": scorer_config.get("institution", 1),
        }

        # Thresholds
        self.description_min_length = scorer_config.get("description_min_length", 100)
        self.downloads_low_threshold = scorer_config.get("downloads_low_threshold", 10)
        self.downloads_high_threshold = scorer_config.get("downloads_high_threshold", 1000)

        # Build institution patterns
        self._institution_patterns = [
            re.compile(rf"\b{inst}\b", re.IGNORECASE) for inst in self.KNOWN_INSTITUTIONS
        ]

    def score_dataset(self, dataset: dict) -> dict:
        """Calculate quality score for a dataset.

        Args:
            dataset: Dataset dictionary with metadata.

        Returns:
            Scoring result with breakdown.
        """
        scores = {
            "description": 0,
            "downloads_low": 0,
            "downloads_high": 0,
            "license": 0,
            "tags": 0,
            "paper": 0,
            "institution": 0,
        }

        # 1. Description/README quality (+2)
        description = self._get_description(dataset)
        if len(description) >= self.description_min_length:
            scores["description"] = self.weights["description"]

        # 2. Downloads (+1 for >10, +2 more for >1000)
        downloads = dataset.get("downloads", 0) or 0
        if downloads > self.downloads_low_threshold:
            scores["downloads_low"] = self.weights["downloads_low"]
        if downloads > self.downloads_high_threshold:
            scores["downloads_high"] = self.weights["downloads_high"]

        # 3. License (+1)
        if self._has_license(dataset):
            scores["license"] = self.weights["license"]

        # 4. Tags/task categories (+1)
        if self._has_tags(dataset):
            scores["tags"] = self.weights["tags"]

        # 5. Associated paper (+2)
        if self._has_paper(dataset):
            scores["paper"] = self.weights["paper"]

        # 6. Known institution author (+1)
        if self._is_known_institution(dataset):
            scores["institution"] = self.weights["institution"]

        total_score = sum(scores.values())

        return {
            "dataset_name": dataset.get("name", dataset.get("id", "unknown")),
            "total_score": total_score,
            "max_score": sum(self.weights.values()),
            "score_breakdown": scores,
            "quality_level": self._get_quality_level(total_score),
        }

    def _get_description(self, dataset: dict) -> str:
        """Extract description from dataset."""
        # Try multiple fields
        description = dataset.get("description", "") or ""
        if not description:
            description = dataset.get("readme", "") or ""
        if not description:
            card_data = dataset.get("card_data", {}) or {}
            description = card_data.get("description", "") or ""
        return description.strip()

    def _has_license(self, dataset: dict) -> bool:
        """Check if dataset has license information."""
        # Direct license field
        if dataset.get("license"):
            return True

        # Check card_data
        card_data = dataset.get("card_data", {}) or {}
        if card_data.get("license"):
            return True

        # Check tags for license
        tags = dataset.get("tags", []) or []
        for tag in tags:
            if isinstance(tag, str) and tag.startswith("license:"):
                return True

        return False

    def _has_tags(self, dataset: dict) -> bool:
        """Check if dataset has meaningful tags."""
        tags = dataset.get("tags", []) or []

        # Filter out generic tags
        meaningful_tags = [
            t
            for t in tags
            if isinstance(t, str)
            and (
                t.startswith("task_categories:")
                or t.startswith("task_ids:")
                or t.startswith("language:")
                or t.startswith("size_categories:")
            )
        ]

        if meaningful_tags:
            return True

        # Check card_data
        card_data = dataset.get("card_data", {}) or {}
        if card_data.get("task_categories") or card_data.get("task_ids"):
            return True

        return False

    def _has_paper(self, dataset: dict) -> bool:
        """Check if dataset has associated paper."""
        # Direct paper URL
        if dataset.get("paper_url") or dataset.get("arxiv_url"):
            return True

        # Check card_data
        card_data = dataset.get("card_data", {}) or {}
        if card_data.get("paper") or card_data.get("arxiv"):
            return True

        # Check description for arxiv links
        description = self._get_description(dataset)
        if "arxiv.org" in description.lower():
            return True

        return False

    def _is_known_institution(self, dataset: dict) -> bool:
        """Check if author is from known institution."""
        author = dataset.get("author", "") or ""

        for pattern in self._institution_patterns:
            if pattern.search(author):
                return True

        return False

    def _get_quality_level(self, score: int) -> str:
        """Get quality level label from score."""
        if score >= 8:
            return "high"
        elif score >= 5:
            return "medium"
        elif score >= 2:
            return "low"
        else:
            return "very_low"

    def batch_score(self, datasets: list[dict]) -> list[dict]:
        """Score multiple datasets.

        Args:
            datasets: List of dataset dictionaries.

        Returns:
            List of scoring results.
        """
        return [self.score_dataset(ds) for ds in datasets]

    def filter_by_quality(
        self,
        datasets: list[dict],
        min_score: int = 2,
    ) -> tuple[list[dict], list[dict]]:
        """Filter datasets by quality score.

        Args:
            datasets: List of datasets to filter.
            min_score: Minimum quality score to keep.

        Returns:
            Tuple of (passed_datasets, filtered_out_datasets).
        """
        passed = []
        filtered = []

        for ds in datasets:
            result = self.score_dataset(ds)
            ds_with_score = {**ds, "_quality_score": result}

            if result["total_score"] >= min_score:
                passed.append(ds_with_score)
            else:
                filtered.append(ds_with_score)

        return passed, filtered

    def get_quality_stars(self, score: int) -> str:
        """Convert score to star rating display.

        Args:
            score: Quality score (0-10).

        Returns:
            Star rating string (e.g., "⭐⭐⭐").
        """
        max_stars = 4
        stars = min(max_stars, (score + 1) // 3)  # 0-2=0, 3-5=1, 6-8=2, 9+=3, max 4
        return "⭐" * max(1, stars) if score > 0 else "-"
