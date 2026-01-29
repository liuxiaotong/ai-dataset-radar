"""Dataset Value Scoring System.

Combines multiple signals to produce a comprehensive value score (0-100)
for each dataset, helping prioritize which datasets to investigate.

Scoring Rules:
- SOTA model usage: +30 points
- Citation monthly growth > 10: +20 points
- Used by 3+ models: +20 points
- From top institution: +15 points
- Has paper + code: +10 points
- Data scale > 10GB: +5 points
"""

from datetime import datetime
from typing import Optional
import re


class ValueScorer:
    """Score datasets by their potential value."""

    # Top institutions that produce high-quality datasets
    TOP_INSTITUTIONS = [
        # US Tech Giants
        "google", "deepmind", "meta", "fair", "openai", "anthropic",
        "microsoft", "nvidia", "apple", "amazon", "aws",
        # US Academia
        "stanford", "berkeley", "mit", "cmu", "princeton", "harvard",
        "caltech", "cornell", "washington", "illinois",
        # Chinese Tech
        "bytedance", "alibaba", "damo", "tencent", "baidu", "huawei",
        # Chinese Academia
        "shanghai ai lab", "tsinghua", "peking", "zhejiang", "fudan",
        "chinese academy", "cuhk", "hkust",
        # European
        "deepmind", "inria", "mpi", "eth zurich", "oxford", "cambridge",
        # AI Labs
        "eleuther", "laion", "together", "stability", "cohere",
    ]

    # Score weights
    WEIGHTS = {
        "sota_usage": 30,
        "citation_growth": 20,
        "model_usage": 20,
        "top_institution": 15,
        "paper_and_code": 10,
        "large_scale": 5,
    }

    def __init__(
        self,
        citation_growth_threshold: float = 10.0,
        model_usage_threshold: int = 3,
        scale_threshold_gb: float = 10.0,
    ):
        """Initialize the value scorer.

        Args:
            citation_growth_threshold: Monthly citation growth to trigger bonus.
            model_usage_threshold: Minimum models using dataset for bonus.
            scale_threshold_gb: Dataset size threshold for scale bonus.
        """
        self.citation_growth_threshold = citation_growth_threshold
        self.model_usage_threshold = model_usage_threshold
        self.scale_threshold_gb = scale_threshold_gb

        # Build institution patterns for matching
        self._institution_patterns = [
            re.compile(rf"\b{inst}\b", re.IGNORECASE)
            for inst in self.TOP_INSTITUTIONS
        ]

    def score_dataset(
        self,
        dataset_name: str,
        sota_model_count: int = 0,
        citation_monthly_growth: float = 0.0,
        model_usage_count: int = 0,
        institution: Optional[str] = None,
        authors: Optional[list[str]] = None,
        has_paper: bool = False,
        has_code: bool = False,
        paper_url: Optional[str] = None,
        code_url: Optional[str] = None,
        size_gb: Optional[float] = None,
    ) -> dict:
        """Calculate value score for a dataset.

        Args:
            dataset_name: Name of the dataset.
            sota_model_count: Number of SOTA models using this dataset.
            citation_monthly_growth: Average monthly citation growth rate.
            model_usage_count: Number of models trained on this dataset.
            institution: Detected institution name.
            authors: List of author names.
            has_paper: Whether dataset has an associated paper.
            has_code: Whether dataset has associated code.
            paper_url: URL to the paper.
            code_url: URL to the code repository.
            size_gb: Dataset size in gigabytes.

        Returns:
            Dictionary with score breakdown and total.
        """
        scores = {
            "sota_usage": 0,
            "citation_growth": 0,
            "model_usage": 0,
            "top_institution": 0,
            "paper_and_code": 0,
            "large_scale": 0,
        }

        # SOTA model usage (+30)
        if sota_model_count > 0:
            scores["sota_usage"] = min(self.WEIGHTS["sota_usage"], sota_model_count * 10)

        # Citation growth (+20)
        if citation_monthly_growth >= self.citation_growth_threshold:
            scores["citation_growth"] = self.WEIGHTS["citation_growth"]
        elif citation_monthly_growth > 0:
            # Partial credit
            ratio = citation_monthly_growth / self.citation_growth_threshold
            scores["citation_growth"] = int(self.WEIGHTS["citation_growth"] * min(ratio, 1.0))

        # Model usage (+20)
        if model_usage_count >= self.model_usage_threshold:
            scores["model_usage"] = self.WEIGHTS["model_usage"]
        elif model_usage_count > 0:
            # Partial credit
            ratio = model_usage_count / self.model_usage_threshold
            scores["model_usage"] = int(self.WEIGHTS["model_usage"] * ratio)

        # Top institution (+15)
        is_top_institution = self._check_top_institution(institution, authors)
        if is_top_institution:
            scores["top_institution"] = self.WEIGHTS["top_institution"]

        # Paper + code (+10)
        if has_paper or paper_url:
            scores["paper_and_code"] += self.WEIGHTS["paper_and_code"] // 2
        if has_code or code_url:
            scores["paper_and_code"] += self.WEIGHTS["paper_and_code"] // 2

        # Large scale (+5)
        if size_gb and size_gb >= self.scale_threshold_gb:
            scores["large_scale"] = self.WEIGHTS["large_scale"]

        total_score = sum(scores.values())

        return {
            "dataset_name": dataset_name,
            "total_score": total_score,
            "score_breakdown": scores,
            "is_top_institution": is_top_institution,
            "signals": {
                "sota_model_count": sota_model_count,
                "citation_monthly_growth": citation_monthly_growth,
                "model_usage_count": model_usage_count,
                "institution": institution,
                "has_paper": has_paper or bool(paper_url),
                "has_code": has_code or bool(code_url),
                "size_gb": size_gb,
            },
        }

    def _check_top_institution(
        self,
        institution: Optional[str],
        authors: Optional[list[str]],
    ) -> bool:
        """Check if dataset is from a top institution.

        Args:
            institution: Explicit institution name.
            authors: List of author names.

        Returns:
            True if from a top institution.
        """
        # Check explicit institution
        if institution:
            for pattern in self._institution_patterns:
                if pattern.search(institution):
                    return True

        # Check author affiliations
        if authors:
            author_text = " ".join(authors)
            for pattern in self._institution_patterns:
                if pattern.search(author_text):
                    return True

        return False

    def batch_score(self, datasets: list[dict]) -> list[dict]:
        """Score multiple datasets and return sorted results.

        Args:
            datasets: List of dataset info dictionaries.

        Returns:
            Sorted list of scored datasets (highest first).
        """
        scored = []

        for ds in datasets:
            result = self.score_dataset(
                dataset_name=ds.get("name", ds.get("dataset_name", "Unknown")),
                sota_model_count=ds.get("sota_model_count", 0),
                citation_monthly_growth=ds.get("citation_monthly_growth", 0),
                model_usage_count=ds.get("model_usage_count", ds.get("usage_count", 0)),
                institution=ds.get("institution"),
                authors=ds.get("authors"),
                has_paper=ds.get("has_paper", False),
                has_code=ds.get("has_code", False),
                paper_url=ds.get("paper_url"),
                code_url=ds.get("code_url"),
                size_gb=ds.get("size_gb"),
            )

            # Merge original data
            result["original_data"] = ds
            scored.append(result)

        # Sort by total score descending
        scored.sort(key=lambda x: x["total_score"], reverse=True)

        return scored

    def filter_by_score(
        self,
        scored_datasets: list[dict],
        min_score: int = 0,
    ) -> list[dict]:
        """Filter datasets by minimum score.

        Args:
            scored_datasets: List of scored dataset dictionaries.
            min_score: Minimum score threshold.

        Returns:
            Filtered list of datasets.
        """
        return [ds for ds in scored_datasets if ds["total_score"] >= min_score]


class ValueAggregator:
    """Aggregate value signals from multiple sources."""

    def __init__(self):
        """Initialize the aggregator."""
        self.scorer = ValueScorer()
        self._datasets = {}

    def add_semantic_scholar_data(self, papers: list[dict]) -> None:
        """Add citation data from Semantic Scholar.

        Args:
            papers: List of paper dictionaries with citation info.
        """
        for paper in papers:
            dataset_name = paper.get("dataset_name")
            if not dataset_name:
                continue

            key = self._normalize_name(dataset_name)
            if key not in self._datasets:
                self._datasets[key] = {"name": dataset_name}

            self._datasets[key]["citation_count"] = paper.get("citation_count", 0)
            self._datasets[key]["citation_monthly_growth"] = paper.get(
                "citation_monthly_growth", 0
            )
            self._datasets[key]["paper_url"] = paper.get("url")
            self._datasets[key]["has_paper"] = True
            self._datasets[key]["authors"] = paper.get("authors", [])

    def add_model_card_data(self, model_results: dict) -> None:
        """Add model usage data from model card analysis.

        Args:
            model_results: Results from ModelCardAnalyzer.
        """
        for ds in model_results.get("valuable_datasets", []):
            key = self._normalize_name(ds.get("name", ""))
            if not key:
                continue

            if key not in self._datasets:
                self._datasets[key] = {"name": ds.get("name")}

            self._datasets[key]["model_usage_count"] = ds.get("usage_count", 0)
            self._datasets[key]["total_model_downloads"] = ds.get(
                "total_model_downloads", 0
            )

    def add_sota_data(self, sota_results: dict) -> None:
        """Add SOTA usage data from Papers with Code.

        Args:
            sota_results: Results from PwCSOTAScraper.
        """
        for ds in sota_results.get("ranked_datasets", []):
            key = self._normalize_name(ds.get("name", ""))
            if not key:
                continue

            if key not in self._datasets:
                self._datasets[key] = {"name": ds.get("name")}

            self._datasets[key]["sota_model_count"] = ds.get("sota_model_count", 0)
            self._datasets[key]["areas"] = ds.get("areas", [])
            self._datasets[key]["code_url"] = ds.get("url")

    def add_huggingface_data(self, datasets: list[dict]) -> None:
        """Add metadata from HuggingFace.

        Args:
            datasets: List of HF dataset dictionaries.
        """
        for ds in datasets:
            name = ds.get("name", ds.get("id", ""))
            key = self._normalize_name(name)
            if not key:
                continue

            if key not in self._datasets:
                self._datasets[key] = {"name": name}

            self._datasets[key]["downloads"] = ds.get("downloads", 0)
            self._datasets[key]["likes"] = ds.get("likes", 0)
            self._datasets[key]["author"] = ds.get("author")
            self._datasets[key]["institution"] = ds.get("author")  # Use author as proxy

    def _normalize_name(self, name: str) -> str:
        """Normalize dataset name for matching.

        Args:
            name: Raw dataset name.

        Returns:
            Normalized name.
        """
        if not name:
            return ""
        # Lowercase, remove special chars, normalize spaces
        normalized = name.lower()
        normalized = re.sub(r"[^\w\s]", "", normalized)
        normalized = re.sub(r"\s+", "_", normalized)
        return normalized

    def get_scored_datasets(self, min_score: int = 0) -> list[dict]:
        """Get all datasets with value scores.

        Args:
            min_score: Minimum score threshold.

        Returns:
            Sorted list of scored datasets.
        """
        datasets = list(self._datasets.values())
        scored = self.scorer.batch_score(datasets)

        if min_score > 0:
            scored = self.scorer.filter_by_score(scored, min_score)

        return scored

    def generate_report(self, min_score: int = 40) -> str:
        """Generate a comprehensive report.

        Args:
            min_score: Minimum score to include in report.

        Returns:
            Formatted report string.
        """
        scored = self.get_scored_datasets(min_score)

        lines = [
            "=" * 70,
            "High-Value Dataset Report",
            "=" * 70,
            "",
            f"Total datasets analyzed: {len(self._datasets)}",
            f"Datasets with score >= {min_score}: {len(scored)}",
            "",
            "Score Legend:",
            "  SOTA usage: +30 | Citation growth: +20 | Model usage: +20",
            "  Top institution: +15 | Paper+Code: +10 | Large scale: +5",
            "",
            "-" * 70,
            f"{'Rank':<5} {'Dataset':<35} {'Score':<6} {'Key Signals'}",
            "-" * 70,
        ]

        for i, ds in enumerate(scored[:30], 1):
            name = ds["dataset_name"][:33]
            score = ds["total_score"]

            # Build signal summary
            signals = []
            breakdown = ds["score_breakdown"]
            if breakdown["sota_usage"] > 0:
                signals.append(f"SOTA:{ds['signals']['sota_model_count']}")
            if breakdown["model_usage"] > 0:
                signals.append(f"Models:{ds['signals']['model_usage_count']}")
            if breakdown["citation_growth"] > 0:
                signals.append(f"Cite/mo:{ds['signals']['citation_monthly_growth']:.1f}")
            if breakdown["top_institution"] > 0:
                signals.append("TopInst")
            if breakdown["paper_and_code"] > 0:
                parts = []
                if ds["signals"]["has_paper"]:
                    parts.append("P")
                if ds["signals"]["has_code"]:
                    parts.append("C")
                signals.append(f"[{'+'.join(parts)}]")

            signal_str = ", ".join(signals)

            lines.append(f"{i:<5} {name:<35} {score:<6} {signal_str}")

        lines.append("")
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

        return "\n".join(lines)
