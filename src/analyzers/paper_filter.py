"""Paper filter for RLHF and data annotation related papers.

Tightened filtering to only keep papers directly related to:
- RLHF / preference learning methodology
- Data annotation and collection methods
- Training data quality and curation
"""

import re
from typing import Optional


# Core keywords - papers must contain at least one
REQUIRED_KEYWORDS = [
    # RLHF related
    "human feedback", "rlhf", "rlaif", "reinforcement learning from human",

    # Preference learning
    "preference learning", "preference data", "preference optimization",
    "dpo", "direct preference", "pairwise comparison", "human preference",

    # Reward models
    "reward model", "reward learning", "reward signal",

    # Annotation methods
    "annotation", "labeling", "data collection", "crowdsourcing",
    "annotator", "human evaluation", "annotation guideline",
    "inter-annotator", "annotation quality",

    # SFT / Instruction
    "instruction tuning", "instruction following", "supervised fine-tuning",
    "instruction dataset", "sft data",

    # Data quality
    "data quality", "data curation", "data filtering",
    "synthetic data generation", "data augmentation for llm",
    "data mixture", "training data"
]

# Exclude patterns - if matched without strong signals, exclude
EXCLUDE_PATTERNS = [
    r"we evaluate on .{0,30} benchmark",
    r"we fine-tune .{0,30} on",
    r"we use .{0,30} dataset",
    r"evaluated on .{0,30} benchmarks?",
    r"trained on .{0,30} dataset",
]

# Strong signals - keep paper even if exclude patterns match
STRONG_SIGNALS = [
    "we release", "we collect", "we annotate", "we create",
    "we introduce a dataset", "we present a dataset",
    "annotation process", "data collection process",
    "our dataset", "our benchmark"
]

# Bonus signals for scoring
BONUS_SIGNALS = {
    "open source": 2,
    "release": 2,
    "dataset": 2,
    "annotation guideline": 3,
    "we collect": 2,
    "we annotate": 2,
    "human annotator": 2,
    "labeling cost": 3,
    "annotation cost": 3,
    "inter-annotator agreement": 3,
    "annotation quality": 2,
    "crowdworker": 2,
    "data recipe": 3,
    "data mixture": 2,
    "preference data": 3,
    "human preference": 3,
}

# Priority organizations - papers from these get bonus
PRIORITY_ORGS = [
    "openai", "anthropic", "deepmind", "meta ai", "google",
    "scale ai", "surge", "snorkel", "hugging face", "huggingface",
    "allen ai", "allenai", "ai2"
]

# Paper categories
PAPER_CATEGORIES = {
    "RLHF/偏好学习": [
        "rlhf", "human feedback", "preference", "dpo", "reward model",
        "pairwise", "chosen", "rejected"
    ],
    "数据集构建": [
        "dataset", "data collection", "we collect", "we release",
        "benchmark", "we create", "corpus"
    ],
    "标注方法论": [
        "annotation", "labeling", "crowdsourcing", "annotator",
        "inter-annotator", "annotation guideline", "label quality"
    ],
    "指令微调": [
        "instruction", "sft", "fine-tuning", "supervised",
        "instruction following", "instruction tuning"
    ],
}


class PaperFilter:
    """Filter papers to keep only RLHF/annotation related ones."""

    def __init__(self, config: dict = None):
        """Initialize paper filter.

        Args:
            config: Optional configuration dict.
        """
        self.config = config or {}
        arxiv_config = self.config.get("arxiv", {})
        self.max_papers = arxiv_config.get("max_papers", 15)

    def filter_papers(self, papers: list[dict]) -> list[dict]:
        """Filter papers to only relevant ones.

        Args:
            papers: List of paper dicts.

        Returns:
            Filtered and scored list of papers.
        """
        results = []

        for paper in papers:
            title = paper.get("title", "")
            abstract = paper.get("summary", "") or paper.get("abstract", "")
            text = f"{title} {abstract}".lower()

            # Must contain at least one required keyword
            has_required = any(kw in text for kw in REQUIRED_KEYWORDS)
            if not has_required:
                continue

            # Check exclude patterns
            is_excluded = any(
                re.search(pattern, text) for pattern in EXCLUDE_PATTERNS
            )

            # Check for strong signals that override exclusion
            has_strong_signal = any(signal in text for signal in STRONG_SIGNALS)

            if is_excluded and not has_strong_signal:
                continue

            # Calculate relevance score
            score = self._calculate_score(paper, text)
            paper["relevance_score"] = score

            # Categorize paper
            paper["category"] = self.categorize_paper(paper)

            results.append(paper)

        # Sort by score and limit
        results.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
        return results[:self.max_papers]

    def _calculate_score(self, paper: dict, text: str) -> int:
        """Calculate relevance score for a paper.

        Args:
            paper: Paper dict.
            text: Combined title and abstract text (lowercase).

        Returns:
            Relevance score.
        """
        score = 0

        # Core keyword hits
        for kw in REQUIRED_KEYWORDS:
            if kw in text:
                score += 1

        # Bonus signals
        for signal, bonus in BONUS_SIGNALS.items():
            if signal in text:
                score += bonus

        # Organization bonus
        authors = " ".join(paper.get("authors", [])).lower()
        affiliations = paper.get("affiliations", "")
        if isinstance(affiliations, list):
            affiliations = " ".join(affiliations)
        affiliations = affiliations.lower()
        org_text = f"{authors} {affiliations}"

        for org in PRIORITY_ORGS:
            if org in org_text:
                score += 3
                break

        return score

    def categorize_paper(self, paper: dict) -> str:
        """Categorize a paper into one of the categories.

        Args:
            paper: Paper dict.

        Returns:
            Category string.
        """
        title = paper.get("title", "")
        abstract = paper.get("summary", "") or paper.get("abstract", "")
        text = f"{title} {abstract}".lower()

        # Score each category
        category_scores = {}
        for cat, keywords in PAPER_CATEGORIES.items():
            score = sum(1 for kw in keywords if kw in text)
            if score > 0:
                category_scores[cat] = score

        if not category_scores:
            return "其他相关"

        # Return highest scoring category
        return max(category_scores.items(), key=lambda x: x[1])[0]

    def get_paper_highlight(self, paper: dict) -> str:
        """Extract a highlight/key point from the paper.

        Args:
            paper: Paper dict.

        Returns:
            Short highlight string.
        """
        abstract = paper.get("summary", "") or paper.get("abstract", "")
        text = abstract.lower()

        # Look for key phrases
        highlights = []

        if "we release" in text or "we introduce" in text:
            highlights.append("发布新数据/方法")
        if "annotation" in text:
            highlights.append("标注方法")
        if "human feedback" in text or "rlhf" in text:
            highlights.append("RLHF")
        if "preference" in text:
            highlights.append("偏好学习")
        if "reward model" in text:
            highlights.append("奖励模型")
        if "benchmark" in text:
            highlights.append("评估基准")

        if highlights:
            return ", ".join(highlights[:2])

        return "-"

    def extract_org(self, paper: dict) -> str:
        """Extract organization from paper authors.

        Args:
            paper: Paper dict.

        Returns:
            Organization name or "-".
        """
        authors = " ".join(paper.get("authors", [])).lower()

        for org in PRIORITY_ORGS:
            if org in authors:
                return org.title()

        return "-"
