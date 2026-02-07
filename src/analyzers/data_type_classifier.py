"""Data type classifier for categorizing datasets by training purpose.

Enhanced v5 classifier with:
- Multi-dimensional matching (name, description, README, features)
- Multi-label classification support
- Better handling of synthetic/versioned datasets
"""

import re
from enum import Enum


class DataType(Enum):
    """Dataset type categories for post-training data."""

    RLHF_PREFERENCE = "rlhf_preference"  # Preference/comparison data
    REWARD_MODEL = "reward_model"  # Reward model training data
    SFT_INSTRUCTION = "sft_instruction"  # Instruction fine-tuning data
    CODE = "code"  # Code generation/execution
    AGENT_TOOL = "agent_tool"  # Agent/tool use
    RL_ENVIRONMENT = "rl_environment"  # RL environment/trajectory
    SYNTHETIC = "synthetic"  # Synthetic/distilled data
    MULTIMODAL = "multimodal"  # Multimodal
    MULTILINGUAL = "multilingual"  # Multilingual
    EVALUATION = "evaluation"  # Evaluation/benchmark
    OTHER = "other"


# Classification rules with keywords, field patterns, and name patterns
CLASSIFICATION_RULES = {
    DataType.RLHF_PREFERENCE: {
        "keywords": [
            "preference",
            "comparison",
            "ranking",
            "chosen",
            "rejected",
            "human feedback",
            "dpo",
            "rlhf",
            "rlaif",
            "pairwise",
            "better",
            "worse",
            "preferred",
            "human preference",
            "preference learning",
            "preference data",
        ],
        "field_patterns": ["chosen", "rejected", "preference", "rating", "score"],
        "name_patterns": [r"pref", r"dpo", r"rlhf", r"comparison", r"ranking"],
        "tags": ["dpo", "rlhf", "preference", "human-feedback"],
    },
    DataType.REWARD_MODEL: {
        "keywords": [
            "reward model",
            "reward learning",
            "ppo",
            "reinforcement",
            "scalar reward",
            "reward signal",
            "reward function",
            "reward modeling",
            "reward training",
        ],
        "field_patterns": ["reward", "score"],
        "name_patterns": [r"reward", r"rm[-_]", r"ppo"],
        "tags": ["reward-model", "ppo", "reinforcement-learning"],
    },
    DataType.SFT_INSTRUCTION: {
        "keywords": [
            "instruction",
            "sft",
            "fine-tuning",
            "supervised",
            "chat",
            "conversation",
            "dialogue",
            "multi-turn",
            "assistant",
            "prompt",
            "response",
            "instruct",
            "question answering",
            "qa pairs",
        ],
        "field_patterns": ["instruction", "input", "output", "response", "answer", "question"],
        "name_patterns": [r"instruct", r"sft", r"chat", r"conv", r"dialog"],
        "tags": ["instruction-tuning", "sft", "chat", "conversational"],
    },
    DataType.CODE: {
        "keywords": [
            "code generation",
            "programming",
            "execution",
            "sandbox",
            "coding",
            "code completion",
            "repository",
            "github",
            "python",
            "javascript",
            "code review",
            "debugging",
        ],
        "field_patterns": ["code", "program", "function", "solution"],
        "name_patterns": [r"code", r"prog", r"exec", r"python", r"coding"],
        "tags": ["code", "programming", "code-generation"],
    },
    DataType.AGENT_TOOL: {
        "keywords": [
            "agent",
            "tool use",
            "function call",
            "api call",
            "web browsing",
            "browser",
            "navigation",
            "action",
            "trajectory",
            "environment",
            "task completion",
            "tool calling",
            "function calling",
            "web agent",
        ],
        "field_patterns": ["action", "tool", "function", "api", "step"],
        "name_patterns": [r"agent", r"tool", r"action", r"web", r"browser", r"function[-_]?call"],
        "tags": ["agent", "tool-use", "function-calling"],
    },
    DataType.RL_ENVIRONMENT: {
        "keywords": [
            "environment",
            "trajectory",
            "episode",
            "state",
            "observation",
            "simulation",
            "gym",
            "rl environment",
            "reinforcement learning environment",
            "game",
        ],
        "field_patterns": ["state", "observation", "action", "reward", "done", "episode"],
        "name_patterns": [r"env", r"traj", r"episode", r"sim", r"gym"],
        "tags": ["reinforcement-learning", "simulation", "environment"],
    },
    DataType.SYNTHETIC: {
        "keywords": [
            "synthetic",
            "generated",
            "distill",
            "distillation",
            "teacher",
            "student",
            "augmented",
            "artificial",
            "self-instruct",
            "evol-instruct",
            "gpt-generated",
        ],
        "field_patterns": [],
        # Name patterns for versioned/synthetic data
        "name_patterns": [
            r"[-_]v\d+",
            r"[-_]T\d+",
            r"synthetic",
            r"distill",
            r"generated",
            r"[-_]\d+[kmb]$",
            r"[-_]full[-_]",
            r"[-_]lite[-_]",
        ],
        "tags": ["synthetic", "generated", "distillation"],
    },
    DataType.MULTIMODAL: {
        "keywords": [
            "vision",
            "image",
            "video",
            "audio",
            "multimodal",
            "vlm",
            "vqa",
            "visual",
            "caption",
            "ocr",
            "image-text",
            "visual question",
            "image understanding",
        ],
        "field_patterns": ["image", "video", "audio", "visual"],
        "name_patterns": [r"vision", r"image", r"video", r"vlm", r"vqa", r"visual", r"molmo"],
        "tags": ["multimodal", "vision", "image", "video", "vqa"],
    },
    DataType.MULTILINGUAL: {
        "keywords": [
            "multilingual",
            "translation",
            "cross-lingual",
            "language",
            "languages",
            "parallel corpus",
            "machine translation",
            "bilingual",
            "polyglot",
        ],
        "field_patterns": ["source_lang", "target_lang", "translation"],
        "name_patterns": [r"multi.*lingual", r"translation", r"parallel", r"nlp", r"lang"],
        "tags": ["multilingual", "translation", "cross-lingual"],
    },
    DataType.EVALUATION: {
        "keywords": [
            "benchmark",
            "evaluation",
            "test set",
            "leaderboard",
            "assessment",
            "metric",
            "eval",
            "test suite",
        ],
        "field_patterns": ["gold", "label", "ground_truth"],
        "name_patterns": [r"bench", r"eval", r"test", r"assess"],
        "tags": ["benchmark", "evaluation", "test"],
    },
}


# Type display configuration
TYPE_DISPLAY = {
    DataType.RLHF_PREFERENCE: ("🎯 RLHF/偏好数据", 1),
    DataType.REWARD_MODEL: ("🏆 奖励模型数据", 2),
    DataType.SFT_INSTRUCTION: ("📝 SFT/指令数据", 3),
    DataType.CODE: ("💻 代码生成/执行", 4),
    DataType.AGENT_TOOL: ("🤖 Agent/工具使用", 5),
    DataType.RL_ENVIRONMENT: ("🎮 RL 环境数据", 6),
    DataType.SYNTHETIC: ("🧪 合成/蒸馏数据", 7),
    DataType.MULTIMODAL: ("🖼️ 多模态", 8),
    DataType.MULTILINGUAL: ("🌍 多语言", 9),
    DataType.EVALUATION: ("📊 评估/Benchmark", 10),
    DataType.OTHER: ("📦 其他", 99),
}


class DataTypeClassifier:
    """Enhanced dataset type classifier with multi-label support."""

    def __init__(self, config: dict = None):
        """Initialize the classifier.

        Args:
            config: Optional configuration dict.
        """
        self.config = config or {}
        classifier_config = self.config.get("classifier", {})
        self.min_score = classifier_config.get("min_score_threshold", 2)
        self.fetch_readme = classifier_config.get("fetch_readme", True)

    def classify(self, dataset: dict) -> list[DataType]:
        """Classify a dataset, returning matched types.

        Args:
            dataset: Dataset dict with metadata.

        Returns:
            List of matched DataTypes, sorted by score.
        """
        # Build search text
        text = self._build_search_text(dataset)
        name = (dataset.get("id", "") or dataset.get("name", "")).lower()
        tags = self._normalize_tags(dataset.get("tags", []))
        features = str(dataset.get("features", "")).lower()

        matches = []

        for dtype, rules in CLASSIFICATION_RULES.items():
            score = 0

            # Keyword matching in combined text
            for kw in rules.get("keywords", []):
                if kw.lower() in text.lower():
                    score += 1

            # Name pattern matching (higher weight)
            for pattern in rules.get("name_patterns", []):
                if re.search(pattern, name, re.IGNORECASE):
                    score += 2

            # Field pattern matching in features
            for field in rules.get("field_patterns", []):
                if field in features:
                    score += 2

            # Tag matching (higher weight)
            for tag in rules.get("tags", []):
                if tag in tags:
                    score += 3

            if score >= self.min_score:
                matches.append((dtype, score))

        # Sort by score descending
        matches.sort(key=lambda x: x[1], reverse=True)

        if not matches:
            return [DataType.OTHER]

        return [m[0] for m in matches]

    def _build_search_text(self, dataset: dict) -> str:
        """Build combined search text from dataset fields."""
        parts = [
            dataset.get("id", "") or dataset.get("name", ""),
            dataset.get("description", "") or "",
            " ".join(dataset.get("tags", []) or []),
            dataset.get("card_data", "") or dataset.get("readme", "") or "",
        ]
        return " ".join(str(p) for p in parts)

    def _normalize_tags(self, tags: list) -> set[str]:
        """Normalize tags to lowercase set."""
        normalized = set()
        for tag in tags or []:
            if isinstance(tag, str):
                # Handle "key:value" format
                if ":" in tag:
                    normalized.add(tag.split(":")[-1].lower())
                else:
                    normalized.add(tag.lower())
        return normalized

    def classify_with_details(self, dataset: dict) -> dict:
        """Classify with full details about matching.

        Args:
            dataset: Dataset dict.

        Returns:
            Classification result with details.
        """
        types = self.classify(dataset)
        primary = types[0] if types else DataType.OTHER

        return {
            "dataset_id": dataset.get("id", "") or dataset.get("name", ""),
            "primary_type": primary,
            "all_types": types,
            "is_relevant": primary != DataType.OTHER,
            "type_count": len([t for t in types if t != DataType.OTHER]),
        }

    def group_by_type(self, datasets: list[dict]) -> dict[DataType, list[dict]]:
        """Group datasets by their primary type.

        Args:
            datasets: List of datasets.

        Returns:
            Dict mapping DataType -> list of datasets.
        """
        groups = {dtype: [] for dtype in DataType}

        for ds in datasets:
            types = self.classify(ds)
            primary = types[0] if types else DataType.OTHER

            # Add classification info to dataset
            ds_with_class = {
                **ds,
                "data_type": primary,
                "all_types": types,
                "signals": self._get_signals(ds, types),
            }
            groups[primary].append(ds_with_class)

        # Remove empty groups
        return {k: v for k, v in groups.items() if v}

    def _get_signals(self, dataset: dict, types: list[DataType]) -> list[str]:
        """Extract signal keywords that matched."""
        text = self._build_search_text(dataset).lower()
        signals = []

        for dtype in types:
            if dtype == DataType.OTHER:
                continue
            rules = CLASSIFICATION_RULES.get(dtype, {})
            for kw in rules.get("keywords", [])[:3]:  # Top 3 keywords
                if kw.lower() in text:
                    signals.append(kw)
                    break

        return signals[:5] if signals else ["-"]

    def summarize(self, datasets: list[dict]) -> dict:
        """Generate summary statistics.

        Args:
            datasets: List of datasets.

        Returns:
            Summary statistics dict.
        """
        groups = self.group_by_type(datasets)

        other_count = len(groups.get(DataType.OTHER, []))
        total = len(datasets)
        relevant = total - other_count

        by_type = {}
        for dtype, ds_list in groups.items():
            # Use the string value for JSON serialization
            key = dtype.value if isinstance(dtype, DataType) else dtype
            by_type[key] = len(ds_list)

        summary = {
            "total": total,
            "relevant": relevant,
            "by_type": by_type,
            "other_ratio": other_count / total if total > 0 else 0,
        }

        return summary

    @staticmethod
    def get_type_display(dtype: DataType) -> str:
        """Get display name for a type."""
        return TYPE_DISPLAY.get(dtype, ("📦 其他", 99))[0]

    @staticmethod
    def get_type_order(dtype: DataType) -> int:
        """Get sort order for a type."""
        return TYPE_DISPLAY.get(dtype, ("", 99))[1]

    @staticmethod
    def get_ordered_types() -> list[tuple[DataType, str]]:
        """Get types in display order."""
        items = [(dtype, display[0]) for dtype, display in TYPE_DISPLAY.items()]
        items.sort(key=lambda x: TYPE_DISPLAY[x[0]][1])
        return items
