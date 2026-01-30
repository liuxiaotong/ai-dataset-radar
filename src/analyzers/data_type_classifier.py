"""Data type classifier for categorizing datasets by training purpose.

Classifies datasets into categories like RLHF, SFT, Agent, Code, etc.
based on tags, description, and other metadata.
"""

import re
from typing import Optional


class DataTypeClassifier:
    """Classify datasets by their training data type.

    Categories:
    - preference: RLHF, DPO, preference learning data
    - reward_model: Reward modeling, PPO training data
    - sft: Supervised fine-tuning, instruction tuning
    - code: Code generation, execution environments
    - agent: Tool use, web browsing, agents
    - embodied: Robotics, simulation, embodied AI
    - safety: Safety, alignment, red-teaming
    """

    def __init__(self, config: dict):
        """Initialize the classifier.

        Args:
            config: Configuration with priority_data_types definitions.
        """
        self.config = config
        self.data_types = config.get("priority_data_types", {})

        # Build compiled patterns for each type
        self._patterns = {}
        for dtype, dtype_config in self.data_types.items():
            keywords = dtype_config.get("keywords", [])
            if keywords:
                # Create case-insensitive pattern
                pattern = r'\b(' + '|'.join(re.escape(kw) for kw in keywords) + r')\b'
                self._patterns[dtype] = re.compile(pattern, re.IGNORECASE)

        # Store tags for each type
        self._tags = {
            dtype: set(dtype_config.get("tags", []))
            for dtype, dtype_config in self.data_types.items()
        }

    def classify(self, dataset: dict) -> dict:
        """Classify a dataset by its training data type.

        Args:
            dataset: Dataset dictionary with metadata.

        Returns:
            Classification result with types and confidence.
        """
        # Extract text to search
        name = dataset.get("id", "") or dataset.get("name", "")
        description = dataset.get("description", "") or ""
        readme = dataset.get("readme", "") or ""
        tags = dataset.get("tags", []) or []

        # Combine text for keyword search
        search_text = f"{name} {description} {readme}".lower()

        # Normalize tags
        tag_set = set()
        for tag in tags:
            if isinstance(tag, str):
                # Handle "key:value" format
                if ":" in tag:
                    tag_set.add(tag.split(":")[-1].lower())
                else:
                    tag_set.add(tag.lower())

        # Classify
        detected_types = []

        for dtype, pattern in self._patterns.items():
            # Check keywords
            matches = pattern.findall(search_text)
            keyword_score = len(matches)

            # Check tags
            dtype_tags = self._tags.get(dtype, set())
            tag_matches = tag_set & dtype_tags
            tag_score = len(tag_matches) * 2  # Tags are stronger signal

            total_score = keyword_score + tag_score

            if total_score > 0:
                detected_types.append({
                    "type": dtype,
                    "score": total_score,
                    "keyword_matches": matches[:5],  # Limit to 5
                    "tag_matches": list(tag_matches),
                })

        # Sort by score
        detected_types.sort(key=lambda x: x["score"], reverse=True)

        # Determine primary type
        primary_type = detected_types[0]["type"] if detected_types else None

        return {
            "dataset_id": name,
            "primary_type": primary_type,
            "all_types": detected_types,
            "type_count": len(detected_types),
            "is_relevant": len(detected_types) > 0,
        }

    def classify_batch(self, datasets: list[dict]) -> list[dict]:
        """Classify multiple datasets.

        Args:
            datasets: List of dataset dictionaries.

        Returns:
            List of classification results.
        """
        results = []
        for ds in datasets:
            classification = self.classify(ds)
            results.append({
                **ds,
                "_classification": classification,
            })
        return results

    def filter_relevant(self, datasets: list[dict]) -> list[dict]:
        """Filter to only relevant datasets (those matching any type).

        Args:
            datasets: List of datasets to filter.

        Returns:
            Filtered list of relevant datasets.
        """
        relevant = []
        for ds in datasets:
            classification = self.classify(ds)
            if classification["is_relevant"]:
                relevant.append({
                    **ds,
                    "_classification": classification,
                })
        return relevant

    def group_by_type(self, datasets: list[dict]) -> dict:
        """Group datasets by their primary type.

        Args:
            datasets: List of datasets.

        Returns:
            Dict mapping type -> list of datasets.
        """
        groups = {dtype: [] for dtype in self.data_types}
        groups["other"] = []

        for ds in datasets:
            classification = self.classify(ds)
            primary = classification["primary_type"]

            if primary and primary in groups:
                groups[primary].append({
                    **ds,
                    "_classification": classification,
                })
            else:
                groups["other"].append(ds)

        # Remove empty groups
        return {k: v for k, v in groups.items() if v}

    def get_type_display_name(self, dtype: str) -> str:
        """Get display-friendly name for a data type.

        Args:
            dtype: Internal type key.

        Returns:
            Human-readable type name.
        """
        display_names = {
            "preference": "RLHF/DPO 偏好数据",
            "reward_model": "Reward Model 训练数据",
            "sft": "SFT 指令微调数据",
            "code": "代码生成/执行环境",
            "agent": "Agent/工具使用",
            "embodied": "具身智能/机器人",
            "safety": "安全/对齐数据",
        }
        return display_names.get(dtype, dtype.title())

    def get_type_emoji(self, dtype: str) -> str:
        """Get emoji for a data type.

        Args:
            dtype: Internal type key.

        Returns:
            Emoji string.
        """
        emojis = {
            "preference": "🎯",
            "reward_model": "🏆",
            "sft": "📚",
            "code": "💻",
            "agent": "🤖",
            "embodied": "🦾",
            "safety": "🛡️",
        }
        return emojis.get(dtype, "📊")

    def summarize(self, datasets: list[dict]) -> dict:
        """Generate summary statistics for classified datasets.

        Args:
            datasets: List of datasets.

        Returns:
            Summary statistics.
        """
        groups = self.group_by_type(datasets)

        summary = {
            "total": len(datasets),
            "relevant": sum(len(v) for k, v in groups.items() if k != "other"),
            "by_type": {dtype: len(ds_list) for dtype, ds_list in groups.items()},
        }

        # Calculate percentages
        if summary["total"] > 0:
            summary["relevance_rate"] = summary["relevant"] / summary["total"]
        else:
            summary["relevance_rate"] = 0

        return summary
