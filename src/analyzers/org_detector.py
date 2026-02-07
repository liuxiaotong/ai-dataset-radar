"""Organization detector for identifying dataset publisher affiliations.

Identifies the organization/institution behind dataset publications
through multiple signals: author ID, description, paper affiliations.
"""

import re
from typing import Optional


class OrgDetector:
    """Detect organization affiliations of dataset publishers.

    Detection methods:
    1. Author ID contains organization name
    2. Description/README mentions organization
    3. Associated paper author affiliations
    """

    # Known organizations with aliases
    DEFAULT_ORGS = {
        # Chinese Tech
        "alibaba": ["alibaba", "aliyun", "damo", "qwen", "tongyi"],
        "bytedance": ["bytedance", "douyin", "tiktok", "doubao"],
        "baidu": ["baidu", "ernie", "wenxin", "paddlepaddle"],
        "tencent": ["tencent", "hunyuan", "youtu", "wechat"],
        "huawei": ["huawei", "pangu", "mindspore"],
        # Chinese AI Labs
        "deepseek": ["deepseek"],
        "yi": ["01-ai", "01ai", "yi-", "yi_"],
        "zhipu": ["zhipu", "glm", "chatglm", "thudm"],
        "moonshot": ["moonshot", "kimi"],
        "minimax": ["minimax"],
        "baichuan": ["baichuan"],
        "sensenova": ["sensetime", "sensenova"],
        # US Tech
        "google": ["google", "deepmind", "brain"],
        "meta": ["meta", "facebook", "fair"],
        "microsoft": ["microsoft", "msft", "azure"],
        "openai": ["openai"],
        "anthropic": ["anthropic", "claude"],
        "nvidia": ["nvidia", "nemo"],
        "apple": ["apple"],
        "amazon": ["amazon", "aws", "alexa"],
        # AI Labs
        "huggingface": ["huggingface", "hf"],
        "stability": ["stability", "stable-diffusion"],
        "cohere": ["cohere"],
        "mistral": ["mistral"],
        "together": ["together", "togethercomputer"],
        "eleuther": ["eleuther"],
        # Academia - China
        "tsinghua": ["tsinghua", "thu", "thudm"],
        "peking": ["peking", "pku"],
        "fudan": ["fudan", "fdu"],
        "sjtu": ["sjtu", "shanghai jiao"],
        "zju": ["zhejiang", "zju"],
        "nju": ["nanjing", "nju"],
        "ustc": ["ustc", "science and technology of china"],
        "shanghai_ai_lab": ["shanghai ai", "shlab", "openmmlab", "opengvlab"],
        # Academia - US/Europe
        "stanford": ["stanford"],
        "berkeley": ["berkeley", "ucb", "bair"],
        "mit": ["mit", "massachusetts institute"],
        "cmu": ["cmu", "carnegie mellon"],
        "princeton": ["princeton"],
        "harvard": ["harvard"],
        "oxford": ["oxford"],
        "cambridge": ["cambridge"],
        "eth": ["eth zurich", "ethz"],
        "mpi": ["max planck", "mpi"],
        "inria": ["inria"],
    }

    def __init__(self, config: Optional[dict] = None):
        """Initialize the organization detector.

        Args:
            config: Optional configuration with custom org definitions.
        """
        config = config or {}

        # Load custom orgs from config, merge with defaults
        custom_orgs = config.get("tracked_orgs", {})
        self.orgs = {**self.DEFAULT_ORGS}

        # Merge custom orgs (extend aliases)
        for org, aliases in custom_orgs.items():
            org_lower = org.lower()
            if org_lower in self.orgs:
                # Extend existing aliases
                existing = set(self.orgs[org_lower])
                existing.update(a.lower() for a in aliases)
                self.orgs[org_lower] = list(existing)
            else:
                # Add new org
                self.orgs[org_lower] = [a.lower() for a in aliases]

        # Build regex patterns for efficient matching
        self._patterns = {}
        for org, aliases in self.orgs.items():
            # Create pattern that matches any alias as word boundary
            pattern_str = r"\b(" + "|".join(re.escape(a) for a in aliases) + r")\b"
            self._patterns[org] = re.compile(pattern_str, re.IGNORECASE)

    def detect_from_author(self, author: str) -> Optional[str]:
        """Detect organization from author/username.

        Args:
            author: Author name or username.

        Returns:
            Organization name if detected, None otherwise.
        """
        if not author:
            return None

        author_lower = author.lower()

        for org, pattern in self._patterns.items():
            if pattern.search(author_lower):
                return org

        return None

    def detect_from_text(self, text: str) -> Optional[str]:
        """Detect organization from text (description, README, etc).

        Args:
            text: Text content to search.

        Returns:
            Organization name if detected, None otherwise.
        """
        if not text:
            return None

        text_lower = text.lower()

        for org, pattern in self._patterns.items():
            if pattern.search(text_lower):
                return org

        return None

    def detect_from_dataset(self, dataset: dict) -> dict:
        """Detect organization from dataset metadata.

        Checks multiple signals and returns detection result.

        Args:
            dataset: Dataset dictionary.

        Returns:
            Detection result with org and confidence.
        """
        detections = []

        # 1. Check author
        author = dataset.get("author", "")
        org = self.detect_from_author(author)
        if org:
            detections.append(("author", org))

        # 2. Check description
        description = dataset.get("description", "") or ""
        org = self.detect_from_text(description)
        if org:
            detections.append(("description", org))

        # 3. Check README
        readme = dataset.get("readme", "") or ""
        if readme and readme != description:
            org = self.detect_from_text(readme)
            if org:
                detections.append(("readme", org))

        # 4. Check card_data
        card_data = dataset.get("card_data", {}) or {}
        if card_data:
            card_text = str(card_data)
            org = self.detect_from_text(card_text)
            if org:
                detections.append(("card_data", org))

        # 5. Check paper URL for arxiv affiliation hints
        paper_url = dataset.get("paper_url", "") or ""
        if paper_url:
            org = self.detect_from_text(paper_url)
            if org:
                detections.append(("paper_url", org))

        # Determine final org (prioritize author > description > others)
        final_org = None
        confidence = "none"
        source = None

        if detections:
            # Count org occurrences
            org_counts = {}
            for src, detected_org in detections:
                org_counts[detected_org] = org_counts.get(detected_org, 0) + 1

            # Get most frequent
            sorted_orgs = sorted(org_counts.items(), key=lambda x: x[1], reverse=True)
            final_org = sorted_orgs[0][0]

            # Determine confidence
            if len(detections) >= 3 or ("author" in [d[0] for d in detections]):
                confidence = "high"
            elif len(detections) >= 2:
                confidence = "medium"
            else:
                confidence = "low"

            # Get primary source
            for src, org in detections:
                if org == final_org:
                    source = src
                    break

        return {
            "organization": final_org,
            "confidence": confidence,
            "source": source,
            "all_detections": detections,
        }

    def classify_datasets(self, datasets: list[dict]) -> dict:
        """Classify datasets by organization.

        Args:
            datasets: List of datasets to classify.

        Returns:
            Dict mapping organization -> list of datasets.
        """
        org_datasets = {org: [] for org in self.orgs}
        org_datasets["unknown"] = []

        for ds in datasets:
            detection = self.detect_from_dataset(ds)
            org = detection["organization"]

            if org:
                ds_with_org = {**ds, "_org_detection": detection}
                org_datasets[org].append(ds_with_org)
            else:
                org_datasets["unknown"].append(ds)

        # Remove empty organizations
        org_datasets = {org: ds_list for org, ds_list in org_datasets.items() if ds_list}

        return org_datasets

    def enrich_datasets(self, datasets: list[dict]) -> list[dict]:
        """Add organization detection to datasets.

        Args:
            datasets: List of datasets.

        Returns:
            Datasets with _detected_org field added.
        """
        enriched = []
        for ds in datasets:
            detection = self.detect_from_dataset(ds)
            enriched.append(
                {
                    **ds,
                    "_detected_org": detection["organization"],
                    "_org_confidence": detection["confidence"],
                }
            )
        return enriched

    def get_org_display_name(self, org: str) -> str:
        """Get display-friendly organization name.

        Args:
            org: Internal org key.

        Returns:
            Formatted display name.
        """
        # Capitalize and format
        display_names = {
            "alibaba": "Alibaba",
            "bytedance": "ByteDance",
            "baidu": "Baidu",
            "tencent": "Tencent",
            "huawei": "Huawei",
            "deepseek": "DeepSeek",
            "yi": "01.AI (Yi)",
            "zhipu": "Zhipu AI",
            "moonshot": "Moonshot",
            "google": "Google",
            "meta": "Meta",
            "microsoft": "Microsoft",
            "openai": "OpenAI",
            "anthropic": "Anthropic",
            "nvidia": "NVIDIA",
            "huggingface": "Hugging Face",
            "stability": "Stability AI",
            "mistral": "Mistral AI",
            "tsinghua": "Tsinghua University",
            "peking": "Peking University",
            "stanford": "Stanford University",
            "berkeley": "UC Berkeley",
            "mit": "MIT",
            "cmu": "CMU",
            "shanghai_ai_lab": "Shanghai AI Lab",
        }
        return display_names.get(org, org.title())
