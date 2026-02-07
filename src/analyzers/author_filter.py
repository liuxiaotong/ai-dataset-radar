"""Author quality filter for detecting suspicious batch upload accounts.

Filters out low-quality/spam accounts to improve data factory signal-to-noise ratio.
"""

import re
from typing import Optional


class AuthorFilter:
    """Filter suspicious batch upload accounts.

    Detection criteria:
    1. Username pattern: letters + 5+ digits at end (e.g., marksgarrett6068025)
    2. Dataset ID pattern: 80%+ datasets have meaningless IDs (10+ random alphanumeric)
    3. Dataset quality: All datasets lack description/downloads/license
    """

    # Pattern for suspicious usernames: letters followed by 5+ digits
    SUSPICIOUS_USERNAME_PATTERN = re.compile(r"^[a-zA-Z]+\d{5,}$")

    # Pattern for random/meaningless dataset IDs: 10+ alphanumeric characters
    RANDOM_ID_PATTERN = re.compile(r"^[a-zA-Z0-9]{10,}$")

    # Common meaningful dataset name patterns (should NOT be filtered)
    MEANINGFUL_PATTERNS = [
        re.compile(r"[-_]"),  # Contains separator
        re.compile(r"\d{4}"),  # Contains year
        re.compile(r"(dataset|data|benchmark|corpus|eval)", re.I),  # Dataset keywords
        re.compile(r"(train|test|val|dev)", re.I),  # Split keywords
        re.compile(r"(v\d|version)", re.I),  # Version indicators
    ]

    def __init__(self, config: Optional[dict] = None):
        """Initialize the author filter.

        Args:
            config: Optional configuration dict with thresholds.
        """
        config = config or {}
        filter_config = config.get("quality_filter", {}).get("author", {})

        # Thresholds
        self.random_id_threshold = filter_config.get("random_id_threshold", 0.8)
        self.min_username_digits = filter_config.get("min_username_digits", 5)
        self.min_description_length = filter_config.get("min_description_length", 20)

    def is_suspicious_username(self, username: str) -> bool:
        """Check if username matches suspicious pattern.

        Args:
            username: Author username to check.

        Returns:
            True if username appears to be auto-generated spam.
        """
        if not username:
            return True

        # Check for pattern: letters + many digits
        if self.SUSPICIOUS_USERNAME_PATTERN.match(username):
            return True

        # Check for very long random-looking usernames
        if len(username) > 20 and username.isalnum():
            # Count digit ratio
            digit_count = sum(1 for c in username if c.isdigit())
            if digit_count >= self.min_username_digits:
                return True

        return False

    def is_random_dataset_id(self, dataset_id: str) -> bool:
        """Check if dataset ID appears to be random/meaningless.

        Args:
            dataset_id: Dataset ID or name to check.

        Returns:
            True if ID appears random/auto-generated.
        """
        if not dataset_id:
            return True

        # Extract just the dataset name (remove author prefix)
        if "/" in dataset_id:
            dataset_id = dataset_id.split("/")[-1]

        # Check if it matches random pattern
        if not self.RANDOM_ID_PATTERN.match(dataset_id):
            return False

        # Check if it has meaningful patterns
        for pattern in self.MEANINGFUL_PATTERNS:
            if pattern.search(dataset_id):
                return False

        return True

    def has_quality_metadata(self, dataset: dict) -> bool:
        """Check if dataset has quality metadata.

        Args:
            dataset: Dataset dictionary.

        Returns:
            True if dataset has meaningful metadata.
        """
        # Check description
        description = dataset.get("description", "") or ""
        if len(description.strip()) >= self.min_description_length:
            return True

        # Check downloads
        downloads = dataset.get("downloads", 0) or 0
        if downloads > 0:
            return True

        # Check license
        license_info = dataset.get("license", "") or dataset.get("tags", [])
        if license_info:
            if isinstance(license_info, list):
                license_tags = [t for t in license_info if "license" in str(t).lower()]
                if license_tags:
                    return True
            elif license_info.strip():
                return True

        # Check if has paper/code reference
        if dataset.get("paper_url") or dataset.get("code_url"):
            return True

        # Check card_data for additional metadata
        card_data = dataset.get("card_data", {}) or {}
        if card_data.get("license") or card_data.get("task_categories"):
            return True

        return False

    def analyze_author(self, author: str, datasets: list[dict]) -> dict:
        """Analyze an author's account quality.

        Args:
            author: Author username.
            datasets: List of datasets by this author.

        Returns:
            Analysis result with quality indicators.
        """
        result = {
            "author": author,
            "is_suspicious": False,
            "reasons": [],
            "dataset_count": len(datasets),
            "quality_score": 0,  # 0-10 score
        }

        # Check username
        if self.is_suspicious_username(author):
            result["reasons"].append("suspicious_username")
            result["is_suspicious"] = True

        # Check dataset IDs
        if datasets:
            random_id_count = sum(
                1 for ds in datasets if self.is_random_dataset_id(ds.get("name", ds.get("id", "")))
            )
            random_ratio = random_id_count / len(datasets)

            if random_ratio >= self.random_id_threshold:
                result["reasons"].append(f"random_ids_{random_ratio:.0%}")
                result["is_suspicious"] = True

        # Check dataset quality
        if datasets:
            quality_count = sum(1 for ds in datasets if self.has_quality_metadata(ds))

            if quality_count == 0:
                result["reasons"].append("no_quality_metadata")
                result["is_suspicious"] = True
            else:
                # Partial credit for some quality datasets
                result["quality_score"] = min(10, int(quality_count / len(datasets) * 10))

        return result

    def filter_authors(
        self,
        author_datasets: dict[str, list[dict]],
    ) -> tuple[dict[str, list[dict]], list[dict]]:
        """Filter suspicious authors from dataset groupings.

        Args:
            author_datasets: Dict mapping author -> list of datasets.

        Returns:
            Tuple of (filtered_author_datasets, filtered_out_authors).
        """
        filtered = {}
        filtered_out = []

        for author, datasets in author_datasets.items():
            analysis = self.analyze_author(author, datasets)

            if analysis["is_suspicious"]:
                filtered_out.append(analysis)
            else:
                filtered[author] = datasets

        return filtered, filtered_out
