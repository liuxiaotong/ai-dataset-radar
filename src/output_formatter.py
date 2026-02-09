"""Dual output formatter for Markdown and JSON reports."""

import json
import os
from datetime import datetime, timedelta
from typing import Optional


# Fields to remove from dataset output
INTERNAL_DATASET_FIELDS = {
    "_id",
    "_org",
    "_hf_id",
    "sha",
    "key",
    "disabled",
    "gated",
    "private",
}

# Fields to keep in clean dataset output
DATASET_OUTPUT_FIELDS = {
    "id",
    "author",
    "name",
    "downloads",
    "likes",
    "description",
    "tags",
    "license",
    "languages",
    "size_category",
    "task_categories",
    "category",
    "all_categories",
    "signals",
    "created_at",
    "last_modified",
    "source_url",
    "url",
    "source",
    "growth_7d",
    "growth_30d",
}


class DualOutputFormatter:
    """Formatter that saves both Markdown and JSON reports."""

    def __init__(self, output_dir: str = "data/reports"):
        """Initialize the formatter.

        Args:
            output_dir: Directory to save reports.
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def save_reports(
        self, markdown_content: str, data: dict, filename_prefix: str = "intel_report"
    ) -> tuple[str, str]:
        """Save both Markdown and JSON reports.

        Args:
            markdown_content: The markdown report content.
            data: Structured data for JSON output.
            filename_prefix: Prefix for output filenames.

        Returns:
            Tuple of (markdown_path, json_path).
        """
        date_str = datetime.now().strftime("%Y-%m-%d")
        base_name = f"{filename_prefix}_{date_str}"

        # Save Markdown
        md_path = os.path.join(self.output_dir, f"{base_name}.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(markdown_content)

        # Save JSON
        json_data = self._format_json_output(data)
        json_path = os.path.join(self.output_dir, f"{base_name}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False, default=str)

        return md_path, json_path

    def _format_json_output(self, data: dict) -> dict:
        """Format data to standard JSON schema.

        Args:
            data: Raw data dictionary.

        Returns:
            Formatted JSON output with standard schema.
        """
        # Get raw data
        datasets = data.get("datasets", [])
        github_activity = data.get("github_activity", [])
        papers = data.get("papers", [])
        blog_posts = data.get("blog_posts", [])
        x_activity = data.get("x_activity", {})
        labs_activity = data.get("labs_activity", {})
        vendor_activity = data.get("vendor_activity", {})
        datasets_by_type = data.get("datasets_by_type", {})
        trend_data = data.get("trend_data", {})

        # Calculate period with proper start date
        period = data.get("period", {})
        days = period.get("days", 7)
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)

        # Count GitHub repos and orgs
        github_stats = self._count_github_stats(github_activity)

        # Count total datasets
        total_datasets = len(datasets)

        # Clean and enrich datasets
        cleaned_datasets = [self._clean_dataset(ds, datasets_by_type) for ds in datasets]

        # Clean labs_activity datasets
        cleaned_labs = self._clean_labs_activity(labs_activity, datasets_by_type)

        # Handle vendor_activity - remove if empty
        cleaned_vendor = self._clean_vendor_activity(vendor_activity)

        # Clean papers (remove duplicate summary/abstract)
        cleaned_papers = [self._clean_paper(p) for p in papers]

        # Convert datasets_by_type to ID references
        datasets_by_type_ids = self._convert_datasets_by_type(datasets_by_type)

        return {
            "generated_at": end_time.isoformat(),
            "period": {
                "days": days,
                "start": start_time.isoformat(),
                "end": end_time.isoformat(),
            },
            "summary": {
                "total_datasets": total_datasets,
                "total_github_orgs": github_stats["orgs"],
                "total_github_repos": github_stats["repos"],
                "total_github_repos_high_relevance": github_stats["high_relevance"],
                "total_papers": len(papers),
                "total_blog_posts": self._count_blog_posts(blog_posts),
                "total_x_tweets": sum(
                    len(a.get("relevant_tweets", []))
                    for a in x_activity.get("accounts", [])
                ),
                "total_trending_datasets": len(trend_data.get("top_growing_7d", [])),
            },
            "labs_activity": cleaned_labs,
            "vendor_activity": cleaned_vendor,
            "datasets": cleaned_datasets,
            "datasets_by_type": datasets_by_type_ids,
            "github_activity": github_activity,
            "papers": cleaned_papers,
            "blog_posts": blog_posts,
            "x_activity": x_activity,
            "featured_trends": trend_data if trend_data else None,
        }

    def _count_github_stats(self, github_activity: list) -> dict:
        """Count GitHub statistics.

        Args:
            github_activity: List of org activity dicts.

        Returns:
            Dict with orgs, repos, high_relevance counts.
        """
        orgs = len(github_activity)
        repos = 0
        high_relevance = 0

        for org_data in github_activity:
            if isinstance(org_data, dict):
                repos_list = org_data.get("repos_updated", [])
                repos += len(repos_list)
                for repo in repos_list:
                    if isinstance(repo, dict) and repo.get("relevance") == "high":
                        high_relevance += 1

        return {
            "orgs": orgs,
            "repos": repos,
            "high_relevance": high_relevance,
        }

    def _clean_dataset(self, ds: dict, datasets_by_type: dict = None) -> dict:
        """Clean a dataset by removing internal fields and enriching with category.

        Args:
            ds: Raw dataset dict.
            datasets_by_type: Mapping of type to datasets for category lookup.

        Returns:
            Cleaned dataset dict.
        """
        # Start with a clean dict
        cleaned = {}

        # Copy allowed fields
        for key, value in ds.items():
            if key in DATASET_OUTPUT_FIELDS and key not in INTERNAL_DATASET_FIELDS:
                cleaned[key] = value

        # Extract structured fields from tags if present
        tags = ds.get("tags", [])
        if tags and isinstance(tags, list):
            cleaned["tags"], extracted = self._extract_from_tags(tags)
            # Only add extracted fields if not already present
            for field, value in extracted.items():
                if field not in cleaned or not cleaned[field]:
                    cleaned[field] = value

        # Parse card_data if it's a string
        card_data = ds.get("card_data")
        if card_data and isinstance(card_data, str):
            # Remove it - it's not useful as a string
            cleaned.pop("card_data", None)
        elif card_data and isinstance(card_data, dict):
            cleaned["card_data"] = card_data

        # Look up category from datasets_by_type
        ds_id = ds.get("id", "")
        if datasets_by_type and ds_id:
            category, all_categories, signals = self._find_dataset_category(ds_id, datasets_by_type)
            if category:
                cleaned["category"] = category
                cleaned["all_categories"] = all_categories
                cleaned["signals"] = signals

        # Ensure source_url exists
        if "source_url" not in cleaned and "url" in cleaned:
            cleaned["source_url"] = cleaned["url"]

        return cleaned

    def _extract_from_tags(self, tags: list) -> tuple[list, dict]:
        """Extract structured fields from HuggingFace tags.

        Args:
            tags: List of tag strings.

        Returns:
            Tuple of (remaining_tags, extracted_fields).
        """
        remaining = []
        extracted = {
            "license": [],
            "languages": [],
            "size_category": None,
            "task_categories": [],
        }

        for tag in tags:
            if not isinstance(tag, str):
                continue

            if tag.startswith("license:"):
                extracted["license"].append(tag.replace("license:", ""))
            elif tag.startswith("language:"):
                extracted["languages"].append(tag.replace("language:", ""))
            elif tag.startswith("size_categories:"):
                extracted["size_category"] = tag.replace("size_categories:", "")
            elif tag.startswith("task_categories:"):
                extracted["task_categories"].append(tag.replace("task_categories:", ""))
            else:
                # Keep other tags
                remaining.append(tag)

        # Convert single-item lists to strings where appropriate
        if len(extracted["license"]) == 1:
            extracted["license"] = extracted["license"][0]
        elif not extracted["license"]:
            del extracted["license"]

        if not extracted["languages"]:
            del extracted["languages"]

        if not extracted["size_category"]:
            del extracted["size_category"]

        if not extracted["task_categories"]:
            del extracted["task_categories"]

        return remaining, extracted

    def _find_dataset_category(
        self, ds_id: str, datasets_by_type: dict
    ) -> tuple[Optional[str], list, list]:
        """Find category for a dataset from datasets_by_type.

        Args:
            ds_id: Dataset ID.
            datasets_by_type: Mapping of type to datasets.

        Returns:
            Tuple of (primary_category, all_categories, signals).
        """
        all_categories = []
        all_signals = []
        primary_category = None

        for dtype, ds_list in datasets_by_type.items():
            if not isinstance(ds_list, list):
                continue

            # Normalize dtype (handle DataType enum repr)
            dtype_str = self._normalize_data_type(dtype)

            for ds in ds_list:
                if not isinstance(ds, dict):
                    continue
                if ds.get("id") == ds_id:
                    all_categories.append(dtype_str)
                    # Get signals from this entry
                    signals = ds.get("signals", [])
                    if signals:
                        all_signals.extend(signals)

        if all_categories:
            primary_category = all_categories[0]

        return primary_category, all_categories, list(set(all_signals))

    def _normalize_data_type(self, dtype) -> str:
        """Normalize data type to string.

        Args:
            dtype: DataType enum or string.

        Returns:
            Lowercase string representation.
        """
        dtype_str = str(dtype)
        # Handle "DataType.SYNTHETIC" -> "synthetic"
        if "." in dtype_str:
            dtype_str = dtype_str.split(".")[-1]
        return dtype_str.lower()

    def _convert_datasets_by_type(self, datasets_by_type: dict) -> dict:
        """Convert datasets_by_type to use ID references only.

        Args:
            datasets_by_type: Original mapping.

        Returns:
            Mapping of type to list of dataset IDs.
        """
        result = {}
        for dtype, ds_list in datasets_by_type.items():
            dtype_str = self._normalize_data_type(dtype)
            if isinstance(ds_list, list):
                ids = [ds.get("id") for ds in ds_list if isinstance(ds, dict) and ds.get("id")]
                if ids:
                    result[dtype_str] = ids
        return result

    def _clean_labs_activity(self, labs_activity: dict, datasets_by_type: dict) -> dict:
        """Clean labs_activity by removing internal fields from datasets.

        Args:
            labs_activity: Raw labs activity dict.
            datasets_by_type: For category lookup.

        Returns:
            Cleaned labs activity.
        """
        if not isinstance(labs_activity, dict):
            return labs_activity

        cleaned = {}
        for key, value in labs_activity.items():
            if isinstance(value, dict):
                cleaned[key] = {}
                for org_key, org_data in value.items():
                    if isinstance(org_data, dict):
                        cleaned_org = dict(org_data)
                        # Clean datasets in this org
                        if "datasets" in cleaned_org:
                            cleaned_org["datasets"] = [
                                self._clean_dataset(ds, datasets_by_type)
                                for ds in cleaned_org.get("datasets", [])
                            ]
                        cleaned[key][org_key] = cleaned_org
                    else:
                        cleaned[key][org_key] = org_data
            else:
                cleaned[key] = value

        return cleaned

    def _clean_vendor_activity(self, vendor_activity: dict) -> Optional[dict]:
        """Clean vendor_activity - remove if empty.

        Args:
            vendor_activity: Raw vendor activity.

        Returns:
            Cleaned vendor activity or None if empty.
        """
        if not isinstance(vendor_activity, dict):
            return None

        # Check if vendors dict is empty or has only empty dicts
        vendors = vendor_activity.get("vendors", {})
        if not vendors:
            return None

        has_data = False
        for tier, tier_data in vendors.items():
            if isinstance(tier_data, dict) and tier_data:
                has_data = True
                break

        if not has_data:
            return None

        return vendor_activity

    def _clean_paper(self, paper: dict) -> dict:
        """Clean a paper by removing duplicate summary/abstract.

        Args:
            paper: Raw paper dict.

        Returns:
            Cleaned paper dict.
        """
        if not isinstance(paper, dict):
            return paper

        cleaned = dict(paper)

        # Remove summary if it's identical to abstract
        abstract = cleaned.get("abstract", "")
        summary = cleaned.get("summary", "")
        if abstract and summary and abstract == summary:
            del cleaned["summary"]

        return cleaned

    def _count_blog_posts(self, blog_posts: list) -> int:
        """Count total blog posts across all sources.

        Args:
            blog_posts: List of blog activity dicts.

        Returns:
            Total number of blog posts.
        """
        total = 0
        for blog in blog_posts:
            if isinstance(blog, dict):
                articles = blog.get("articles", [])
                total += len(articles) if isinstance(articles, list) else 0
        return total

    def format_summary(self, data: dict) -> str:
        """Generate a summary string for console output.

        Args:
            data: Structured data dictionary.

        Returns:
            Summary string.
        """
        json_data = self._format_json_output(data)
        summary = json_data["summary"]

        return (
            f"Summary: {summary['total_datasets']} datasets, "
            f"{summary['total_github_repos']} repos ({summary['total_github_repos_high_relevance']} high relevance), "
            f"{summary['total_papers']} papers, "
            f"{summary['total_blog_posts']} blog posts"
        )
