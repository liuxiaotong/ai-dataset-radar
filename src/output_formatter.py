"""Dual output formatter for Markdown and JSON reports."""

import json
import os
from datetime import datetime
from typing import Optional


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
        self,
        markdown_content: str,
        data: dict,
        filename_prefix: str = "intel_report"
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
        # Extract counts
        datasets = data.get("datasets", [])
        github_activity = data.get("github_activity", [])
        papers = data.get("papers", [])
        blog_posts = data.get("blog_posts", [])
        labs_activity = data.get("labs_activity", {})
        vendor_activity = data.get("vendor_activity", {})

        # Count datasets across all sources
        total_datasets = len(datasets)
        if isinstance(labs_activity, dict):
            for category in labs_activity.values():
                if isinstance(category, dict):
                    for org_data in category.values():
                        if isinstance(org_data, dict):
                            total_datasets += len(org_data.get("datasets", []))

        # Count repos
        total_repos = len(github_activity)
        if isinstance(labs_activity, dict):
            for category in labs_activity.values():
                if isinstance(category, dict):
                    for org_data in category.values():
                        if isinstance(org_data, dict):
                            total_repos += len(org_data.get("repos", []))

        return {
            "generated_at": datetime.now().isoformat(),
            "period": data.get("period", {
                "days": 7,
                "start": None,
                "end": datetime.now().isoformat()
            }),
            "summary": {
                "total_datasets": total_datasets,
                "total_repos": total_repos,
                "total_papers": len(papers),
                "total_blog_posts": self._count_blog_posts(blog_posts),
            },
            "labs_activity": labs_activity,
            "vendor_activity": vendor_activity,
            "datasets": datasets,
            "datasets_by_type": data.get("datasets_by_type", {}),
            "github_activity": github_activity,
            "papers": papers,
            "blog_posts": blog_posts,
        }

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
            f"{summary['total_repos']} repos, "
            f"{summary['total_papers']} papers, "
            f"{summary['total_blog_posts']} blog posts"
        )
