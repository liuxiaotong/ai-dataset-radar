"""Papers with Code SOTA Tracker - Extract datasets from SOTA models.

Connects SOTA model performance to training/evaluation datasets,
enabling discovery of datasets that produce top results.

API Documentation: https://paperswithcode.com/api/v1/docs/
"""

import time
import random
from datetime import datetime
from typing import Optional
import requests


class PwCSOTAScraper:
    """Scraper for Papers with Code SOTA results and associated datasets."""

    BASE_URL = "https://paperswithcode.com/api/v1"

    # Priority areas for dataset discovery
    PRIORITY_AREAS = [
        "robotics",
        "code-generation",
        "question-answering",
        "text-generation",
        "image-classification",
        "object-detection",
        "semantic-segmentation",
        "machine-translation",
        "named-entity-recognition",
        "visual-question-answering",
    ]

    def __init__(self, areas: Optional[list[str]] = None, top_n: int = 10):
        """Initialize the SOTA scraper.

        Args:
            areas: List of areas to track (defaults to PRIORITY_AREAS).
            top_n: Number of top SOTA results to fetch per task.
        """
        self.areas = areas or self.PRIORITY_AREAS
        self.top_n = top_n
        self.session = requests.Session()
        self.session.headers.update({
            "Accept": "application/json",
            "User-Agent": "AI-Dataset-Radar/3.0 (https://github.com/liuxiaotong/ai-dataset-radar)",
        })
        self._last_request_time = 0
        self._base_delay = 1.5

    def _rate_limit_wait(self) -> None:
        """Wait to respect rate limits."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self._base_delay:
            time.sleep(self._base_delay - elapsed + random.uniform(0.1, 0.5))
        self._last_request_time = time.time()

    def _request_with_retry(
        self,
        url: str,
        params: dict,
        max_retries: int = 3,
    ) -> Optional[dict]:
        """Make a request with retry logic.

        Args:
            url: Request URL.
            params: Query parameters.
            max_retries: Maximum retry attempts.

        Returns:
            JSON response data or None on failure.
        """
        for attempt in range(max_retries + 1):
            self._rate_limit_wait()

            try:
                response = self.session.get(url, params=params, timeout=30)

                # Check content type
                content_type = response.headers.get("Content-Type", "")
                if "application/json" not in content_type:
                    if attempt < max_retries:
                        time.sleep(2 ** attempt)
                        continue
                    return None

                if response.status_code == 200:
                    return response.json()

                elif response.status_code == 429:
                    wait_time = (2 ** attempt) * 3 + random.uniform(1, 2)
                    if attempt < max_retries:
                        print(f"  Rate limited, waiting {wait_time:.1f}s...")
                        time.sleep(wait_time)
                        continue
                    return None

                elif response.status_code == 404:
                    # Resource not found - don't retry
                    return None

                elif response.status_code >= 500:
                    if attempt < max_retries:
                        time.sleep(2 ** attempt)
                        continue

                response.raise_for_status()

            except requests.exceptions.Timeout:
                if attempt < max_retries:
                    time.sleep(2 ** attempt)
                    continue
                return None

            except requests.RequestException as e:
                if attempt < max_retries:
                    time.sleep(2 ** attempt)
                    continue
                return None

            except ValueError:
                return None

        return None

    def fetch(self) -> list[dict]:
        """Fetch SOTA results and associated datasets.

        Returns:
            List of SOTA-dataset associations.
        """
        all_results = []

        for i, area in enumerate(self.areas):
            print(f"  Fetching SOTA for: {area} ({i+1}/{len(self.areas)})")
            area_results = self._fetch_area_sota(area)
            all_results.extend(area_results)

        return all_results

    def _fetch_area_sota(self, area: str) -> list[dict]:
        """Fetch SOTA results for a specific area.

        Args:
            area: Area/task name.

        Returns:
            List of SOTA results with dataset info.
        """
        # Get tasks in this area
        tasks = self._get_area_tasks(area)
        if not tasks:
            return []

        results = []
        for task in tasks[:5]:  # Top 5 tasks per area
            task_id = task.get("id")
            task_name = task.get("name", "")

            # Get datasets used in this task
            datasets = self._get_task_datasets(task_id)

            # Get top SOTA results
            sota_results = self._get_task_sota(task_id)

            for ds in datasets:
                ds_info = {
                    "area": area,
                    "task_id": task_id,
                    "task_name": task_name,
                    "dataset_id": ds.get("id"),
                    "dataset_name": ds.get("name"),
                    "dataset_url": ds.get("url"),
                    "sota_count": len(sota_results),
                    "sota_models": [],
                    "source": "pwc_sota",
                }

                # Match SOTA results to this dataset
                for sota in sota_results:
                    if sota.get("dataset") == ds.get("name"):
                        ds_info["sota_models"].append(
                            {
                                "model_name": sota.get("model_name"),
                                "paper_title": sota.get("paper_title"),
                                "paper_url": sota.get("paper_url"),
                                "metric_value": sota.get("metric_value"),
                                "metric_name": sota.get("metric_name"),
                            }
                        )

                if ds_info["sota_models"]:
                    results.append(ds_info)

        return results

    def _get_area_tasks(self, area: str) -> list[dict]:
        """Get tasks in a specific area.

        Args:
            area: Area name.

        Returns:
            List of task dictionaries.
        """
        url = f"{self.BASE_URL}/tasks/"
        params = {
            "area": area,
            "items_per_page": 20,
        }

        data = self._request_with_retry(url, params)
        if not data:
            return []

        return data.get("results", [])

    def _get_task_datasets(self, task_id: str) -> list[dict]:
        """Get datasets associated with a task.

        Args:
            task_id: Task identifier.

        Returns:
            List of dataset dictionaries.
        """
        url = f"{self.BASE_URL}/tasks/{task_id}/datasets/"
        params = {"items_per_page": 20}

        data = self._request_with_retry(url, params)
        if not data:
            return []

        return data.get("results", [])

    def _get_task_sota(self, task_id: str) -> list[dict]:
        """Get SOTA results for a task.

        Args:
            task_id: Task identifier.

        Returns:
            List of SOTA result dictionaries.
        """
        url = f"{self.BASE_URL}/evaluations/"
        params = {
            "task": task_id,
            "items_per_page": self.top_n,
            "ordering": "-metric_value",
        }

        data = self._request_with_retry(url, params)
        if not data:
            return []

        results = []
        for item in data.get("results", []):
            results.append(
                {
                    "model_name": item.get("model_name", ""),
                    "paper_title": item.get("paper", {}).get("title", "")
                    if item.get("paper")
                    else "",
                    "paper_url": item.get("paper", {}).get("url", "")
                    if item.get("paper")
                    else "",
                    "dataset": item.get("dataset", {}).get("name", "")
                    if item.get("dataset")
                    else "",
                    "metric_name": item.get("metric", {}).get("name", "")
                    if item.get("metric")
                    else "",
                    "metric_value": item.get("metric_value"),
                }
            )
        return results

    def get_dataset_benchmarks(self, dataset_name: str) -> list[dict]:
        """Get all benchmarks/tasks that use a dataset.

        Args:
            dataset_name: Name of the dataset.

        Returns:
            List of benchmark information.
        """
        url = f"{self.BASE_URL}/datasets/"
        params = {
            "q": dataset_name,
            "items_per_page": 10,
        }

        data = self._request_with_retry(url, params)
        if not data:
            return []

        results = []
        for ds in data.get("results", []):
            if dataset_name.lower() in ds.get("name", "").lower():
                results.append(
                    {
                        "id": ds.get("id"),
                        "name": ds.get("name"),
                        "full_name": ds.get("full_name"),
                        "url": ds.get("url"),
                        "paper_count": ds.get("num_papers", 0),
                        "description": ds.get("description", ""),
                    }
                )
        return results

    def get_dataset_papers(self, dataset_id: str, limit: int = 20) -> list[dict]:
        """Get papers that use a specific dataset.

        Args:
            dataset_id: Dataset identifier from PwC.
            limit: Maximum number of papers to return.

        Returns:
            List of paper dictionaries.
        """
        url = f"{self.BASE_URL}/datasets/{dataset_id}/papers/"
        params = {"items_per_page": limit}

        data = self._request_with_retry(url, params)
        if not data:
            return []

        papers = []
        for item in data.get("results", []):
            papers.append(
                {
                    "id": item.get("id"),
                    "title": item.get("title"),
                    "url": item.get("url"),
                    "arxiv_id": item.get("arxiv_id"),
                    "proceeding": item.get("proceeding"),
                    "date": item.get("date"),
                }
            )
        return papers

    def analyze_sota_datasets(self) -> dict:
        """Run full SOTA dataset analysis.

        Returns:
            Analysis results with dataset rankings.
        """
        print("Fetching SOTA results from Papers with Code...")
        results = self.fetch()
        print(f"  Found {len(results)} dataset-SOTA associations")

        # Aggregate by dataset
        dataset_stats = {}
        for r in results:
            ds_name = r.get("dataset_name", "")
            if not ds_name:
                continue

            if ds_name not in dataset_stats:
                dataset_stats[ds_name] = {
                    "name": ds_name,
                    "url": r.get("dataset_url"),
                    "areas": set(),
                    "tasks": set(),
                    "sota_model_count": 0,
                    "sota_models": [],
                }

            dataset_stats[ds_name]["areas"].add(r.get("area", ""))
            dataset_stats[ds_name]["tasks"].add(r.get("task_name", ""))
            dataset_stats[ds_name]["sota_model_count"] += len(r.get("sota_models", []))
            dataset_stats[ds_name]["sota_models"].extend(r.get("sota_models", []))

        # Convert sets to lists for JSON serialization
        for ds in dataset_stats.values():
            ds["areas"] = list(ds["areas"])
            ds["tasks"] = list(ds["tasks"])
            # Keep only top 10 SOTA models
            ds["sota_models"] = ds["sota_models"][:10]

        # Sort by SOTA model count
        ranked_datasets = sorted(
            dataset_stats.values(),
            key=lambda x: x["sota_model_count"],
            reverse=True,
        )

        return {
            "total_associations": len(results),
            "unique_datasets": len(dataset_stats),
            "areas_covered": list(self.areas),
            "ranked_datasets": ranked_datasets,
            "analysis_date": datetime.now().isoformat(),
        }

    def generate_report(self, results: dict) -> str:
        """Generate a human-readable report.

        Args:
            results: Analysis results dictionary.

        Returns:
            Formatted report string.
        """
        lines = [
            "=" * 60,
            "SOTA Dataset Analysis Report",
            "=" * 60,
            "",
            f"Total SOTA-dataset associations: {results['total_associations']}",
            f"Unique datasets: {results['unique_datasets']}",
            f"Areas covered: {', '.join(results['areas_covered'])}",
            "",
            "Top Datasets by SOTA Model Count:",
            "-" * 40,
        ]

        for i, ds in enumerate(results["ranked_datasets"][:20], 1):
            areas_str = ", ".join(ds["areas"][:3])
            lines.append(
                f"{i:2}. {ds['name']:<30} " f"({ds['sota_model_count']} SOTA models)"
            )
            lines.append(f"      Areas: {areas_str}")
            if ds["sota_models"]:
                top_model = ds["sota_models"][0].get("model_name", "N/A")
                lines.append(f"      Top model: {top_model}")

        return "\n".join(lines)
