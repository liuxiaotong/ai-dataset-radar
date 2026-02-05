"""Database operations for AI Dataset Radar."""

import os
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from utils.logging_config import get_logger

logger = get_logger("db")

# Valid column names for dynamic SQL (whitelist for security)
VALID_GROWTH_COLUMNS = {"downloads_7d_growth", "downloads_30d_growth"}
def _get_growth_column(days: int) -> str:
    """Get validated growth column name.

    Args:
        days: 7 or 30 for corresponding growth column.

    Returns:
        Valid column name.

    Raises:
        ValueError: If days is not 7 or 30.
    """
    if days not in (7, 30):
        raise ValueError(f"Invalid days value: {days}, must be 7 or 30")
    column = "downloads_7d_growth" if days == 7 else "downloads_30d_growth"
    return column
class RadarDatabase:
    """SQLite database for storing radar data and historical trends.

    Uses thread-local connections for thread safety and context managers
    for proper resource cleanup.
    """

    def __init__(self, db_path: str = "data/radar.db"):
        """Initialize database connection.

        Args:
            db_path: Path to SQLite database file.
        """
        self.db_path = db_path
        self._local = threading.local()
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_tables()

    def _get_connection(self) -> sqlite3.Connection:
        """Get a thread-local database connection with row factory.

        Reuses existing connection for the current thread if available.
        """
        if not hasattr(self._local, 'connection') or self._local.connection is None:
            self._local.connection = sqlite3.connect(self.db_path)
            self._local.connection.row_factory = sqlite3.Row
        return self._local.connection

    @contextmanager
    def _transaction(self):
        """Context manager for database transactions.

        Automatically commits on success, rolls back on error.
        """
        conn = self._get_connection()
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error("Database transaction failed: %s", e)
            raise

    def close(self):
        """Close the thread-local connection if open."""
        if hasattr(self._local, 'connection') and self._local.connection:
            self._local.connection.close()
            self._local.connection = None

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - close connection."""
        self.close()
        return False

    def _init_tables(self) -> None:
        """Initialize database tables."""
        conn = self._get_connection()
        cursor = conn.cursor()

        # Datasets table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS datasets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT NOT NULL,
                dataset_id TEXT NOT NULL,
                name TEXT NOT NULL,
                author TEXT,
                url TEXT,
                created_at TEXT,
                first_seen TEXT NOT NULL,
                UNIQUE(source, dataset_id)
            )
        """)

        # Daily statistics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS daily_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                dataset_id INTEGER NOT NULL,
                date TEXT NOT NULL,
                downloads INTEGER DEFAULT 0,
                likes INTEGER DEFAULT 0,
                stars INTEGER DEFAULT 0,
                FOREIGN KEY (dataset_id) REFERENCES datasets(id),
                UNIQUE(dataset_id, date)
            )
        """)

        # Models table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS models (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id TEXT NOT NULL UNIQUE,
                name TEXT NOT NULL,
                author TEXT,
                downloads INTEGER DEFAULT 0,
                likes INTEGER DEFAULT 0,
                pipeline_tag TEXT,
                url TEXT,
                first_seen TEXT NOT NULL,
                last_updated TEXT NOT NULL
            )
        """)

        # Model-Dataset relationships
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS model_datasets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id INTEGER NOT NULL,
                dataset_id INTEGER NOT NULL,
                relationship TEXT DEFAULT 'training',
                discovered_at TEXT NOT NULL,
                FOREIGN KEY (model_id) REFERENCES models(id),
                FOREIGN KEY (dataset_id) REFERENCES datasets(id),
                UNIQUE(model_id, dataset_id)
            )
        """)

        # Trends table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trends (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                dataset_id INTEGER NOT NULL,
                date TEXT NOT NULL,
                downloads_7d_growth REAL,
                downloads_30d_growth REAL,
                FOREIGN KEY (dataset_id) REFERENCES datasets(id),
                UNIQUE(dataset_id, date)
            )
        """)

        # Valuable datasets table (v3)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS valuable_datasets (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                source TEXT,
                url TEXT,
                value_score INTEGER DEFAULT 0,
                sota_model_count INTEGER DEFAULT 0,
                citation_count INTEGER DEFAULT 0,
                citation_growth_rate REAL DEFAULT 0,
                model_usage_count INTEGER DEFAULT 0,
                institution TEXT,
                is_top_institution BOOLEAN DEFAULT 0,
                paper_url TEXT,
                code_url TEXT,
                domain TEXT,
                first_seen TEXT NOT NULL,
                last_updated TEXT NOT NULL
            )
        """)

        # SOTA model-dataset relationships (v3)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sota_model_datasets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT NOT NULL,
                dataset_id TEXT NOT NULL,
                usage_type TEXT DEFAULT 'evaluation',
                area TEXT,
                metric_name TEXT,
                metric_value REAL,
                paper_url TEXT,
                discovered_at TEXT NOT NULL,
                UNIQUE(model_name, dataset_id)
            )
        """)

        # Citation tracking table (v3)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS citation_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                paper_id TEXT NOT NULL,
                date TEXT NOT NULL,
                citation_count INTEGER DEFAULT 0,
                UNIQUE(paper_id, date)
            )
        """)

        # Create indices for better query performance
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_daily_stats_date ON daily_stats(date)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_daily_stats_dataset ON daily_stats(dataset_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_trends_date ON trends(date)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_models_downloads ON models(downloads DESC)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_valuable_datasets_score ON valuable_datasets(value_score DESC)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_sota_model_datasets_dataset ON sota_model_datasets(dataset_id)")
    # Dataset operations
    def upsert_dataset(
        self,
        source: str,
        dataset_id: str,
        name: str,
        author: Optional[str] = None,
        url: Optional[str] = None,
        created_at: Optional[str] = None,
    ) -> int:
        """Insert or update a dataset record.

        Returns:
            The database ID of the dataset.
        """
        with self._transaction() as conn:
            cursor = conn.cursor()
            now = datetime.now().isoformat()

            cursor.execute(
                """
                INSERT INTO datasets (source, dataset_id, name, author, url, created_at, first_seen)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(source, dataset_id) DO UPDATE SET
                    name = excluded.name,
                    author = excluded.author,
                    url = excluded.url
                """,
                (source, dataset_id, name, author, url, created_at, now),
            )

            # Get the ID of the inserted/updated record
            cursor.execute(
                "SELECT id FROM datasets WHERE source = ? AND dataset_id = ?",
                (source, dataset_id),
            )
            result = cursor.fetchone()
            return result["id"]

    def get_dataset(self, source: str, dataset_id: str) -> Optional[dict]:
        """Get a dataset by source and ID."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM datasets WHERE source = ? AND dataset_id = ?",
            (source, dataset_id),
        )
        row = cursor.fetchone()
        return dict(row) if row else None

    def get_dataset_by_id(self, db_id: int) -> Optional[dict]:
        """Get a dataset by database ID."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM datasets WHERE id = ?", (db_id,))
        row = cursor.fetchone()
        return dict(row) if row else None

    def get_all_datasets(self, source: Optional[str] = None) -> list[dict]:
        """Get all datasets, optionally filtered by source."""
        conn = self._get_connection()
        cursor = conn.cursor()
        if source:
            cursor.execute("SELECT * FROM datasets WHERE source = ?", (source,))
        else:
            cursor.execute("SELECT * FROM datasets")
        rows = cursor.fetchall()
        return [dict(row) for row in rows]

    # Daily stats operations
    def record_daily_stats(
        self,
        dataset_db_id: int,
        downloads: int = 0,
        likes: int = 0,
        stars: int = 0,
        date: Optional[str] = None,
    ) -> None:
        """Record daily statistics for a dataset."""
        conn = self._get_connection()
        cursor = conn.cursor()
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")

        cursor.execute(
            """
            INSERT INTO daily_stats (dataset_id, date, downloads, likes, stars)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(dataset_id, date) DO UPDATE SET
                downloads = excluded.downloads,
                likes = excluded.likes,
                stars = excluded.stars
            """,
            (dataset_db_id, date, downloads, likes, stars),
        )

    def get_stats_history(
        self,
        dataset_db_id: int,
        days: int = 30,
    ) -> list[dict]:
        """Get download history for a dataset."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

        cursor.execute(
            """
            SELECT * FROM daily_stats
            WHERE dataset_id = ? AND date >= ?
            ORDER BY date ASC
            """,
            (dataset_db_id, cutoff),
        )
        rows = cursor.fetchall()

        return [dict(row) for row in rows]

    # Model operations
    def upsert_model(
        self,
        model_id: str,
        name: str,
        author: Optional[str] = None,
        downloads: int = 0,
        likes: int = 0,
        pipeline_tag: Optional[str] = None,
        url: Optional[str] = None,
    ) -> int:
        """Insert or update a model record."""
        conn = self._get_connection()
        cursor = conn.cursor()
        now = datetime.now().isoformat()

        cursor.execute(
            """
            INSERT INTO models (model_id, name, author, downloads, likes, pipeline_tag, url, first_seen, last_updated)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(model_id) DO UPDATE SET
                name = excluded.name,
                author = excluded.author,
                downloads = excluded.downloads,
                likes = excluded.likes,
                pipeline_tag = excluded.pipeline_tag,
                url = excluded.url,
                last_updated = excluded.last_updated
            """,
            (model_id, name, author, downloads, likes, pipeline_tag, url, now, now),
        )

        cursor.execute("SELECT id FROM models WHERE model_id = ?", (model_id,))
        result = cursor.fetchone()
        return result["id"]

    def get_model(self, model_id: str) -> Optional[dict]:
        """Get a model by ID."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM models WHERE model_id = ?", (model_id,))
        row = cursor.fetchone()

        return dict(row) if row else None

    def get_top_models(self, limit: int = 100) -> list[dict]:
        """Get top models by download count."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM models ORDER BY downloads DESC LIMIT ?",
            (limit,),
        )
        rows = cursor.fetchall()

        return [dict(row) for row in rows]

    # Model-Dataset relationship operations
    def add_model_dataset_link(
        self,
        model_db_id: int,
        dataset_db_id: int,
        relationship: str = "training",
    ) -> None:
        """Link a model to a dataset."""
        conn = self._get_connection()
        cursor = conn.cursor()
        now = datetime.now().isoformat()

        cursor.execute(
            """
            INSERT INTO model_datasets (model_id, dataset_id, relationship, discovered_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(model_id, dataset_id) DO UPDATE SET
                relationship = excluded.relationship
            """,
            (model_db_id, dataset_db_id, relationship, now),
        )

    def get_datasets_for_model(self, model_db_id: int) -> list[dict]:
        """Get all datasets linked to a model."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT d.*, md.relationship
            FROM datasets d
            JOIN model_datasets md ON d.id = md.dataset_id
            WHERE md.model_id = ?
            """,
            (model_db_id,),
        )
        rows = cursor.fetchall()

        return [dict(row) for row in rows]

    def get_models_using_dataset(self, dataset_db_id: int) -> list[dict]:
        """Get all models that use a specific dataset."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT m.*, md.relationship
            FROM models m
            JOIN model_datasets md ON m.id = md.model_id
            WHERE md.dataset_id = ?
            ORDER BY m.downloads DESC
            """,
            (dataset_db_id,),
        )
        rows = cursor.fetchall()

        return [dict(row) for row in rows]

    def get_most_used_datasets(self, limit: int = 20) -> list[dict]:
        """Get datasets ranked by how many models use them."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT d.*, COUNT(md.model_id) as model_count,
                   SUM(m.downloads) as total_model_downloads
            FROM datasets d
            JOIN model_datasets md ON d.id = md.dataset_id
            JOIN models m ON md.model_id = m.id
            GROUP BY d.id
            ORDER BY model_count DESC, total_model_downloads DESC
            LIMIT ?
            """,
            (limit,),
        )
        rows = cursor.fetchall()

        return [dict(row) for row in rows]

    # Trend operations
    def record_trend(
        self,
        dataset_db_id: int,
        downloads_7d_growth: Optional[float] = None,
        downloads_30d_growth: Optional[float] = None,
        date: Optional[str] = None,
    ) -> None:
        """Record calculated trend for a dataset."""
        conn = self._get_connection()
        cursor = conn.cursor()
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")

        cursor.execute(
            """
            INSERT INTO trends (dataset_id, date, downloads_7d_growth, downloads_30d_growth)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(dataset_id, date) DO UPDATE SET
                downloads_7d_growth = excluded.downloads_7d_growth,
                downloads_30d_growth = excluded.downloads_30d_growth
            """,
            (dataset_db_id, date, downloads_7d_growth, downloads_30d_growth),
        )

    def get_rising_datasets(
        self,
        min_growth: float = 0.5,
        days: int = 7,
        limit: int = 20,
    ) -> list[dict]:
        """Get datasets with growth above threshold.

        Args:
            min_growth: Minimum growth rate (0.5 = 50%).
            days: Use 7 or 30 day growth.
            limit: Maximum results to return.

        Returns:
            List of datasets with their trend data and current downloads.
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        today = datetime.now().strftime("%Y-%m-%d")

        growth_column = _get_growth_column(days)

        cursor.execute(
            f"""
            SELECT d.*, t.{growth_column} as growth, t.date as trend_date,
                   ds.downloads as current_downloads, ds.likes as current_likes
            FROM datasets d
            JOIN trends t ON d.id = t.dataset_id
            LEFT JOIN daily_stats ds ON d.id = ds.dataset_id AND ds.date = ?
            WHERE t.date = ? AND t.{growth_column} >= ?
            ORDER BY t.{growth_column} DESC
            LIMIT ?
            """,
            (today, today, min_growth, limit),
        )
        rows = cursor.fetchall()

        return [dict(row) for row in rows]

    def get_breakthrough_datasets(
        self,
        threshold: int = 1000,
        days: int = 7,
        limit: int = 20,
    ) -> list[dict]:
        """Get datasets that broke through from near-zero to significant downloads.

        Identifies datasets that went from <100 downloads to 1000+ downloads.

        Args:
            threshold: Download count to consider as "breakthrough" (default: 1000).
            days: Lookback period for detecting breakthroughs.
            limit: Maximum results to return.

        Returns:
            List of breakthrough datasets with growth details.
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        today = datetime.now().strftime("%Y-%m-%d")
        cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

        cursor.execute(
            """
            SELECT d.*,
                   old_stats.downloads as old_downloads,
                   new_stats.downloads as current_downloads,
                   (new_stats.downloads - COALESCE(old_stats.downloads, 0)) as download_increase
            FROM datasets d
            JOIN daily_stats new_stats ON d.id = new_stats.dataset_id AND new_stats.date = ?
            LEFT JOIN daily_stats old_stats ON d.id = old_stats.dataset_id AND old_stats.date = ?
            WHERE new_stats.downloads >= ?
              AND COALESCE(old_stats.downloads, 0) < 100
            ORDER BY download_increase DESC
            LIMIT ?
            """,
            (today, cutoff, threshold, limit),
        )
        rows = cursor.fetchall()

        return [dict(row) for row in rows]

    def get_top_growing_datasets(
        self,
        days: int = 7,
        limit: int = 20,
        min_downloads: int = 0,
    ) -> list[dict]:
        """Get datasets sorted by absolute download increase.

        Args:
            days: Period for growth calculation.
            limit: Maximum results.
            min_downloads: Minimum current downloads to include.

        Returns:
            List of datasets sorted by download increase.
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        today = datetime.now().strftime("%Y-%m-%d")

        growth_column = _get_growth_column(days)

        cursor.execute(
            f"""
            SELECT d.*, t.{growth_column} as growth, t.date as trend_date,
                   ds.downloads as current_downloads, ds.likes as current_likes,
                   CASE
                       WHEN t.{growth_column} = 'inf' THEN ds.downloads
                       ELSE CAST(ds.downloads * t.{growth_column} / (1 + t.{growth_column}) AS INTEGER)
                   END as download_increase
            FROM datasets d
            JOIN trends t ON d.id = t.dataset_id
            JOIN daily_stats ds ON d.id = ds.dataset_id AND ds.date = ?
            WHERE t.date = ? AND ds.downloads >= ?
                  AND t.{growth_column} IS NOT NULL AND t.{growth_column} > 0
            ORDER BY t.{growth_column} DESC
            LIMIT ?
            """,
            (today, today, min_downloads, limit),
        )
        rows = cursor.fetchall()

        return [dict(row) for row in rows]

    def calculate_growth_rate(
        self,
        dataset_db_id: int,
        days: int = 7,
    ) -> Optional[float]:
        """Calculate download growth rate for a dataset.

        Args:
            dataset_db_id: Database ID of the dataset.
            days: Number of days to calculate growth over.

        Returns:
            Growth rate as a decimal (0.5 = 50% growth), or None if insufficient data.
        """
        history = self.get_stats_history(dataset_db_id, days=days + 1)

        if len(history) < 2:
            return None

        # Get earliest and latest stats
        earliest = history[0]
        latest = history[-1]

        old_downloads = earliest.get("downloads", 0)
        new_downloads = latest.get("downloads", 0)

        if old_downloads == 0:
            return None if new_downloads == 0 else float("inf")

        return (new_downloads - old_downloads) / old_downloads

    # Valuable dataset operations (v3)
    def upsert_valuable_dataset(
        self,
        dataset_id: str,
        name: str,
        value_score: int = 0,
        sota_model_count: int = 0,
        citation_count: int = 0,
        citation_growth_rate: float = 0,
        model_usage_count: int = 0,
        institution: Optional[str] = None,
        is_top_institution: bool = False,
        paper_url: Optional[str] = None,
        code_url: Optional[str] = None,
        domain: Optional[str] = None,
        source: Optional[str] = None,
        url: Optional[str] = None,
    ) -> None:
        """Insert or update a valuable dataset record."""
        conn = self._get_connection()
        cursor = conn.cursor()
        now = datetime.now().isoformat()

        cursor.execute(
            """
            INSERT INTO valuable_datasets (
                id, name, source, url, value_score, sota_model_count,
                citation_count, citation_growth_rate, model_usage_count,
                institution, is_top_institution, paper_url, code_url,
                domain, first_seen, last_updated
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                name = excluded.name,
                value_score = excluded.value_score,
                sota_model_count = excluded.sota_model_count,
                citation_count = excluded.citation_count,
                citation_growth_rate = excluded.citation_growth_rate,
                model_usage_count = excluded.model_usage_count,
                institution = excluded.institution,
                is_top_institution = excluded.is_top_institution,
                paper_url = excluded.paper_url,
                code_url = excluded.code_url,
                domain = excluded.domain,
                last_updated = excluded.last_updated
            """,
            (
                dataset_id, name, source, url, value_score, sota_model_count,
                citation_count, citation_growth_rate, model_usage_count,
                institution, is_top_institution, paper_url, code_url,
                domain, now, now,
            ),
        )

    def get_valuable_datasets(
        self,
        min_score: int = 0,
        domain: Optional[str] = None,
        top_institution_only: bool = False,
        limit: int = 50,
    ) -> list[dict]:
        """Get valuable datasets filtered by criteria.

        Args:
            min_score: Minimum value score threshold.
            domain: Filter by domain (e.g., "robotics").
            top_institution_only: Only return datasets from top institutions.
            limit: Maximum results.

        Returns:
            List of valuable dataset records.
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        query = "SELECT * FROM valuable_datasets WHERE value_score >= ?"
        params = [min_score]

        if domain:
            query += " AND domain = ?"
            params.append(domain)

        if top_institution_only:
            query += " AND is_top_institution = 1"

        query += " ORDER BY value_score DESC LIMIT ?"
        params.append(limit)

        cursor.execute(query, params)
        rows = cursor.fetchall()

        return [dict(row) for row in rows]

    def get_top_valuable_datasets(self, limit: int = 20) -> list[dict]:
        """Get top valuable datasets by score."""
        return self.get_valuable_datasets(min_score=0, limit=limit)

    # SOTA model-dataset operations (v3)
    def add_sota_model_dataset(
        self,
        model_name: str,
        dataset_id: str,
        usage_type: str = "evaluation",
        area: Optional[str] = None,
        metric_name: Optional[str] = None,
        metric_value: Optional[float] = None,
        paper_url: Optional[str] = None,
    ) -> None:
        """Record a SOTA model's use of a dataset."""
        conn = self._get_connection()
        cursor = conn.cursor()
        now = datetime.now().isoformat()

        cursor.execute(
            """
            INSERT INTO sota_model_datasets (
                model_name, dataset_id, usage_type, area,
                metric_name, metric_value, paper_url, discovered_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(model_name, dataset_id) DO UPDATE SET
                usage_type = excluded.usage_type,
                metric_name = excluded.metric_name,
                metric_value = excluded.metric_value
            """,
            (model_name, dataset_id, usage_type, area, metric_name, metric_value, paper_url, now),
        )

    def get_sota_models_for_dataset(self, dataset_id: str) -> list[dict]:
        """Get all SOTA models that use a dataset."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT * FROM sota_model_datasets
            WHERE dataset_id = ?
            ORDER BY metric_value DESC NULLS LAST
            """,
            (dataset_id,),
        )
        rows = cursor.fetchall()

        return [dict(row) for row in rows]

    def get_datasets_by_sota_count(self, limit: int = 20) -> list[dict]:
        """Get datasets ranked by number of SOTA models using them."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT dataset_id, COUNT(*) as sota_count,
                   GROUP_CONCAT(DISTINCT area) as areas
            FROM sota_model_datasets
            GROUP BY dataset_id
            ORDER BY sota_count DESC
            LIMIT ?
            """,
            (limit,),
        )
        rows = cursor.fetchall()

        return [dict(row) for row in rows]

    # Citation history operations (v3)
    def record_citation(
        self,
        paper_id: str,
        citation_count: int,
        date: Optional[str] = None,
    ) -> None:
        """Record citation count for a paper."""
        conn = self._get_connection()
        cursor = conn.cursor()
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")

        cursor.execute(
            """
            INSERT INTO citation_history (paper_id, date, citation_count)
            VALUES (?, ?, ?)
            ON CONFLICT(paper_id, date) DO UPDATE SET
                citation_count = excluded.citation_count
            """,
            (paper_id, date, citation_count),
        )

    def get_citation_history(self, paper_id: str, days: int = 30) -> list[dict]:
        """Get citation history for a paper."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

        cursor.execute(
            """
            SELECT * FROM citation_history
            WHERE paper_id = ? AND date >= ?
            ORDER BY date ASC
            """,
            (paper_id, cutoff),
        )
        rows = cursor.fetchall()

        return [dict(row) for row in rows]

    def get_fast_growing_papers(
        self,
        min_growth_per_month: int = 10,
        limit: int = 20,
    ) -> list[dict]:
        """Get papers with rapid citation growth."""
        conn = self._get_connection()
        cursor = conn.cursor()
        today = datetime.now().strftime("%Y-%m-%d")
        month_ago = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

        cursor.execute(
            """
            SELECT new.paper_id,
                   new.citation_count as current_citations,
                   COALESCE(old.citation_count, 0) as old_citations,
                   (new.citation_count - COALESCE(old.citation_count, 0)) as growth
            FROM citation_history new
            LEFT JOIN citation_history old ON new.paper_id = old.paper_id AND old.date = ?
            WHERE new.date = ?
              AND (new.citation_count - COALESCE(old.citation_count, 0)) >= ?
            ORDER BY growth DESC
            LIMIT ?
            """,
            (month_ago, today, min_growth_per_month, limit),
        )
        rows = cursor.fetchall()

        return [dict(row) for row in rows]


def get_database(config: dict) -> RadarDatabase:
    """Get database instance from configuration.

    Args:
        config: Configuration dictionary.

    Returns:
        RadarDatabase instance.
    """
    db_config = config.get("database", {})
    db_path = db_config.get("path", "data/radar.db")
    return RadarDatabase(db_path)
