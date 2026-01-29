"""Database operations for AI Dataset Radar."""

import os
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional


class RadarDatabase:
    """SQLite database for storing radar data and historical trends."""

    def __init__(self, db_path: str = "data/radar.db"):
        """Initialize database connection.

        Args:
            db_path: Path to SQLite database file.
        """
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_tables()

    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection with row factory."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

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

        # Create indices for better query performance
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_daily_stats_date ON daily_stats(date)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_daily_stats_dataset ON daily_stats(dataset_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_trends_date ON trends(date)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_models_downloads ON models(downloads DESC)")

        conn.commit()
        conn.close()

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
        conn = self._get_connection()
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
        conn.commit()
        conn.close()
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
        conn.close()
        return dict(row) if row else None

    def get_dataset_by_id(self, db_id: int) -> Optional[dict]:
        """Get a dataset by database ID."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM datasets WHERE id = ?", (db_id,))
        row = cursor.fetchone()
        conn.close()
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
        conn.close()
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
        conn.commit()
        conn.close()

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
        conn.close()
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
        conn.commit()
        conn.close()
        return result["id"]

    def get_model(self, model_id: str) -> Optional[dict]:
        """Get a model by ID."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM models WHERE model_id = ?", (model_id,))
        row = cursor.fetchone()
        conn.close()
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
        conn.close()
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
        conn.commit()
        conn.close()

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
        conn.close()
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
        conn.close()
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
        conn.close()
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
        conn.commit()
        conn.close()

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
            List of datasets with their trend data.
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        today = datetime.now().strftime("%Y-%m-%d")

        growth_column = "downloads_7d_growth" if days == 7 else "downloads_30d_growth"

        cursor.execute(
            f"""
            SELECT d.*, t.{growth_column} as growth, t.date as trend_date
            FROM datasets d
            JOIN trends t ON d.id = t.dataset_id
            WHERE t.date = ? AND t.{growth_column} >= ?
            ORDER BY t.{growth_column} DESC
            LIMIT ?
            """,
            (today, min_growth, limit),
        )
        rows = cursor.fetchall()
        conn.close()
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

    def close(self) -> None:
        """Close all database connections (no-op for SQLite with per-call connections)."""
        pass


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
