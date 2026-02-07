"""Simple file-based cache for API responses."""

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Optional

from utils.logging_config import get_logger

logger = get_logger("cache")


class FileCache:
    """File-based cache with TTL support.

    Stores cached data as JSON files in a cache directory.
    """

    def __init__(self, cache_dir: str = "data/cache", default_ttl: int = 3600):
        """Initialize cache.

        Args:
            cache_dir: Directory to store cache files.
            default_ttl: Default time-to-live in seconds (1 hour).
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.default_ttl = default_ttl

    def _get_cache_path(self, key: str) -> Path:
        """Get cache file path for a key."""
        # Use MD5 hash for filename to handle special characters
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.json"

    def get(self, key: str) -> Optional[Any]:
        """Get cached value if not expired.

        Args:
            key: Cache key.

        Returns:
            Cached value or None if not found/expired.
        """
        cache_path = self._get_cache_path(key)

        if not cache_path.exists():
            return None

        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Check expiration
            if time.time() > data.get("expires_at", 0):
                cache_path.unlink()  # Remove expired cache
                return None

            return data.get("value")
        except (json.JSONDecodeError, IOError) as e:
            logger.warning("Cache read error for %s: %s", key, e)
            return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set cached value with TTL.

        Args:
            key: Cache key.
            value: Value to cache (must be JSON-serializable).
            ttl: Time-to-live in seconds, uses default if not specified.
        """
        if ttl is None:
            ttl = self.default_ttl

        cache_path = self._get_cache_path(key)
        data = {
            "key": key,
            "value": value,
            "created_at": time.time(),
            "expires_at": time.time() + ttl,
        }

        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)
        except (IOError, TypeError) as e:
            logger.warning("Cache write error for %s: %s", key, e)

    def delete(self, key: str) -> bool:
        """Delete cached value.

        Args:
            key: Cache key.

        Returns:
            True if deleted, False if not found.
        """
        cache_path = self._get_cache_path(key)
        if cache_path.exists():
            cache_path.unlink()
            return True
        return False

    def clear(self) -> int:
        """Clear all cached values.

        Returns:
            Number of cache files deleted.
        """
        count = 0
        for cache_file in self.cache_dir.glob("*.json"):
            cache_file.unlink()
            count += 1
        return count

    def clear_expired(self) -> int:
        """Clear only expired cache entries.

        Returns:
            Number of expired files deleted.
        """
        count = 0
        now = time.time()

        for cache_file in self.cache_dir.glob("*.json"):
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if now > data.get("expires_at", 0):
                    cache_file.unlink()
                    count += 1
            except (json.JSONDecodeError, IOError):
                cache_file.unlink()
                count += 1

        return count


# Global cache instance
_cache: Optional[FileCache] = None


def get_cache(cache_dir: str = "data/cache", ttl: int = 3600) -> FileCache:
    """Get or create global cache instance.

    Args:
        cache_dir: Cache directory path.
        ttl: Default TTL in seconds.

    Returns:
        FileCache instance.
    """
    global _cache
    if _cache is None:
        _cache = FileCache(cache_dir, ttl)
    return _cache


def cached(key_prefix: str, ttl: int = 3600):
    """Decorator for caching function results.

    Args:
        key_prefix: Prefix for cache key.
        ttl: Time-to-live in seconds.

    Returns:
        Decorator function.
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            # Build cache key from function name and arguments
            key_parts = [key_prefix, func.__name__]
            key_parts.extend(str(arg) for arg in args)
            key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
            cache_key = ":".join(key_parts)

            cache = get_cache()
            result = cache.get(cache_key)

            if result is not None:
                logger.debug("Cache hit: %s", cache_key)
                return result

            logger.debug("Cache miss: %s", cache_key)
            result = func(*args, **kwargs)

            if result is not None:
                cache.set(cache_key, result, ttl)

            return result

        return wrapper

    return decorator
