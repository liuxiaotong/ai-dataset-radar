"""Utility modules for AI Dataset Radar."""

from .logging_config import get_logger, setup_logging
from .keywords import match_keywords, count_keyword_matches, calculate_relevance
from .cache import FileCache, get_cache, cached
from .http import (
    create_session,
    get_with_retry,
    get_json,
    get_session,
    DEFAULT_TIMEOUT,
    DEFAULT_CONNECT_TIMEOUT,
    DEFAULT_READ_TIMEOUT,
)

__all__ = [
    "get_logger",
    "setup_logging",
    "match_keywords",
    "count_keyword_matches",
    "calculate_relevance",
    "FileCache",
    "get_cache",
    "cached",
    "create_session",
    "get_with_retry",
    "get_json",
    "get_session",
    "DEFAULT_TIMEOUT",
    "DEFAULT_CONNECT_TIMEOUT",
    "DEFAULT_READ_TIMEOUT",
]
