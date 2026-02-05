"""Keyword matching utilities for AI Dataset Radar."""

import re
from typing import Iterable


def match_keywords(text: str, keywords: Iterable[str], case_sensitive: bool = False) -> list[str]:
    """Find all matching keywords in text.

    Args:
        text: Text to search in.
        keywords: Keywords to look for.
        case_sensitive: Whether matching is case-sensitive.

    Returns:
        List of matched keywords.
    """
    if not text:
        return []

    if not case_sensitive:
        text = text.lower()

    matches = []
    for kw in keywords:
        search_kw = kw if case_sensitive else kw.lower()
        if search_kw in text:
            matches.append(kw)

    return matches


def count_keyword_matches(text: str, keywords: Iterable[str], case_sensitive: bool = False) -> int:
    """Count how many keywords match in text.

    Args:
        text: Text to search in.
        keywords: Keywords to look for.
        case_sensitive: Whether matching is case-sensitive.

    Returns:
        Number of matching keywords.
    """
    return len(match_keywords(text, keywords, case_sensitive))


def calculate_relevance(
    name: str,
    description: str,
    topics: list[str],
    keywords: Iterable[str],
    high_threshold: int = 2
) -> tuple[str, list[str]]:
    """Calculate relevance score based on keyword matches.

    Args:
        name: Item name.
        description: Item description.
        topics: List of topic tags.
        keywords: Keywords to match against.
        high_threshold: Minimum matches for "high" relevance.

    Returns:
        Tuple of (relevance level, matched signals).
    """
    text = f"{name} {description} {' '.join(topics)}".lower()
    matches = match_keywords(text, keywords)

    if len(matches) >= high_threshold:
        return "high", matches
    elif matches:
        return "medium", matches
    else:
        return "low", []
