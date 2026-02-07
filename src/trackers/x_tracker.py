"""X (Twitter) tracker for monitoring AI research accounts.

Supports multiple backends:
- RSSHub: Self-hosted or public instance converting X feeds to RSS
- X API v2: Official API (requires bearer token)

Monitors AI labs, researchers, and data vendors for:
- Dataset release announcements
- Model release announcements
- Research paper shares
- Industry news and trends
"""

import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from typing import Optional

import feedparser
import requests

from utils.logging_config import get_logger

logger = get_logger("x_tracker")

# Signal keywords for detecting relevant tweets
SIGNAL_KEYWORDS = [
    # Dataset & Training
    "dataset", "training data", "benchmark", "evaluation",
    "open source", "open-source", "releasing", "we release",
    "announcing", "introducing", "new paper", "new model",
    # AI Research
    "fine-tuning", "fine-tune", "rlhf", "human feedback",
    "instruction tuning", "alignment", "synthetic data",
    "pre-training", "pretraining", "language model", "llm",
    "multimodal", "vision-language", "reasoning",
    # Chinese keywords
    "开源", "发布", "数据集", "模型", "论文", "训练",
]


class XTracker:
    """Track X (Twitter) accounts for AI-related announcements.

    Uses RSSHub or X API v2 as backend to fetch tweets.
    """

    def __init__(self, config: dict):
        """Initialize X tracker.

        Args:
            config: Configuration dict with x_tracker settings.
        """
        self.config = config
        x_config = config.get("x_tracker", {})

        # Backend: "rsshub" or "api"
        self.backend = x_config.get("backend", "rsshub")

        # RSSHub settings
        self.rsshub_base = x_config.get("rsshub_url", "https://rsshub.app").rstrip("/")

        # X API v2 settings
        self.bearer_token = x_config.get("bearer_token", "")

        # Watched accounts
        self.accounts = x_config.get("accounts", [])

        # Search keywords for X API
        self.search_keywords = x_config.get("search_keywords", [])

        self.session = requests.Session()
        self.session.headers["User-Agent"] = "AI-Dataset-Radar/5.0"

    def _fetch_rsshub_feed(self, username: str, days: int = 7) -> list[dict]:
        """Fetch tweets via RSSHub RSS feed.

        Args:
            username: X/Twitter username (without @).
            days: Look back period in days.

        Returns:
            List of tweet dicts.
        """
        # RSSHub route: /twitter/user/:id
        # Also supports: /twitter/keyword/:keyword
        feed_url = f"{self.rsshub_base}/twitter/user/{username}"

        try:
            feed = feedparser.parse(feed_url)
        except Exception as e:
            logger.warning("Failed to parse RSSHub feed for @%s: %s", username, e)
            return []

        if not feed.entries:
            return []

        cutoff = datetime.utcnow() - timedelta(days=days)
        tweets = []

        for entry in feed.entries[:30]:
            # Parse date
            published = None
            if hasattr(entry, "published_parsed") and entry.published_parsed:
                try:
                    published = datetime(*entry.published_parsed[:6])
                except (TypeError, ValueError):
                    pass

            if published and published < cutoff:
                continue

            # Get text content
            title = entry.get("title", "").strip()
            summary = entry.get("summary", "").strip()
            # Clean HTML from summary
            text = re.sub(r"<[^>]+>", "", summary) if summary else title

            tweet = {
                "username": username,
                "text": text[:500],
                "url": entry.get("link", ""),
                "date": published.strftime("%Y-%m-%d") if published else "",
                "source": "x_rsshub",
            }

            tweet["signals"] = self._extract_signals(tweet)
            tweets.append(tweet)

        return tweets

    def _fetch_api_user_tweets(self, username: str, days: int = 7) -> list[dict]:
        """Fetch tweets via X API v2.

        Args:
            username: X/Twitter username (without @).
            days: Look back period in days.

        Returns:
            List of tweet dicts.
        """
        if not self.bearer_token:
            return []

        headers = {
            "Authorization": f"Bearer {self.bearer_token}",
            "User-Agent": "AI-Dataset-Radar/5.0",
        }

        # Step 1: Get user ID
        user_url = f"https://api.twitter.com/2/users/by/username/{username}"
        try:
            resp = requests.get(user_url, headers=headers, timeout=15)
            if resp.status_code != 200:
                logger.warning("X API user lookup failed for @%s: %s", username, resp.status_code)
                return []
            user_id = resp.json().get("data", {}).get("id")
            if not user_id:
                return []
        except requests.RequestException as e:
            logger.warning("X API error for @%s: %s", username, e)
            return []

        # Step 2: Get recent tweets
        cutoff = datetime.utcnow() - timedelta(days=days)
        start_time = cutoff.strftime("%Y-%m-%dT%H:%M:%SZ")

        tweets_url = f"https://api.twitter.com/2/users/{user_id}/tweets"
        params = {
            "max_results": 20,
            "start_time": start_time,
            "tweet.fields": "created_at,text,public_metrics",
        }

        try:
            resp = requests.get(tweets_url, headers=headers, params=params, timeout=15)
            if resp.status_code == 429:
                logger.warning("X API rate limited")
                return []
            if resp.status_code != 200:
                return []

            data = resp.json().get("data", [])
        except requests.RequestException as e:
            logger.warning("X API tweets error for @%s: %s", username, e)
            return []

        tweets = []
        for item in data:
            text = item.get("text", "")
            created_at = item.get("created_at", "")
            date_str = ""
            if created_at:
                try:
                    dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                    date_str = dt.strftime("%Y-%m-%d")
                except ValueError:
                    pass

            tweet = {
                "username": username,
                "text": text[:500],
                "url": f"https://x.com/{username}/status/{item.get('id', '')}",
                "date": date_str,
                "source": "x_api",
                "metrics": item.get("public_metrics", {}),
            }

            tweet["signals"] = self._extract_signals(tweet)
            tweets.append(tweet)

        return tweets

    def _fetch_api_search(self, query: str, days: int = 7) -> list[dict]:
        """Search tweets via X API v2.

        Args:
            query: Search query string.
            days: Look back period in days.

        Returns:
            List of tweet dicts.
        """
        if not self.bearer_token:
            return []

        headers = {
            "Authorization": f"Bearer {self.bearer_token}",
        }

        cutoff = datetime.utcnow() - timedelta(days=days)
        start_time = cutoff.strftime("%Y-%m-%dT%H:%M:%SZ")

        url = "https://api.twitter.com/2/tweets/search/recent"
        params = {
            "query": query,
            "max_results": 20,
            "start_time": start_time,
            "tweet.fields": "created_at,text,author_id,public_metrics",
        }

        try:
            resp = requests.get(url, headers=headers, params=params, timeout=15)
            if resp.status_code != 200:
                return []
            data = resp.json().get("data", [])
        except requests.RequestException:
            return []

        tweets = []
        for item in data:
            tweet = {
                "username": "",
                "text": item.get("text", "")[:500],
                "url": f"https://x.com/i/status/{item.get('id', '')}",
                "date": item.get("created_at", "")[:10],
                "source": "x_api_search",
                "query": query,
            }
            tweet["signals"] = self._extract_signals(tweet)
            tweets.append(tweet)

        return tweets

    def _extract_signals(self, tweet: dict) -> list[str]:
        """Extract signal keywords from tweet text.

        Args:
            tweet: Tweet dict with text field.

        Returns:
            List of matched signal keywords.
        """
        text = tweet.get("text", "").lower()
        signals = []
        for keyword in SIGNAL_KEYWORDS:
            if keyword.lower() in text:
                signals.append(keyword)
        return list(set(signals))

    def fetch_account(self, username: str, days: int = 7) -> dict:
        """Fetch tweets from a single account.

        Args:
            username: X/Twitter username (without @).
            days: Look back period in days.

        Returns:
            Account activity dict.
        """
        username = username.lstrip("@")

        if self.backend == "api" and self.bearer_token:
            tweets = self._fetch_api_user_tweets(username, days)
        else:
            tweets = self._fetch_rsshub_feed(username, days)

        # Filter to relevant tweets only
        relevant = [t for t in tweets if t.get("signals")]

        return {
            "username": username,
            "total_tweets": len(tweets),
            "relevant_tweets": relevant,
            "has_activity": len(relevant) > 0,
        }

    def fetch_all(self, days: int = 7) -> dict:
        """Fetch tweets from all configured accounts (parallelized).

        Args:
            days: Look back period in days.

        Returns:
            Dict with account activities and search results.
        """
        logger.info("Tracking %d X/Twitter accounts...", len(self.accounts))

        results = {
            "accounts": [],
            "search_results": [],
        }

        # Fetch accounts in parallel
        if self.accounts:
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = {
                    executor.submit(self.fetch_account, acct, days): acct
                    for acct in self.accounts
                }
                for future in futures:
                    try:
                        activity = future.result()
                        if activity["has_activity"]:
                            results["accounts"].append(activity)
                    except Exception as e:
                        logger.warning("Error fetching @%s: %s", futures[future], e)

        # Run keyword searches (API only)
        if self.backend == "api" and self.bearer_token and self.search_keywords:
            for query in self.search_keywords:
                try:
                    tweets = self._fetch_api_search(query, days)
                    if tweets:
                        results["search_results"].extend(tweets)
                except Exception as e:
                    logger.warning("X search error for '%s': %s", query, e)

        active = len(results["accounts"])
        tweet_count = sum(len(a["relevant_tweets"]) for a in results["accounts"])
        logger.info("Found %d active accounts with %d relevant tweets", active, tweet_count)

        return results
