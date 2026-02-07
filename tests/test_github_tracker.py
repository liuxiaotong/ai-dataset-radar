"""Tests for GitHub organization tracker."""

import sys
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from trackers.github_tracker import GitHubTracker, DEFAULT_SIGNAL_KEYWORDS


class TestGitHubTrackerInit:
    """Tests for GitHubTracker initialization."""

    def test_init_defaults(self):
        """Test initialization with minimal config."""
        tracker = GitHubTracker({})
        assert tracker.token == ""
        assert tracker.vendor_orgs == []
        assert tracker.lab_orgs == []
        assert "Accept" in tracker.headers
        assert "User-Agent" in tracker.headers

    def test_init_with_token(self):
        """Test initialization with explicit token."""
        config = {"github": {"token": "ghp_test123"}}
        tracker = GitHubTracker(config)
        assert tracker.token == "ghp_test123"
        assert "Authorization" in tracker.headers

    def test_init_with_env_var_syntax(self):
        """Test ${VAR} syntax resolves from environment."""
        config = {"github": {"token": "${GITHUB_TOKEN}"}}
        with patch.dict("os.environ", {"GITHUB_TOKEN": "ghp_env_token"}):
            tracker = GitHubTracker(config)
            assert tracker.token == "ghp_env_token"

    def test_init_with_env_fallback(self):
        """Test fallback to GITHUB_TOKEN env var."""
        config = {"github": {"token": ""}}
        with patch.dict("os.environ", {"GITHUB_TOKEN": "ghp_fallback"}):
            tracker = GitHubTracker(config)
            assert tracker.token == "ghp_fallback"

    def test_init_no_auth_header_without_token(self):
        """Test no Authorization header when no token."""
        tracker = GitHubTracker({})
        assert "Authorization" not in tracker.headers

    def test_init_orgs_from_config(self):
        """Test org lists from config."""
        config = {
            "github": {
                "orgs": {
                    "data_vendors": ["scaleapi", "argilla-io"],
                    "ai_labs": ["openai", "deepseek-ai"],
                }
            }
        }
        tracker = GitHubTracker(config)
        assert tracker.vendor_orgs == ["scaleapi", "argilla-io"]
        assert tracker.lab_orgs == ["openai", "deepseek-ai"]

    def test_init_custom_relevance_keywords(self):
        """Test custom relevance keywords from config."""
        config = {"sources": {"github": {"relevance_keywords": ["custom1", "custom2"]}}}
        tracker = GitHubTracker(config)
        assert tracker.relevance_keywords == ["custom1", "custom2"]

    def test_init_default_relevance_keywords(self):
        """Test default keywords used when not configured."""
        tracker = GitHubTracker({})
        assert tracker.relevance_keywords == DEFAULT_SIGNAL_KEYWORDS

    def test_session_has_connection_pooling(self):
        """Test that session is created for connection pooling."""
        tracker = GitHubTracker({})
        assert tracker.session is not None


class TestExtractSignals:
    """Tests for legacy signal extraction."""

    @pytest.fixture
    def tracker(self):
        return GitHubTracker({})

    def test_extract_from_name(self, tracker):
        """Test extracting signals from repo name."""
        repo = {"name": "rlhf-toolkit", "description": "", "topics": []}
        signals = tracker._extract_signals(repo)
        assert "rlhf" in signals

    def test_extract_from_description(self, tracker):
        """Test extracting signals from description."""
        repo = {"name": "project", "description": "A dataset for fine-tuning", "topics": []}
        signals = tracker._extract_signals(repo)
        assert "dataset" in signals
        assert "fine-tuning" in signals or "fine-tune" in signals

    def test_extract_from_topics(self, tracker):
        """Test extracting signals from topics."""
        repo = {"name": "project", "description": "", "topics": ["synthetic-data", "llm"]}
        signals = tracker._extract_signals(repo)
        assert "synthetic-data" in signals
        assert "llm" in signals

    def test_no_signals(self, tracker):
        """Test repo with no matching signals."""
        repo = {"name": "my-website", "description": "Personal blog", "topics": []}
        signals = tracker._extract_signals(repo)
        assert signals == []

    def test_hyphen_matching(self, tracker):
        """Test hyphenated keyword matching."""
        repo = {"name": "project", "description": "human in the loop annotation", "topics": []}
        signals = tracker._extract_signals(repo)
        assert "human-in-the-loop" in signals

    def test_deduplication(self, tracker):
        """Test signals are deduplicated."""
        repo = {"name": "dataset-dataset", "description": "dataset", "topics": ["dataset"]}
        signals = tracker._extract_signals(repo)
        assert signals.count("dataset") == 1


class TestExtractRelevanceSignals:
    """Tests for configurable relevance signal extraction."""

    @pytest.fixture
    def tracker(self):
        return GitHubTracker({})

    def test_match_in_name(self, tracker):
        """Test keyword match in repo name."""
        repo = {"name": "rlhf-training", "description": "", "topics": []}
        signals = tracker._extract_relevance_signals(repo)
        assert "rlhf" in signals

    def test_match_in_description(self, tracker):
        """Test keyword match in description."""
        repo = {"name": "project", "description": "Synthetic data generation tool", "topics": []}
        signals = tracker._extract_relevance_signals(repo)
        assert "synthetic-data" in signals

    def test_match_in_topics(self, tracker):
        """Test keyword match in topics."""
        repo = {"name": "project", "description": "", "topics": ["active-learning", "evaluation"]}
        signals = tracker._extract_relevance_signals(repo)
        assert "active-learning" in signals
        assert "evaluation" in signals

    def test_case_insensitive(self, tracker):
        """Test case-insensitive matching."""
        repo = {"name": "RLHF-Project", "description": "", "topics": []}
        signals = tracker._extract_relevance_signals(repo)
        assert "rlhf" in signals

    def test_sorted_output(self, tracker):
        """Test output is sorted."""
        repo = {"name": "rlhf-dataset", "description": "evaluation benchmark", "topics": []}
        signals = tracker._extract_relevance_signals(repo)
        assert signals == sorted(signals)

    def test_custom_keywords(self):
        """Test with custom relevance keywords."""
        config = {"sources": {"github": {"relevance_keywords": ["custom-keyword"]}}}
        tracker = GitHubTracker(config)
        repo = {"name": "custom-keyword-project", "description": "", "topics": []}
        signals = tracker._extract_relevance_signals(repo)
        assert "custom-keyword" in signals


class TestCalculateRelevance:
    """Tests for weighted relevance scoring."""

    @pytest.fixture
    def tracker(self):
        return GitHubTracker({})

    def test_high_relevance_many_signals(self, tracker):
        """Test high relevance with many keyword matches."""
        signals = ["rlhf", "dataset", "fine-tuning"]  # 3 * 10 = 30
        relevance = tracker._calculate_relevance(signals)
        assert relevance == "high"

    def test_medium_relevance(self, tracker):
        """Test medium relevance with one keyword."""
        signals = ["dataset"]  # 1 * 10 = 10
        relevance = tracker._calculate_relevance(signals)
        assert relevance == "medium"

    def test_low_relevance_no_signals(self, tracker):
        """Test low relevance with no signals."""
        signals = []
        relevance = tracker._calculate_relevance(signals)
        assert relevance == "low"

    def test_star_boost(self, tracker):
        """Test stars contribute to score."""
        signals = []  # 0 base
        repo = {"stars": 5000, "updated_at": "2020-01-01"}  # +50 from stars
        relevance = tracker._calculate_relevance(signals, repo)
        assert relevance == "high"

    def test_recency_boost(self, tracker):
        """Test recent update adds +5."""
        signals = []
        today = datetime.utcnow().strftime("%Y-%m-%d")
        repo = {"stars": 0, "updated_at": today}
        relevance = tracker._calculate_relevance(signals, repo)
        assert relevance == "medium"  # 0 + 5 (recency) = 5

    def test_negative_pattern_penalty(self, tracker):
        """Test negative patterns reduce score."""
        signals = ["dataset"]  # 10
        repo = {"name": "example-dataset", "stars": 0, "updated_at": "2020-01-01"}
        relevance = tracker._calculate_relevance(signals, repo)
        # 10 - 15 = -5 → low
        assert relevance == "low"

    def test_negative_patterns_list(self, tracker):
        """Test all negative patterns are checked."""
        for pattern in GitHubTracker.NEGATIVE_PATTERNS:
            signals = ["dataset"]  # 10
            repo = {"name": f"{pattern}test", "stars": 0, "updated_at": "2020-01-01"}
            relevance = tracker._calculate_relevance(signals, repo)
            assert relevance == "low", f"Pattern '{pattern}' should cause penalty"

    def test_combined_scoring(self, tracker):
        """Test combined scoring from signals + stars + recency."""
        signals = ["dataset"]  # 10
        today = datetime.utcnow().strftime("%Y-%m-%d")
        repo = {"stars": 1000, "updated_at": today, "name": "real-project"}
        relevance = tracker._calculate_relevance(signals, repo)
        # 10 + 10 (stars) + 5 (recency) = 25 → high
        assert relevance == "high"


class TestMakeRequest:
    """Tests for API request handling."""

    @pytest.fixture
    def tracker(self):
        return GitHubTracker({})

    def test_successful_request(self, tracker):
        """Test successful API request."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = [{"name": "repo1"}]
        tracker.session.get = MagicMock(return_value=mock_resp)

        result = tracker._make_request("https://api.github.com/test")
        assert result == [{"name": "repo1"}]

    def test_rate_limited_returns_none(self, tracker):
        """Test 403 rate limit returns None."""
        mock_resp = MagicMock()
        mock_resp.status_code = 403
        mock_resp.headers = {"X-RateLimit-Reset": str(int(datetime.utcnow().timestamp()) + 60)}
        tracker.session.get = MagicMock(return_value=mock_resp)

        result = tracker._make_request("https://api.github.com/test")
        assert result is None

    def test_not_found_returns_none(self, tracker):
        """Test 404 returns None."""
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        tracker.session.get = MagicMock(return_value=mock_resp)

        result = tracker._make_request("https://api.github.com/test")
        assert result is None

    @patch("time.sleep")
    def test_server_error_retries(self, mock_sleep, tracker):
        """Test 5xx triggers retry."""
        mock_resp_500 = MagicMock()
        mock_resp_500.status_code = 500
        mock_resp_ok = MagicMock()
        mock_resp_ok.status_code = 200
        mock_resp_ok.json.return_value = {"data": "ok"}

        tracker.session.get = MagicMock(side_effect=[mock_resp_500, mock_resp_ok])

        result = tracker._make_request("https://api.github.com/test")
        assert result == {"data": "ok"}
        assert mock_sleep.called

    @patch("time.sleep")
    def test_request_exception_retries(self, mock_sleep, tracker):
        """Test network error triggers retry."""
        import requests

        tracker.session.get = MagicMock(
            side_effect=[
                requests.ConnectionError("timeout"),
                MagicMock(status_code=200, json=MagicMock(return_value={"ok": True})),
            ]
        )

        result = tracker._make_request("https://api.github.com/test")
        assert result == {"ok": True}

    @patch("time.sleep")
    def test_all_retries_exhausted(self, mock_sleep, tracker):
        """Test returns None when all retries fail."""
        import requests

        tracker.session.get = MagicMock(side_effect=requests.ConnectionError("timeout"))

        result = tracker._make_request("https://api.github.com/test", max_retries=2)
        assert result is None


class TestGetOrgRepos:
    """Tests for organization repo fetching."""

    @pytest.fixture
    def tracker(self):
        return GitHubTracker({})

    def test_empty_response(self, tracker):
        """Test empty response returns empty list."""
        tracker._make_request = MagicMock(return_value=None)
        result = tracker.get_org_repos("test-org")
        assert result == []

    def test_filters_old_repos(self, tracker):
        """Test repos older than cutoff are filtered out."""
        old_date = (datetime.utcnow() - timedelta(days=30)).strftime("%Y-%m-%dT%H:%M:%SZ")
        recent_date = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

        tracker._make_request = MagicMock(
            return_value=[
                {
                    "name": "old-repo",
                    "full_name": "org/old-repo",
                    "description": "old",
                    "html_url": "https://github.com/org/old-repo",
                    "updated_at": old_date,
                    "stargazers_count": 10,
                    "language": "Python",
                    "topics": [],
                },
                {
                    "name": "new-repo",
                    "full_name": "org/new-repo",
                    "description": "dataset tool",
                    "html_url": "https://github.com/org/new-repo",
                    "updated_at": recent_date,
                    "stargazers_count": 100,
                    "language": "Python",
                    "topics": ["dataset"],
                },
            ]
        )

        result = tracker.get_org_repos("org", days=7)
        assert len(result) == 1
        assert result[0]["name"] == "new-repo"

    def test_repo_info_structure(self, tracker):
        """Test returned repo dict has expected fields."""
        now = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        tracker._make_request = MagicMock(
            return_value=[
                {
                    "name": "test",
                    "full_name": "org/test",
                    "description": "A dataset",
                    "html_url": "https://github.com/org/test",
                    "updated_at": now,
                    "stargazers_count": 42,
                    "language": "Python",
                    "topics": ["nlp"],
                },
            ]
        )

        result = tracker.get_org_repos("org")
        assert len(result) == 1
        repo = result[0]
        assert "name" in repo
        assert "full_name" in repo
        assert "description" in repo
        assert "url" in repo
        assert "stars" in repo
        assert "relevance" in repo
        assert "relevance_signals" in repo
        assert "signals" in repo


class TestGetOrgActivity:
    """Tests for organization activity summary."""

    @pytest.fixture
    def tracker(self):
        return GitHubTracker({})

    def test_activity_structure(self, tracker):
        """Test activity summary has expected structure."""
        tracker.get_org_repos = MagicMock(
            return_value=[
                {"name": "repo1", "signals": ["dataset"], "stars": 200, "relevance": "high"},
            ]
        )

        result = tracker.get_org_activity("test-org")
        assert result["org"] == "test-org"
        assert result["repos_count"] == 1
        assert result["has_activity"] is True
        assert len(result["repos_updated"]) == 1

    def test_no_activity(self, tracker):
        """Test no activity when no relevant repos."""
        tracker.get_org_repos = MagicMock(
            return_value=[
                {"name": "unrelated", "signals": [], "stars": 5, "relevance": "low"},
            ]
        )

        result = tracker.get_org_activity("test-org")
        assert result["has_activity"] is False


class TestDefaultSignalKeywords:
    """Tests for the default signal keywords list."""

    def test_keywords_not_empty(self):
        assert len(DEFAULT_SIGNAL_KEYWORDS) > 0

    def test_essential_keywords_present(self):
        essential = ["rlhf", "dataset", "fine-tuning", "synthetic-data", "evaluation", "llm"]
        for kw in essential:
            assert kw in DEFAULT_SIGNAL_KEYWORDS, f"Missing essential keyword: {kw}"

    def test_keywords_lowercase(self):
        for kw in DEFAULT_SIGNAL_KEYWORDS:
            assert kw == kw.lower(), f"Keyword not lowercase: {kw}"

    def test_no_duplicates(self):
        assert len(DEFAULT_SIGNAL_KEYWORDS) == len(set(DEFAULT_SIGNAL_KEYWORDS))
