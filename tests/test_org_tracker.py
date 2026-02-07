"""Tests for HuggingFace organization tracker."""

import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from trackers.org_tracker import OrgTracker


@pytest.fixture
def sample_config():
    """Standard test configuration."""
    return {
        "watched_orgs": {
            "frontier_labs": {
                "OpenAI": {
                    "hf_ids": ["openai"],
                    "keywords": ["gpt", "chatgpt"],
                    "priority": "high",
                },
                "Meta": {
                    "hf_ids": ["meta-llama", "facebook"],
                    "keywords": ["llama"],
                    "priority": "high",
                },
            },
            "emerging_labs": {
                "Mistral": {
                    "hf_ids": ["mistralai"],
                    "keywords": ["mistral"],
                    "priority": "medium",
                },
            },
        },
        "watched_vendors": {
            "premium": {
                "ScaleAI": {
                    "hf_ids": ["scaleai"],
                    "github": ["scaleapi"],
                    "keywords": ["scale"],
                    "blog_url": "https://scale.com/blog",
                },
            },
            "specialized": {
                "SurgeAI": {
                    "hf_ids": ["surge-ai"],
                    "github": ["surge-ai"],
                    "keywords": ["surge"],
                },
            },
        },
    }


class TestOrgTrackerInit:
    """Tests for OrgTracker initialization."""

    def test_init_empty_config(self):
        """Test initialization with empty config."""
        tracker = OrgTracker({})
        assert tracker.watched_orgs == {}
        assert tracker.watched_vendors == {}

    def test_init_parses_orgs(self, sample_config):
        """Test watched_orgs parsed correctly."""
        tracker = OrgTracker(sample_config)
        assert "OpenAI" in tracker.watched_orgs
        assert "Meta" in tracker.watched_orgs
        assert "Mistral" in tracker.watched_orgs
        assert len(tracker.watched_orgs) == 3

    def test_init_org_structure(self, sample_config):
        """Test parsed org has correct fields."""
        tracker = OrgTracker(sample_config)
        openai = tracker.watched_orgs["OpenAI"]
        assert openai["hf_ids"] == ["openai"]
        assert openai["keywords"] == ["gpt", "chatgpt"]
        assert openai["category"] == "frontier_labs"
        assert openai["priority"] == "high"

    def test_init_multi_hf_ids(self, sample_config):
        """Test org with multiple HF IDs."""
        tracker = OrgTracker(sample_config)
        meta = tracker.watched_orgs["Meta"]
        assert meta["hf_ids"] == ["meta-llama", "facebook"]

    def test_init_parses_vendors(self, sample_config):
        """Test watched_vendors parsed correctly."""
        tracker = OrgTracker(sample_config)
        assert "ScaleAI" in tracker.watched_vendors
        assert "SurgeAI" in tracker.watched_vendors

    def test_init_vendor_structure(self, sample_config):
        """Test parsed vendor has correct fields."""
        tracker = OrgTracker(sample_config)
        scale = tracker.watched_vendors["ScaleAI"]
        assert scale["hf_ids"] == ["scaleai"]
        assert scale["github"] == ["scaleapi"]
        assert scale["tier"] == "premium"
        assert scale["blog_url"] == "https://scale.com/blog"

    def test_init_session_created(self, sample_config):
        """Test HTTP session is initialized."""
        tracker = OrgTracker(sample_config)
        assert tracker.session is not None
        assert "User-Agent" in tracker.session.headers

    def test_init_rate_limit_defaults(self, sample_config):
        """Test rate limiting defaults."""
        tracker = OrgTracker(sample_config)
        assert tracker._request_delay == 0.1


class TestParseOrgs:
    """Tests for _parse_orgs method."""

    def test_parse_empty(self):
        tracker = OrgTracker({})
        result = tracker._parse_orgs({})
        assert result == {}

    def test_parse_non_dict_values(self):
        tracker = OrgTracker({})
        result = tracker._parse_orgs({"category": "not_a_dict"})
        assert result == {}

    def test_parse_nested_non_dict(self):
        tracker = OrgTracker({})
        result = tracker._parse_orgs({"category": {"org": "not_a_dict"}})
        assert result == {}

    def test_parse_default_priority(self):
        tracker = OrgTracker({})
        result = tracker._parse_orgs({
            "test_category": {
                "TestOrg": {
                    "hf_ids": ["test"],
                    "keywords": [],
                }
            }
        })
        assert result["TestOrg"]["priority"] == "medium"


class TestParseVendors:
    """Tests for _parse_vendors method."""

    def test_parse_empty(self):
        tracker = OrgTracker({})
        result = tracker._parse_vendors({})
        assert result == {}

    def test_parse_vendor_fields(self):
        tracker = OrgTracker({})
        result = tracker._parse_vendors({
            "premium": {
                "TestVendor": {
                    "hf_ids": ["test-vendor"],
                    "github": ["test-vendor-gh"],
                    "keywords": ["test"],
                    "blog_url": "https://test.com/blog",
                }
            }
        })
        assert "TestVendor" in result
        assert result["TestVendor"]["tier"] == "premium"
        assert result["TestVendor"]["blog_url"] == "https://test.com/blog"


class TestRateLimit:
    """Tests for rate limiting."""

    def test_rate_limit_enforced(self, sample_config):
        """Test rate limit delays requests."""
        tracker = OrgTracker(sample_config)
        tracker._request_delay = 0.05  # Short delay for testing

        start = time.time()
        tracker._rate_limit()
        tracker._rate_limit()
        elapsed = time.time() - start

        # Should have at least one delay
        assert elapsed >= 0.04

    def test_rate_limit_thread_safe(self, sample_config):
        """Test rate limit uses lock."""
        tracker = OrgTracker(sample_config)
        assert tracker._rate_lock is not None


class TestRequestWithRetry:
    """Tests for _request_with_retry method."""

    @pytest.fixture
    def tracker(self, sample_config):
        tracker = OrgTracker(sample_config)
        tracker._request_delay = 0  # Disable rate limit for tests
        return tracker

    def test_successful_request(self, tracker):
        """Test successful request returns data and caches."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = [{"id": "test/dataset"}]
        tracker.session.get = MagicMock(return_value=mock_resp)
        tracker._cache = MagicMock()
        tracker._cache.get.return_value = None

        result = tracker._request_with_retry(
            url="https://huggingface.co/api/datasets",
            params={"author": "test"},
            cache_key="test:key",
            description="test",
        )
        assert result == [{"id": "test/dataset"}]
        tracker._cache.set.assert_called_once()

    def test_cache_hit(self, tracker):
        """Test cached data returned without request."""
        tracker._cache = MagicMock()
        tracker._cache.get.return_value = [{"id": "cached"}]

        result = tracker._request_with_retry(
            url="https://huggingface.co/api/datasets",
            params={},
            cache_key="cached:key",
            description="test",
        )
        assert result == [{"id": "cached"}]
        tracker.session.get.assert_not_called() if hasattr(tracker.session.get, 'assert_not_called') else None

    @patch("time.sleep")
    def test_server_error_retries(self, mock_sleep, tracker):
        """Test 5xx error triggers retry."""
        mock_500 = MagicMock()
        mock_500.status_code = 500
        mock_ok = MagicMock()
        mock_ok.status_code = 200
        mock_ok.json.return_value = [{"id": "retried"}]

        tracker.session.get = MagicMock(side_effect=[mock_500, mock_ok])
        tracker._cache = MagicMock()
        tracker._cache.get.return_value = None

        result = tracker._request_with_retry(
            url="https://huggingface.co/api/test",
            params={},
            cache_key="retry:key",
            description="test",
        )
        assert result == [{"id": "retried"}]

    def test_non_200_returns_empty(self, tracker):
        """Test non-200, non-5xx returns empty list."""
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        tracker.session.get = MagicMock(return_value=mock_resp)
        tracker._cache = MagicMock()
        tracker._cache.get.return_value = None

        result = tracker._request_with_retry(
            url="https://huggingface.co/api/test",
            params={},
            cache_key="404:key",
            description="test",
        )
        assert result == []

    @patch("time.sleep")
    def test_connection_error_retries(self, mock_sleep, tracker):
        """Test network error triggers retry."""
        import requests
        mock_ok = MagicMock()
        mock_ok.status_code = 200
        mock_ok.json.return_value = []

        tracker.session.get = MagicMock(
            side_effect=[requests.ConnectionError("timeout"), mock_ok]
        )
        tracker._cache = MagicMock()
        tracker._cache.get.return_value = None

        result = tracker._request_with_retry(
            url="https://huggingface.co/api/test",
            params={},
            cache_key="err:key",
            description="test",
        )
        assert result == []


class TestFetchOrgDatasets:
    """Tests for _fetch_org_datasets."""

    def test_calls_correct_url(self, sample_config):
        tracker = OrgTracker(sample_config)
        tracker._request_with_retry = MagicMock(return_value=[])

        tracker._fetch_org_datasets("openai", limit=50)
        tracker._request_with_retry.assert_called_once()
        call_args = tracker._request_with_retry.call_args
        assert "datasets" in call_args.kwargs.get("url", call_args[1].get("url", ""))
        assert call_args.kwargs.get("params", call_args[1].get("params", {}))["author"] == "openai"


class TestFetchOrgModels:
    """Tests for _fetch_org_models."""

    def test_calls_correct_url(self, sample_config):
        tracker = OrgTracker(sample_config)
        tracker._request_with_retry = MagicMock(return_value=[])

        tracker._fetch_org_models("meta-llama", limit=25)
        tracker._request_with_retry.assert_called_once()
        call_args = tracker._request_with_retry.call_args
        assert "models" in call_args.kwargs.get("url", call_args[1].get("url", ""))


class TestGetAllHfIds:
    """Tests for get_all_hf_ids."""

    def test_returns_all_ids(self, sample_config):
        tracker = OrgTracker(sample_config)
        ids = tracker.get_all_hf_ids()
        assert "openai" in ids
        assert "meta-llama" in ids
        assert "facebook" in ids
        assert "mistralai" in ids
        assert "scaleai" in ids
        assert "surge-ai" in ids

    def test_deduplicates_ids(self):
        """Test duplicate IDs are removed."""
        config = {
            "watched_orgs": {
                "cat1": {
                    "Org1": {"hf_ids": ["shared-id"], "keywords": [], "priority": "high"},
                }
            },
            "watched_vendors": {
                "tier1": {
                    "Vendor1": {"hf_ids": ["shared-id"], "github": [], "keywords": []},
                }
            },
        }
        tracker = OrgTracker(config)
        ids = tracker.get_all_hf_ids()
        assert ids.count("shared-id") == 1


class TestFetchSingleOrg:
    """Tests for _fetch_single_org."""

    @pytest.fixture
    def tracker(self, sample_config):
        return OrgTracker(sample_config)

    def test_returns_none_when_no_data(self, tracker):
        """Test returns None when org has no recent activity."""
        tracker._fetch_org_datasets = MagicMock(return_value=[])
        tracker._fetch_org_models = MagicMock(return_value=[])

        cutoff = datetime.now() - timedelta(days=7)
        org_info = {"hf_ids": ["openai"], "category": "frontier_labs", "priority": "high"}

        result = tracker._fetch_single_org("OpenAI", org_info, cutoff)
        assert result is None

    def test_returns_data_when_recent(self, tracker):
        """Test returns data when org has recent datasets."""
        recent_date = datetime.utcnow().isoformat() + "Z"
        tracker._fetch_org_datasets = MagicMock(return_value=[
            {"id": "openai/new-dataset", "lastModified": recent_date},
        ])
        tracker._fetch_org_models = MagicMock(return_value=[])

        cutoff = datetime.now() - timedelta(days=7)
        org_info = {"hf_ids": ["openai"], "category": "frontier_labs", "priority": "high"}

        result = tracker._fetch_single_org("OpenAI", org_info, cutoff)
        assert result is not None
        category, org_name, org_data = result
        assert category == "frontier_labs"
        assert org_name == "OpenAI"
        assert len(org_data["datasets"]) == 1

    def test_filters_old_datasets(self, tracker):
        """Test old datasets are filtered out."""
        old_date = (datetime.utcnow() - timedelta(days=30)).isoformat() + "Z"
        tracker._fetch_org_datasets = MagicMock(return_value=[
            {"id": "openai/old-dataset", "lastModified": old_date},
        ])
        tracker._fetch_org_models = MagicMock(return_value=[])

        cutoff = datetime.now() - timedelta(days=7)
        org_info = {"hf_ids": ["openai"], "category": "frontier_labs", "priority": "high"}

        result = tracker._fetch_single_org("OpenAI", org_info, cutoff)
        assert result is None  # No recent data → None


class TestFetchLabActivity:
    """Tests for fetch_lab_activity."""

    def test_returns_structured_results(self, sample_config):
        """Test returns dict with expected categories."""
        tracker = OrgTracker(sample_config)
        tracker._fetch_single_org = MagicMock(return_value=None)

        result = tracker.fetch_lab_activity(days=7)
        assert "frontier_labs" in result
        assert "emerging_labs" in result
        assert "research_labs" in result


class TestFetchVendorActivity:
    """Tests for fetch_vendor_activity."""

    def test_returns_structured_results(self, sample_config):
        """Test returns dict with expected tiers."""
        tracker = OrgTracker(sample_config)
        tracker._fetch_org_datasets = MagicMock(return_value=[])

        result = tracker.fetch_vendor_activity(days=7)
        assert "premium" in result
        assert "specialized" in result
