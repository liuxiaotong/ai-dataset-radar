"""Tests for the notifiers module."""

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from notifiers import (
    BusinessIntelNotifier,
    ConsoleNotifier,
    MarkdownNotifier,
    WebhookNotifier,
    create_notifiers,
    expand_env_vars,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def empty_data():
    """Return data dict with all sources empty."""
    return {
        "huggingface": [],
        "paperswithcode": [],
        "arxiv": [],
        "github": [],
        "hf_papers": [],
    }


@pytest.fixture
def rich_data():
    """Return data dict populated with representative items for every source."""
    return {
        "huggingface": [
            {
                "name": "test-dataset",
                "author": "test-author",
                "downloads": 12345,
                "url": "https://huggingface.co/datasets/test-dataset",
                "tags": ["nlp", "text-classification", "en"],
            },
            {
                "name": "another-ds",
                "author": "org2",
                "downloads": 99,
                "url": "https://huggingface.co/datasets/another-ds",
                "tags": [],
            },
        ],
        "paperswithcode": [
            {
                "name": "PWC-Dataset",
                "description": "A short description of the dataset for testing purposes",
                "paper_count": 5,
                "url": "https://paperswithcode.com/dataset/pwc",
            },
        ],
        "arxiv": [
            {
                "title": "An Interesting Paper Title",
                "authors": ["Alice", "Bob", "Charlie", "Dave"],
                "categories": ["cs.CL", "cs.AI"],
                "url": "https://arxiv.org/abs/2401.00001",
                "summary": "This paper presents something remarkable " * 20,
            },
        ],
        "github": [
            {
                "full_name": "org/dataset-repo",
                "description": "A dataset repository on GitHub",
                "stars": 500,
                "language": "Python",
                "url": "https://github.com/org/dataset-repo",
                "is_dataset": True,
            },
            {
                "full_name": "org/non-dataset-repo",
                "description": "Not a dataset",
                "stars": 10,
                "language": "Go",
                "url": "https://github.com/org/non-dataset-repo",
                "is_dataset": False,
            },
        ],
        "hf_papers": [
            {
                "title": "HF Daily Paper About Datasets",
                "upvotes": 42,
                "arxiv_id": "2401.99999",
                "url": "https://huggingface.co/papers/2401.99999",
                "is_dataset_paper": True,
            },
            {
                "title": "HF Paper Not About Datasets",
                "upvotes": 10,
                "arxiv_id": "2401.00002",
                "url": "https://huggingface.co/papers/2401.00002",
                "is_dataset_paper": False,
            },
        ],
    }


@pytest.fixture
def sample_trend_results():
    """Trend results as returned by TrendAnalyzer.analyze()."""
    return {
        "top_growing_7d": [
            {
                "name": "growing-ds",
                "dataset_id": "growing-ds",
                "url": "https://huggingface.co/datasets/growing-ds",
                "growth": 0.45,
                "current_downloads": 5000,
                "domains": ["NLP"],
            },
        ],
        "breakthroughs": [
            {
                "name": "breakthrough-ds",
                "dataset_id": "breakthrough-ds",
                "url": "https://huggingface.co/datasets/breakthrough-ds",
                "old_downloads": 0,
                "current_downloads": 2000,
                "download_increase": 2000,
            },
        ],
    }


@pytest.fixture
def sample_opportunity_results_list():
    """Opportunity results where data_factories is a plain list."""
    return {
        "data_factories": [
            {
                "author": "prolific-author",
                "dataset_count": 7,
                "datasets": [
                    {"name": "ds-1", "id": "ds-1"},
                    {"name": "ds-2", "id": "ds-2"},
                    {"name": "ds-3", "id": "ds-3"},
                    {"name": "ds-4", "id": "ds-4"},
                ],
                "possible_org": "SomeOrg",
            },
        ],
        "annotation_opportunities": [
            {
                "title": "Paper Needing Annotation Services For Real-World Data",
                "signals": ["human annotation", "manual labeling", "crowd-sourcing", "quality check"],
                "detected_org": "BigCorp",
                "arxiv_id": "2401.11111",
            },
        ],
        "org_activity": {
            "google": {
                "total_items": 2,
                "datasets": [{"name": "google-ds", "url": "https://example.com/google-ds"}],
                "papers": [
                    {
                        "title": "Google Paper About Something",
                        "url": "https://arxiv.org/abs/2401.22222",
                    }
                ],
            },
            "meta": {
                "total_items": 0,
                "datasets": [],
                "papers": [],
            },
        },
        "summary": {
            "annotation_opportunity_count": 1,
            "data_factory_count": 1,
        },
    }


@pytest.fixture
def sample_opportunity_results_dict():
    """Opportunity results where data_factories is a dict with org/individual keys.

    This is the shape that triggered a previous bug -- the code must handle
    both dict and list forms.
    """
    return {
        "data_factories": {
            "org_factories": [
                {
                    "author": "BigOrg",
                    "dataset_count": 12,
                    "datasets": [
                        {"name": "org-ds-1"},
                        {"name": "org-ds-2"},
                    ],
                    "possible_org": "BigOrg Inc.",
                },
            ],
            "individual_factories": [
                {
                    "author": "indie-author",
                    "dataset_count": 5,
                    "datasets": [
                        {"name": "indie-ds-1"},
                        {"name": "indie-ds-2"},
                        {"name": "indie-ds-3"},
                        {"name": "indie-ds-4"},
                    ],
                    "possible_org": None,
                },
            ],
        },
        "annotation_opportunities": [],
        "org_activity": {},
        "summary": {
            "annotation_opportunity_count": 0,
            "data_factory_count": 2,
        },
    }


@pytest.fixture
def sample_domain_data():
    """Domain classification data as returned by DomainFilter."""
    return {
        "robotics": [
            {
                "name": "robot-manipulation",
                "source": "huggingface",
                "url": "https://huggingface.co/datasets/robot-manipulation",
                "downloads": 300,
                "tags": ["manipulation", "sim2real"],
                "growth": 0.15,
            },
            {
                "name": "Robotics Paper",
                "source": "arxiv",
                "url": "https://arxiv.org/abs/2401.55555",
            },
        ],
    }


# ===========================================================================
# ConsoleNotifier
# ===========================================================================


class TestConsoleNotifier:
    """Tests for ConsoleNotifier."""

    def test_notify_with_empty_data(self, capsys, empty_data):
        """ConsoleNotifier.notify must not crash with empty data."""
        notifier = ConsoleNotifier(use_color=False)
        notifier.notify(empty_data)
        captured = capsys.readouterr()
        assert "AI Dataset Radar Report" in captured.out

    def test_notify_with_rich_data(self, capsys, rich_data):
        """Console output contains expected dataset names."""
        notifier = ConsoleNotifier(use_color=False)
        notifier.notify(rich_data)
        captured = capsys.readouterr()
        assert "test-dataset" in captured.out
        assert "PWC-Dataset" in captured.out
        assert "An Interesting Paper Title" in captured.out
        assert "org/dataset-repo" in captured.out
        assert "HF Daily Paper About Datasets" in captured.out

    def test_notify_with_color(self, capsys, rich_data):
        """Console output includes ANSI codes when color is enabled."""
        notifier = ConsoleNotifier(use_color=True)
        notifier.notify(rich_data)
        captured = capsys.readouterr()
        # ANSI escape code for bold
        assert "\033[1m" in captured.out

    def test_color_method_disabled(self):
        """_color returns plain text when color is disabled."""
        notifier = ConsoleNotifier(use_color=False)
        assert notifier._color("hello", "bold") == "hello"

    def test_color_method_enabled(self):
        """_color wraps text in ANSI codes when enabled."""
        notifier = ConsoleNotifier(use_color=True)
        result = notifier._color("hello", "bold")
        assert result.startswith("\033[1m")
        assert result.endswith("\033[0m")
        assert "hello" in result

    def test_color_method_unknown_color(self):
        """_color returns plain text for an unknown color name."""
        notifier = ConsoleNotifier(use_color=True)
        assert notifier._color("text", "nonexistent") == "text"

    def test_notify_with_missing_keys(self, capsys):
        """ConsoleNotifier handles a data dict that lacks some keys."""
        notifier = ConsoleNotifier(use_color=False)
        notifier.notify({"huggingface": []})  # missing other keys
        captured = capsys.readouterr()
        assert "AI Dataset Radar Report" in captured.out


# ===========================================================================
# MarkdownNotifier
# ===========================================================================


class TestMarkdownNotifier:
    """Tests for MarkdownNotifier."""

    def test_notify_creates_file(self, tmp_path, rich_data):
        """notify() creates a markdown file in the output directory."""
        notifier = MarkdownNotifier(output_dir=str(tmp_path))
        filepath = notifier.notify(rich_data)
        assert os.path.isfile(filepath)
        assert filepath.endswith(".md")

    def test_markdown_contains_sections(self, tmp_path, rich_data):
        """Generated markdown includes all expected section headers."""
        notifier = MarkdownNotifier(output_dir=str(tmp_path))
        filepath = notifier.notify(rich_data)
        content = Path(filepath).read_text(encoding="utf-8")
        assert "# AI Dataset Radar Report" in content
        assert "## Summary" in content
        assert "## Hugging Face Datasets" in content
        assert "## Papers with Code Datasets" in content
        assert "## arXiv Papers" in content
        assert "## GitHub Repos (Early Signal)" in content
        assert "## HF Daily Papers (Early Signal)" in content

    def test_markdown_empty_data(self, tmp_path, empty_data):
        """Markdown handles empty data without crashing."""
        notifier = MarkdownNotifier(output_dir=str(tmp_path))
        filepath = notifier.notify(empty_data)
        content = Path(filepath).read_text(encoding="utf-8")
        assert "No datasets found" in content

    def test_markdown_summary_counts(self, tmp_path, rich_data):
        """Summary section reflects correct item counts."""
        notifier = MarkdownNotifier(output_dir=str(tmp_path))
        content = notifier._generate_markdown(rich_data)
        # 2 hf + 1 pwc + 1 arxiv + 2 github + 2 hf_papers = 8
        assert "**Total items found:** 8" in content
        assert "**Hugging Face datasets:** 2" in content

    def test_generate_markdown_output_dir_created(self, tmp_path, empty_data):
        """notify() creates the output directory if it does not exist."""
        new_dir = tmp_path / "nested" / "reports"
        notifier = MarkdownNotifier(output_dir=str(new_dir))
        filepath = notifier.notify(empty_data)
        assert os.path.isdir(str(new_dir))
        assert os.path.isfile(filepath)


# ===========================================================================
# BusinessIntelNotifier -- critical data_factories dict-vs-list tests
# ===========================================================================


class TestBusinessIntelNotifier:
    """Tests for BusinessIntelNotifier, including the data_factories fix."""

    def test_notify_creates_file(self, tmp_path, rich_data):
        """notify() creates an intel report file."""
        notifier = BusinessIntelNotifier(output_dir=str(tmp_path))
        filepath = notifier.notify(rich_data)
        assert os.path.isfile(filepath)
        assert "intel_report_" in filepath

    def test_report_with_no_optional_data(self, tmp_path, rich_data):
        """Report generates fine when trend/opportunity/domain are None."""
        notifier = BusinessIntelNotifier(output_dir=str(tmp_path))
        filepath = notifier.notify(rich_data)
        content = Path(filepath).read_text(encoding="utf-8")
        assert "商业情报周报" in content
        assert "需要多天数据才能计算增长趋势" in content
        assert "本周未检测到数据工厂活动" in content

    def test_data_factories_as_list(
        self, tmp_path, rich_data, sample_opportunity_results_list
    ):
        """When data_factories is a list, the report renders correctly."""
        notifier = BusinessIntelNotifier(output_dir=str(tmp_path))
        filepath = notifier.notify(
            rich_data, opportunity_results=sample_opportunity_results_list
        )
        content = Path(filepath).read_text(encoding="utf-8")
        assert "prolific-author" in content
        assert "SomeOrg" in content

    def test_data_factories_as_dict(
        self, tmp_path, rich_data, sample_opportunity_results_dict
    ):
        """When data_factories is a dict with org/individual keys, report works.

        This is the critical test for the bug fix where data_factories
        changed from a list to a dict with 'org_factories' and
        'individual_factories' keys.
        """
        notifier = BusinessIntelNotifier(output_dir=str(tmp_path))
        filepath = notifier.notify(
            rich_data, opportunity_results=sample_opportunity_results_dict
        )
        content = Path(filepath).read_text(encoding="utf-8")
        # Both org and individual factories should appear
        assert "BigOrg" in content
        assert "indie-author" in content

    def test_data_factories_dict_merges_both_lists(
        self, tmp_path, rich_data, sample_opportunity_results_dict
    ):
        """Verify that org_factories and individual_factories are concatenated."""
        notifier = BusinessIntelNotifier(output_dir=str(tmp_path))
        content = notifier._generate_report(
            rich_data,
            trend_results=None,
            opportunity_results=sample_opportunity_results_dict,
            domain_data=None,
        )
        # The table should contain rows for both factory types
        assert "BigOrg" in content
        assert "indie-author" in content
        # Dataset names from both should appear
        assert "org-ds-1" in content
        assert "indie-ds-1" in content

    def test_data_factories_dict_empty_sublists(self, tmp_path, rich_data):
        """Dict form with empty org_factories and individual_factories."""
        opp = {
            "data_factories": {
                "org_factories": [],
                "individual_factories": [],
            },
            "annotation_opportunities": [],
            "org_activity": {},
            "summary": {},
        }
        notifier = BusinessIntelNotifier(output_dir=str(tmp_path))
        # Should not crash -- empty table, no rows
        content = notifier._generate_report(rich_data, None, opp, None)
        assert "数据工厂动态" in content

    def test_data_factories_dict_missing_keys(self, tmp_path, rich_data):
        """Dict form where one of the sub-keys is missing entirely."""
        opp = {
            "data_factories": {
                "org_factories": [
                    {
                        "author": "OnlyOrg",
                        "dataset_count": 3,
                        "datasets": [{"name": "x"}],
                        "possible_org": "OrgCo",
                    }
                ],
                # 'individual_factories' key is missing
            },
            "annotation_opportunities": [],
            "org_activity": {},
            "summary": {},
        }
        notifier = BusinessIntelNotifier(output_dir=str(tmp_path))
        content = notifier._generate_report(rich_data, None, opp, None)
        assert "OnlyOrg" in content

    def test_trend_results_rendered(
        self, tmp_path, rich_data, sample_trend_results
    ):
        """Top growing datasets and breakthroughs appear in the report."""
        notifier = BusinessIntelNotifier(output_dir=str(tmp_path))
        content = notifier._generate_report(
            rich_data, sample_trend_results, None, None
        )
        assert "growing-ds" in content
        assert "45.0%" in content
        assert "breakthrough-ds" in content

    def test_domain_data_robotics(
        self, tmp_path, rich_data, sample_domain_data
    ):
        """Robotics section renders when domain_data is supplied."""
        notifier = BusinessIntelNotifier(output_dir=str(tmp_path))
        content = notifier._generate_report(
            rich_data, None, None, sample_domain_data
        )
        assert "robot-manipulation" in content
        assert "manipulation" in content

    def test_org_activity_section(
        self, tmp_path, rich_data, sample_opportunity_results_list
    ):
        """Organization activity section renders active orgs only."""
        notifier = BusinessIntelNotifier(output_dir=str(tmp_path))
        content = notifier._generate_report(
            rich_data, None, sample_opportunity_results_list, None
        )
        # google has total_items=2, meta has total_items=0
        assert "GOOGLE" in content
        # meta should NOT appear (0 items)
        assert "META" not in content

    def test_annotation_opportunities_section(
        self, tmp_path, rich_data, sample_opportunity_results_list
    ):
        """Annotation opportunities render with truncated titles and signal list."""
        notifier = BusinessIntelNotifier(output_dir=str(tmp_path))
        content = notifier._generate_report(
            rich_data, None, sample_opportunity_results_list, None
        )
        assert "BigCorp" in content
        assert "2401.11111" in content

    def test_statistics_summary(
        self, tmp_path, rich_data, sample_opportunity_results_list, sample_domain_data
    ):
        """Statistics section includes key metrics."""
        notifier = BusinessIntelNotifier(output_dir=str(tmp_path))
        content = notifier._generate_report(
            rich_data, None, sample_opportunity_results_list, sample_domain_data
        )
        assert "本周新增数据集" in content
        assert "检测到潜在商机" in content


# ===========================================================================
# WebhookNotifier
# ===========================================================================


class TestWebhookNotifier:
    """Tests for WebhookNotifier."""

    @patch("notifiers.requests.post")
    def test_notify_success(self, mock_post, rich_data):
        """Successful webhook POST returns True."""
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        notifier = WebhookNotifier(url="https://hooks.example.com/webhook")
        result = notifier.notify(rich_data)

        assert result is True
        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args
        assert call_kwargs[1]["json"]["summary"]["huggingface_count"] == 2

    @patch("notifiers.requests.post")
    def test_notify_failure(self, mock_post, rich_data):
        """Failed webhook POST returns False."""
        import requests as req

        mock_post.side_effect = req.RequestException("connection refused")

        notifier = WebhookNotifier(url="https://hooks.example.com/webhook")
        result = notifier.notify(rich_data)

        assert result is False

    @patch("notifiers.requests.post")
    def test_notify_payload_structure(self, mock_post, empty_data):
        """Webhook payload includes timestamp, summary, and data."""
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        notifier = WebhookNotifier(url="https://hooks.example.com/webhook")
        notifier.notify(empty_data)

        payload = mock_post.call_args[1]["json"]
        assert "timestamp" in payload
        assert "summary" in payload
        assert "data" in payload
        assert payload["summary"]["huggingface_count"] == 0
        assert payload["summary"]["paperswithcode_count"] == 0
        assert payload["summary"]["arxiv_count"] == 0


# ===========================================================================
# expand_env_vars
# ===========================================================================


class TestExpandEnvVars:
    """Tests for the expand_env_vars helper function."""

    def test_simple_expansion(self):
        """Single env var is expanded."""
        with patch.dict(os.environ, {"MY_VAR": "hello"}):
            assert expand_env_vars("${MY_VAR}") == "hello"

    def test_multiple_expansions(self):
        """Multiple env vars in one string are all expanded."""
        with patch.dict(os.environ, {"HOST": "localhost", "PORT": "8080"}):
            result = expand_env_vars("http://${HOST}:${PORT}/api")
            assert result == "http://localhost:8080/api"

    def test_missing_var_returns_empty(self):
        """Missing env var is replaced with empty string."""
        env = os.environ.copy()
        env.pop("NONEXISTENT_VAR_12345", None)
        with patch.dict(os.environ, env, clear=True):
            result = expand_env_vars("prefix_${NONEXISTENT_VAR_12345}_suffix", warn_missing=False)
            assert result == "prefix__suffix"

    def test_missing_var_prints_warning(self, capsys):
        """Missing env var triggers a warning when warn_missing=True."""
        env = os.environ.copy()
        env.pop("MISSING_XYZ", None)
        with patch.dict(os.environ, env, clear=True):
            expand_env_vars("${MISSING_XYZ}", warn_missing=True)
            captured = capsys.readouterr()
            assert "MISSING_XYZ" in captured.out

    def test_no_vars_returns_unchanged(self):
        """String without ${...} patterns is returned unchanged."""
        assert expand_env_vars("plain string") == "plain string"

    def test_non_string_passthrough(self):
        """Non-string values are returned as-is."""
        assert expand_env_vars(42) == 42
        assert expand_env_vars(None) is None
        assert expand_env_vars(["a", "b"]) == ["a", "b"]


# ===========================================================================
# create_notifiers
# ===========================================================================


class TestCreateNotifiers:
    """Tests for the create_notifiers factory function."""

    def test_defaults_create_console_markdown_intel(self):
        """With an empty config, console, markdown, and intel notifiers are created."""
        notifiers = create_notifiers({})
        types = [type(n).__name__ for n in notifiers]
        assert "ConsoleNotifier" in types
        assert "MarkdownNotifier" in types
        assert "BusinessIntelNotifier" in types

    def test_disable_console(self):
        """Console notifier can be disabled."""
        config = {"console": {"enabled": False}}
        notifiers = create_notifiers(config)
        types = [type(n).__name__ for n in notifiers]
        assert "ConsoleNotifier" not in types

    def test_enable_webhook(self):
        """Webhook notifier is created when enabled."""
        config = {"webhook": {"enabled": True, "url": "https://example.com/hook"}}
        notifiers = create_notifiers(config)
        types = [type(n).__name__ for n in notifiers]
        assert "WebhookNotifier" in types

    def test_enable_email(self):
        """Email notifier is created when enabled with required fields."""
        config = {
            "email": {
                "enabled": True,
                "smtp_server": "smtp.example.com",
                "smtp_port": 587,
                "username": "user@example.com",
                "password": "secret",
                "from_addr": "user@example.com",
                "to_addrs": ["dest@example.com"],
            }
        }
        notifiers = create_notifiers(config)
        types = [type(n).__name__ for n in notifiers]
        assert "EmailNotifier" in types

    def test_full_config_passed_to_intel(self):
        """BusinessIntelNotifier receives the full_config argument."""
        full_cfg = {"some": "config"}
        notifiers = create_notifiers({}, full_config=full_cfg)
        intel = [n for n in notifiers if isinstance(n, BusinessIntelNotifier)][0]
        assert intel.config == full_cfg

    def test_markdown_output_dir(self):
        """MarkdownNotifier picks up custom output_dir from config."""
        config = {"markdown": {"enabled": True, "output_dir": "/custom/path"}}
        notifiers = create_notifiers(config)
        md = [n for n in notifiers if isinstance(n, MarkdownNotifier)][0]
        assert md.output_dir == "/custom/path"

    def test_all_disabled_returns_empty(self):
        """When every notifier is disabled, an empty list is returned."""
        config = {
            "console": {"enabled": False},
            "markdown": {"enabled": False},
            "business_intel": {"enabled": False},
            "email": {"enabled": False},
            "webhook": {"enabled": False},
        }
        notifiers = create_notifiers(config)
        assert notifiers == []
