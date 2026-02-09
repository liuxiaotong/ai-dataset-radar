"""Tests for intel_report module — IntelReportGenerator."""

import sys
from pathlib import Path

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from analyzers.data_type_classifier import DataType
from intel_report import IntelReportGenerator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def generator():
    """Return an IntelReportGenerator with default config."""
    return IntelReportGenerator(config={})


@pytest.fixture
def generator_with_limits():
    """Return an IntelReportGenerator with explicit report limits."""
    return IntelReportGenerator(
        config={"report": {"limits": {"labs_per_category": 3, "datasets_per_type": 2}}}
    )


@pytest.fixture
def empty_lab_activity():
    return {"labs": {}}


@pytest.fixture
def empty_vendor_activity():
    return {"vendors": {}}


@pytest.fixture
def empty_datasets_by_type():
    return {}


@pytest.fixture
def sample_lab_activity():
    return {
        "labs": {
            "frontier_labs": {
                "openai": {
                    "datasets": [
                        {"id": "openai/gpt4-data", "downloads": 1500},
                        {"id": "openai/whisper-data", "downloads": 800},
                    ],
                    "models": [{"id": "openai/gpt-4o"}],
                },
                "anthropic": {
                    "datasets": [],
                    "models": [{"id": "anthropic/claude-3"}],
                },
            },
            "emerging_labs": {
                "mistral": {
                    "datasets": [{"id": "mistral/instruct-v3", "downloads": 300}],
                    "models": [],
                },
            },
            "research_labs": {
                "allen_ai": {
                    "datasets": [{"id": "allen_ai/olmo-data", "downloads": 200}],
                    "models": [],
                },
            },
        }
    }


@pytest.fixture
def sample_vendor_activity():
    return {
        "vendors": {
            "tier1": {
                "scale_ai": {
                    "datasets": [
                        {"id": "scale_ai/rlhf-pairs", "downloads": 5000},
                    ],
                },
            },
            "tier2": {
                "labelbox": {
                    "datasets": [],
                },
            },
        }
    }


@pytest.fixture
def sample_datasets_by_type():
    return {
        DataType.SFT_INSTRUCTION: [
            {
                "id": "org/sft-chat-v2",
                "downloads": 4200,
                "signals": ["instruction", "chat"],
            },
        ],
        DataType.RLHF_PREFERENCE: [
            {
                "id": "org/preference-pairs",
                "downloads": 3100,
                "signals": ["preference"],
            },
            {
                "id": "org/dpo-mix",
                "downloads": 900,
                "signals": ["dpo"],
            },
        ],
        DataType.OTHER: [
            {"id": "org/misc-data", "downloads": 10, "signals": ["-"]},
        ],
    }


@pytest.fixture
def sample_github_activity():
    return [
        {
            "org": "scale-ai",
            "repos_updated": [
                {
                    "name": "llm-engine",
                    "description": "Open-source LLM engine",
                    "url": "https://github.com/scale-ai/llm-engine",
                    "stars": 120,
                    "signals": ["llm", "engine"],
                },
            ],
        },
    ]


@pytest.fixture
def sample_blog_activity():
    return [
        {
            "source": "Scale AI",
            "articles": [
                {
                    "title": "Scaling RLHF with New Data Pipeline",
                    "url": "https://scale.com/blog/rlhf",
                    "summary": "A new approach to scaling RLHF annotation",
                },
            ],
        },
    ]


@pytest.fixture
def sample_papers():
    return [
        {
            "title": "Direct Preference Optimization: Your Language Model is Secretly a Reward Model",
            "category": "RLHF/偏好学习",
            "url": "https://arxiv.org/abs/2305.18290",
            "highlight": "DPO simplifies RLHF",
        },
        {
            "title": "Self-Instruct: Aligning Language Models",
            "category": "指令微调",
            "url": "https://arxiv.org/abs/2212.10560",
            "_matched_keywords": ["instruction", "self-instruct"],
        },
    ]


# ---------------------------------------------------------------------------
# Initialization tests
# ---------------------------------------------------------------------------


class TestIntelReportGeneratorInit:
    """Tests for IntelReportGenerator initialization."""

    def test_default_config(self):
        """IntelReportGenerator should accept an empty config without errors."""
        gen = IntelReportGenerator(config={})
        assert gen.config == {}
        assert gen.limits == {}

    def test_config_with_report_limits(self):
        """Limits should be extracted from config['report']['limits']."""
        config = {"report": {"limits": {"labs_per_category": 5, "datasets_per_type": 8}}}
        gen = IntelReportGenerator(config=config)
        assert gen.limits["labs_per_category"] == 5
        assert gen.limits["datasets_per_type"] == 8

    def test_config_missing_report_key(self):
        """Missing 'report' key should produce empty limits, not an error."""
        gen = IntelReportGenerator(config={"other_key": 42})
        assert gen.limits == {}


# ---------------------------------------------------------------------------
# generate() — empty data
# ---------------------------------------------------------------------------


class TestGenerateEmptyData:
    """Tests for generate() when all inputs are empty."""

    def test_empty_data_returns_markdown(self, generator, empty_lab_activity,
                                         empty_vendor_activity, empty_datasets_by_type):
        """generate() with empty inputs should still return a valid markdown string."""
        report = generator.generate(
            lab_activity=empty_lab_activity,
            vendor_activity=empty_vendor_activity,
            datasets_by_type=empty_datasets_by_type,
            papers=[],
        )
        assert isinstance(report, str)
        assert len(report) > 0

    def test_empty_data_contains_header(self, generator, empty_lab_activity,
                                        empty_vendor_activity, empty_datasets_by_type):
        """Report should start with the main header even with empty data."""
        report = generator.generate(
            lab_activity=empty_lab_activity,
            vendor_activity=empty_vendor_activity,
            datasets_by_type=empty_datasets_by_type,
            papers=[],
        )
        assert "# AI 数据情报周报" in report

    def test_empty_data_contains_all_sections(self, generator, empty_lab_activity,
                                              empty_vendor_activity, empty_datasets_by_type):
        """All section headers should be present even when data is empty."""
        report = generator.generate(
            lab_activity=empty_lab_activity,
            vendor_activity=empty_vendor_activity,
            datasets_by_type=empty_datasets_by_type,
            papers=[],
        )
        assert "本周摘要" in report
        assert "美国 AI Labs 动态" in report
        assert "数据供应商动态" in report
        assert "高价值数据集" in report
        assert "相关论文" in report

    def test_empty_labs_shows_no_activity(self, generator, empty_lab_activity,
                                          empty_vendor_activity, empty_datasets_by_type):
        """Empty labs should show the 'no activity' message."""
        report = generator.generate(
            lab_activity=empty_lab_activity,
            vendor_activity=empty_vendor_activity,
            datasets_by_type=empty_datasets_by_type,
            papers=[],
        )
        assert "本周无监控目标的新活动" in report

    def test_empty_papers_shows_no_papers(self, generator, empty_lab_activity,
                                          empty_vendor_activity, empty_datasets_by_type):
        """Empty papers list should show the 'no papers' message."""
        report = generator.generate(
            lab_activity=empty_lab_activity,
            vendor_activity=empty_vendor_activity,
            datasets_by_type=empty_datasets_by_type,
            papers=[],
        )
        assert "本周无高度相关论文" in report


# ---------------------------------------------------------------------------
# generate() — with real data
# ---------------------------------------------------------------------------


class TestGenerateWithData:
    """Tests for generate() with populated inputs."""

    def test_report_contains_lab_names(self, generator, sample_lab_activity,
                                       sample_vendor_activity, sample_datasets_by_type,
                                       sample_papers):
        """Lab organization names should appear in the report."""
        report = generator.generate(
            lab_activity=sample_lab_activity,
            vendor_activity=sample_vendor_activity,
            datasets_by_type=sample_datasets_by_type,
            papers=sample_papers,
        )
        # org names are title-cased with underscores replaced
        assert "Openai" in report
        assert "Mistral" in report

    def test_report_contains_vendor_datasets(self, generator, sample_lab_activity,
                                              sample_vendor_activity, sample_datasets_by_type,
                                              sample_papers):
        """Vendor dataset names should appear in the report."""
        report = generator.generate(
            lab_activity=sample_lab_activity,
            vendor_activity=sample_vendor_activity,
            datasets_by_type=sample_datasets_by_type,
            papers=sample_papers,
        )
        assert "rlhf-pairs" in report

    def test_report_contains_dataset_types(self, generator, sample_lab_activity,
                                            sample_vendor_activity, sample_datasets_by_type,
                                            sample_papers):
        """Typed datasets should produce section headers with display names."""
        report = generator.generate(
            lab_activity=sample_lab_activity,
            vendor_activity=sample_vendor_activity,
            datasets_by_type=sample_datasets_by_type,
            papers=sample_papers,
        )
        assert "SFT/指令数据" in report
        assert "RLHF/偏好数据" in report

    def test_report_contains_paper_titles(self, generator, sample_lab_activity,
                                           sample_vendor_activity, sample_datasets_by_type,
                                           sample_papers):
        """Paper titles should appear in the report."""
        report = generator.generate(
            lab_activity=sample_lab_activity,
            vendor_activity=sample_vendor_activity,
            datasets_by_type=sample_datasets_by_type,
            papers=sample_papers,
        )
        # Title is truncated to 55 chars
        assert "Direct Preference Optimization" in report
        assert "Self-Instruct" in report

    def test_report_contains_github_activity(self, generator, sample_lab_activity,
                                              sample_vendor_activity, sample_datasets_by_type,
                                              sample_papers, sample_github_activity):
        """GitHub activity should appear when provided."""
        report = generator.generate(
            lab_activity=sample_lab_activity,
            vendor_activity=sample_vendor_activity,
            datasets_by_type=sample_datasets_by_type,
            papers=sample_papers,
            github_activity=sample_github_activity,
        )
        assert "llm-engine" in report
        assert "GitHub 活动" in report

    def test_report_contains_blog_activity(self, generator, sample_lab_activity,
                                            sample_vendor_activity, sample_datasets_by_type,
                                            sample_papers, sample_blog_activity):
        """Blog activity should appear when provided."""
        report = generator.generate(
            lab_activity=sample_lab_activity,
            vendor_activity=sample_vendor_activity,
            datasets_by_type=sample_datasets_by_type,
            papers=sample_papers,
            blog_activity=sample_blog_activity,
        )
        assert "Scaling RLHF" in report
        assert "博客动态" in report

    def test_report_footer(self, generator, sample_lab_activity,
                           sample_vendor_activity, sample_datasets_by_type,
                           sample_papers):
        """Report should end with the versioned footer."""
        report = generator.generate(
            lab_activity=sample_lab_activity,
            vendor_activity=sample_vendor_activity,
            datasets_by_type=sample_datasets_by_type,
            papers=sample_papers,
        )
        assert "AI Dataset Radar v" in report

    def test_summary_counts_active_labs(self, generator, sample_lab_activity,
                                        sample_vendor_activity, sample_datasets_by_type,
                                        sample_papers):
        """Summary should reflect correct count of active labs."""
        report = generator.generate(
            lab_activity=sample_lab_activity,
            vendor_activity=sample_vendor_activity,
            datasets_by_type=sample_datasets_by_type,
            papers=sample_papers,
        )
        # openai has datasets+models, anthropic has models only, mistral has datasets,
        # allen_ai has datasets => 4 active labs
        assert "4 家" in report

    def test_high_other_ratio_triggers_warning(self, generator, empty_lab_activity,
                                                empty_vendor_activity):
        """When >30% datasets are OTHER, a classification warning should appear."""
        datasets_by_type = {
            DataType.OTHER: [
                {"id": f"org/unclassified-{i}", "downloads": 1, "signals": ["-"]}
                for i in range(7)
            ],
            DataType.CODE: [
                {"id": "org/code-ds", "downloads": 10, "signals": ["code"]},
            ],
        }
        report = generator.generate(
            lab_activity=empty_lab_activity,
            vendor_activity=empty_vendor_activity,
            datasets_by_type=datasets_by_type,
            papers=[],
        )
        assert "分类覆盖率" in report


# ---------------------------------------------------------------------------
# generate_console_summary()
# ---------------------------------------------------------------------------


class TestGenerateConsoleSummary:
    """Tests for generate_console_summary()."""

    def test_console_summary_returns_string(self, generator, empty_lab_activity,
                                             empty_vendor_activity, empty_datasets_by_type):
        """Console summary should return a non-empty string."""
        result = generator.generate_console_summary(
            lab_activity=empty_lab_activity,
            vendor_activity=empty_vendor_activity,
            datasets_by_type=empty_datasets_by_type,
        )
        assert isinstance(result, str)
        assert len(result) > 0

    def test_console_summary_header(self, generator, empty_lab_activity,
                                     empty_vendor_activity, empty_datasets_by_type):
        """Console summary should contain the versioned header."""
        result = generator.generate_console_summary(
            lab_activity=empty_lab_activity,
            vendor_activity=empty_vendor_activity,
            datasets_by_type=empty_datasets_by_type,
        )
        assert "AI Dataset Radar v" in result
        assert "竞争情报摘要" in result

    def test_console_summary_shows_lab_names(self, generator, sample_lab_activity,
                                              empty_vendor_activity, sample_datasets_by_type):
        """Console summary should list active lab names."""
        result = generator.generate_console_summary(
            lab_activity=sample_lab_activity,
            vendor_activity=empty_vendor_activity,
            datasets_by_type=sample_datasets_by_type,
        )
        assert "openai" in result
        assert "Frontier Labs" in result

    def test_console_summary_shows_type_distribution(self, generator, empty_lab_activity,
                                                      empty_vendor_activity,
                                                      sample_datasets_by_type):
        """Console summary should show dataset type distribution."""
        result = generator.generate_console_summary(
            lab_activity=empty_lab_activity,
            vendor_activity=empty_vendor_activity,
            datasets_by_type=sample_datasets_by_type,
        )
        assert "数据集类型分布" in result
        assert "sft_instruction: 1" in result
        assert "rlhf_preference: 2" in result

    def test_console_summary_shows_github_and_blog(self, generator, sample_lab_activity,
                                                    empty_vendor_activity,
                                                    sample_datasets_by_type,
                                                    sample_github_activity,
                                                    sample_blog_activity):
        """Console summary should mention GitHub and blog sources when provided."""
        result = generator.generate_console_summary(
            lab_activity=sample_lab_activity,
            vendor_activity=empty_vendor_activity,
            datasets_by_type=sample_datasets_by_type,
            github_activity=sample_github_activity,
            blog_activity=sample_blog_activity,
        )
        assert "GitHub活跃" in result
        assert "scale-ai" in result
        assert "博客更新" in result
        assert "Scale AI" in result
