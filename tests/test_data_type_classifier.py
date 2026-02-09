"""Tests for data_type_classifier module."""

import importlib.util
import sys
from pathlib import Path

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import the module directly to avoid pulling in the full analyzers package
# (the analyzers/__init__.py imports heavy dependencies we don't need here).
_mod_path = Path(__file__).parent.parent / "src" / "analyzers" / "data_type_classifier.py"
_spec = importlib.util.spec_from_file_location("data_type_classifier", _mod_path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

DataType = _mod.DataType
DataTypeClassifier = _mod.DataTypeClassifier
CLASSIFICATION_RULES = _mod.CLASSIFICATION_RULES
TYPE_DISPLAY = _mod.TYPE_DISPLAY


# ---------------------------------------------------------------------------
# DataType enum tests
# ---------------------------------------------------------------------------


class TestDataTypeEnum:
    """Tests for DataType enum values and properties."""

    def test_all_expected_members_exist(self):
        """Verify every expected enum member is defined."""
        expected = [
            "RLHF_PREFERENCE",
            "REWARD_MODEL",
            "SFT_INSTRUCTION",
            "CODE",
            "AGENT_TOOL",
            "ROBOTICS",
            "RL_ENVIRONMENT",
            "SYNTHETIC",
            "MULTIMODAL",
            "MULTILINGUAL",
            "EVALUATION",
            "OTHER",
        ]
        for name in expected:
            assert hasattr(DataType, name), f"DataType.{name} is missing"

    def test_enum_values_are_strings(self):
        """Each enum value should be a lowercase string."""
        for member in DataType:
            assert isinstance(member.value, str)
            assert member.value == member.value.lower()

    def test_enum_member_count(self):
        """Ensure no unexpected members were added silently."""
        assert len(DataType) == 12

    @pytest.mark.parametrize(
        "member,value",
        [
            (DataType.RLHF_PREFERENCE, "rlhf_preference"),
            (DataType.REWARD_MODEL, "reward_model"),
            (DataType.SFT_INSTRUCTION, "sft_instruction"),
            (DataType.CODE, "code"),
            (DataType.AGENT_TOOL, "agent_tool"),
            (DataType.ROBOTICS, "robotics"),
            (DataType.RL_ENVIRONMENT, "rl_environment"),
            (DataType.SYNTHETIC, "synthetic"),
            (DataType.MULTIMODAL, "multimodal"),
            (DataType.MULTILINGUAL, "multilingual"),
            (DataType.EVALUATION, "evaluation"),
            (DataType.OTHER, "other"),
        ],
    )
    def test_enum_value_mapping(self, member, value):
        """Each member should map to its expected string value."""
        assert member.value == value


# ---------------------------------------------------------------------------
# Classification rules / display config sanity checks
# ---------------------------------------------------------------------------


class TestClassificationRulesAndDisplay:
    """Ensure classification rules and display config are consistent."""

    def test_every_type_except_other_has_rules(self):
        """All non-OTHER types should have classification rules."""
        for dtype in DataType:
            if dtype is DataType.OTHER:
                continue
            assert dtype in CLASSIFICATION_RULES, f"Missing rules for {dtype}"

    def test_every_type_has_display_config(self):
        """All types should appear in TYPE_DISPLAY."""
        for dtype in DataType:
            assert dtype in TYPE_DISPLAY, f"Missing display config for {dtype}"

    def test_display_order_values_are_unique(self):
        """Display sort-order values must not collide."""
        orders = [v[1] for v in TYPE_DISPLAY.values()]
        assert len(orders) == len(set(orders))


# ---------------------------------------------------------------------------
# DataTypeClassifier instantiation
# ---------------------------------------------------------------------------


class TestClassifierInit:
    """Tests for DataTypeClassifier construction."""

    def test_default_init(self):
        """Classifier can be created without arguments."""
        clf = DataTypeClassifier()
        assert clf.config == {}
        assert clf.min_score == 2
        assert clf.fetch_readme is True

    def test_custom_config(self):
        """Custom config values are respected."""
        cfg = {"classifier": {"min_score_threshold": 5, "fetch_readme": False}}
        clf = DataTypeClassifier(config=cfg)
        assert clf.min_score == 5
        assert clf.fetch_readme is False


# ---------------------------------------------------------------------------
# classify() -- core classification tests
# ---------------------------------------------------------------------------


class TestClassify:
    """Tests for DataTypeClassifier.classify()."""

    @pytest.fixture
    def clf(self):
        """Classifier with default settings."""
        return DataTypeClassifier()

    # -- Edge cases ----------------------------------------------------------

    def test_empty_dataset_returns_other(self, clf):
        """An empty dict should fall back to OTHER."""
        result = clf.classify({})
        assert result == [DataType.OTHER]

    def test_no_matching_signals_returns_other(self, clf):
        """Completely irrelevant metadata should yield OTHER."""
        ds = {
            "id": "foo/random-stuff",
            "description": "nothing special here",
            "tags": [],
        }
        result = clf.classify(ds)
        assert result == [DataType.OTHER]

    def test_none_values_do_not_crash(self, clf):
        """None values in fields should not raise exceptions."""
        ds = {
            "id": None,
            "name": None,
            "description": None,
            "tags": None,
            "features": None,
        }
        result = clf.classify(ds)
        assert isinstance(result, list)
        assert all(isinstance(t, DataType) for t in result)

    def test_none_id_with_valid_name(self, clf):
        """When id is None but name is valid, classification should work."""
        ds = {"id": None, "name": "dpo-preference-set", "tags": []}
        result = clf.classify(ds)
        assert DataType.RLHF_PREFERENCE in result

    # -- Name-pattern matching -----------------------------------------------

    @pytest.mark.parametrize(
        "dataset_id,expected_type",
        [
            ("org/my-dpo-dataset", DataType.RLHF_PREFERENCE),
            ("org/rlhf-reward-mix", DataType.RLHF_PREFERENCE),
            ("org/sft-instruct-v2", DataType.SFT_INSTRUCTION),
            ("org/chat-conversations", DataType.SFT_INSTRUCTION),
            ("org/code-exercises", DataType.CODE),
            ("org/python-solutions", DataType.CODE),
            ("org/agent-trajectories", DataType.AGENT_TOOL),
            ("org/vision-qa", DataType.MULTIMODAL),
            ("org/image-caption-set", DataType.MULTIMODAL),
            ("org/eval-benchmark-2024", DataType.EVALUATION),
            ("org/multilingual-corpus", DataType.MULTILINGUAL),
        ],
    )
    def test_classify_by_name_pattern(self, clf, dataset_id, expected_type):
        """Name patterns should trigger the correct primary type."""
        ds = {"id": dataset_id}
        result = clf.classify(ds)
        assert expected_type in result, (
            f"Expected {expected_type} for id='{dataset_id}', got {result}"
        )

    # -- Description keyword matching ----------------------------------------

    @pytest.mark.parametrize(
        "description,expected_type",
        [
            (
                "This dataset contains human preference comparisons for RLHF",
                DataType.RLHF_PREFERENCE,
            ),
            (
                "Instruction fine-tuning data for supervised training",
                DataType.SFT_INSTRUCTION,
            ),
            (
                "A code generation and programming exercise corpus",
                DataType.CODE,
            ),
            (
                "Agent tool use traces from web browsing sessions",
                DataType.AGENT_TOOL,
            ),
            (
                "Vision and image understanding multimodal QA pairs",
                DataType.MULTIMODAL,
            ),
            (
                "Synthetic distillation data generated by a teacher model",
                DataType.SYNTHETIC,
            ),
            (
                "Benchmark evaluation test set with leaderboard metrics",
                DataType.EVALUATION,
            ),
            (
                "Reward model training data with scalar reward signals",
                DataType.REWARD_MODEL,
            ),
            (
                "Multilingual parallel corpus for machine translation",
                DataType.MULTILINGUAL,
            ),
            (
                "RL environment trajectory episodes with observations",
                DataType.RL_ENVIRONMENT,
            ),
            (
                "Robotic manipulation teleoperation data for embodied AI grasping tasks",
                DataType.ROBOTICS,
            ),
        ],
    )
    def test_classify_by_description(self, clf, description, expected_type):
        """Keywords in descriptions should trigger the correct type."""
        ds = {"id": "org/generic-name", "description": description}
        result = clf.classify(ds)
        assert expected_type in result, (
            f"Expected {expected_type} for description snippet, got {result}"
        )

    # -- Tag matching --------------------------------------------------------

    @pytest.mark.parametrize(
        "tags,expected_type",
        [
            (["dpo", "human-feedback"], DataType.RLHF_PREFERENCE),
            (["sft", "instruction-tuning"], DataType.SFT_INSTRUCTION),
            (["code", "programming"], DataType.CODE),
            (["agent", "function-calling"], DataType.AGENT_TOOL),
            (["multimodal", "vision"], DataType.MULTIMODAL),
            (["synthetic", "generated"], DataType.SYNTHETIC),
            (["benchmark", "evaluation"], DataType.EVALUATION),
            (["reward-model", "ppo"], DataType.REWARD_MODEL),
            (["multilingual", "translation"], DataType.MULTILINGUAL),
            (["robotics", "embodied-ai"], DataType.ROBOTICS),
            (["reinforcement-learning", "environment"], DataType.RL_ENVIRONMENT),
        ],
    )
    def test_classify_by_tags(self, clf, tags, expected_type):
        """Matching tags should trigger the correct primary type."""
        ds = {"id": "org/generic-name", "tags": tags}
        result = clf.classify(ds)
        assert expected_type in result, (
            f"Expected {expected_type} for tags={tags}, got {result}"
        )

    def test_tags_with_key_value_format(self, clf):
        """Tags in 'key:value' format should be normalised correctly."""
        ds = {"id": "org/generic", "tags": ["task:dpo", "license:mit"]}
        result = clf.classify(ds)
        assert DataType.RLHF_PREFERENCE in result

    # -- Multi-label classification ------------------------------------------

    def test_multi_label_returns_multiple_types(self, clf):
        """A dataset with signals for several types returns them all."""
        ds = {
            "id": "org/code-instruct-dpo",
            "description": "Instruction-tuned code generation with preference pairs",
            "tags": ["code", "dpo", "sft"],
        }
        result = clf.classify(ds)
        # Should detect at least two distinct categories
        assert len(result) >= 2
        type_set = set(result)
        # Code and preference should both be present
        assert DataType.CODE in type_set
        assert DataType.RLHF_PREFERENCE in type_set

    def test_results_sorted_by_score_descending(self, clf):
        """Types with higher scores should appear first."""
        ds = {
            "id": "org/preference-dpo-ranked",
            "description": "human preference comparison ranking chosen rejected",
            "tags": ["dpo", "rlhf", "preference", "human-feedback"],
        }
        result = clf.classify(ds)
        # RLHF_PREFERENCE should be first because it has by far the most signals
        assert result[0] == DataType.RLHF_PREFERENCE

    # -- Case insensitivity --------------------------------------------------

    def test_description_case_insensitive(self, clf):
        """Keyword matching in descriptions should be case-insensitive."""
        ds = {
            "id": "org/foo",
            "description": "PREFERENCE COMPARISON with HUMAN FEEDBACK for DPO training",
        }
        result = clf.classify(ds)
        assert DataType.RLHF_PREFERENCE in result

    def test_tags_case_insensitive(self, clf):
        """Tag matching should be case-insensitive."""
        ds = {"id": "org/bar", "tags": ["DPO", "RLHF", "Preference"]}
        result = clf.classify(ds)
        assert DataType.RLHF_PREFERENCE in result

    def test_name_pattern_case_insensitive(self, clf):
        """Name-pattern matching should be case-insensitive."""
        ds = {"id": "org/My-DPO-Dataset"}
        result = clf.classify(ds)
        assert DataType.RLHF_PREFERENCE in result

    # -- Feature / field pattern matching ------------------------------------

    def test_feature_field_matching(self, clf):
        """Field names inside features should boost the score."""
        ds = {
            "id": "org/generic",
            "description": "A dataset with chosen and rejected columns",
            "features": {"chosen": "string", "rejected": "string"},
        }
        result = clf.classify(ds)
        assert DataType.RLHF_PREFERENCE in result

    # -- min_score threshold -------------------------------------------------

    def test_high_threshold_filters_weak_matches(self):
        """Raising min_score should drop borderline classifications."""
        clf_strict = DataTypeClassifier(
            config={"classifier": {"min_score_threshold": 50}}
        )
        ds = {
            "id": "org/vaguely-code-related",
            "description": "some code examples",
        }
        result = clf_strict.classify(ds)
        assert result == [DataType.OTHER]

    # -- Specific dataset type scenarios -------------------------------------

    def test_sft_instruction_dataset(self, clf):
        """Realistic SFT/instruction dataset should be classified correctly."""
        ds = {
            "id": "HuggingFaceH4/ultrachat_200k",
            "description": "Multi-turn conversational instruction dataset for SFT",
            "tags": ["chat", "sft", "conversational", "instruction-tuning"],
            "features": {"instruction": "string", "response": "string"},
        }
        result = clf.classify(ds)
        assert result[0] == DataType.SFT_INSTRUCTION

    def test_rlhf_preference_dataset(self, clf):
        """Realistic RLHF/preference dataset should be classified correctly."""
        ds = {
            "id": "Anthropic/hh-rlhf",
            "description": "Human preference data about helpfulness and harmlessness",
            "tags": ["rlhf", "human-feedback", "preference"],
            "features": {"chosen": "string", "rejected": "string"},
        }
        result = clf.classify(ds)
        assert result[0] == DataType.RLHF_PREFERENCE

    def test_synthetic_dataset(self, clf):
        """Realistic synthetic dataset should be classified correctly."""
        ds = {
            "id": "teknium/OpenHermes-2.5-v2",
            "description": "Synthetic GPT-generated instruction data via distillation",
            "tags": ["synthetic", "generated"],
        }
        result = clf.classify(ds)
        assert DataType.SYNTHETIC in result

    def test_code_dataset(self, clf):
        """Realistic code dataset should be classified correctly."""
        ds = {
            "id": "bigcode/the-stack",
            "description": "Code generation dataset with Python and JavaScript",
            "tags": ["code", "programming", "code-generation"],
        }
        result = clf.classify(ds)
        assert result[0] == DataType.CODE

    def test_multimodal_dataset(self, clf):
        """Realistic multimodal dataset should be classified correctly."""
        ds = {
            "id": "HuggingFaceM4/the-cauldron",
            "description": "Vision and image understanding visual question answering",
            "tags": ["multimodal", "vision", "vqa", "image"],
        }
        result = clf.classify(ds)
        assert result[0] == DataType.MULTIMODAL

    # -- readme / card_data --------------------------------------------------

    def test_readme_content_contributes_to_matching(self, clf):
        """Keywords found in card_data / readme should influence classification."""
        ds = {
            "id": "org/generic",
            "description": "",
            "tags": [],
            "card_data": "This dataset is designed for reward model training with PPO.",
        }
        result = clf.classify(ds)
        assert DataType.REWARD_MODEL in result

    # -- name fallback -------------------------------------------------------

    def test_name_field_used_when_id_missing(self, clf):
        """When 'id' is absent, 'name' should be used for name-pattern matching."""
        ds = {"name": "my-dpo-preference-set"}
        result = clf.classify(ds)
        assert DataType.RLHF_PREFERENCE in result


# ---------------------------------------------------------------------------
# classify_with_details()
# ---------------------------------------------------------------------------


class TestClassifyWithDetails:
    """Tests for DataTypeClassifier.classify_with_details()."""

    @pytest.fixture
    def clf(self):
        return DataTypeClassifier()

    def test_returns_expected_keys(self, clf):
        """Result dict should contain the documented keys."""
        ds = {"id": "org/something", "tags": ["dpo"]}
        details = clf.classify_with_details(ds)
        assert "dataset_id" in details
        assert "primary_type" in details
        assert "all_types" in details
        assert "is_relevant" in details
        assert "type_count" in details

    def test_primary_type_matches_first_classify_result(self, clf):
        """primary_type should equal the first element of classify()."""
        ds = {"id": "org/dpo-pref", "tags": ["dpo", "preference"]}
        details = clf.classify_with_details(ds)
        direct = clf.classify(ds)
        assert details["primary_type"] == direct[0]

    def test_is_relevant_false_for_other(self, clf):
        """is_relevant should be False when primary type is OTHER."""
        ds = {"id": "org/unrelated"}
        details = clf.classify_with_details(ds)
        assert details["is_relevant"] is False

    def test_is_relevant_true_for_known_type(self, clf):
        """is_relevant should be True for a real classification."""
        ds = {"id": "org/instruct-sft", "tags": ["sft"]}
        details = clf.classify_with_details(ds)
        assert details["is_relevant"] is True

    def test_type_count_excludes_other(self, clf):
        """type_count should not count OTHER."""
        ds = {"id": "org/unrelated"}
        details = clf.classify_with_details(ds)
        assert details["type_count"] == 0


# ---------------------------------------------------------------------------
# group_by_type()
# ---------------------------------------------------------------------------


class TestGroupByType:
    """Tests for DataTypeClassifier.group_by_type()."""

    @pytest.fixture
    def clf(self):
        return DataTypeClassifier()

    def test_empty_list_returns_empty_dict(self, clf):
        """No datasets should produce no groups."""
        assert clf.group_by_type([]) == {}

    def test_datasets_grouped_correctly(self, clf):
        """Datasets should land in the correct primary-type bucket."""
        datasets = [
            {"id": "org/dpo-pref", "tags": ["dpo", "preference"]},
            {"id": "org/sft-chat", "tags": ["sft", "chat"]},
            {"id": "org/random-stuff"},
        ]
        groups = clf.group_by_type(datasets)
        assert DataType.RLHF_PREFERENCE in groups
        assert DataType.SFT_INSTRUCTION in groups
        assert DataType.OTHER in groups

    def test_grouped_datasets_carry_classification_info(self, clf):
        """Each grouped dataset should have data_type, all_types, signals."""
        datasets = [{"id": "org/dpo-pref", "tags": ["dpo", "preference"]}]
        groups = clf.group_by_type(datasets)
        ds = groups[DataType.RLHF_PREFERENCE][0]
        assert "data_type" in ds
        assert "all_types" in ds
        assert "signals" in ds


# ---------------------------------------------------------------------------
# summarize()
# ---------------------------------------------------------------------------


class TestSummarize:
    """Tests for DataTypeClassifier.summarize()."""

    @pytest.fixture
    def clf(self):
        return DataTypeClassifier()

    def test_summarize_empty(self, clf):
        """Summarising no datasets should return zero counts."""
        summary = clf.summarize([])
        assert summary["total"] == 0
        assert summary["relevant"] == 0
        assert summary["other_ratio"] == 0

    def test_summarize_counts(self, clf):
        """Counts should reflect the classified distribution."""
        datasets = [
            {"id": "org/dpo-pref", "tags": ["dpo", "preference"]},
            {"id": "org/sft-chat", "tags": ["sft", "chat"]},
            {"id": "org/random-stuff"},
        ]
        summary = clf.summarize(datasets)
        assert summary["total"] == 3
        assert summary["relevant"] == 2
        assert "rlhf_preference" in summary["by_type"]
        assert "sft_instruction" in summary["by_type"]


# ---------------------------------------------------------------------------
# Static / display helpers
# ---------------------------------------------------------------------------


class TestDisplayHelpers:
    """Tests for static display helper methods."""

    def test_get_type_display_known(self):
        """Known types should return their configured display string."""
        display = DataTypeClassifier.get_type_display(DataType.CODE)
        assert isinstance(display, str)
        assert len(display) > 0

    def test_get_type_display_other_fallback(self):
        """OTHER should still return a display string."""
        display = DataTypeClassifier.get_type_display(DataType.OTHER)
        assert isinstance(display, str)

    def test_get_type_order_returns_int(self):
        """get_type_order should return an integer."""
        order = DataTypeClassifier.get_type_order(DataType.SFT_INSTRUCTION)
        assert isinstance(order, int)

    def test_get_ordered_types_sorted(self):
        """get_ordered_types should return items sorted by display order."""
        ordered = DataTypeClassifier.get_ordered_types()
        orders = [TYPE_DISPLAY[item[0]][1] for item in ordered]
        assert orders == sorted(orders)

    def test_get_ordered_types_complete(self):
        """get_ordered_types should include every DataType."""
        ordered = DataTypeClassifier.get_ordered_types()
        returned_types = {item[0] for item in ordered}
        assert returned_types == set(DataType)


# ---------------------------------------------------------------------------
# Internal helper methods
# ---------------------------------------------------------------------------


class TestInternalHelpers:
    """Tests for _build_search_text and _normalize_tags."""

    @pytest.fixture
    def clf(self):
        return DataTypeClassifier()

    def test_build_search_text_combines_fields(self, clf):
        """_build_search_text should concatenate id, description, tags, card_data."""
        ds = {
            "id": "org/my-dataset",
            "description": "A fine dataset",
            "tags": ["sft", "chat"],
            "card_data": "Extra readme content",
        }
        text = clf._build_search_text(ds)
        assert "org/my-dataset" in text
        assert "A fine dataset" in text
        assert "sft" in text
        assert "Extra readme content" in text

    def test_build_search_text_handles_missing_fields(self, clf):
        """Missing fields should not cause errors."""
        text = clf._build_search_text({})
        assert isinstance(text, str)

    def test_normalize_tags_basic(self, clf):
        """Tags should be lowercased and returned as a set."""
        tags = ["DPO", "SFT", "Chat"]
        result = clf._normalize_tags(tags)
        assert result == {"dpo", "sft", "chat"}

    def test_normalize_tags_key_value(self, clf):
        """Tags with 'key:value' format should extract the value part."""
        tags = ["task:dpo", "license:mit"]
        result = clf._normalize_tags(tags)
        assert "dpo" in result
        assert "mit" in result
        # The full key:value should NOT be in the set
        assert "task:dpo" not in result

    def test_normalize_tags_none_input(self, clf):
        """None input should return an empty set."""
        assert clf._normalize_tags(None) == set()

    def test_normalize_tags_non_string_items(self, clf):
        """Non-string items in tags list should be skipped."""
        tags = ["dpo", 123, None, "sft"]
        result = clf._normalize_tags(tags)
        assert result == {"dpo", "sft"}
