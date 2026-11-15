"""Regression tests for weekly issue titles.

The retired template was a fixed slot: a concrete event, then a second clause
that human judgment matters more. Titles now follow the issue's own most
important finding, and that slot must not come back.
"""

from pathlib import Path

from insight_titles import (
    extract_candidate_titles,
    is_fixed_judgment_slot,
    select_issue_title,
)

ROOT = Path(__file__).resolve().parents[1]
PIPELINE = ROOT / "pipeline-CLAUDE.md"
RUN_WEEKLY = ROOT / "run_weekly.sh"

REPORT = """
### 1. 关键发现

#### 1.1 11月11日论文比较固定预算下的人类与合成偏好 [P0]

- **详情**：2026-11-12 的论文把比较从样本数改成固定预算。
- **商业意义**：需要知道哪些样本值得交给人看。

#### 1.2 Surge AI 更新三项专业评估基准 [P0]

- **详情**：Handbook、ComplexConstraints、Chartography。
- **商业意义**：任务和评分准则被做成了基准。

## 候选标题

1. 新增一万余条楼梯运动｜物理数据扩容更需人类判断
2. 哪些任务值得测，哪些样本值得人看
3. NVIDIA数据栈领跑具身智能，人类判断决定安全落地

## 不应被读入的后文

1. 这条不是候选标题
"""


def test_prompt_drops_the_fixed_title_slot() -> None:
    text = PIPELINE.read_text(encoding="utf-8")
    start = text.index("### 章节 5：候选标题")
    section = text[start:]
    assert "结构不固定" in section
    assert "独有" in section
    assert "最重要" in section
    assert "后半句" not in section
    assert "人类判断/数据价值" not in section
    assert "集识光年使命" not in section
    assert "20-35" not in section
    assert "用全角 `｜` 分隔前后两部分" not in section
    assert "不要用全角 `｜` 或半角 `|` 把标题拆成「事实 + 立场」" in section
    assert "标题 A 前半句" not in section


def test_publisher_no_longer_appends_the_canned_second_clause() -> None:
    script = RUN_WEEKLY.read_text(encoding="utf-8")
    assert "select_issue_title" in script
    assert "AI 数据行业周度洞察" not in script
    assert "AI 数据行业前沿洞察" not in script


def test_retired_slot_is_recognized_and_specific_titles_are_kept() -> None:
    assert is_fixed_judgment_slot("新增一万余条楼梯运动｜物理数据扩容更需人类判断")
    assert is_fixed_judgment_slot("新增一万余条楼梯运动 | 物理数据扩容更需人类判断")
    assert is_fixed_judgment_slot("多语种语音推出高保真筛选，人类判断连接质量与收入")
    assert is_fixed_judgment_slot("视频理解数据进入工业化供给｜Apple 论证人类判断不可替代")
    assert is_fixed_judgment_slot("人类判断更重要")
    assert not is_fixed_judgment_slot("哪些任务值得测，哪些样本值得人看")
    assert not is_fixed_judgment_slot("11月11日论文比较固定预算下的人类与合成偏好")


def test_selector_skips_the_slogan_slot_for_the_unique_finding() -> None:
    assert extract_candidate_titles(REPORT)[0].startswith("新增一万余条楼梯运动")
    assert select_issue_title(REPORT) == "哪些任务值得测，哪些样本值得人看"


def test_selector_keeps_a_specific_first_candidate() -> None:
    text = REPORT.replace(
        "1. 新增一万余条楼梯运动｜物理数据扩容更需人类判断\n",
        "1. GRAIL 新增 12512 条楼梯运动\n",
    )
    assert select_issue_title(text) == "GRAIL 新增 12512 条楼梯运动"


def test_selector_falls_back_to_the_finding_without_a_slogan() -> None:
    text = REPORT.replace("2. 哪些任务值得测，哪些样本值得人看\n", "")
    text = text.replace(
        "3. NVIDIA数据栈领跑具身智能，人类判断决定安全落地\n",
        "2. NVIDIA数据栈领跑具身智能，人类判断决定安全落地\n",
    )
    assert select_issue_title(text) == "11月11日论文比较固定预算下的人类与合成偏好"
    assert "人类判断" not in select_issue_title(text)
    assert "周度洞察" not in select_issue_title(text)


def test_explicit_title_overrides_candidates_without_rewriting_them() -> None:
    approved = "哪些任务值得测，哪些样本值得人看"
    assert select_issue_title(REPORT, override=approved) == approved
    assert select_issue_title("没有候选标题", override=approved) == approved
