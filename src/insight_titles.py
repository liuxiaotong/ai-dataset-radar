"""Choose the weekly issue title from the insight report.

The title states the most important finding that belongs to this issue.
It does not use a fixed two-part slot of "event | human judgment matters
more", and it does not append a company position or a human-value conclusion
the finding itself did not state.
"""

from __future__ import annotations

import re

_CANDIDATE_BLOCK = re.compile(
    r"^## 候选标题\s*\n([\s\S]*?)(?=^##\s|\Z)",
    re.MULTILINE,
)
_CANDIDATE_LINE = re.compile(r"^\d+[.、) ]+\s*(.+)$")
_FINDING_HEADING = re.compile(r"^#{3,4}\s+1\.1\s+(.+?)\s*$", re.MULTILINE)
_PRIORITY_SUFFIX = re.compile(r"\s*\[P\d\]\s*$")
_SLOGAN_TAIL = re.compile(
    r"(?:"
    r"更需人类判断"
    r"|人类判断更重要"
    r"|人类判断不可替代"
    r"|人类判断决定"
    r"|人类判断定义"
    r"|人类判断连接"
    r"|人类判断正?成为"
    r"|人类判断成"
    r")"
)
_CANNED_TITLES = frozenset(
    {
        "人类判断更重要",
        "更需人类判断",
        "人类判断不可替代",
    }
)
_FALLBACK = "本期最重要的数据发现"


def title_parts(title: str) -> list[str]:
    """Split a title on the retired fact/stance separators."""

    text = (title or "").replace("｜", "|").strip()
    if "|" in text:
        return [part.strip() for part in text.split("|") if part.strip()]
    if "，" in text:
        return [part.strip() for part in text.split("，") if part.strip()]
    return [text] if text else []


def is_fixed_judgment_slot(title: str) -> bool:
    """Return whether a title is the retired event-plus-slogan template.

    A concrete title may mention people when that is the finding. This only
    matches the bolted-on second clause, or a title that is nothing but the
    canned conclusion.
    """

    parts = title_parts(title)
    if len(parts) >= 2 and _SLOGAN_TAIL.search(parts[-1]):
        return True
    compact = re.sub(r"\s+", "", title or "")
    return compact in _CANNED_TITLES


def extract_candidate_titles(text: str) -> list[str]:
    """Return candidate titles in the order the report lists them."""

    match = _CANDIDATE_BLOCK.search(text or "")
    if not match:
        return []
    titles: list[str] = []
    for line in match.group(1).splitlines():
        found = _CANDIDATE_LINE.match(line.strip())
        if found:
            title = found.group(1).strip()
            if title:
                titles.append(title)
    return titles


def finding_heading(text: str) -> str:
    """Return the first finding heading, without a priority tag."""

    match = _FINDING_HEADING.search(text or "")
    if not match:
        return ""
    return _PRIORITY_SUFFIX.sub("", match.group(1)).strip()


def _without_slogan_slot(title: str) -> str:
    if not is_fixed_judgment_slot(title):
        return title.strip()
    parts = title_parts(title)
    if parts and not _SLOGAN_TAIL.search(parts[0]):
        return parts[0]
    return ""


def select_issue_title(text: str, override: str | None = None) -> str:
    """Pick the title that will be published for one issue.

    An explicit override is kept as given. That is how an already approved
    title is applied without rewriting the issue body. Automatic selection
    uses the first candidate that is not the retired slogan slot, then the
    first finding's own heading.
    """

    chosen = (override or "").strip()
    if chosen:
        return chosen
    for candidate in extract_candidate_titles(text):
        if not is_fixed_judgment_slot(candidate):
            return candidate
    heading = _without_slogan_slot(finding_heading(text))
    return heading or _FALLBACK
