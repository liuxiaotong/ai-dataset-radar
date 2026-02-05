#!/usr/bin/env python3
"""Code quality fixes for AI Dataset Radar.

This script automatically applies code quality improvements:
1. Replace print() with logger calls
2. Add DB input validation
3. Extract duplicate keyword matching logic
4. Run tests to verify changes

Run: python scripts/fix_code_quality.py
"""

import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).parent.parent
SRC_DIR = PROJECT_ROOT / "src"
REPORT_FILE = PROJECT_ROOT / "fix_report.md"

# Track changes
changes = []
errors = []


def log(msg: str):
    """Log message to console and report."""
    print(msg)
    changes.append(msg)


def replace_prints_with_logger(file_path: Path) -> int:
    """Replace print() calls with logger calls in a file.

    Returns number of replacements made.
    """
    content = file_path.read_text(encoding="utf-8")
    original = content

    # Check if logger already imported
    has_logger_import = "from utils.logging_config import" in content or "import logging" in content

    # Patterns to replace
    replacements = [
        # print(f"...") -> logger.info("...", ...)
        (r'print\(f"([^"]*?)"\)', r'logger.info("\1")'),
        # print("...") -> logger.info("...")
        (r'print\("([^"]*?)"\)', r'logger.info("\1")'),
        # print(f'...') -> logger.info('...', ...)
        (r"print\(f'([^']*?)'\)", r"logger.info('\1')"),
    ]

    count = 0
    for pattern, replacement in replacements:
        new_content, n = re.subn(pattern, replacement, content)
        if n > 0:
            content = new_content
            count += n

    # Add logger import if we made changes and it's not there
    if count > 0 and not has_logger_import:
        # Find the imports section and add logger import
        if "import " in content:
            # Add after last import
            lines = content.split("\n")
            last_import_idx = 0
            for i, line in enumerate(lines):
                if line.startswith("import ") or line.startswith("from "):
                    last_import_idx = i

            lines.insert(last_import_idx + 1, "")
            lines.insert(last_import_idx + 2, "from utils.logging_config import get_logger")
            lines.insert(last_import_idx + 3, "")
            lines.insert(last_import_idx + 4, 'logger = get_logger(__name__)')
            content = "\n".join(lines)

    if content != original:
        file_path.write_text(content, encoding="utf-8")
        return count
    return 0


def add_db_validation():
    """Add input validation to db.py."""
    db_file = SRC_DIR / "db.py"
    content = db_file.read_text(encoding="utf-8")

    # Check if validation already exists
    if "def _validate_required" in content:
        log("  DB validation already exists, skipping")
        return 0

    # Add validation helper function after imports
    validation_code = '''
def _validate_required(name: str, value, max_length: int = None):
    """Validate required field is not empty and within length limit."""
    if value is None or (isinstance(value, str) and not value.strip()):
        raise ValueError(f"{name} cannot be empty")
    if max_length and isinstance(value, str) and len(value) > max_length:
        raise ValueError(f"{name} exceeds max length of {max_length}")
    return value

'''

    # Find class definition and insert before it
    class_match = re.search(r'\nclass DatasetDatabase:', content)
    if class_match:
        insert_pos = class_match.start()
        content = content[:insert_pos] + "\n" + validation_code + content[insert_pos:]
        db_file.write_text(content, encoding="utf-8")
        log("  Added _validate_required() helper function")
        return 1

    return 0


def create_keyword_utils():
    """Create utility function for keyword matching."""
    utils_file = SRC_DIR / "utils" / "keywords.py"

    if utils_file.exists():
        log("  keywords.py already exists, skipping")
        return 0

    utils_file.parent.mkdir(parents=True, exist_ok=True)

    code = '''"""Keyword matching utilities for AI Dataset Radar."""

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
'''

    utils_file.write_text(code, encoding="utf-8")

    # Update __init__.py
    init_file = SRC_DIR / "utils" / "__init__.py"
    init_content = init_file.read_text(encoding="utf-8")
    if "keywords" not in init_content:
        init_content += "\nfrom .keywords import match_keywords, count_keyword_matches, calculate_relevance\n"
        init_file.write_text(init_content, encoding="utf-8")

    log("  Created utils/keywords.py with match_keywords(), count_keyword_matches(), calculate_relevance()")
    return 1


def add_config_validation():
    """Add config validation to main_intel.py."""
    main_file = SRC_DIR / "main_intel.py"
    content = main_file.read_text(encoding="utf-8")

    if "def validate_config" in content:
        log("  Config validation already exists, skipping")
        return 0

    # Add validation function
    validation_code = '''
def validate_config(config: dict) -> list[str]:
    """Validate configuration has required sections.

    Args:
        config: Configuration dictionary.

    Returns:
        List of warning messages.
    """
    warnings = []

    if not config:
        warnings.append("Configuration is empty, using defaults")
        return warnings

    # Check for watched orgs
    watched_orgs = config.get("watched_orgs", {})
    if not watched_orgs:
        warnings.append("No watched_orgs configured - no HuggingFace orgs will be tracked")

    # Check for watched vendors
    watched_vendors = config.get("watched_vendors", {})
    if not watched_vendors:
        warnings.append("No watched_vendors configured - no vendors will be tracked")

    # Check for blogs
    blogs = watched_vendors.get("blogs", [])
    if not blogs:
        warnings.append("No blogs configured - blog tracking disabled")

    return warnings

'''

    # Insert after load_config function
    load_config_end = content.find("def fetch_dataset_readmes")
    if load_config_end > 0:
        content = content[:load_config_end] + validation_code + "\n" + content[load_config_end:]
        main_file.write_text(content, encoding="utf-8")
        log("  Added validate_config() function")
        return 1

    return 0


def run_tests() -> tuple[bool, str]:
    """Run pytest and return success status and output."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=120
        )
        success = result.returncode == 0
        output = result.stdout + result.stderr
        return success, output
    except subprocess.TimeoutExpired:
        return False, "Tests timed out after 120 seconds"
    except Exception as e:
        return False, f"Error running tests: {e}"


def generate_report(test_success: bool, test_output: str):
    """Generate markdown report of all changes."""
    report = f"""# Code Quality Fix Report

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Summary

| Metric | Value |
|--------|-------|
| Files Modified | {len([c for c in changes if "Modified" in c or "Added" in c or "Created" in c])} |
| Tests Passing | {"Yes" if test_success else "No"} |

## Changes Made

"""
    for change in changes:
        report += f"- {change}\n"

    if errors:
        report += "\n## Errors\n\n"
        for error in errors:
            report += f"- {error}\n"

    report += f"""
## Test Results

```
{"PASSED" if test_success else "FAILED"}
```

<details>
<summary>Full Test Output</summary>

```
{test_output[-3000:] if len(test_output) > 3000 else test_output}
```

</details>

## Next Steps

"""
    if test_success:
        report += """1. Review changes: `git diff`
2. Commit: `git add -A && git commit -m "Apply code quality fixes"`
3. Push: `git push origin main`
"""
    else:
        report += """1. Check test failures above
2. Fix any issues
3. Re-run: `python scripts/fix_code_quality.py`
"""

    REPORT_FILE.write_text(report, encoding="utf-8")
    print(f"\nReport saved to: {REPORT_FILE}")


def main():
    """Run all fixes."""
    print("=" * 60)
    print("  AI Dataset Radar - Code Quality Fixes")
    print("=" * 60)
    print()

    # 1. Replace prints with logger in scrapers
    print("[1/4] Replacing print() with logger...")
    scraper_files = list((SRC_DIR / "scrapers").glob("*.py"))
    tracker_files = list((SRC_DIR / "trackers").glob("*.py"))

    total_replacements = 0
    for f in scraper_files + tracker_files:
        if f.name == "__init__.py":
            continue
        count = replace_prints_with_logger(f)
        if count > 0:
            log(f"  Modified {f.name}: {count} replacements")
            total_replacements += count

    if total_replacements == 0:
        log("  No print statements to replace")

    # 2. Add DB validation
    print("\n[2/4] Adding DB input validation...")
    add_db_validation()

    # 3. Create keyword utils
    print("\n[3/4] Creating keyword matching utilities...")
    create_keyword_utils()

    # 4. Add config validation
    print("\n[4/4] Adding config validation...")
    add_config_validation()

    # Run tests
    print("\n[5/5] Running tests...")
    test_success, test_output = run_tests()

    if test_success:
        print("Tests PASSED")
    else:
        print("Tests FAILED - check report for details")
        errors.append("Some tests failed")

    # Generate report
    generate_report(test_success, test_output)

    print()
    print("=" * 60)
    print("  Done!")
    print("=" * 60)
    print(f"\nSee full report: {REPORT_FILE}")

    return 0 if test_success else 1


if __name__ == "__main__":
    sys.exit(main())
