#!/usr/bin/env python3
"""Debug script for vendor monitoring."""

import requests


# 测试 GitHub API
def test_github():
    orgs = ["scaleapi", "snorkel-ai", "argilla-io", "Labelbox", "humanloop", "openai", "meta-llama"]
    for org in orgs:
        url = f"https://api.github.com/orgs/{org}/repos?sort=updated&per_page=5"
        resp = requests.get(url)
        if resp.ok:
            repos = resp.json()
            print(f"{org}: {resp.status_code}, repos: {len(repos)}")
            if repos:
                print(f"  Latest: {repos[0]['name']} (updated: {repos[0]['updated_at'][:10]})")
        else:
            print(f"{org}: {resp.status_code}, error: {resp.json().get('message', 'unknown')}")

        # Check rate limit
        remaining = resp.headers.get("X-RateLimit-Remaining", "?")
        print(f"  Rate limit remaining: {remaining}")


# 测试博客 RSS
def test_blogs():
    feeds = [
        ("Argilla RSS", "https://argilla.io/blog/feed.xml"),
        ("Argilla Blog", "https://argilla.io/blog/"),
        ("Scale Blog", "https://scale.com/blog"),
        ("Snorkel Blog", "https://snorkel.ai/blog/"),
        ("Anthropic Research", "https://www.anthropic.com/research"),
    ]
    for name, url in feeds:
        try:
            resp = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
            content_type = resp.headers.get("Content-Type", "")[:50]
            print(f"{name}: {resp.status_code}, type: {content_type}, length: {len(resp.text)}")

            # Check if it's RSS/XML
            if "xml" in content_type or resp.text.strip().startswith("<?xml"):
                print("  -> Valid RSS/XML feed")
            elif "<html" in resp.text.lower():
                print("  -> HTML page (need scraping)")
        except Exception as e:
            print(f"{name}: error - {e}")


if __name__ == "__main__":
    print("=== GitHub API Test ===")
    test_github()
    print("\n=== Blog/RSS Test ===")
    test_blogs()
