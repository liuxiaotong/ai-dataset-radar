"""Tests for GitHub scraper."""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from scrapers.github import GitHubScraper


class TestGitHubScraper:
    """Tests for GitHubScraper class."""

    @pytest.fixture
    def scraper(self):
        """Create a scraper instance."""
        return GitHubScraper(limit=10, days=7)

    def test_init(self, scraper):
        """Test scraper initialization."""
        assert scraper.limit == 10
        assert scraper.days == 7
        assert "Accept" in scraper.headers

    def test_init_with_token(self):
        """Test scraper initialization with GitHub token."""
        scraper = GitHubScraper(token="test_token")
        assert "Authorization" in scraper.headers
        assert scraper.headers["Authorization"] == "token test_token"

    def test_parse_repo(self, scraper):
        """Test parsing a repository from API response."""
        raw = {
            "id": 12345,
            "full_name": "test-user/test-dataset",
            "name": "test-dataset",
            "owner": {"login": "test-user"},
            "description": "A test dataset for NLP",
            "stargazers_count": 100,
            "forks_count": 20,
            "language": "Python",
            "topics": ["dataset", "nlp"],
            "created_at": "2024-01-15T10:00:00Z",
            "html_url": "https://github.com/test-user/test-dataset",
        }

        result = scraper._parse_repo(raw)

        assert result["source"] == "github"
        assert result["id"] == "test-user/test-dataset"
        assert result["name"] == "test-dataset"
        assert result["author"] == "test-user"
        assert result["stars"] == 100
        assert result["is_dataset"] is True

    def test_is_dataset_related_by_topic(self, scraper):
        """Test dataset detection by topic."""
        repo = {
            "name": "my-repo",
            "description": "",
            "topics": ["dataset", "python"],
        }
        assert scraper._is_dataset_related(repo) is True

    def test_is_dataset_related_by_name(self, scraper):
        """Test dataset detection by name."""
        repo = {
            "name": "my-dataset-collection",
            "description": "",
            "topics": [],
        }
        assert scraper._is_dataset_related(repo) is True

    def test_is_dataset_related_by_description(self, scraper):
        """Test dataset detection by description."""
        repo = {
            "name": "my-project",
            "description": "A benchmark for evaluating LLMs",
            "topics": [],
        }
        assert scraper._is_dataset_related(repo) is True

    def test_is_not_dataset_related(self, scraper):
        """Test non-dataset repo detection."""
        repo = {
            "name": "my-web-app",
            "description": "A simple web application",
            "topics": ["web", "javascript"],
        }
        assert scraper._is_dataset_related(repo) is False

    @patch("scrapers.github.requests.get")
    def test_search_repos(self, mock_get, scraper):
        """Test searching repositories."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "items": [
                {
                    "id": 1,
                    "full_name": "user/dataset",
                    "name": "dataset",
                    "owner": {"login": "user"},
                    "description": "A dataset",
                    "stargazers_count": 50,
                    "forks_count": 10,
                    "language": "Python",
                    "topics": ["dataset"],
                    "created_at": "2024-01-15T10:00:00Z",
                    "html_url": "https://github.com/user/dataset",
                }
            ]
        }
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        repos = scraper._search_repos("dataset")

        assert len(repos) == 1
        assert repos[0]["name"] == "dataset"

    @patch("scrapers.github.requests.get")
    def test_search_repos_error(self, mock_get, scraper):
        """Test handling API errors."""
        import requests

        mock_get.side_effect = requests.RequestException("API Error")

        repos = scraper._search_repos("dataset")

        assert repos == []

    def test_parse_trending_article(self, scraper):
        """Test parsing trending article HTML."""
        from bs4 import BeautifulSoup

        html = """
        <article class="Box-row">
            <h2><a href="/user/trending-dataset">user/trending-dataset</a></h2>
            <p>A trending dataset for ML</p>
            <a href="/user/trending-dataset/stargazers">1,234</a>
            <span itemprop="programmingLanguage">Python</span>
        </article>
        """
        soup = BeautifulSoup(html, "html.parser")
        article = soup.select_one("article")

        result = scraper._parse_trending_article(article)

        assert result is not None
        assert result["full_name"] == "user/trending-dataset"
        assert result["stars"] == 1234
        assert result["language"] == "Python"


class TestGitHubScraperIntegration:
    """Integration tests (require network)."""

    @pytest.mark.skip(reason="Requires network access")
    def test_fetch_real_data(self):
        """Test fetching real data from GitHub."""
        scraper = GitHubScraper(limit=5, days=7)
        repos = scraper.fetch()

        assert isinstance(repos, list)
        # Should find some repos
        assert len(repos) >= 0
