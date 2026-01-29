"""Notification handlers for AI Dataset Radar."""

import json
import os
import smtplib
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Optional

import requests


class ConsoleNotifier:
    """Output results to console with optional color formatting."""

    # ANSI color codes
    COLORS = {
        "reset": "\033[0m",
        "bold": "\033[1m",
        "blue": "\033[94m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "cyan": "\033[96m",
        "magenta": "\033[95m",
    }

    def __init__(self, use_color: bool = True):
        self.use_color = use_color

    def _color(self, text: str, color: str) -> str:
        """Apply color to text if color is enabled."""
        if self.use_color and color in self.COLORS:
            return f"{self.COLORS[color]}{text}{self.COLORS['reset']}"
        return text

    def notify(self, data: dict) -> None:
        """Print formatted results to console.

        Args:
            data: Dictionary containing datasets from each source.
        """
        print("\n" + "=" * 60)
        print(self._color("  AI Dataset Radar Report", "bold"))
        print(self._color(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", "cyan"))
        print("=" * 60)

        # Hugging Face datasets
        hf_data = data.get("huggingface", [])
        print(f"\n{self._color('Hugging Face Datasets', 'blue')} ({len(hf_data)} found)")
        print("-" * 40)
        for ds in hf_data[:10]:  # Show top 10
            print(f"  {self._color(ds.get('name', 'N/A'), 'green')}")
            print(f"    Author: {ds.get('author', 'N/A')}")
            print(f"    Downloads: {ds.get('downloads', 0):,}")
            print(f"    URL: {ds.get('url', 'N/A')}")
            print()

        # Papers with Code datasets
        pwc_data = data.get("paperswithcode", [])
        print(f"\n{self._color('Papers with Code Datasets', 'yellow')} ({len(pwc_data)} found)")
        print("-" * 40)
        for ds in pwc_data[:10]:
            print(f"  {self._color(ds.get('name', 'N/A'), 'green')}")
            desc = ds.get("description", "")
            if desc:
                desc = desc[:100] + "..." if len(desc) > 100 else desc
                print(f"    {desc}")
            print(f"    Papers: {ds.get('paper_count', 0)}")
            print(f"    URL: {ds.get('url', 'N/A')}")
            print()

        # arXiv papers
        arxiv_data = data.get("arxiv", [])
        print(f"\n{self._color('arXiv Papers', 'magenta')} ({len(arxiv_data)} found)")
        print("-" * 40)
        for paper in arxiv_data[:10]:
            print(f"  {self._color(paper.get('title', 'N/A'), 'green')}")
            authors = paper.get("authors", [])
            if authors:
                author_str = ", ".join(authors[:3])
                if len(authors) > 3:
                    author_str += f" et al. ({len(authors)} authors)"
                print(f"    Authors: {author_str}")
            print(f"    Categories: {', '.join(paper.get('categories', []))}")
            print(f"    URL: {paper.get('url', 'N/A')}")
            print()

        print("=" * 60)
        total = len(hf_data) + len(pwc_data) + len(arxiv_data)
        print(f"Total: {total} items found")
        print("=" * 60 + "\n")


class MarkdownNotifier:
    """Generate Markdown reports."""

    def __init__(self, output_dir: str = "data"):
        self.output_dir = output_dir

    def notify(self, data: dict) -> str:
        """Generate and save a Markdown report.

        Args:
            data: Dictionary containing datasets from each source.

        Returns:
            Path to the generated report file.
        """
        os.makedirs(self.output_dir, exist_ok=True)

        date_str = datetime.now().strftime("%Y-%m-%d")
        filename = f"report_{date_str}.md"
        filepath = os.path.join(self.output_dir, filename)

        content = self._generate_markdown(data)

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)

        print(f"Markdown report saved to: {filepath}")
        return filepath

    def _generate_markdown(self, data: dict) -> str:
        """Generate Markdown content from data."""
        lines = []
        lines.append("# AI Dataset Radar Report")
        lines.append(f"\n*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")

        # Summary
        hf_count = len(data.get("huggingface", []))
        pwc_count = len(data.get("paperswithcode", []))
        arxiv_count = len(data.get("arxiv", []))
        total = hf_count + pwc_count + arxiv_count

        lines.append("## Summary\n")
        lines.append(f"- **Total items found:** {total}")
        lines.append(f"- **Hugging Face datasets:** {hf_count}")
        lines.append(f"- **Papers with Code datasets:** {pwc_count}")
        lines.append(f"- **arXiv papers:** {arxiv_count}")
        lines.append("")

        # Hugging Face
        lines.append("## Hugging Face Datasets\n")
        hf_data = data.get("huggingface", [])
        if hf_data:
            lines.append("| Name | Author | Downloads | Tags |")
            lines.append("|------|--------|-----------|------|")
            for ds in hf_data:
                name = f"[{ds.get('name', 'N/A')}]({ds.get('url', '#')})"
                author = ds.get("author", "N/A")
                downloads = f"{ds.get('downloads', 0):,}"
                tags = ", ".join(ds.get("tags", [])[:3])
                lines.append(f"| {name} | {author} | {downloads} | {tags} |")
        else:
            lines.append("*No datasets found*")
        lines.append("")

        # Papers with Code
        lines.append("## Papers with Code Datasets\n")
        pwc_data = data.get("paperswithcode", [])
        if pwc_data:
            lines.append("| Name | Description | Papers |")
            lines.append("|------|-------------|--------|")
            for ds in pwc_data:
                name = f"[{ds.get('name', 'N/A')}]({ds.get('url', '#')})"
                desc = ds.get("description", "")
                desc = desc[:80] + "..." if len(desc) > 80 else desc
                desc = desc.replace("|", "\\|").replace("\n", " ")
                papers = ds.get("paper_count", 0)
                lines.append(f"| {name} | {desc} | {papers} |")
        else:
            lines.append("*No datasets found*")
        lines.append("")

        # arXiv
        lines.append("## arXiv Papers\n")
        arxiv_data = data.get("arxiv", [])
        if arxiv_data:
            for paper in arxiv_data:
                title = paper.get("title", "N/A")
                url = paper.get("url", "#")
                lines.append(f"### [{title}]({url})\n")
                authors = paper.get("authors", [])
                if authors:
                    author_str = ", ".join(authors[:5])
                    if len(authors) > 5:
                        author_str += f" et al."
                    lines.append(f"**Authors:** {author_str}\n")
                categories = paper.get("categories", [])
                if categories:
                    lines.append(f"**Categories:** {', '.join(categories)}\n")
                summary = paper.get("summary", "")
                if summary:
                    summary = summary[:300] + "..." if len(summary) > 300 else summary
                    lines.append(f"> {summary}\n")
                lines.append("")
        else:
            lines.append("*No papers found*")

        lines.append("---")
        lines.append("*Report generated by [AI Dataset Radar](https://github.com/your-username/ai-dataset-radar)*")

        return "\n".join(lines)


class EmailNotifier:
    """Send email notifications via SMTP."""

    def __init__(
        self,
        smtp_server: str,
        smtp_port: int,
        username: str,
        password: str,
        from_addr: str,
        to_addrs: list[str],
    ):
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.from_addr = from_addr
        self.to_addrs = to_addrs

    def notify(self, data: dict) -> bool:
        """Send email notification with report.

        Args:
            data: Dictionary containing datasets from each source.

        Returns:
            True if email was sent successfully.
        """
        # Generate markdown content
        md_notifier = MarkdownNotifier()
        content = md_notifier._generate_markdown(data)

        # Create email
        msg = MIMEMultipart("alternative")
        date_str = datetime.now().strftime("%Y-%m-%d")
        msg["Subject"] = f"AI Dataset Radar Report - {date_str}"
        msg["From"] = self.from_addr
        msg["To"] = ", ".join(self.to_addrs)

        # Attach as plain text (markdown)
        msg.attach(MIMEText(content, "plain", "utf-8"))

        try:
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.username, self.password)
                server.sendmail(self.from_addr, self.to_addrs, msg.as_string())
            print(f"Email sent to: {', '.join(self.to_addrs)}")
            return True
        except Exception as e:
            print(f"Error sending email: {e}")
            return False


class WebhookNotifier:
    """Send notifications via webhook (POST JSON)."""

    def __init__(self, url: str):
        self.url = url

    def notify(self, data: dict) -> bool:
        """Send webhook notification.

        Args:
            data: Dictionary containing datasets from each source.

        Returns:
            True if webhook was sent successfully.
        """
        payload = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "huggingface_count": len(data.get("huggingface", [])),
                "paperswithcode_count": len(data.get("paperswithcode", [])),
                "arxiv_count": len(data.get("arxiv", [])),
            },
            "data": data,
        }

        try:
            response = requests.post(
                self.url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=30,
            )
            response.raise_for_status()
            print(f"Webhook sent successfully to: {self.url}")
            return True
        except requests.RequestException as e:
            print(f"Error sending webhook: {e}")
            return False


def expand_env_vars(value: str) -> str:
    """Expand environment variables in a string.

    Args:
        value: String potentially containing ${VAR} patterns.

    Returns:
        String with environment variables expanded.
    """
    if not isinstance(value, str):
        return value

    import re

    pattern = r"\$\{([^}]+)\}"

    def replace(match):
        var_name = match.group(1)
        return os.environ.get(var_name, "")

    return re.sub(pattern, replace, value)


def create_notifiers(config: dict) -> list:
    """Create notifier instances based on configuration.

    Args:
        config: Notification configuration dictionary.

    Returns:
        List of enabled notifier instances.
    """
    notifiers = []

    # Console notifier
    console_cfg = config.get("console", {})
    if console_cfg.get("enabled", True):
        notifiers.append(ConsoleNotifier(use_color=console_cfg.get("color", True)))

    # Markdown notifier
    md_cfg = config.get("markdown", {})
    if md_cfg.get("enabled", True):
        notifiers.append(MarkdownNotifier(output_dir=md_cfg.get("output_dir", "data")))

    # Email notifier
    email_cfg = config.get("email", {})
    if email_cfg.get("enabled", False):
        notifiers.append(
            EmailNotifier(
                smtp_server=expand_env_vars(email_cfg.get("smtp_server", "")),
                smtp_port=email_cfg.get("smtp_port", 587),
                username=expand_env_vars(email_cfg.get("username", "")),
                password=expand_env_vars(email_cfg.get("password", "")),
                from_addr=expand_env_vars(email_cfg.get("from_addr", "")),
                to_addrs=[expand_env_vars(addr) for addr in email_cfg.get("to_addrs", [])],
            )
        )

    # Webhook notifier
    webhook_cfg = config.get("webhook", {})
    if webhook_cfg.get("enabled", False):
        notifiers.append(WebhookNotifier(url=expand_env_vars(webhook_cfg.get("url", ""))))

    return notifiers
