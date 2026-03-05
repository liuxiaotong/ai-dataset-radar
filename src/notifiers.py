"""Notification handlers for AI Dataset Radar."""

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

        # GitHub repos (early signal)
        github_data = data.get("github", [])
        dataset_repos = [r for r in github_data if r.get("is_dataset")]
        if github_data:
            print(
                f"\n{self._color('GitHub Repos', 'cyan')} ({len(dataset_repos)} dataset-related / {len(github_data)} total)"
            )
            print("-" * 40)
            for repo in dataset_repos[:10]:
                print(f"  {self._color(repo.get('full_name', 'N/A'), 'green')}")
                desc = repo.get("description", "")
                if desc:
                    desc = desc[:80] + "..." if len(desc) > 80 else desc
                    print(f"    {desc}")
                print(
                    f"    Stars: {repo.get('stars', 0):,} | Language: {repo.get('language', 'N/A')}"
                )
                print(f"    URL: {repo.get('url', 'N/A')}")
                print()

        # HuggingFace Papers (early signal)
        hf_papers_data = data.get("hf_papers", [])
        dataset_papers = [p for p in hf_papers_data if p.get("is_dataset_paper")]
        if hf_papers_data:
            print(
                f"\n{self._color('HF Daily Papers', 'blue')} ({len(dataset_papers)} dataset-related / {len(hf_papers_data)} total)"
            )
            print("-" * 40)
            for paper in dataset_papers[:10]:
                print(f"  {self._color(paper.get('title', 'N/A'), 'green')}")
                print(
                    f"    Upvotes: {paper.get('upvotes', 0)} | arXiv: {paper.get('arxiv_id', 'N/A')}"
                )
                print(f"    URL: {paper.get('url', 'N/A')}")
                print()

        print("=" * 60)
        total = (
            len(hf_data) + len(arxiv_data) + len(github_data) + len(hf_papers_data)
        )
        print(f"Total: {total} items found")
        print("=" * 60 + "\n")


class MarkdownNotifier:
    """Generate Markdown reports (legacy format)."""

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
        arxiv_count = len(data.get("arxiv", []))
        github_data = data.get("github", [])
        github_count = len(github_data)
        github_dataset_count = len([r for r in github_data if r.get("is_dataset")])
        hf_papers_data = data.get("hf_papers", [])
        hf_papers_count = len(hf_papers_data)
        hf_papers_dataset_count = len([p for p in hf_papers_data if p.get("is_dataset_paper")])
        total = hf_count + arxiv_count + github_count + hf_papers_count

        lines.append("## Summary\n")
        lines.append(f"- **Total items found:** {total}")
        lines.append(f"- **Hugging Face datasets:** {hf_count}")
        lines.append(f"- **arXiv papers:** {arxiv_count}")
        lines.append(f"- **GitHub repos:** {github_count} ({github_dataset_count} dataset-related)")
        lines.append(
            f"- **HF Daily Papers:** {hf_papers_count} ({hf_papers_dataset_count} dataset-related)"
        )
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
            lines.append("No datasets found")
        lines.append("")

        # Papers with Code
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
                        author_str += " et al."
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
            lines.append("No papers found")
        lines.append("")

        # GitHub repos (early signal)
        lines.append("## GitHub Repos (Early Signal)\n")
        github_data = data.get("github", [])
        dataset_repos = [r for r in github_data if r.get("is_dataset")]
        if dataset_repos:
            lines.append("| Repository | Description | Stars | Language |")
            lines.append("|------------|-------------|-------|----------|")
            for repo in dataset_repos:
                name = f"[{repo.get('full_name', 'N/A')}]({repo.get('url', '#')})"
                desc = repo.get("description", "")
                desc = desc[:60] + "..." if len(desc) > 60 else desc
                desc = desc.replace("|", "\\|").replace("\n", " ")
                stars = f"{repo.get('stars', 0):,}"
                lang = repo.get("language", "N/A")
                lines.append(f"| {name} | {desc} | {stars} | {lang} |")
        else:
            lines.append("No dataset-related repos found")
        lines.append("")

        # HuggingFace Papers (early signal)
        lines.append("## HF Daily Papers (Early Signal)\n")
        hf_papers_data = data.get("hf_papers", [])
        dataset_papers = [p for p in hf_papers_data if p.get("is_dataset_paper")]
        if dataset_papers:
            lines.append("| Title | Upvotes | arXiv |")
            lines.append("|-------|---------|-------|")
            for paper in dataset_papers:
                title = paper.get("title", "N/A")
                title = title[:60] + "..." if len(title) > 60 else title
                title = title.replace("|", "\\|")
                title_link = f"[{title}]({paper.get('url', '#')})"
                upvotes = paper.get("upvotes", 0)
                arxiv = paper.get("arxiv_id", "N/A")
                lines.append(f"| {title_link} | {upvotes} | {arxiv} |")
        else:
            lines.append("No dataset-related papers found")

        lines.append("")
        lines.append("---")
        lines.append(
            "> Report generated by [AI Dataset Radar](https://github.com/your-username/ai-dataset-radar)"
        )

        return "\n".join(lines)


class BusinessIntelNotifier:
    """Generate business intelligence markdown reports."""

    def __init__(self, output_dir: str = "data", config: Optional[dict] = None):
        self.output_dir = output_dir
        self.config = config or {}

    def notify(
        self,
        data: dict,
        trend_results: Optional[dict] = None,
        opportunity_results: Optional[dict] = None,
        domain_data: Optional[dict] = None,
    ) -> str:
        """Generate and save a business intelligence report.

        Args:
            data: Dictionary containing datasets from each source.
            trend_results: Results from TrendAnalyzer.analyze().
            opportunity_results: Results from OpportunityAnalyzer.analyze().
            domain_data: Results from DomainFilter.classify_all().

        Returns:
            Path to the generated report file.
        """
        os.makedirs(self.output_dir, exist_ok=True)

        date_str = datetime.now().strftime("%Y-%m-%d")
        filename = f"intel_report_{date_str}.md"
        filepath = os.path.join(self.output_dir, filename)

        content = self._generate_report(data, trend_results, opportunity_results, domain_data)

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)

        print(f"Business intelligence report saved to: {filepath}")
        return filepath

    def _generate_report(
        self,
        data: dict,
        trend_results: Optional[dict],
        opportunity_results: Optional[dict],
        domain_data: Optional[dict],
    ) -> str:
        """Generate the business intelligence report content."""
        lines = []
        date_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        lines.append("# AI Dataset Radar 商业情报周报")
        lines.append(f"> Generated: {date_str}\n")

        # Section 1: Top Growing Datasets
        lines.append("## 🔥 增长最快的数据集 (Top 10)\n")
        if trend_results and trend_results.get("top_growing_7d"):
            lines.append("| 排名 | 数据集 | 7天增长率 | 当前下载 | 领域标签 |")
            lines.append("|------|--------|-----------|----------|----------|")
            for i, ds in enumerate(trend_results["top_growing_7d"][:10], 1):
                name = ds.get("name", ds.get("dataset_id", "Unknown"))
                name_link = f"[{name}]({ds.get('url', '#')})"
                growth = ds.get("growth", 0)
                growth_str = f"{growth * 100:.1f}%" if growth != float("inf") else "New"
                downloads = ds.get("current_downloads", 0) or 0
                domains = ds.get("domains", [])
                domain_str = ", ".join(domains[:2]) if domains else "-"
                lines.append(f"| {i} | {name_link} | {growth_str} | {downloads:,} | {domain_str} |")
        else:
            lines.append("需要多天数据才能计算增长趋势")
        lines.append("")

        # Breakthrough datasets
        if trend_results and trend_results.get("breakthroughs"):
            lines.append("### 🚀 破圈数据集 (0 → 1000+ 下载)\n")
            lines.append("| 数据集 | 起始下载 | 当前下载 | 增量 |")
            lines.append("|--------|----------|----------|------|")
            for ds in trend_results["breakthroughs"][:5]:
                name = ds.get("name", ds.get("dataset_id", "Unknown"))
                name_link = f"[{name}]({ds.get('url', '#')})"
                old = ds.get("old_downloads", 0) or 0
                current = ds.get("current_downloads", 0)
                increase = ds.get("download_increase", current - old)
                lines.append(f"| {name_link} | {old:,} | {current:,} | +{increase:,} |")
            lines.append("")

        # Section 2: Data Factories
        lines.append("## 🏭 数据工厂动态\n")
        if opportunity_results and opportunity_results.get("data_factories"):
            data_factories = opportunity_results["data_factories"]
            if isinstance(data_factories, dict):
                all_factories = data_factories.get(
                    "org_factories", []
                ) + data_factories.get("individual_factories", [])
            else:
                all_factories = data_factories
            lines.append("| 作者/机构 | 本周发布数量 | 数据集列表 | 可能归属 |")
            lines.append("|-----------|--------------|------------|----------|")
            for factory in all_factories[:10]:
                author = factory["author"]
                count = factory["dataset_count"]
                datasets = [
                    ds.get("name", ds.get("id", "?"))[:20] for ds in factory["datasets"][:3]
                ]
                datasets_str = ", ".join(datasets)
                if len(factory["datasets"]) > 3:
                    datasets_str += f" +{len(factory['datasets']) - 3}..."
                org = factory.get("possible_org", "-") or "-"
                lines.append(f"| {author} | {count} | {datasets_str} | {org} |")
        else:
            lines.append("本周未检测到数据工厂活动")
        lines.append("")

        # Section 3: Domain Focus - Robotics
        lines.append("## 🤖 具身智能专区\n")
        robotics_data = domain_data.get("robotics", []) if domain_data else []

        # Filter to robotics datasets
        robotics_datasets = [ds for ds in robotics_data if ds.get("source") == "huggingface"]
        if robotics_datasets:
            lines.append("### 新增机器人数据集\n")
            lines.append("| 数据集 | 任务类型 | 数据规模 | 增长趋势 |")
            lines.append("|--------|----------|----------|----------|")
            for ds in robotics_datasets[:10]:
                name = ds.get("name", "N/A")
                name_link = f"[{name}]({ds.get('url', '#')})"
                tags = ds.get("tags", [])[:2]
                task = ", ".join(tags) if tags else "-"
                downloads = ds.get("downloads", 0)
                size = f"{downloads:,} downloads"
                growth = ds.get("growth", None)
                growth_str = f"{growth * 100:.1f}%" if growth else "-"
                lines.append(f"| {name_link} | {task} | {size} | {growth_str} |")
        else:
            lines.append("### 新增机器人数据集\n")
            lines.append("本周无新增")
        lines.append("")

        # Section 4: Papers with Annotation Signals
        lines.append("## 📄 有标注需求的论文\n")
        if opportunity_results and opportunity_results.get("annotation_opportunities"):
            lines.append("| 论文 | 检测到的信号 | 机构 | arXiv链接 |")
            lines.append("|------|--------------|------|-----------|")
            for opp in opportunity_results["annotation_opportunities"][:15]:
                title = opp["title"][:50] + "..." if len(opp["title"]) > 50 else opp["title"]
                title = title.replace("|", "\\|")
                signals = ", ".join(opp["signals"][:3])
                if len(opp["signals"]) > 3:
                    signals += "..."
                org = opp.get("detected_org", "-") or "-"
                arxiv_id = opp.get("arxiv_id", "")
                arxiv_link = f"[{arxiv_id}](https://arxiv.org/abs/{arxiv_id})" if arxiv_id else "-"
                lines.append(f"| {title} | {signals} | {org} | {arxiv_link} |")
        else:
            lines.append("本周未检测到有标注需求的论文")
        lines.append("")

        # Section 5: Organization Activity
        lines.append("## 🏢 大厂动态\n")
        if opportunity_results and opportunity_results.get("org_activity"):
            org_activity = opportunity_results["org_activity"]
            # Sort by activity
            sorted_orgs = sorted(
                [
                    (org, org_data)
                    for org, org_data in org_activity.items()
                    if org_data["total_items"] > 0
                ],
                key=lambda x: x[1]["total_items"],
                reverse=True,
            )

            for org, org_data in sorted_orgs[:6]:
                lines.append(f"### {org.upper()}\n")
                if org_data["datasets"]:
                    lines.append("**相关数据集:**")
                    for ds in org_data["datasets"][:3]:
                        name = ds.get("name", ds.get("id", "Unknown"))
                        url = ds.get("url", "#")
                        lines.append(f"- [{name}]({url})")
                if org_data["papers"]:
                    lines.append("\n**相关论文:**")
                    for p in org_data["papers"][:3]:
                        title = p.get("title", "Unknown")[:60]
                        url = p.get("url", p.get("arxiv_url", "#"))
                        lines.append(f"- [{title}]({url})")
                lines.append("")

            if not sorted_orgs:
                lines.append("本周未检测到大厂相关活动")
        else:
            lines.append("本周未检测到大厂相关活动")
        lines.append("")

        # Section 6: Statistics Summary
        lines.append("## 📊 统计摘要\n")

        hf_count = len(data.get("huggingface", []))
        github_data = data.get("github", [])
        github_dataset_count = len([r for r in github_data if r.get("is_dataset")])
        hf_papers = data.get("hf_papers", [])
        hf_papers_dataset_count = len([p for p in hf_papers if p.get("is_dataset_paper")])

        # Calculate domain distribution (only count HF datasets for percentage)
        robotics_hf_count = 0
        if domain_data:
            for item in domain_data.get("robotics", []):
                if item.get("source") == "huggingface":
                    robotics_hf_count += 1

        lines.append(f"- **本周新增数据集:** {hf_count} 个")
        lines.append(f"- **GitHub 数据集仓库:** {github_dataset_count} 个")
        lines.append(f"- **数据集相关论文:** {hf_papers_dataset_count} 篇")

        if robotics_hf_count > 0 and hf_count > 0:
            robotics_pct = robotics_hf_count / hf_count * 100
            lines.append(f"- **机器人领域占比:** {robotics_pct:.1f}%")

        if opportunity_results:
            opp_count = opportunity_results.get("summary", {}).get(
                "annotation_opportunity_count", 0
            )
            factory_count = opportunity_results.get("summary", {}).get("data_factory_count", 0)
            lines.append(f"- **检测到潜在商机:** {opp_count} 个")
            lines.append(f"- **数据工厂:** {factory_count} 个")

        lines.append("")
        lines.append("---")
        lines.append("> Report generated by AI Dataset Radar v2 — Business Intelligence System")

        return "\n".join(lines)


class EmailNotifier:
    """Send email notifications via SMTP.

    Security note: Password should be passed via environment variable reference
    (e.g., ${SMTP_PASSWORD}) and will be resolved at send time, not stored.
    """

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
        # Store password reference, not the actual password
        # If it's an env var reference like ${SMTP_PASSWORD}, it will be resolved at send time
        self._password_ref = password
        self.from_addr = from_addr
        self.to_addrs = to_addrs

    @property
    def password(self) -> str:
        """Get password, expanding environment variables if needed."""
        return expand_env_vars(self._password_ref)

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


def expand_env_vars(value: str, warn_missing: bool = True) -> str:
    """Expand environment variables in a string.

    Args:
        value: String potentially containing ${VAR} patterns.
        warn_missing: If True, print warning for missing env vars.

    Returns:
        String with environment variables expanded.
    """
    if not isinstance(value, str):
        return value

    import re

    pattern = r"\$\{([^}]+)\}"
    missing_vars = []

    def replace(match):
        var_name = match.group(1)
        env_value = os.environ.get(var_name)
        if env_value is None:
            missing_vars.append(var_name)
            return ""
        return env_value

    result = re.sub(pattern, replace, value)

    if warn_missing and missing_vars:
        print(f"Warning: Missing environment variables: {', '.join(missing_vars)}")

    return result


def create_notifiers(config: dict, full_config: Optional[dict] = None) -> list:
    """Create notifier instances based on configuration.

    Args:
        config: Notification configuration dictionary.
        full_config: Full application configuration (for BusinessIntelNotifier).

    Returns:
        List of enabled notifier instances.
    """
    notifiers = []

    # Console notifier
    console_cfg = config.get("console", {})
    if console_cfg.get("enabled", True):
        notifiers.append(ConsoleNotifier(use_color=console_cfg.get("color", True)))

    # Markdown notifier (legacy format)
    md_cfg = config.get("markdown", {})
    if md_cfg.get("enabled", True):
        notifiers.append(MarkdownNotifier(output_dir=md_cfg.get("output_dir", "data")))

    # Business intelligence notifier (new format)
    intel_cfg = config.get("business_intel", {})
    if intel_cfg.get("enabled", True):
        notifiers.append(
            BusinessIntelNotifier(
                output_dir=intel_cfg.get("output_dir", "data"),
                config=full_config,
            )
        )

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
