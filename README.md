<p align="center">
  <h1 align="center">AI Dataset Radar</h1>
  <p align="center">
    <strong>Competitive intelligence for AI training data</strong><br>
    <strong>AI 训练数据竞争情报系统</strong>
  </p>
  <p align="center">
    <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
    <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python 3.10+"></a>
    <a href="#mcp-server"><img src="https://img.shields.io/badge/MCP-7_Tools-purple.svg" alt="MCP Server"></a>
  </p>
</p>

---

**What is this?** A monitoring system that tracks AI training data activity from 30+ organizations. It aggregates datasets from HuggingFace, repos from GitHub, papers from arXiv, and blog posts from 17 company blogs — then outputs both human-readable Markdown and structured JSON for LLM consumption.

**这是什么？** 一个监控系统，追踪 30+ 组织的 AI 训练数据动态。它聚合 HuggingFace 数据集、GitHub 仓库、arXiv 论文和 17 个公司博客的文章，输出人类可读的 Markdown 和供 LLM 使用的结构化 JSON。

**Use cases / 使用场景:**
- Stay updated on new datasets from frontier labs (数据集追踪)
- Monitor competitor announcements (竞品动态)
- Feed structured data to Claude/GPT for analysis (喂给大模型分析)
- Research what data others are using for training (训练数据研究)

## What You Get / 输出概览

| Source | Coverage | What's Tracked |
|--------|----------|----------------|
| **HuggingFace** | 30+ orgs | Datasets from OpenAI, Google, Meta, Qwen, DeepSeek, Mistral... |
| **Blogs** | 17 sources | OpenAI, Anthropic, DeepMind, Mistral, Scale AI, Stanford HAI, Tencent... |
| **GitHub** | 15+ orgs | Repos from argilla-io, scaleapi, openai, anthropics, deepseek-ai... |
| **Papers** | 2 sources | arXiv (cs.CL/AI/LG) + HuggingFace Daily Papers |

**Output formats / 输出格式:**
- `intel_report_YYYY-MM-DD.md` — Human-readable report
- `intel_report_YYYY-MM-DD.json` — Structured data for LLM agents

---

## Quick Start / 快速开始

### Command Line / 命令行

```bash
git clone https://github.com/liuxiaotong/ai-dataset-radar.git
cd ai-dataset-radar
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Optional: Install Playwright for JavaScript-rendered blogs
# 可选：安装 Playwright 以抓取 JS 渲染的博客
playwright install chromium

# Run scan / 运行扫描
python src/main_intel.py --days 7
```

Reports saved to `data/reports/`:
- `intel_report_YYYY-MM-DD.md` — Human-readable / 人类可读
- `intel_report_YYYY-MM-DD.json` — For LLMs/scripts / 供 LLM 使用

### Claude Desktop (MCP Server)

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "ai-dataset-radar": {
      "command": "/path/to/ai-dataset-radar/.venv/bin/python",
      "args": ["/path/to/ai-dataset-radar/mcp_server/server.py"]
    }
  }
}
```

Then ask Claude: *"Scan for new AI datasets"* / 然后问 Claude：*"扫描新的 AI 数据集"*

### Claude Code

```bash
/radar    # Project context / 项目上下文
/scan     # Run scan / 运行扫描
```

---

## Data Sources / 数据源

### HuggingFace Datasets

| Category | Organizations |
|----------|---------------|
| **Frontier Labs** | OpenAI, Google/DeepMind, Meta, Anthropic |
| **Emerging Labs** | Mistral, Cohere, AI21, Together |
| **Research Labs** | EleutherAI, Allen AI, HuggingFace, NVIDIA |
| **China Labs** | Qwen, DeepSeek, Baichuan, Yi, InternLM, Zhipu |

### Blog Monitoring (17 active sources)

| Category | Blogs |
|----------|-------|
| **US Frontier** | OpenAI, Google AI, DeepMind, Meta AI Research |
| **US Emerging** | Mistral AI, Scale AI, Together AI, AI21 Labs |
| **Research** | Stanford HAI, Berkeley BAIR, Anthropic Research |
| **China** | Qwen, Tencent Hunyuan, Zhipu AI, 01.AI, Baidu AI |
| **Data Vendors** | Argilla, Scale AI |

### GitHub Organizations

Monitors repos from: `openai`, `anthropics`, `deepseek-ai`, `argilla-io`, `scaleapi`, `EleutherAI`, and more.

### Papers

- arXiv (cs.CL, cs.AI, cs.LG) filtered by keywords
- HuggingFace Daily Papers

---

## Output Format / 输出格式

### JSON (for LLMs)

```json
{
  "summary": {
    "total_datasets": 15,
    "total_github_repos": 137,
    "total_papers": 21,
    "total_blog_posts": 56
  },
  "datasets": [
    {
      "id": "allenai/Dolci-Instruct-SFT",
      "category": "sft_instruction",
      "downloads": 2610,
      "signals": ["sft", "multilingual"]
    }
  ],
  "blog_posts": [
    {
      "source": "OpenAI Blog",
      "articles": [
        {"title": "Introducing the Codex app", "url": "https://openai.com/index/..."}
      ]
    }
  ]
}
```

### Markdown (for humans)

```markdown
## US AI Labs

### OpenAI Blog
- [Introducing the Codex app](https://openai.com/index/introducing-the-codex-app)
- [Inside OpenAI's in-house data agent](https://openai.com/index/inside-our-in-house-data-agent)

## High-Value Datasets

### Synthetic
| Dataset | Publisher | Downloads |
|---------|-----------|-----------|
| Sera-4.5A-Lite-T1 | allenai | 226 |
```

---

## Configuration / 配置

Edit `config.yaml`:

```yaml
# HuggingFace organizations to track
watched_orgs:
  frontier_labs:
    openai: { hf_ids: ["openai"], keywords: ["gpt"] }
    google_deepmind: { hf_ids: ["google", "deepmind"] }
  china_opensource:
    qwen: { hf_ids: ["Qwen"], keywords: ["qwen"] }
    deepseek: { hf_ids: ["deepseek-ai"] }

# Blog sources (supports RSS, scraping, and Playwright)
watched_vendors:
  blogs:
    - name: "OpenAI Blog"
      url: "https://openai.com/blog"
      type: "auto"
    - name: "Tencent Hunyuan Research"
      url: "https://hy.tencent.com/research"
      type: "browser"  # Uses Playwright for JS-rendered pages
      selector: ".blog-item"

# Data types to classify
priority_data_types:
  preference: { keywords: ["rlhf", "dpo", "preference"] }
  sft: { keywords: ["instruction", "chat", "sft"] }
  synthetic: { keywords: ["synthetic", "generated"] }
```

Set `GITHUB_TOKEN` environment variable for higher API rate limits.

---

## Architecture / 架构

```
ai-dataset-radar/
├── src/
│   ├── main_intel.py        # Entry point
│   ├── scrapers/            # HuggingFace, GitHub, arXiv
│   ├── trackers/            # Blog tracker (RSS + Playwright)
│   ├── analyzers/           # Dataset classification
│   └── output_formatter.py  # Markdown + JSON output
├── mcp_server/server.py     # Claude Desktop MCP server
├── .claude/commands/        # Claude Code skills
├── config.yaml              # Monitoring configuration
└── data/reports/            # Generated reports
```

---

## MCP Server Tools

| Tool | Description |
|------|-------------|
| `radar_scan` | Run full intelligence scan |
| `radar_summary` | Get latest report summary |
| `radar_datasets` | List datasets by category |
| `radar_github` | View GitHub activity |
| `radar_papers` | View recent papers |
| `radar_blogs` | View blog articles from 17 sources |
| `radar_config` | Show monitoring configuration |

---

## Dataset Categories / 数据集分类

| Category | Examples | Description |
|----------|----------|-------------|
| **SFT** | Alpaca, ShareGPT | Instruction-following |
| **Preference** | UltraFeedback, HelpSteer | RLHF/DPO training |
| **Synthetic** | Sera, Magpie | AI-generated data |
| **Agent** | SWE-bench, WebArena | Tool use & agents |
| **Multimodal** | Action100M, VoxPopuli | Image/Audio/Video |
| **Multilingual** | WaxalNLP | Multiple languages |
| **Code** | StarCoder | Programming data |

---

## Roadmap / 路线图

- [x] Multi-source aggregation (HF, GitHub, arXiv, blogs)
- [x] Dual output format (Markdown + JSON)
- [x] MCP Server for Claude Desktop
- [x] Playwright support for JS-rendered blogs
- [x] 17 active blog sources across US/China/Research
- [ ] Scheduled execution & alerts
- [ ] Web dashboard
- [ ] LLM-powered summarization

---

## Contributing / 贡献

PRs welcome! Areas where help is needed:

- New blog sources (especially China closed-source labs)
- Improved scraping selectors for complex SPAs
- Web UI dashboard
- More language support

---

## License

MIT — see [LICENSE](LICENSE)

---

<p align="center">
  <sub>Built for AI researchers, data teams, and anyone tracking the training data landscape</sub><br>
  <sub>为 AI 研究者、数据团队和所有关注训练数据动态的人而建</sub>
</p>
