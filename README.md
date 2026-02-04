<p align="center">
  <h1 align="center">AI Dataset Radar</h1>
  <p align="center">
    <strong>Competitive intelligence for AI training data</strong><br>
    <strong>AI 训练数据竞争情报系统</strong>
  </p>
  <p align="center">
    <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
    <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python 3.10+"></a>
    <a href="#mcp-server"><img src="https://img.shields.io/badge/MCP-Server-purple.svg" alt="MCP Server"></a>
  </p>
</p>

---

Track datasets, papers, and announcements from 30+ AI labs and data vendors across HuggingFace, GitHub, arXiv, and company blogs.

追踪 30+ AI 实验室和数据供应商在 HuggingFace、GitHub、arXiv 和公司博客上发布的数据集、论文和公告。

## What You Get / 输出概览

| Source | What's Tracked |
|--------|----------------|
| **HuggingFace** | New datasets from 30+ AI labs (OpenAI, Google, Meta, Qwen, DeepSeek...) |
| **Blogs** | 17 sources: OpenAI, Anthropic, Mistral, Scale AI, Stanford HAI, Tencent... |
| **GitHub** | Repo activity from data vendors & AI labs (argilla-io, scaleapi, openai...) |
| **Papers** | arXiv + HuggingFace Papers filtered by RLHF, SFT, dataset keywords |

**Output formats / 输出格式:**
- `intel_report_YYYY-MM-DD.md` — Markdown for humans
- `intel_report_YYYY-MM-DD.json` — Structured JSON for LLMs

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
  <sub>Built for the AI data community</sub><br>
  <sub>为 AI 数据社区而建</sub>
</p>
