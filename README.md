<p align="center">
  <h1 align="center">🛰️ AI Dataset Radar</h1>
  <p align="center">
    <strong>Track AI training datasets across HuggingFace, GitHub, arXiv & blogs</strong>
  </p>
  <p align="center">
    <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
    <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python 3.10+"></a>
    <a href="#mcp-server"><img src="https://img.shields.io/badge/MCP-Server-purple.svg" alt="MCP Server"></a>
  </p>
</p>

---

Monitor 30+ AI labs and data vendors. Get structured reports on new datasets, GitHub repos, papers, and blog posts — delivered as Markdown for humans or JSON for LLMs.

## ✨ What You Get

```
┌────────────────────────────────────────────────────────────────┐
│  12 datasets │ 138 repos │ 28 papers │ 4 blog posts            │
│  ─────────────────────────────────────────────────────────────  │
│  • OpenAI, Anthropic, Google, Meta, DeepSeek, Qwen...          │
│  • Scale AI, Argilla, Snorkel, Labelbox...                     │
│  • RLHF, SFT, Synthetic, Agent, Evaluation datasets            │
└────────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Option 1: Command Line

```bash
git clone https://github.com/liuxiaotong/ai-dataset-radar.git
cd ai-dataset-radar
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run scan
python src/main_intel.py --days 7
```

Reports saved to `data/reports/`:
- `intel_report_2024-01-15.md` — Human-readable
- `intel_report_2024-01-15.json` — For LLMs/scripts

### Option 2: Claude Desktop (MCP)

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

Then ask Claude: *"Scan for new AI datasets"* or *"What's new from OpenAI?"*

### Option 3: Claude Code

```bash
/radar    # Get project context
/scan     # Run intelligence scan
```

---

## 📊 Output Example

### JSON (for LLMs)

```json
{
  "summary": {
    "total_datasets": 12,
    "total_github_repos": 138,
    "total_github_repos_high_relevance": 2,
    "total_papers": 28
  },
  "datasets": [
    {
      "id": "google/WaxalNLP",
      "category": "multilingual",
      "downloads": 1539,
      "license": "cc-by-4.0",
      "signals": ["multilingual", "audio"]
    }
  ],
  "github_activity": [
    {
      "org": "argilla-io",
      "repos_updated": [
        {"name": "argilla", "relevance": "high", "relevance_signals": ["annotation", "rlhf"]}
      ]
    }
  ]
}
```

### Markdown (for humans)

```markdown
## AI Labs Activity

### google_deepmind
- **WaxalNLP** (1.5K downloads) - ASR/TTS for African languages

## GitHub Activity
### argilla-io
- **argilla** ⭐ 8.2K [HIGH] - Data curation for LLMs
```

---

## ⚙️ Configuration

Edit `config.yaml` to customize:

```yaml
# Organizations to monitor
watched_orgs:
  frontier_labs:
    openai: { hf_ids: ["openai"], keywords: ["gpt"] }
    anthropic: { hf_ids: ["anthropic"], keywords: ["claude"] }
  china_opensource:
    qwen: { hf_ids: ["Qwen"], keywords: ["qwen"] }
    deepseek: { hf_ids: ["deepseek-ai"], keywords: ["deepseek"] }

# Data types to track
priority_data_types:
  preference: { keywords: ["rlhf", "dpo", "preference"] }
  sft: { keywords: ["instruction", "chat", "alpaca"] }
  agent: { keywords: ["tool use", "function calling"] }

# GitHub relevance keywords
sources:
  github:
    relevance_keywords: [dataset, annotation, benchmark, rlhf]
```

**Optional:** Set `GITHUB_TOKEN` for higher API rate limits.

---

## 🏗️ Architecture

```
ai-dataset-radar/
├── src/
│   ├── main_intel.py      # Entry point
│   ├── scrapers/          # HuggingFace, GitHub, arXiv, RSS
│   ├── trackers/          # Org & blog monitors
│   ├── analyzers/         # Dataset classification
│   └── output_formatter.py
├── mcp_server/            # Claude Desktop integration
│   └── server.py
├── .claude/commands/      # Claude Code skills
│   ├── radar.md
│   └── scan.md
├── config.yaml            # Watchlist configuration
└── data/reports/          # Generated reports
```

---

## 🔌 MCP Server Tools

When using Claude Desktop:

| Tool | Description |
|------|-------------|
| `radar_scan` | Run full intelligence scan |
| `radar_summary` | Get latest report summary |
| `radar_datasets` | List datasets (filter by category) |
| `radar_github` | View GitHub activity (filter by relevance) |
| `radar_papers` | View recent papers |
| `radar_config` | Show current watchlist |

---

## 📦 Dataset Categories

| Category | Examples |
|----------|----------|
| **SFT** | Alpaca, ShareGPT, OpenOrca |
| **Preference** | UltraFeedback, HelpSteer, HH-RLHF |
| **Synthetic** | Sera, Magpie |
| **Agent** | SWE-bench, WebArena, ToolBench |
| **Evaluation** | MMLU, HumanEval, GPQA |
| **Multimodal** | Action100M, VoxPopuli |
| **Code** | StarCoder, CodeParrot |

---

## 🧪 Development

```bash
# Run tests
python -m pytest tests/ -v

# Add a new scraper
# 1. Create src/scrapers/my_source.py
# 2. Inherit from BaseScraper
# 3. Register with @register_scraper("my_source")
```

<details>
<summary>Example: Custom Scraper</summary>

```python
from src.scrapers.base import BaseScraper
from src.scrapers.registry import register_scraper

@register_scraper("my_source")
class MySourceScraper(BaseScraper):
    name = "my_source"
    source_type = "dataset_registry"

    def scrape(self, config=None) -> list[dict]:
        return [{"source": "my_source", "id": "dataset-1"}]
```

</details>

---

## 🗺️ Roadmap

- [x] Multi-source aggregation (HF, GitHub, arXiv, blogs)
- [x] Dual output (Markdown + JSON)
- [x] MCP Server for Claude Desktop
- [x] Claude Code skills
- [ ] Scheduled execution & alerts
- [ ] Web dashboard
- [ ] LLM-powered summarization

---

## 🤝 Contributing

PRs welcome! Areas where help is needed:

- New data sources (e.g., Twitter/X, Discord)
- Improved classification heuristics
- Web UI
- Documentation translations

---

## 📄 License

MIT — see [LICENSE](LICENSE)

---

<p align="center">
  <sub>Built for the AI data community. Star ⭐ if useful!</sub>
</p>
