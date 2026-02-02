<p align="center">
  <h1 align="center">AI Dataset Radar</h1>
  <p align="center">
    <strong>Discover high-value AI datasets before they go mainstream</strong>
  </p>
  <p align="center">
    <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
    <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.8+-blue.svg" alt="Python 3.8+"></a>
    <a href="https://github.com/liuxiaotong/ai-dataset-radar"><img src="https://img.shields.io/badge/version-5.0-green.svg" alt="Version"></a>
  </p>
</p>

---

AI Dataset Radar is a neutral data collection layer that aggregates signals from HuggingFace, GitHub, arXiv, and blogs. It provides structured Markdown and JSON output for downstream analysis and LLM consumption.

## Quick Start

```bash
# Clone and install
git clone https://github.com/liuxiaotong/ai-dataset-radar.git
cd ai-dataset-radar && pip install -r requirements.txt

# Run competitive intelligence
python src/main_intel.py

# Run value analysis
python src/main.py --value-analysis
```

## Features

- **Plugin-Based Scrapers** — Modular architecture with `BaseScraper` class and registry for easy extension
- **Multi-Source Aggregation** — Collect from HuggingFace Hub, GitHub orgs, arXiv, Papers with Code, and RSS feeds
- **Post-Training Classification** — Auto-categorize datasets into SFT, Preference/RLHF, Agent, and Evaluation types
- **Competitive Intelligence** — Monitor 30+ organizations: US labs, China labs, and data vendors
- **Dual Output Format** — Generate both Markdown reports and structured JSON for LLM consumption
- **Watchlist Mechanism** — Configure orgs, keywords, and feeds via `config.yaml`

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         main_intel.py                           │
│                    (Orchestration Layer)                        │
└─────────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌───────────────┐   ┌─────────────────┐   ┌─────────────────┐
│   Scrapers    │   │    Trackers     │   │   Analyzers     │
│ (Plugin-based)│   │  (Org Monitor)  │   │ (Classification)│
└───────────────┘   └─────────────────┘   └─────────────────┘
        │                     │                     │
        ▼                     ▼                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                     DualOutputFormatter                         │
│              (Markdown + JSON Report Generation)                │
└─────────────────────────────────────────────────────────────────┘
```

### Scraper Registry

All scrapers inherit from `BaseScraper` and are registered via decorator:

```python
from src.scrapers import get_all_scrapers, get_scraper

# List all registered scrapers
scrapers = get_all_scrapers()
# {'huggingface': <HuggingFaceScraper>, 'arxiv': <ArxivScraper>, ...}

# Get specific scraper
hf = get_scraper("huggingface")
datasets = hf.scrape()
```

## Data Sources

| Source | Type | Update Frequency | Content |
|--------|------|------------------|---------|
| HuggingFace Hub | `dataset_registry` | 1-3 days | Datasets, models, metadata |
| arXiv | `paper` | 7-14 days | Preprints, abstracts |
| Papers with Code | `dataset_registry` | 1-3 days | Benchmarks, datasets |
| GitHub Search | `code_host` | 1-3 days | Repository metadata |
| GitHub Org Monitor | `code_host` | Real-time | Org-specific repos |
| Blog RSS | `blog` | 1-7 days | Research updates |

## Configuration

### Watchlist Mechanism

Configure organizations and keywords in `config.yaml`:

```yaml
sources:
  huggingface:
    enabled: true
    watch_orgs:
      - allenai
      - google
      - facebook
      - nvidia
      - Qwen

  github:
    enabled: true
    watch_orgs:
      - openai
      - anthropics
      - deepseek-ai
      - argilla-io
    relevance_keywords:
      - dataset
      - annotation
      - benchmark
      - rlhf

  blogs:
    enabled: true
    feeds:
      - name: Argilla
        url: https://argilla.io/blog/feed
      - name: Qwen
        url: https://qwenlm.github.io/feed.xml

  arxiv:
    enabled: true
    categories:
      - cs.CL
      - cs.AI
      - cs.LG
    keywords:
      - human feedback
      - RLHF
      - preference learning
```

### API Keys (Optional)

```bash
export GITHUB_TOKEN=your_token  # Higher rate limits
```

## Output Formats

### Markdown Report

Human-readable reports saved to `data/reports/intel_report_YYYY-MM-DD.md`:

```markdown
# AI Dataset Radar - Intelligence Report

## Executive Summary
- 42 datasets from tracked organizations
- 15 relevant papers
- 8 blog posts with signals

## US AI Labs Activity
### OpenAI
- **new-dataset** (1.2K downloads) - Description...
```

### JSON Output

Structured data for LLM consumption at `data/reports/intel_report_YYYY-MM-DD.json`:

```json
{
  "generated_at": "2024-01-15T10:30:00",
  "summary": {
    "total_datasets": 42,
    "total_repos": 15,
    "total_papers": 8,
    "total_blog_posts": 5
  },
  "datasets": [...],
  "papers": [...],
  "github_activity": [...],
  "blog_posts": [...]
}
```

## Usage

### Competitive Intelligence

```bash
# Full report (default: last 7 days)
python src/main_intel.py

# Custom time range
python src/main_intel.py --days 14

# Skip specific sources
python src/main_intel.py --no-blogs --no-papers

# Custom output path
python src/main_intel.py --output my_report.md
```

### Value Analysis

```bash
# Full analysis
python src/main.py --value-analysis

# Focus on specific types
python src/main.py --focus sft
python src/main.py --focus preference

# Filter by score
python src/main.py --value-analysis --min-score 60
```

## Organizations Tracked

| Category | Organizations |
|----------|---------------|
| **Frontier Labs** | OpenAI, Anthropic, Google/DeepMind, Meta, xAI |
| **Emerging Labs** | Mistral, Cohere, AI21, Together, Databricks |
| **Research Labs** | EleutherAI, HuggingFace, Allen AI, LMSys, NVIDIA |
| **China Open Source** | Qwen, DeepSeek, ChatGLM, Baichuan, Yi, InternLM |
| **China Closed Source** | Baidu ERNIE, ByteDance Doubao, Tencent Hunyuan, Moonshot Kimi |
| **Data Vendors** | Scale AI, Surge AI, Argilla, Snorkel, Labelbox |

## Dataset Categories

| Category | Description | Examples |
|----------|-------------|----------|
| **SFT** | Instruction-following data | Alpaca, ShareGPT, OpenOrca |
| **Preference** | Human preference pairs for RLHF/DPO | UltraFeedback, HelpSteer, HH-RLHF |
| **Agent** | Tool use and trajectory data | SWE-bench, WebArena, ToolBench |
| **Evaluation** | Benchmark test sets | MMLU, HumanEval, GPQA |

## Project Structure

```
ai-dataset-radar/
├── src/
│   ├── main.py              # Value analysis entry point
│   ├── main_intel.py        # Competitive intelligence entry point
│   ├── output_formatter.py  # Dual output (MD + JSON)
│   ├── scrapers/
│   │   ├── base.py          # BaseScraper abstract class
│   │   ├── registry.py      # Scraper registration
│   │   ├── huggingface.py   # HuggingFace Hub scraper
│   │   ├── arxiv.py         # arXiv paper scraper
│   │   ├── github.py        # GitHub search scraper
│   │   ├── github_org.py    # GitHub org monitor
│   │   ├── blog_rss.py      # RSS feed scraper
│   │   └── ...
│   ├── analyzers/           # Scoring & classification
│   └── trackers/            # Org & blog monitoring
├── tests/                   # Test suite
├── config.yaml              # Configuration
└── requirements.txt         # Dependencies
```

## Roadmap

- [x] Core infrastructure (database, scrapers)
- [x] Multi-source aggregation
- [x] Value scoring system
- [x] Post-training classification
- [x] Competitive intelligence (US & China labs)
- [x] Plugin-based scraper architecture
- [x] Dual output format (Markdown + JSON)
- [x] Config-driven watchlist mechanism
- [ ] Deep analysis (PDF extraction, LLM summarization)
- [ ] Automation (scheduled execution, alerting)

## Contributing

Contributions welcome! Please feel free to submit a Pull Request.

### Adding a New Scraper

```python
from src.scrapers.base import BaseScraper
from src.scrapers.registry import register_scraper

@register_scraper("my_source")
class MySourceScraper(BaseScraper):
    name = "my_source"
    source_type = "dataset_registry"  # or "paper", "code_host", "blog"

    def scrape(self, config=None) -> list[dict]:
        # Your scraping logic
        return [{"source": "my_source", "id": "...", ...}]
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

Built with data from [Hugging Face](https://huggingface.co/), [GitHub](https://github.com/), [arXiv](https://arxiv.org/), and [Papers with Code](https://paperswithcode.com/).

---

<p align="center">
  <sub>If you find this useful, please consider giving it a star</sub>
</p>
