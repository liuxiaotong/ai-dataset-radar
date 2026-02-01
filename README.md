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

AI Dataset Radar aggregates signals from Semantic Scholar, HuggingFace, GitHub, and arXiv to score datasets by research impact, model adoption, and commercial potential. Track what OpenAI, Anthropic, DeepSeek, and 30+ labs are releasing.

## Quick Start

```bash
# Clone and install
git clone https://github.com/liuxiaotong/ai-dataset-radar.git
cd ai-dataset-radar && pip install -r requirements.txt

# Run value analysis
python src/main.py --value-analysis

# Run competitive intelligence
python src/main_intel.py
```

## Features

- **Multi-Signal Scoring** — Composite scores (0-100) from 6 weighted signals: SOTA usage, citation velocity, model adoption, institution prestige, reproducibility, scale
- **Post-Training Classification** — Auto-categorize datasets into SFT, Preference/RLHF, Agent, and Evaluation types
- **Competitive Intelligence** — Monitor 30+ organizations: US labs (OpenAI, Anthropic, Meta), China labs (Qwen, DeepSeek, ChatGLM), and data vendors (Scale AI, Argilla)
- **Trend Analysis** — Distinguish leading indicators (citation growth) from lagging metrics (current adoption)

## Why This Project?

Thousands of datasets are published annually. Finding high-value ones before they go mainstream is hard:

| Problem | Traditional Approach | Our Solution |
|---------|---------------------|--------------|
| Information overload | Keyword search | Multi-signal scoring |
| Missing emerging trends | Manual curation | Citation velocity tracking |
| Unknown production value | Guesswork | Model card reverse-engineering |

## Scoring System

```
Score = Σ (weight × indicator)
```

| Signal | Weight | Criterion |
|--------|--------|-----------|
| SOTA Model Usage | 30 | Referenced by state-of-the-art models |
| Citation Velocity | 20 | Monthly citation growth ≥ 10 |
| Model Adoption | 20 | Used by ≥ 3 HuggingFace models |
| Institution Prestige | 15 | Origin: top-tier research labs |
| Reproducibility | 10 | Paper + code available |
| Scale | 5 | Dataset size > 10GB |

## Dataset Categories

| Category | Description | Examples |
|----------|-------------|----------|
| **SFT** | Instruction-following data | Alpaca, ShareGPT, OpenOrca |
| **Preference** | Human preference pairs for RLHF/DPO | UltraFeedback, HelpSteer, HH-RLHF |
| **Agent** | Tool use and trajectory data | SWE-bench, WebArena, ToolBench |
| **Evaluation** | Benchmark test sets | MMLU, HumanEval, GPQA |

## Usage

### Value Analysis

```bash
# Full analysis
python src/main.py --value-analysis

# Focus on specific types
python src/main.py --focus sft
python src/main.py --focus preference
python src/main.py --focus agent
python src/main.py --focus evaluation

# Filter by score
python src/main.py --value-analysis --min-score 60
```

### Competitive Intelligence

```bash
python src/main_intel.py
```

Generates reports with:
- US Labs Activity (OpenAI, Anthropic, Google, Meta)
- China Labs Activity (Qwen, DeepSeek, ChatGLM, Baichuan)
- Data Vendor Activity (Scale AI, Argilla, Snorkel)
- Datasets by Type

## Organizations Tracked

| Category | Organizations |
|----------|---------------|
| **Frontier Labs** | OpenAI, Anthropic, Google/DeepMind, Meta, xAI |
| **Emerging Labs** | Mistral, Cohere, AI21, Together, Databricks |
| **Research Labs** | EleutherAI, HuggingFace, Allen AI, LMSys, NVIDIA |
| **China Open Source** | Qwen, DeepSeek, ChatGLM, Baichuan, Yi, InternLM |
| **China Closed Source** | Baidu ERNIE, ByteDance Doubao, Tencent Hunyuan, Moonshot Kimi |
| **Data Vendors** | Scale AI, Surge AI, Argilla, Snorkel, Labelbox |

## Data Sources

| Source | Update Frequency | Content |
|--------|------------------|---------|
| Semantic Scholar | Real-time | Citations, paper metadata |
| HuggingFace Hub | 1-3 days | Datasets, models |
| GitHub Trending | 1-3 days | Repository metadata |
| arXiv | 7-14 days | Preprints |
| Blog Monitoring | 1-7 days | Research updates |

## Configuration

Optional API keys for higher rate limits:

```bash
export SEMANTIC_SCHOLAR_API_KEY=your_key  # Recommended
export GITHUB_TOKEN=your_token            # Optional
```

See [`config.yaml`](config.yaml) for full configuration options.

## Project Structure

```
ai-dataset-radar/
├── src/
│   ├── main.py              # Value analysis
│   ├── main_intel.py        # Competitive intelligence
│   ├── scrapers/            # Data collection
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
- [ ] Deep analysis (PDF extraction, LLM summarization)
- [ ] Automation (scheduled execution, alerting)

## Contributing

Contributions welcome! Please feel free to submit a Pull Request.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

Built with data from [Semantic Scholar](https://www.semanticscholar.org/), [Hugging Face](https://huggingface.co/), [GitHub](https://github.com/), and [arXiv](https://arxiv.org/).

---

<p align="center">
  <sub>If you find this useful, please consider giving it a ⭐</sub>
</p>
