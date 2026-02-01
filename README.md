# AI Dataset Radar

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Version](https://img.shields.io/badge/version-v5.0-green.svg)](https://github.com/liuxiaotong/ai-dataset-radar)

**A Multi-Signal Intelligence System for High-Value AI Dataset Discovery**

**面向高价值 AI 数据集发现的多信号情报系统**

---

## Abstract | 摘要

AI Dataset Radar is an automated intelligence system designed to identify and evaluate high-value datasets for machine learning research and development. The system aggregates heterogeneous signals from multiple authoritative sources—including citation metrics, model adoption patterns, and benchmark associations—to compute composite value scores that quantify a dataset's research impact, adoption trajectory, and commercial potential.

AI Dataset Radar 是一个自动化情报系统，旨在识别和评估机器学习研究与开发中的高价值数据集。该系统聚合来自多个权威来源的异构信号——包括引用指标、模型采用模式和基准关联——计算综合价值评分，以量化数据集的研究影响力、采用轨迹和商业潜力。

**Key Contributions | 主要贡献:**

1. A weighted multi-factor scoring model integrating six orthogonal signals for dataset valuation
2. Specialized filtering mechanisms for post-training datasets (SFT, RLHF, Agent, Evaluation)
3. Organization-level competitive intelligence tracking across 30+ research institutions (US & China)
4. Temporal signal analysis distinguishing leading indicators from lagging metrics
5. Comprehensive China AI ecosystem monitoring (open source & closed source models)

---

## 1. Introduction | 引言

The proliferation of AI research has created an information asymmetry problem: while thousands of datasets are published annually, identifying those with high research impact or commercial value remains challenging. Traditional discovery methods—keyword search, manual curation—fail to capture emerging trends or quantify relative value.

AI 研究的快速发展造成了信息不对称问题：尽管每年发布数千个数据集，但识别具有高研究影响力或商业价值的数据集仍然具有挑战性。传统发现方法——关键词搜索、人工筛选——无法捕捉新兴趋势或量化相对价值。

This system addresses three fundamental questions:
- **What datasets are gaining research traction?** (Citation velocity analysis)
- **Which datasets power production models?** (Model card reverse-engineering)
- **Where are annotation opportunities?** (Post-training data demand signals)

本系统解决三个基本问题：
- **哪些数据集正在获得研究关注？**（引用增速分析）
- **哪些数据集支撑生产模型？**（模型卡逆向工程）
- **哪里存在标注机会？**（后训练数据需求信号）

---

## 2. Methodology | 方法论

### 2.1 Value Scoring Framework | 价值评分框架

The system employs a weighted additive scoring model (Score ∈ [0, 100]):

系统采用加权加法评分模型（评分 ∈ [0, 100]）：

```
Score = Σ (weight_i × indicator_i)
```

**English:**

| Signal | Weight | Criterion | Rationale |
|--------|--------|-----------|-----------|
| SOTA Model Usage | 30 | Referenced by state-of-the-art models | Indicates benchmark relevance |
| Citation Velocity | 20 | Monthly citation growth ≥ 10 | Leading indicator of research interest |
| Model Adoption | 20 | Used by ≥ 3 HuggingFace models | Proxy for practical utility |
| Institution Prestige | 15 | Origin: top-tier research labs | Quality signal |
| Reproducibility | 10 | Associated paper + code available | Scientific rigor |
| Scale | 5 | Dataset size > 10GB | Resource investment indicator |

**中文:**

| 信号 | 权重 | 标准 | 依据 |
|------|------|------|------|
| SOTA 模型使用 | 30 | 被 SOTA 模型引用 | 表明基准相关性 |
| 引用增速 | 20 | 月引用增长 ≥ 10 | 研究兴趣的领先指标 |
| 模型采用度 | 20 | 被 ≥ 3 个 HuggingFace 模型使用 | 实用性代理指标 |
| 机构声誉 | 15 | 来源：顶级研究实验室 | 质量信号 |
| 可复现性 | 10 | 有配套论文和代码 | 科学严谨性 |
| 规模 | 5 | 数据集大小 > 10GB | 资源投入指标 |

### 2.2 Post-Training Dataset Classification | 后训练数据集分类

A specialized `PostTrainingFilter` module classifies datasets into four categories critical for LLM development:

专门的 `PostTrainingFilter` 模块将数据集分类为 LLM 开发的四个关键类别：

**English:**

| Category | Description | Example Datasets |
|----------|-------------|------------------|
| **SFT** (Supervised Fine-Tuning) | Instruction-following data | Alpaca, ShareGPT, OpenOrca, FLAN |
| **Preference** (RLHF/DPO) | Human preference pairs | UltraFeedback, HelpSteer, Nectar, HH-RLHF |
| **Agent** | Tool use and trajectory data | WebArena, SWE-bench, ToolBench, GAIA |
| **Evaluation** | Benchmark test sets | MMLU, HumanEval, GPQA, GSM8K |

**中文:**

| 类别 | 描述 | 示例数据集 |
|------|------|-----------|
| **SFT** (监督微调) | 指令遵循数据 | Alpaca, ShareGPT, OpenOrca, FLAN |
| **Preference** (RLHF/DPO) | 人类偏好配对 | UltraFeedback, HelpSteer, Nectar, HH-RLHF |
| **Agent** | 工具使用和轨迹数据 | WebArena, SWE-bench, ToolBench, GAIA |
| **Evaluation** | 基准测试集 | MMLU, HumanEval, GPQA, GSM8K |

Classification employs a confidence-weighted signal matching approach:

分类采用置信度加权信号匹配方法：

```
Confidence Score = 0.6 × |strong_signals| + 0.3 × |medium_signals| + 0.1 × |weak_signals|
```

### 2.3 Temporal Signal Analysis | 时序信号分析

**English:**

| Signal Type | Source | Temporal Characteristic | Business Implication |
|-------------|--------|------------------------|---------------------|
| Citation Velocity | Semantic Scholar | Leading (6-12 months) | Predicts future industry demand |
| Model Adoption | HuggingFace | Concurrent | Reflects current production use |
| SOTA Association | Benchmarks | Concurrent | Indicates premium positioning |

**中文:**

| 信号类型 | 来源 | 时序特征 | 商业含义 |
|----------|------|----------|----------|
| 引用增速 | Semantic Scholar | 领先（6-12个月） | 预测未来产业需求 |
| 模型采用 | HuggingFace | 同步 | 反映当前生产使用 |
| SOTA 关联 | 基准测试 | 同步 | 表明溢价定位 |

---

## 3. System Architecture | 系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│                      AI Dataset Radar                           │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │  Semantic   │  │ HuggingFace │  │   GitHub    │   Data      │
│  │  Scholar    │  │  Hub API    │  │  Trending   │   Sources   │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘             │
│         │                │                │                     │
│         └────────────────┼────────────────┘                     │
│                          ▼                                      │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │                    Scraper Layer                          │ │
│  │  semantic_scholar.py │ huggingface.py │ github.py │ arxiv │ │
│  └───────────────────────────────────────────────────────────┘ │
│                          │                                      │
│                          ▼                                      │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │                   Analysis Layer                          │ │
│  │  ValueScorer │ PostTrainingFilter │ TrendAnalyzer │ Opps  │ │
│  └───────────────────────────────────────────────────────────┘ │
│                          │                                      │
│                          ▼                                      │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │               Persistence & Reporting                     │ │
│  │          SQLite DB │ Markdown Reports │ JSON Export       │ │
│  └───────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### 3.1 Data Sources | 数据来源

**English:**

| Source | Update Frequency | Content Type | API Requirements |
|--------|------------------|--------------|------------------|
| Semantic Scholar | Real-time | Citation metrics, paper metadata | API key recommended |
| HuggingFace Hub | 1-3 days | Datasets, models, papers | Public API |
| GitHub Trending | 1-3 days | Repository metadata | Token optional |
| arXiv | 7-14 days | Preprint papers | Public feed |
| Blog Monitoring | 1-7 days | Research updates, product news | Web scraping |

**中文:**

| 来源 | 更新频率 | 内容类型 | API 要求 |
|------|----------|----------|----------|
| Semantic Scholar | 实时 | 引用指标、论文元数据 | 建议配置 API Key |
| HuggingFace Hub | 1-3 天 | 数据集、模型、论文 | 公开 API |
| GitHub Trending | 1-3 天 | 仓库元数据 | Token 可选 |
| arXiv | 7-14 天 | 预印本论文 | 公开 Feed |
| Blog Monitoring | 1-7 天 | 研究动态、产品更新 | 网页抓取 |

**Blog Monitoring Targets | 博客监控目标:**

Scale AI, Snorkel AI, Argilla, Anthropic Research, DeepSeek, Qwen, 智谱 AI

### 3.2 Organization Tracking | 组织追踪

The system monitors dataset publications from 30+ organizations across five categories:

系统监控 30+ 组织的数据集发布，覆盖五大类别：

**English:**

| Category | Organizations | Priority |
|----------|---------------|----------|
| **Frontier Labs** | OpenAI, Anthropic, Google/DeepMind, Meta, xAI | High |
| **Emerging Labs** | Mistral, Cohere, AI21, Together, Databricks | Medium |
| **Research Labs** | EleutherAI, HuggingFace, Allen AI, LMSys, NVIDIA | Medium |
| **China Open Source** | Qwen (通义千问), DeepSeek (深度求索), ChatGLM (智谱), Baichuan (百川), Yi (零一万物), InternLM (书生), MiniMax, Stepfun (阶跃星辰) | High |
| **China Closed Source** | Baidu ERNIE (文心一言), ByteDance Doubao (豆包), Tencent Hunyuan (混元), iFlytek Spark (星火), Moonshot Kimi (月之暗面), SenseTime (商汤) | Medium |

**中文:**

| 类别 | 组织 | 优先级 |
|------|------|--------|
| **一线实验室** | OpenAI, Anthropic, Google/DeepMind, Meta, xAI | 高 |
| **新兴实验室** | Mistral, Cohere, AI21, Together, Databricks | 中 |
| **研究实验室** | EleutherAI, HuggingFace, Allen AI, LMSys, NVIDIA | 中 |
| **中国开源大模型** | Qwen (通义千问), DeepSeek (深度求索), ChatGLM (智谱), Baichuan (百川), Yi (零一万物), InternLM (书生), MiniMax, Stepfun (阶跃星辰) | 高 |
| **中国闭源大模型** | Baidu ERNIE (文心一言), ByteDance Doubao (豆包), Tencent Hunyuan (混元), iFlytek Spark (星火), Moonshot Kimi (月之暗面), SenseTime (商汤) | 中 |

**Data Vendors | 数据供应商:**

| Tier | Vendors |
|------|---------|
| **Premium** | Scale AI, Surge AI, Appen, Sama |
| **Specialized** | Argilla, Snorkel, Labelbox, Humanloop |

---

## 4. Installation | 安装

### 4.1 Requirements | 环境要求

- Python 3.8+
- SQLite 3.35+
- 512 MB available memory

### 4.2 Setup | 配置

```bash
git clone https://github.com/liuxiaotong/ai-dataset-radar.git
cd ai-dataset-radar

python -m venv venv
source venv/bin/activate  # Linux/macOS
# or: venv\Scripts\activate  # Windows

pip install -r requirements.txt
```

### 4.3 API Configuration | API 配置

```bash
# Semantic Scholar API (recommended for citation analysis)
# 申请地址: https://www.semanticscholar.org/product/api
export SEMANTIC_SCHOLAR_API_KEY=your_key_here

# GitHub API (optional, for higher rate limits)
export GITHUB_TOKEN=your_token_here
```

**Rate Limits | 速率限制:**

| Service | Without Key | With Key |
|---------|-------------|----------|
| Semantic Scholar | 100 req / 5 min | 1 req / sec |
| GitHub | 60 req / hour | 5000 req / hour |

---

## 5. Usage | 使用方法

### 5.1 Basic Analysis | 基础分析

```bash
# Full analysis pipeline
python src/main.py --value-analysis

# Post-training dataset discovery
python src/main.py --focus sft           # SFT datasets
python src/main.py --focus preference    # RLHF/DPO datasets
python src/main.py --focus agent         # Agent datasets
python src/main.py --focus evaluation    # Benchmark datasets
```

### 5.2 Competitive Intelligence | 竞争情报

```bash
# Generate competitive intelligence report
# 生成竞争情报报告
python src/main_intel.py

# Output includes | 报告包含:
# - US Labs Activity (美国实验室动态)
# - China Labs Activity (中国大模型厂商动态)
# - Data Vendor Activity (数据供应商动态)
# - Datasets by Type (按类型分类的数据集)
# - Relevant Papers (相关论文)
```

### 5.3 Filtered Analysis | 过滤分析

```bash
# High-value datasets only (score ≥ 60)
python src/main.py --value-analysis --min-score 60

# Top-tier institutions only
python src/main.py --value-analysis --top-institutions

# Positive growth trend only
python src/main.py --value-analysis --growth-only

# Domain-specific analysis
python src/main.py --focus robotics
python src/main.py --focus multimodal
```

### 5.4 Command Reference | 命令参考

| Option | Description | Default |
|--------|-------------|---------|
| `--value-analysis` | Enable multi-signal scoring | Off |
| `--focus DOMAIN` | Filter by domain (sft, preference, agent, evaluation, robotics, rlhf, multimodal) | None |
| `--min-score N` | Minimum value score threshold | 0 |
| `--top-institutions` | Restrict to top-tier institutions | Off |
| `--growth-only` | Positive growth trend only | Off |
| `--opportunities` | Detect annotation opportunities | Off |
| `--quick` | Data collection only (skip analysis) | Off |

---

## 6. Output Specification | 输出规范

### 6.1 Value Report | 价值报告

Generated at `data/value_report_YYYY-MM-DD.md`:

```markdown
# High-Value Dataset Report | 高价值数据集报告

## Executive Summary | 执行摘要
- High-value (≥60): 15 datasets
- Medium-value (40-59): 23 datasets
- Post-training datasets: 12 identified

## Top Datasets by Category | 分类排行

### SFT Datasets
| Rank | Dataset | Score | Downloads | Institution |
|------|---------|-------|-----------|-------------|
| 1    | OpenOrca | 82    | 125,000   | OpenOrca    |

### Preference Datasets
| Rank | Dataset | Score | Downloads | Institution |
|------|---------|-------|-----------|-------------|
| 1    | UltraFeedback | 78 | 89,000 | OpenBMB |
```

### 6.2 Intelligence Report | 竞争情报报告

Generated at `data/intel_report_YYYY-MM-DD.md`:

生成于 `data/intel_report_YYYY-MM-DD.md`:

```markdown
# AI Data Intelligence Report | AI 数据情报报告

## US Labs Activity | 美国实验室动态
- OpenAI: 2 new datasets, 5 new models
- Anthropic: 1 research paper on constitutional AI

## China Labs Activity | 中国大模型厂商动态
- Qwen: Released Qwen2.5-Coder series
- DeepSeek: New reasoning dataset published

## Data Vendor Activity | 数据供应商动态
- Scale AI: Blog post on synthetic data generation
- Argilla: New distilabel release

## Datasets by Type | 数据集分类
### SFT (Supervised Fine-Tuning)
| Dataset | Organization | Downloads |
|---------|--------------|-----------|
| ... | ... | ... |
```

### 6.3 JSON Export | JSON 导出

```json
{
  "datasets": [...],
  "analysis_timestamp": "2026-01-30T12:00:00Z",
  "post_training_summary": {
    "sft": {"count": 5, "items": [...]},
    "preference": {"count": 3, "items": [...]},
    "agent": {"count": 2, "items": [...]},
    "evaluation": {"count": 4, "items": [...]}
  }
}
```

---

## 7. Configuration | 配置

### 7.1 Focus Areas | 聚焦领域

```yaml
# config.yaml
focus_areas:
  sft:
    enabled: true
    keywords:
      - instruction tuning
      - supervised fine-tuning
      - ShareGPT
      - Alpaca
    hf_tags:
      - task_categories:conversational

  preference:
    enabled: true
    keywords:
      - DPO
      - RLHF
      - chosen rejected
      - human feedback
      - UltraFeedback

  agent:
    enabled: true
    keywords:
      - function calling
      - tool use
      - trajectory
      - SWE-bench
      - WebArena

  evaluation:
    enabled: true
    keywords:
      - benchmark
      - MMLU
      - HumanEval
      - GPQA
```

### 7.2 Organization Tracking | 组织追踪

```yaml
# config.yaml
watched_orgs:
  # Frontier Labs - 一线实验室
  frontier_labs:
    openai:
      hf_ids: ["openai"]
      keywords: ["openai", "gpt", "chatgpt"]
      priority: high
    anthropic:
      hf_ids: ["anthropic", "Anthropic"]
      keywords: ["anthropic", "claude", "constitutional"]
      priority: high

  # China Open Source Labs - 中国开源大模型
  china_opensource:
    alibaba_qwen:
      hf_ids: ["Qwen", "qwen"]
      github: ["QwenLM"]
      keywords: ["qwen", "通义千问", "tongyi"]
      priority: high
    deepseek:
      hf_ids: ["deepseek-ai"]
      github: ["deepseek-ai"]
      keywords: ["deepseek", "深度求索"]
      priority: high

  # China Closed Source Labs - 中国闭源大模型（关键词监控）
  china_closedsource:
    baidu_ernie:
      hf_ids: []
      keywords: ["文心一言", "ernie", "wenxin", "百度"]
      priority: medium
```

---

## 8. Development | 开发

### 8.1 Project Structure | 项目结构

```
ai-dataset-radar/
├── src/
│   ├── main.py                    # Value analysis entry point | 价值分析入口
│   ├── main_intel.py              # Competitive intelligence entry point | 竞争情报入口
│   ├── db.py                      # SQLite persistence layer
│   ├── filters.py                 # Dataset filtering & classification
│   ├── report.py                  # Value report generation
│   ├── intel_report.py            # Intelligence report generation
│   ├── notifiers.py               # Notification system
│   ├── scrapers/
│   │   ├── semantic_scholar.py    # Citation tracking
│   │   ├── huggingface.py         # HF datasets/models
│   │   ├── github.py              # Trending repositories
│   │   ├── arxiv.py               # Paper retrieval
│   │   ├── hf_papers.py           # HF daily papers
│   │   └── pwc_sota.py            # PapersWithCode SOTA tracking
│   ├── analyzers/
│   │   ├── value_scorer.py        # Multi-factor scoring
│   │   ├── model_card_analyzer.py # Model card parsing
│   │   ├── trend.py               # Growth analysis
│   │   ├── opportunities.py       # Business signal detection
│   │   ├── data_type_classifier.py# Post-training data classification
│   │   ├── org_detector.py        # Organization detection
│   │   └── quality_scorer.py      # Quality scoring
│   └── trackers/
│       ├── org_tracker.py         # Organization activity tracking | 组织活动追踪
│       ├── github_tracker.py      # GitHub repository tracking | GitHub 仓库追踪
│       └── blog_tracker.py        # Blog/RSS monitoring | 博客监控
├── tests/                         # Test suite (50+ test cases)
├── config.yaml                    # Configuration file
└── requirements.txt               # Dependencies
```

### 8.2 Testing | 测试

```bash
# Run full test suite
python -m pytest tests/ -v

# Run with coverage report
python -m pytest tests/ --cov=src --cov-report=html

# Run specific test module
python -m pytest tests/test_business_intel.py -v
```

---

## 9. Roadmap | 路线图

**English:**

| Phase | Status | Description |
|-------|--------|-------------|
| Phase 1 | ✅ Complete | Core infrastructure (database, scrapers, trend analysis) |
| Phase 2 | ✅ Complete | Multi-source aggregation (GitHub, HF Papers, org tracking) |
| Phase 3 | ✅ Complete | Value scoring system (citations, SOTA, model cards) |
| Phase 3.5 | ✅ Complete | Post-training dataset classification (SFT, RLHF, Agent, Eval) |
| Phase 4 | ✅ Complete | Competitive intelligence (China labs monitoring, blog tracking, intel reports) |
| Phase 5 | 🔄 Planned | Deep analysis (PDF extraction, LLM summarization) |
| Phase 6 | 🔄 Planned | Automation (scheduled execution, alerting, monitoring) |

**中文:**

| 阶段 | 状态 | 描述 |
|------|------|------|
| 阶段 1 | ✅ 完成 | 核心基础设施（数据库、爬虫、趋势分析） |
| 阶段 2 | ✅ 完成 | 多源聚合（GitHub、HF 论文、机构追踪） |
| 阶段 3 | ✅ 完成 | 价值评分系统（引用、SOTA、模型卡） |
| 阶段 3.5 | ✅ 完成 | 后训练数据集分类（SFT、RLHF、Agent、Eval） |
| 阶段 4 | ✅ 完成 | 竞争情报增强（中国大模型监控、博客追踪、情报报告） |
| 阶段 5 | 🔄 计划中 | 深度分析（PDF 提取、LLM 摘要） |
| 阶段 6 | 🔄 计划中 | 自动化（定时执行、告警、监控） |

---

## 10. Citation | 引用

If you use this system in your research, please cite:

如果您在研究中使用本系统，请引用：

```bibtex
@software{ai_dataset_radar,
  author = {Liu, Xiaotong},
  title = {AI Dataset Radar: A Multi-Signal Intelligence System for High-Value AI Dataset Discovery},
  year = {2026},
  url = {https://github.com/liuxiaotong/ai-dataset-radar}
}
```

---

## License | 许可证

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件。

---

## Acknowledgments | 致谢

This system builds upon APIs and data from:
- [Semantic Scholar](https://www.semanticscholar.org/) - Citation data
- [Hugging Face](https://huggingface.co/) - Dataset and model metadata
- [GitHub](https://github.com/) - Repository trending data
- [arXiv](https://arxiv.org/) - Preprint papers

本系统基于以下平台的 API 和数据构建：
- [Semantic Scholar](https://www.semanticscholar.org/) - 引用数据
- [Hugging Face](https://huggingface.co/) - 数据集和模型元数据
- [GitHub](https://github.com/) - 仓库趋势数据
- [arXiv](https://arxiv.org/) - 预印本论文
