# AI Dataset Radar

**English** | [中文](#中文文档)

> Data Recipe Intelligence System for AI dataset discovery and analysis.

A business intelligence tool for data labeling companies to discover valuable data recipes, research dataset construction methods, and track industry trends.

## Features

### Core Capabilities

| Feature | Description |
|---------|-------------|
| **Early Signal Detection** | GitHub Trending + HF Daily Papers for earliest discovery |
| **Multi-source Tracking** | Monitors 5 data sources for comprehensive coverage |
| **Model-Dataset Analysis** | Discovers which datasets are used by popular models |
| **Trend Detection** | Tracks download growth and identifies rising datasets |
| **Persistent Storage** | SQLite database for historical data and trends |
| **Smart Filtering** | Auto-detect dataset-related content via keywords |

### Data Sources

| Source | Discovery Latency | Content |
|--------|-------------------|---------|
| **GitHub Trending** | Day 1-3 | New dataset repos |
| **HF Daily Papers** | Day 3-7 | Trending AI papers |
| Hugging Face Hub | Day 7+ | Datasets & Models |
| Papers with Code | Day 14+ | Benchmarks & SOTA |
| arXiv | Day 14+ | Research Papers |

### Discovery Timeline

```
Day 0   ████ Researcher announces on Twitter
Day 1-3 ████ GitHub repo created        ← We catch it here
Day 3-7 ████ HF Daily Papers features   ← We catch it here
Day 7+  ████ HuggingFace Hub upload
Day 14+ ████ arXiv / Papers with Code
```

## Quick Start

### Installation

```bash
git clone https://github.com/liuxiaotong/ai-dataset-radar.git
cd ai-dataset-radar

python -m venv venv
source venv/bin/activate  # Linux/Mac

pip install -r requirements.txt
```

### Basic Usage

```bash
# Full analysis (fetch + trend + model analysis)
python src/main.py

# Quick mode (fetch only)
python src/main.py --quick

# Skip specific analysis
python src/main.py --no-models    # Skip model-dataset analysis
python src/main.py --no-trends    # Skip trend analysis
python src/main.py --no-notify    # Skip notifications
```

### Configuration

Edit `config.yaml`:

```yaml
database:
  path: data/radar.db

sources:
  huggingface:
    enabled: true
    limit: 50
  github:
    enabled: true
    limit: 30
    token: ${GITHUB_TOKEN}  # Optional: for higher rate limits
  hf_papers:
    enabled: true
    limit: 50

models:
  enabled: true
  limit: 100
  min_downloads: 1000

analysis:
  trend_days: [7, 30]
  min_growth_alert: 0.5
```

## Architecture

```
ai-dataset-radar/
├── src/
│   ├── main.py                 # Entry point
│   ├── db.py                   # SQLite database layer
│   ├── filters.py              # Filtering logic
│   ├── notifiers.py            # Notification handlers
│   ├── scrapers/
│   │   ├── huggingface.py      # HF datasets + models
│   │   ├── github.py           # GitHub trending repos
│   │   ├── hf_papers.py        # HF daily papers
│   │   ├── paperswithcode.py   # Benchmarks
│   │   └── arxiv.py            # Papers
│   └── analyzers/
│       ├── model_dataset.py    # Model-dataset relationships
│       └── trend.py            # Growth trend analysis
├── tests/                      # Test suite (61 tests)
├── data/                       # Runtime data (gitignored)
└── config.yaml
```

## Output Example

```
============================================================
  AI Dataset Radar v2
============================================================

Fetching data from sources...
  Hugging Face datasets: 50 found
  arXiv papers: 50 found
  GitHub repos: 30 found (5 dataset-related)
  HF Daily Papers: 50 found (36 dataset-related)

Top Datasets by Model Usage:
1. wikipedia - Used by 6 models
2. eli5 - Used by 5 models
3. squad - Used by 4 models
```

## Development

```bash
# Run tests
python -m pytest tests/ -v

# Run with custom config
python src/main.py --config my-config.yaml
```

## Roadmap

- [x] Phase 1: Infrastructure (database, model tracking, trend analysis)
- [x] Phase 1.5: Early signals (GitHub Trending, HF Daily Papers)
- [ ] Phase 2: Deep analysis (PDF parsing, GitHub code analysis, LLM summarization)
- [ ] Phase 3: Intelligence (competitor monitoring, weekly reports, alerts)

## License

MIT License

---

<a name="中文文档"></a>
# AI Dataset Radar

[English](#ai-dataset-radar) | **中文**

> 数据配方情报系统 - AI 数据集发现与分析工具

为数据标注公司打造的商业情报工具，用于发现有价值的数据配方、研究数据集构建方法、追踪行业趋势。

## 功能特性

### 核心能力

| 功能 | 说明 |
|------|------|
| **早期信号检测** | GitHub Trending + HF Daily Papers 实现最早发现 |
| **多源追踪** | 5 个数据源全覆盖 |
| **模型-数据集分析** | 发现热门模型使用的数据集 |
| **趋势检测** | 追踪下载量增长，识别上升期数据集 |
| **持久化存储** | SQLite 数据库存储历史数据和趋势 |
| **智能过滤** | 自动识别数据集相关内容 |

### 数据源

| 来源 | 发现延迟 | 内容 |
|------|----------|------|
| **GitHub Trending** | Day 1-3 | 新数据集仓库 |
| **HF Daily Papers** | Day 3-7 | 热门 AI 论文 |
| Hugging Face Hub | Day 7+ | 数据集和模型 |
| Papers with Code | Day 14+ | 基准测试和 SOTA |
| arXiv | Day 14+ | 研究论文 |

### 发现时间线

```
Day 0   ████ 研究者在 Twitter 公告
Day 1-3 ████ GitHub 仓库创建     ← 我们在这里捕获
Day 3-7 ████ HF Daily Papers    ← 我们在这里捕获
Day 7+  ████ HuggingFace Hub 上传
Day 14+ ████ arXiv / Papers with Code
```

## 快速开始

### 安装

```bash
git clone https://github.com/liuxiaotong/ai-dataset-radar.git
cd ai-dataset-radar

python -m venv venv
source venv/bin/activate  # Linux/Mac

pip install -r requirements.txt
```

### 基本用法

```bash
# 完整分析（抓取 + 趋势 + 模型分析）
python src/main.py

# 快速模式（仅抓取）
python src/main.py --quick

# 跳过特定分析
python src/main.py --no-models    # 跳过模型-数据集分析
python src/main.py --no-trends    # 跳过趋势分析
python src/main.py --no-notify    # 跳过通知
```

### 配置说明

编辑 `config.yaml`：

```yaml
database:
  path: data/radar.db

sources:
  huggingface:
    enabled: true
    limit: 50
  github:
    enabled: true
    limit: 30
    token: ${GITHUB_TOKEN}  # 可选：提高 API 速率限制
  hf_papers:
    enabled: true
    limit: 50

models:
  enabled: true
  limit: 100
  min_downloads: 1000

analysis:
  trend_days: [7, 30]
  min_growth_alert: 0.5
```

## 项目结构

```
ai-dataset-radar/
├── src/
│   ├── main.py                 # 主入口
│   ├── db.py                   # SQLite 数据库层
│   ├── filters.py              # 过滤逻辑
│   ├── notifiers.py            # 通知处理
│   ├── scrapers/
│   │   ├── huggingface.py      # HF 数据集 + 模型
│   │   ├── github.py           # GitHub 热门仓库
│   │   ├── hf_papers.py        # HF 每日论文
│   │   ├── paperswithcode.py   # 基准测试
│   │   └── arxiv.py            # 论文
│   └── analyzers/
│       ├── model_dataset.py    # 模型-数据集关联
│       └── trend.py            # 增长趋势分析
├── tests/                      # 测试套件 (61 个测试)
├── data/                       # 运行时数据 (已 gitignore)
└── config.yaml
```

## 输出示例

```
============================================================
  AI Dataset Radar v2
============================================================

Fetching data from sources...
  Hugging Face datasets: 50 found
  arXiv papers: 50 found
  GitHub repos: 30 found (5 dataset-related)
  HF Daily Papers: 50 found (36 dataset-related)

按使用量排名的数据集:
1. wikipedia - 被 6 个模型使用
2. eli5 - 被 5 个模型使用
3. squad - 被 4 个模型使用
```

## 开发指南

```bash
# 运行测试
python -m pytest tests/ -v

# 使用自定义配置
python src/main.py --config my-config.yaml
```

## 路线图

- [x] 阶段 1：基础设施（数据库、模型抓取、趋势分析）
- [x] 阶段 1.5：早期信号（GitHub Trending、HF Daily Papers）
- [ ] 阶段 2：深度分析（论文 PDF 解析、GitHub 代码分析、LLM 配方总结）
- [ ] 阶段 3：情报整合（竞品监控、周报/月报、告警机制）

## 许可证

MIT License
