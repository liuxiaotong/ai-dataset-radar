# AI Dataset Radar

**English** | [中文](#中文文档)

> Data Recipe Intelligence System for AI dataset discovery and analysis.

A business intelligence tool for data labeling companies to discover valuable data recipes, research dataset construction methods, and track industry trends.

## Features

### Core Capabilities

| Feature | Description |
|---------|-------------|
| **Multi-source Tracking** | Monitors Hugging Face, Papers with Code, and arXiv |
| **Model-Dataset Analysis** | Discovers which datasets are used by popular models |
| **Trend Detection** | Tracks download growth and identifies rising datasets |
| **Persistent Storage** | SQLite database for historical data and trends |
| **Smart Filtering** | Filter by downloads, keywords, date range |
| **Flexible Output** | Console, Markdown reports, Email, Webhook |

### Data Sources

| Source | Content | API |
|--------|---------|-----|
| Hugging Face | Datasets & Models | REST API |
| Papers with Code | Benchmarks & SOTA | REST API |
| arXiv | Research Papers | Atom Feed |

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/ai-dataset-radar.git
cd ai-dataset-radar

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```bash
# Full analysis (fetch data + trend analysis + model analysis)
python src/main.py

# Quick mode (fetch data only, skip analysis)
python src/main.py --quick

# Skip specific analysis
python src/main.py --no-models    # Skip model-dataset analysis
python src/main.py --no-trends    # Skip trend analysis
python src/main.py --no-notify    # Skip notifications
```

### Configuration

Edit `config.yaml`:

```yaml
# Database
database:
  path: data/radar.db

# Data sources
sources:
  huggingface:
    enabled: true
    limit: 50

# Model tracking
models:
  enabled: true
  limit: 100
  min_downloads: 1000

# Analysis settings
analysis:
  trend_days: [7, 30]
  min_growth_alert: 0.5  # Alert on 50%+ growth
```

## Architecture

```
ai-dataset-radar/
├── src/
│   ├── main.py                 # Entry point
│   ├── db.py                   # SQLite database layer
│   ├── filters.py              # Filtering logic
│   ├── notifiers.py            # Notification handlers
│   ├── scrapers/               # Data source scrapers
│   │   ├── huggingface.py      # HF datasets + models
│   │   ├── paperswithcode.py   # Benchmarks
│   │   └── arxiv.py            # Papers
│   └── analyzers/              # Analysis modules
│       ├── model_dataset.py    # Model-dataset relationships
│       └── trend.py            # Growth trend analysis
├── tests/                      # Test suite
├── data/
│   ├── radar.db               # SQLite database
│   └── reports/               # Generated reports
└── config.yaml                # Configuration
```

## Database Schema

```
radar.db
├── datasets        # Dataset metadata
├── daily_stats     # Daily download/like counts
├── models          # Model information
├── model_datasets  # Model → Dataset relationships
└── trends          # Calculated growth rates
```

## Output Examples

### Model-Dataset Analysis

```
============================================================
  Model-Dataset Relationship Analysis
============================================================

Models analyzed: 100
Total links found: 149
Unique datasets: 93

Top Datasets by Model Usage:
1. wikipedia - Used by 6 models
2. eli5 - Used by 5 models
3. squad - Used by 4 models
```

### Trend Analysis

```
============================================================
  Dataset Trend Analysis
============================================================

Rising Datasets (7-day growth):
  popular-dataset
    Growth: 75.0%
    URL: https://huggingface.co/datasets/...
```

## Development

```bash
# Run tests
python -m pytest tests/ -v

# Run with custom config
python src/main.py --config my-config.yaml
```

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
| **多源追踪** | 监控 Hugging Face、Papers with Code、arXiv |
| **模型-数据集分析** | 发现热门模型使用的数据集 |
| **趋势检测** | 追踪下载量增长，识别上升期数据集 |
| **持久化存储** | SQLite 数据库存储历史数据和趋势 |
| **智能过滤** | 按下载量、关键词、时间范围过滤 |
| **灵活输出** | 控制台、Markdown 报告、邮件、Webhook |

### 数据源

| 来源 | 内容 | 接口 |
|------|------|------|
| Hugging Face | 数据集和模型 | REST API |
| Papers with Code | 基准测试和 SOTA | REST API |
| arXiv | 研究论文 | Atom Feed |

## 快速开始

### 安装

```bash
# 克隆仓库
git clone https://github.com/your-username/ai-dataset-radar.git
cd ai-dataset-radar

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或: venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 基本用法

```bash
# 完整分析（抓取数据 + 趋势分析 + 模型分析）
python src/main.py

# 快速模式（仅抓取数据，跳过分析）
python src/main.py --quick

# 跳过特定分析
python src/main.py --no-models    # 跳过模型-数据集分析
python src/main.py --no-trends    # 跳过趋势分析
python src/main.py --no-notify    # 跳过通知
```

### 配置说明

编辑 `config.yaml`：

```yaml
# 数据库配置
database:
  path: data/radar.db

# 数据源配置
sources:
  huggingface:
    enabled: true
    limit: 50

# 模型追踪配置
models:
  enabled: true
  limit: 100
  min_downloads: 1000  # 只追踪有热度的模型

# 分析配置
analysis:
  trend_days: [7, 30]       # 计算 7 天和 30 天趋势
  min_growth_alert: 0.5     # 增长超过 50% 时标记
```

## 项目结构

```
ai-dataset-radar/
├── src/
│   ├── main.py                 # 主入口
│   ├── db.py                   # SQLite 数据库层
│   ├── filters.py              # 过滤逻辑
│   ├── notifiers.py            # 通知处理
│   ├── scrapers/               # 数据抓取
│   │   ├── huggingface.py      # HF 数据集 + 模型
│   │   ├── paperswithcode.py   # 基准测试
│   │   └── arxiv.py            # 论文
│   └── analyzers/              # 分析模块
│       ├── model_dataset.py    # 模型-数据集关联
│       └── trend.py            # 增长趋势分析
├── tests/                      # 测试套件
├── data/
│   ├── radar.db               # SQLite 数据库
│   └── reports/               # 生成的报告
└── config.yaml                # 配置文件
```

## 数据库结构

```
radar.db
├── datasets        # 数据集元信息
├── daily_stats     # 每日下载量/点赞数
├── models          # 模型信息
├── model_datasets  # 模型 → 数据集 关联
└── trends          # 计算的增长率
```

## 输出示例

### 模型-数据集分析

```
============================================================
  模型-数据集关联分析
============================================================

分析模型数: 100
发现关联数: 149
涉及数据集: 93

按使用量排名的数据集:
1. wikipedia - 被 6 个模型使用
2. eli5 - 被 5 个模型使用
3. squad - 被 4 个模型使用
```

### 趋势分析

```
============================================================
  数据集趋势分析
============================================================

上升期数据集（7天增长）:
  popular-dataset
    增长率: 75.0%
    链接: https://huggingface.co/datasets/...
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
- [ ] 阶段 2：深度分析（论文 PDF 解析、GitHub 代码分析、LLM 配方总结）
- [ ] 阶段 3：情报整合（竞品监控、周报/月报、告警机制）

## 许可证

MIT License

---

Made with ❤️ for the AI research community
