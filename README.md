# AI Dataset Radar

**English** | [中文](#中文文档)

> Business Intelligence System for AI dataset discovery and opportunity detection.

A business intelligence tool for data labeling companies to discover valuable data recipes, detect annotation opportunities, track industry trends, and monitor competitor activity.

## Features

### Core Capabilities

| Feature | Description |
|---------|-------------|
| **Early Signal Detection** | GitHub Trending + HF Daily Papers for earliest discovery |
| **Growth Trend Analysis** | Identify breakthrough datasets (0 → 1000+ downloads) |
| **Domain Focus Filtering** | Filter by robotics, RLHF, multimodal, and more |
| **Opportunity Detection** | Detect data factories and annotation signals in papers |
| **Organization Tracking** | Monitor activity from major AI labs |
| **Business Intelligence Reports** | Weekly reports with actionable insights |

### Data Sources

| Source | Discovery Latency | Content |
|--------|-------------------|---------|
| **GitHub Trending** | Day 1-3 | New dataset repos |
| **HF Daily Papers** | Day 3-7 | Trending AI papers |
| Hugging Face Hub | Day 7+ | Datasets & Models |
| Papers with Code | Day 14+ | Benchmarks & SOTA |
| arXiv | Day 14+ | Research Papers |

### Business Intelligence

| Signal | Description |
|--------|-------------|
| **Data Factories** | Authors publishing 3+ datasets in 7 days |
| **Annotation Signals** | Papers mentioning "human annotation", "crowdsourced", etc. |
| **Breakthrough Datasets** | Datasets growing from 0 to 1000+ downloads |
| **Organization Activity** | Datasets/papers from ByteDance, Google, OpenAI, etc. |

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
# Full analysis (fetch + trend + opportunities)
python src/main.py

# Quick mode (fetch only)
python src/main.py --quick

# Business intelligence options
python src/main.py --focus robotics    # Filter by domain
python src/main.py --growth-only       # Only show growing datasets
python src/main.py --min-growth 0.5    # Minimum 50% growth rate
python src/main.py --opportunities     # Focus on business signals

# Skip specific analysis
python src/main.py --no-models         # Skip model-dataset analysis
python src/main.py --no-trends         # Skip trend analysis
python src/main.py --no-opportunities  # Skip opportunity detection
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
    token: ${GITHUB_TOKEN}
  hf_papers:
    enabled: true
    limit: 50

# Domain focus areas
focus_areas:
  robotics:
    enabled: true
    keywords: [robotics, manipulation, embodied, gripper]
    hf_tags: [task_categories:robotics]
  rlhf:
    enabled: true
    keywords: [preference, human feedback, RLHF, DPO]
  multimodal:
    enabled: true
    keywords: [vision-language, VLM, multimodal]

# Organization tracking
tracked_orgs:
  bytedance: [ByteDance, 字节, TikTok]
  google: [Google, DeepMind]
  openai: [OpenAI]

# Opportunity detection
opportunities:
  annotation_signals:
    - human annotation
    - crowdsourced
    - data collection
  data_factory:
    min_datasets: 3
    days: 7

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
│   ├── filters.py              # Domain & organization filters
│   ├── notifiers.py            # Report generators
│   ├── scrapers/
│   │   ├── huggingface.py      # HF datasets + models
│   │   ├── github.py           # GitHub trending repos
│   │   ├── hf_papers.py        # HF daily papers
│   │   ├── paperswithcode.py   # Benchmarks
│   │   └── arxiv.py            # Papers
│   └── analyzers/
│       ├── model_dataset.py    # Model-dataset relationships
│       ├── trend.py            # Growth trend analysis
│       └── opportunities.py    # Business opportunity detection
├── tests/                      # Test suite (85 tests)
├── data/                       # Runtime data (gitignored)
└── config.yaml
```

## Output Example

### Console Output

```
============================================================
  AI Dataset Radar v2 - Business Intelligence System
============================================================

Fetching data from sources...
  Hugging Face datasets: 50 found
  GitHub repos: 30 found (5 dataset-related)
  HF Daily Papers: 50 found (36 dataset-related)

Domain Classification
  robotics: 8 items
  rlhf: 12 items
  multimodal: 15 items

Business Opportunity Analysis
  Data factories detected: 2
  Papers with annotation signals: 18
  Active tracked organizations: 4
```

### Business Intelligence Report (`data/intel_report_*.md`)

```markdown
# AI Dataset Radar 商业情报周报

## 🔥 增长最快的数据集 (Top 10)
| 排名 | 数据集 | 7天增长率 | 当前下载 | 领域标签 |
|------|--------|-----------|----------|----------|
| 1 | lerobot-data | 156.3% | 12,450 | robotics |

## 🏭 数据工厂动态
| 作者/机构 | 本周发布数量 | 数据集列表 | 可能归属 |

## 📄 有标注需求的论文
| 论文 | 检测到的信号 | 机构 | arXiv链接 |

## 🏢 大厂动态
### GOOGLE
- Datasets: gemini-robotics-data
- Papers: Scaling Robot Learning...
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
- [x] Phase 2: Business intelligence (domain filtering, opportunity detection, org tracking)
- [ ] Phase 3: Deep analysis (PDF parsing, GitHub code analysis, LLM summarization)
- [ ] Phase 4: Automation (scheduled runs, alerts, competitor monitoring)

## License

MIT License

---

<a name="中文文档"></a>
# AI Dataset Radar

[English](#ai-dataset-radar) | **中文**

> 商业情报系统 - AI 数据集发现与商机检测

为数据标注公司打造的商业情报工具，用于发现有价值的数据配方、检测标注商机、追踪行业趋势、监控竞争对手动态。

## 功能特性

### 核心能力

| 功能 | 说明 |
|------|------|
| **早期信号检测** | GitHub Trending + HF Daily Papers 实现最早发现 |
| **增长趋势分析** | 识别破圈数据集（0 → 1000+ 下载） |
| **领域聚焦过滤** | 按机器人、RLHF、多模态等领域筛选 |
| **商机检测** | 检测数据工厂和论文中的标注需求信号 |
| **机构追踪** | 监控主要 AI 实验室的活动 |
| **商业情报报告** | 周报形式输出可执行洞察 |

### 数据源

| 来源 | 发现延迟 | 内容 |
|------|----------|------|
| **GitHub Trending** | Day 1-3 | 新数据集仓库 |
| **HF Daily Papers** | Day 3-7 | 热门 AI 论文 |
| Hugging Face Hub | Day 7+ | 数据集和模型 |
| Papers with Code | Day 14+ | 基准测试和 SOTA |
| arXiv | Day 14+ | 研究论文 |

### 商业情报信号

| 信号 | 说明 |
|------|------|
| **数据工厂** | 7 天内发布 3+ 个数据集的作者 |
| **标注需求信号** | 论文中提到 "human annotation"、"crowdsourced" 等 |
| **破圈数据集** | 下载量从 0 增长到 1000+ |
| **机构动态** | 来自字节、Google、OpenAI 等的数据集/论文 |

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
# 完整分析（抓取 + 趋势 + 商机检测）
python src/main.py

# 快速模式（仅抓取）
python src/main.py --quick

# 商业情报选项
python src/main.py --focus robotics    # 按领域过滤
python src/main.py --growth-only       # 只看有增长的
python src/main.py --min-growth 0.5    # 最低 50% 增长率
python src/main.py --opportunities     # 聚焦商业信号

# 跳过特定分析
python src/main.py --no-models         # 跳过模型-数据集分析
python src/main.py --no-trends         # 跳过趋势分析
python src/main.py --no-opportunities  # 跳过商机检测
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
    token: ${GITHUB_TOKEN}
  hf_papers:
    enabled: true
    limit: 50

# 领域聚焦配置
focus_areas:
  robotics:
    enabled: true
    keywords: [robotics, manipulation, embodied, gripper]
    hf_tags: [task_categories:robotics]
  rlhf:
    enabled: true
    keywords: [preference, human feedback, RLHF, DPO]
  multimodal:
    enabled: true
    keywords: [vision-language, VLM, multimodal]

# 机构追踪
tracked_orgs:
  bytedance: [ByteDance, 字节, TikTok]
  google: [Google, DeepMind]
  openai: [OpenAI]

# 商机检测设置
opportunities:
  annotation_signals:
    - human annotation
    - crowdsourced
    - data collection
  data_factory:
    min_datasets: 3
    days: 7

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
│   ├── filters.py              # 领域和机构过滤器
│   ├── notifiers.py            # 报告生成器
│   ├── scrapers/
│   │   ├── huggingface.py      # HF 数据集 + 模型
│   │   ├── github.py           # GitHub 热门仓库
│   │   ├── hf_papers.py        # HF 每日论文
│   │   ├── paperswithcode.py   # 基准测试
│   │   └── arxiv.py            # 论文
│   └── analyzers/
│       ├── model_dataset.py    # 模型-数据集关联
│       ├── trend.py            # 增长趋势分析
│       └── opportunities.py    # 商机检测
├── tests/                      # 测试套件 (85 个测试)
├── data/                       # 运行时数据 (已 gitignore)
└── config.yaml
```

## 输出示例

### 控制台输出

```
============================================================
  AI Dataset Radar v2 - Business Intelligence System
============================================================

Fetching data from sources...
  Hugging Face datasets: 50 found
  GitHub repos: 30 found (5 dataset-related)
  HF Daily Papers: 50 found (36 dataset-related)

Domain Classification
  robotics: 8 items
  rlhf: 12 items
  multimodal: 15 items

Business Opportunity Analysis
  Data factories detected: 2
  Papers with annotation signals: 18
  Active tracked organizations: 4
```

### 商业情报报告 (`data/intel_report_*.md`)

```markdown
# AI Dataset Radar 商业情报周报

## 🔥 增长最快的数据集 (Top 10)
| 排名 | 数据集 | 7天增长率 | 当前下载 | 领域标签 |
|------|--------|-----------|----------|----------|
| 1 | lerobot-data | 156.3% | 12,450 | robotics |

## 🏭 数据工厂动态
| 作者/机构 | 本周发布数量 | 数据集列表 | 可能归属 |

## 📄 有标注需求的论文
| 论文 | 检测到的信号 | 机构 | arXiv链接 |

## 🏢 大厂动态
### GOOGLE
- 相关数据集: gemini-robotics-data
- 相关论文: Scaling Robot Learning...
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
- [x] 阶段 2：商业情报（领域过滤、商机检测、机构追踪）
- [ ] 阶段 3：深度分析（论文 PDF 解析、GitHub 代码分析、LLM 配方总结）
- [ ] 阶段 4：自动化（定时运行、告警、竞品监控）

## 许可证

MIT License
