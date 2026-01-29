# AI Dataset Radar

A multi-signal intelligence system for discovering high-value AI datasets through citation analysis, SOTA model tracking, and cross-platform data aggregation.

一个通过引用分析、SOTA 模型追踪和跨平台数据聚合发现高价值 AI 数据集的多信号情报系统。

---

## Overview | 概述

AI Dataset Radar addresses the challenge of identifying valuable datasets in the rapidly evolving AI research landscape. By aggregating signals from multiple authoritative sources—Semantic Scholar citations, HuggingFace model cards, Papers with Code benchmarks—the system computes a composite value score that reflects a dataset's research impact and adoption trajectory.

AI Dataset Radar 解决了在快速发展的 AI 研究领域中识别有价值数据集的挑战。通过聚合来自多个权威来源的信号——Semantic Scholar 引用、HuggingFace 模型卡、Papers with Code 基准测试——系统计算出反映数据集研究影响力和采用轨迹的综合价值评分。

## Value Scoring Methodology | 价值评分方法

The system employs a weighted multi-factor scoring model (0-100):

系统采用加权多因子评分模型 (0-100)：

| Signal | Weight | Criterion |
|--------|--------|-----------|
| SOTA Model Usage | +30 | Referenced by state-of-the-art models |
| Citation Velocity | +20 | Monthly citation growth ≥ 10 |
| Model Adoption | +20 | Used by ≥ 3 HuggingFace models |
| Institution Prestige | +15 | Origin: Google, Stanford, OpenAI, etc. |
| Reproducibility | +10 | Associated paper and code available |
| Scale | +5 | Dataset size > 10GB |

| 信号 | 权重 | 标准 |
|------|------|------|
| SOTA 模型使用 | +30 | 被 SOTA 模型引用 |
| 引用增速 | +20 | 月引用增长 ≥ 10 |
| 模型采用度 | +20 | 被 ≥ 3 个 HuggingFace 模型使用 |
| 机构声誉 | +15 | 来源：Google、Stanford、OpenAI 等 |
| 可复现性 | +10 | 有配套论文和代码 |
| 规模 | +5 | 数据集大小 > 10GB |

## Data Sources | 数据来源

| Source | Latency | Content |
|--------|---------|---------|
| Semantic Scholar API | Real-time | Citation metrics and growth rates |
| HuggingFace Model Cards | 1-3 days | Model-dataset relationships |
| Papers with Code | 1-7 days | SOTA benchmarks and evaluations |
| GitHub Trending | 1-3 days | Emerging dataset repositories |
| HuggingFace Daily Papers | 3-7 days | Curated research papers |
| arXiv | 7-14 days | Preprint publications |

| 来源 | 延迟 | 内容 |
|------|------|------|
| Semantic Scholar API | 实时 | 引用指标和增长率 |
| HuggingFace 模型卡 | 1-3 天 | 模型-数据集关系 |
| Papers with Code | 1-7 天 | SOTA 基准和评测 |
| GitHub Trending | 1-3 天 | 新兴数据集仓库 |
| HuggingFace Daily Papers | 3-7 天 | 精选研究论文 |
| arXiv | 7-14 天 | 预印本发表 |

## Installation | 安装

```bash
git clone https://github.com/liuxiaotong/ai-dataset-radar.git
cd ai-dataset-radar

python -m venv venv
source venv/bin/activate  # Linux/macOS

pip install -r requirements.txt
```

## Usage | 使用方法

### Basic Commands | 基本命令

```bash
# Standard analysis pipeline
# 标准分析流程
python src/main.py

# Value-focused analysis (v3)
# 价值导向分析 (v3)
python src/main.py --value-analysis

# Filter by minimum score threshold
# 按最低评分阈值过滤
python src/main.py --value-analysis --min-score 60

# Restrict to top-tier institutions
# 限定顶级机构
python src/main.py --value-analysis --top-institutions

# Domain-specific analysis
# 特定领域分析
python src/main.py --value-analysis --domain robotics
python src/main.py --focus rlhf
```

### Analysis Options | 分析选项

```bash
--value-analysis       # Enable multi-signal value scoring | 启用多信号价值评分
--min-score N          # Minimum score threshold (0-100) | 最低评分阈值
--domain DOMAIN        # Filter: robotics, nlp, vision, code | 领域过滤
--top-institutions     # Top-tier institutions only | 仅顶级机构
--growth-only          # Positive growth trend only | 仅正增长趋势
--min-growth N         # Minimum growth rate (e.g., 0.5 = 50%) | 最低增长率

--no-value-analysis    # Skip value scoring | 跳过价值评分
--no-trends            # Skip trend analysis | 跳过趋势分析
--no-models            # Skip model-dataset analysis | 跳过模型-数据集分析
--quick                # Data collection only | 仅数据采集
```

## Configuration | 配置

```yaml
# config.yaml
database:
  path: data/radar.db

sources:
  semantic_scholar:
    enabled: true
    min_citations: 20
    min_monthly_growth: 10
  huggingface:
    enabled: true
    limit: 50

value_analysis:
  min_score: 0
  model_cards:
    enabled: true
    min_downloads: 1000
    min_usage: 3
  sota:
    enabled: true
    areas: [robotics, code-generation, question-answering]

focus_areas:
  robotics:
    enabled: true
    keywords: [robotics, manipulation, embodied, gripper]
    hf_tags: [task_categories:robotics]
```

## Architecture | 架构

```
ai-dataset-radar/
├── src/
│   ├── main.py                    # Entry point | 主入口
│   ├── db.py                      # SQLite persistence | 数据持久化
│   ├── report.py                  # Report generation | 报告生成
│   ├── scrapers/
│   │   ├── semantic_scholar.py    # Citation tracking | 引用追踪
│   │   ├── pwc_sota.py            # SOTA associations | SOTA 关联
│   │   ├── huggingface.py         # HF datasets/models | HF 数据集/模型
│   │   ├── github.py              # Trending repos | 热门仓库
│   │   └── arxiv.py               # Paper retrieval | 论文检索
│   └── analyzers/
│       ├── value_scorer.py        # Scoring system | 评分系统
│       ├── model_card_analyzer.py # Model card parsing | 模型卡解析
│       ├── trend.py               # Growth analysis | 增长分析
│       └── opportunities.py       # Signal detection | 信号检测
├── tests/                         # Test suite (118 tests) | 测试套件
└── config.yaml
```

## Output | 输出

### Console Output | 控制台输出

```
============================================================
  AI Dataset Radar v3 - High-Value Dataset Discovery
============================================================

Fetching Semantic Scholar citations... 85 papers
Analyzing model cards... 500 models, 42 datasets with ≥3 uses
Analyzing SOTA associations... 28 datasets linked

Value Analysis Summary:
  High-value (≥60):    15 datasets
  Medium-value (40-59): 23 datasets
  Total analyzed:       89 datasets
```

### Value Report | 价值报告

Generated at `data/value_report_YYYY-MM-DD.md`:

```markdown
# 高价值数据集周报

## Top 10 High-Value Datasets
| Rank | Dataset | Score | SOTA | Citations/mo | Domain | Institution |
|------|---------|-------|------|--------------|--------|-------------|
| 1 | OpenWebText | 85 | 12 | 45.2 | NLP | EleutherAI |
| 2 | LAION-5B | 78 | 8 | 32.1 | Vision | LAION |
```

## Development | 开发

```bash
# Run test suite | 运行测试
python -m pytest tests/ -v

# Run with coverage | 覆盖率测试
python -m pytest tests/ --cov=src
```

## Roadmap | 路线图

- [x] Phase 1: Core infrastructure (database, scrapers, trend analysis)
- [x] Phase 2: Multi-source aggregation (GitHub, HF Papers, organization tracking)
- [x] Phase 3: Value scoring system (citations, SOTA, model cards)
- [ ] Phase 4: Deep analysis (PDF extraction, code analysis, LLM summarization)
- [ ] Phase 5: Automation (scheduled execution, alerting, monitoring)

---

- [x] 阶段 1：核心基础设施（数据库、爬虫、趋势分析）
- [x] 阶段 2：多源聚合（GitHub、HF 论文、机构追踪）
- [x] 阶段 3：价值评分系统（引用、SOTA、模型卡）
- [ ] 阶段 4：深度分析（PDF 提取、代码分析、LLM 摘要）
- [ ] 阶段 5：自动化（定时执行、告警、监控）

## License | 许可证

MIT License
