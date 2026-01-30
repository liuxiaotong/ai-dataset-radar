# AI Dataset Radar: A Competitive Intelligence System for AI Training Data Discovery

# AI Dataset Radar：面向人工智能训练数据发现的竞争情报系统

---

## Abstract | 摘要

**English:**
We present AI Dataset Radar, a competitive intelligence system designed to monitor and analyze the AI training data ecosystem. The system addresses a critical need in the data annotation industry: systematic tracking of dataset publications from leading AI laboratories and data vendors. By aggregating signals from multiple authoritative sources—including HuggingFace, arXiv, and GitHub—the system enables stakeholders to identify emerging data requirements, monitor competitor activities, and discover high-value dataset opportunities. Our multi-signal approach combines organization tracking, data type classification, and quality filtering to produce actionable intelligence reports. Experimental results demonstrate the system's capability to effectively filter noise and surface relevant datasets across seven priority categories: preference learning, reward modeling, supervised fine-tuning, code generation, agent training, embodied AI, and safety alignment.

**中文：**
本文介绍 AI Dataset Radar，一个面向人工智能训练数据生态系统监控与分析的竞争情报系统。该系统解决了数据标注行业的关键需求：对领先 AI 实验室和数据供应商发布的数据集进行系统化追踪。通过聚合来自 HuggingFace、arXiv 和 GitHub 等多个权威来源的信号，系统帮助利益相关者识别新兴数据需求、监控竞争对手动态，并发现高价值数据集机会。我们的多信号方法结合了组织追踪、数据类型分类和质量过滤，以生成可操作的情报报告。实验结果表明，该系统能够有效过滤噪声，并在七个优先类别中呈现相关数据集：偏好学习、奖励建模、监督微调、代码生成、智能体训练、具身智能和安全对齐。

---

## 1. Introduction | 引言

### 1.1 Background | 研究背景

The rapid advancement of large language models (LLMs) has created unprecedented demand for high-quality training data. Post-training techniques—including Supervised Fine-Tuning (SFT), Reinforcement Learning from Human Feedback (RLHF), and Direct Preference Optimization (DPO)—require carefully curated datasets that are increasingly becoming strategic assets for AI organizations.

大型语言模型（LLMs）的快速发展对高质量训练数据产生了前所未有的需求。后训练技术——包括监督微调（SFT）、基于人类反馈的强化学习（RLHF）和直接偏好优化（DPO）——需要精心策划的数据集，这些数据集日益成为 AI 组织的战略资产。

### 1.2 Problem Statement | 问题陈述

Data annotation companies face significant challenges in:
1. **Information Asymmetry**: Limited visibility into what datasets leading AI labs are producing and consuming
2. **Market Intelligence**: Difficulty tracking competitor activities in the data vendor space
3. **Technology Trends**: Identifying emerging data requirements before they become mainstream

数据标注公司面临以下重大挑战：
1. **信息不对称**：对领先 AI 实验室正在生产和消费的数据集缺乏可见性
2. **市场情报**：难以追踪数据供应商领域的竞争对手活动
3. **技术趋势**：在数据需求成为主流之前识别新兴需求

### 1.3 Contributions | 主要贡献

This work makes the following contributions:
- A systematic framework for monitoring AI training data publications across multiple platforms
- A hierarchical classification system for post-training data types
- Quality filtering mechanisms to reduce noise from low-value dataset publications
- An open-source implementation with comprehensive test coverage

本工作的主要贡献包括：
- 跨多平台监控 AI 训练数据发布的系统框架
- 后训练数据类型的层次化分类系统
- 降低低价值数据集发布噪声的质量过滤机制
- 具有全面测试覆盖的开源实现

---

## 2. Related Work | 相关工作

### 2.1 Dataset Discovery Platforms | 数据集发现平台

Existing platforms such as HuggingFace Hub, Papers with Code, and Kaggle provide dataset discovery capabilities but lack competitive intelligence features tailored to the data annotation industry.

现有平台如 HuggingFace Hub、Papers with Code 和 Kaggle 提供数据集发现功能，但缺乏针对数据标注行业的竞争情报功能。

### 2.2 Research Trend Analysis | 研究趋势分析

Tools like Semantic Scholar and Google Scholar provide citation metrics but do not specifically track dataset-related publications or provide industry-specific insights.

Semantic Scholar 和 Google Scholar 等工具提供引用指标，但不专门追踪与数据集相关的出版物或提供行业特定洞察。

---

## 3. System Architecture | 系统架构

### 3.1 Overview | 系统概览

```
┌─────────────────────────────────────────────────────────────────┐
│                    AI Dataset Radar v4                          │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │ HuggingFace │  │   arXiv     │  │   GitHub    │  Data       │
│  │     API     │  │     API     │  │     API     │  Sources    │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘             │
│         │                │                │                     │
│         └────────────────┼────────────────┘                     │
│                          ▼                                      │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │                   Organization Tracker                     │ │
│  │  • Frontier Labs (OpenAI, Anthropic, Google, Meta)        │ │
│  │  • Emerging Labs (Mistral, Cohere, Together)              │ │
│  │  • Data Vendors (Scale AI, Surge AI, Argilla)             │ │
│  └───────────────────────────────────────────────────────────┘ │
│                          │                                      │
│                          ▼                                      │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │                  Data Type Classifier                      │ │
│  │  preference | reward_model | sft | code | agent | safety  │ │
│  └───────────────────────────────────────────────────────────┘ │
│                          │                                      │
│                          ▼                                      │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │                Intelligence Report Generator               │ │
│  └───────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Module Descriptions | 模块说明

| Module | Description | 模块说明 |
|--------|-------------|----------|
| `trackers/org_tracker.py` | Monitors specific organizations on HuggingFace | 监控特定组织在 HuggingFace 上的活动 |
| `analyzers/data_type_classifier.py` | Classifies datasets by training purpose | 按训练目的分类数据集 |
| `analyzers/quality_scorer.py` | Scores dataset quality (0-10 scale) | 评估数据集质量（0-10 分制） |
| `analyzers/author_filter.py` | Filters suspicious batch-upload accounts | 过滤可疑的批量上传账号 |
| `intel_report.py` | Generates structured intelligence reports | 生成结构化情报报告 |

### 3.3 Directory Structure | 目录结构

```
ai-dataset-radar/
├── src/
│   ├── main_intel.py              # Primary entry point | 主入口
│   ├── intel_report.py            # Report generation | 报告生成
│   ├── trackers/
│   │   └── org_tracker.py         # Organization monitoring | 组织监控
│   ├── analyzers/
│   │   ├── data_type_classifier.py
│   │   ├── quality_scorer.py
│   │   ├── author_filter.py
│   │   └── org_detector.py
│   └── scrapers/
│       ├── huggingface.py
│       ├── arxiv.py
│       └── github.py
├── tests/                         # Test suite (130 tests) | 测试套件
├── data/                          # Output directory | 输出目录
└── config.yaml                    # Configuration | 配置文件
```

---

## 4. Methodology | 方法论

### 4.1 Organization Tracking | 组织追踪

The system maintains a curated list of monitoring targets organized into three tiers:

系统维护一个分为三个层级的监控目标列表：

**Tier 1: Frontier Labs | 一线实验室**
- OpenAI, Anthropic, Google DeepMind, Meta AI, xAI

**Tier 2: Emerging Labs | 新兴实验室**
- Mistral AI, Cohere, AI21 Labs, Together AI, Databricks

**Tier 3: Data Vendors | 数据供应商**
- Scale AI, Surge AI, Appen, Sama, Argilla

### 4.2 Data Type Classification | 数据类型分类

We define seven priority categories aligned with post-training requirements:

我们定义了与后训练需求对齐的七个优先类别：

| Category | Keywords | Description |
|----------|----------|-------------|
| `preference` | RLHF, DPO, comparison, chosen/rejected | Human preference data for alignment |
| `reward_model` | reward, PPO, trajectory | Training data for reward models |
| `sft` | instruction, chat, dialogue | Supervised fine-tuning data |
| `code` | code, execution, sandbox | Code generation and execution |
| `agent` | tool use, function calling, web browsing | Agent training data |
| `embodied` | robot, simulation, manipulation | Embodied AI and robotics |
| `safety` | harmful, toxic, red team | Safety and alignment data |

| 类别 | 关键词 | 描述 |
|------|--------|------|
| `preference` | RLHF, DPO, 对比, chosen/rejected | 用于对齐的人类偏好数据 |
| `reward_model` | reward, PPO, trajectory | 奖励模型训练数据 |
| `sft` | instruction, chat, dialogue | 监督微调数据 |
| `code` | code, execution, sandbox | 代码生成与执行 |
| `agent` | tool use, function calling, web browsing | 智能体训练数据 |
| `embodied` | robot, simulation, manipulation | 具身智能与机器人 |
| `safety` | harmful, toxic, red team | 安全与对齐数据 |

### 4.3 Quality Filtering | 质量过滤

To address the noise problem from spam accounts, we implement a multi-factor quality scoring system:

为解决垃圾账号带来的噪声问题，我们实现了多因子质量评分系统：

```
Quality Score (0-10) = Σ weights × indicators

Indicators:
  - Description length ≥ 100 chars    (+2)
  - Downloads > 10                     (+1)
  - Downloads > 1000                   (+2)
  - Explicit license                   (+1)
  - Task tags defined                  (+1)
  - Associated paper                   (+2)
  - Known institution author           (+1)
```

---

## 5. Installation | 安装

### 5.1 Requirements | 环境要求

- Python ≥ 3.10
- Dependencies: `requests`, `pyyaml`, `beautifulsoup4`

### 5.2 Setup | 安装步骤

```bash
git clone https://github.com/liuxiaotong/ai-dataset-radar.git
cd ai-dataset-radar

python -m venv venv
source venv/bin/activate  # Linux/macOS
# or: venv\Scripts\activate  # Windows

pip install -r requirements.txt
```

---

## 6. Usage | 使用方法

### 6.1 Basic Execution | 基本执行

```bash
# Run competitive intelligence analysis
# 运行竞争情报分析
python src/main_intel.py

# Specify analysis period
# 指定分析周期
python src/main_intel.py --days 14

# Export raw data as JSON
# 导出原始数据为 JSON
python src/main_intel.py --json

# Skip specific components
# 跳过特定组件
python src/main_intel.py --no-labs      # Skip AI labs | 跳过 AI 实验室
python src/main_intel.py --no-vendors   # Skip vendors | 跳过供应商
python src/main_intel.py --no-papers    # Skip papers | 跳过论文
```

### 6.2 Configuration | 配置

The system is configured via `config.yaml`:

系统通过 `config.yaml` 进行配置：

```yaml
# Monitoring targets | 监控目标
watched_orgs:
  frontier_labs:
    openai:
      hf_ids: ["openai"]
      keywords: ["openai", "gpt"]
      priority: high

# Priority data types | 优先数据类型
priority_data_types:
  preference:
    keywords: [preference, RLHF, DPO, chosen, rejected]
    tags: [dpo, rlhf]
```

---

## 7. Output Format | 输出格式

### 7.1 Intelligence Report Structure | 情报报告结构

The system generates markdown reports with the following sections:

系统生成包含以下章节的 Markdown 报告：

```markdown
# AI 数据情报周报

## 📊 本周摘要
- 活跃 AI Labs: N 家
- 活跃数据供应商: N 家
- 高价值数据集: N 个

## 🔬 美国 AI Labs 动态
### Frontier Labs
| 机构 | 本周数据集 | 本周模型 |
|------|-----------|---------|

## 🏢 数据供应商动态（竞品监控）

## 📊 高价值数据集（按类型）
### 🎯 RLHF/DPO 偏好数据
### 💻 代码生成/执行
### 🤖 Agent/工具使用

## 📄 相关论文
```

---

## 8. Evaluation | 评估

### 8.1 Test Coverage | 测试覆盖

```bash
# Run test suite | 运行测试套件
python -m pytest tests/ -v

# Results: 130 passed, 2 skipped
```

### 8.2 Performance Metrics | 性能指标

| Metric | Value |
|--------|-------|
| Organizations tracked | 23 |
| Data types classified | 7 |
| Test cases | 130 |
| API rate limit handling | Exponential backoff |

---

## 9. Limitations and Future Work | 局限性与未来工作

### 9.1 Current Limitations | 当前局限

1. **API Dependencies**: Reliance on third-party APIs with rate limits
2. **Keyword-Based Classification**: May miss semantically similar but lexically different content
3. **English-Centric**: Primary focus on English-language publications

1. **API 依赖**：依赖有速率限制的第三方 API
2. **基于关键词的分类**：可能遗漏语义相似但词汇不同的内容
3. **以英语为中心**：主要关注英语出版物

### 9.2 Future Directions | 未来方向

- Integration of LLM-based semantic classification
- Real-time alerting for high-priority publications
- Historical trend analysis and forecasting
- Multi-language support

- 集成基于 LLM 的语义分类
- 高优先级发布的实时告警
- 历史趋势分析与预测
- 多语言支持

---

## 10. Conclusion | 结论

AI Dataset Radar provides a systematic approach to competitive intelligence in the AI training data space. By combining organization tracking, data type classification, and quality filtering, the system enables data annotation companies to make informed strategic decisions based on comprehensive market intelligence.

AI Dataset Radar 为 AI 训练数据领域的竞争情报提供了系统化方法。通过结合组织追踪、数据类型分类和质量过滤，该系统使数据标注公司能够基于全面的市场情报做出明智的战略决策。

---

## References | 参考文献

1. Ouyang, L., et al. (2022). Training language models to follow instructions with human feedback. *NeurIPS*.
2. Rafailov, R., et al. (2023). Direct Preference Optimization: Your Language Model is Secretly a Reward Model. *NeurIPS*.
3. Wang, Y., et al. (2023). Self-Instruct: Aligning Language Models with Self-Generated Instructions. *ACL*.

---

## License | 许可证

MIT License

## Citation | 引用

```bibtex
@software{ai_dataset_radar,
  title = {AI Dataset Radar: A Competitive Intelligence System for AI Training Data Discovery},
  author = {Liu, Xiaotong},
  year = {2026},
  url = {https://github.com/liuxiaotong/ai-dataset-radar}
}
```
