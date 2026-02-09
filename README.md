<div align="center">

# AI Dataset Radar

**面向 AI Agent 的训练数据竞争情报系统**
**Competitive intelligence system for AI training datasets**

[![CI](https://github.com/liuxiaotong/ai-dataset-radar/actions/workflows/ci.yml/badge.svg)](https://github.com/liuxiaotong/ai-dataset-radar/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-855_passed-brightgreen.svg)](#开发路线)
[![Agent Ready](https://img.shields.io/badge/Agent-Ready-orange.svg)](docs/agent-integration.md)
[![MCP](https://img.shields.io/badge/MCP-16_Tools-purple.svg)](docs/mcp.md)

[快速开始](#快速开始) · [使用方式](#使用方式) · [数据源](#数据源) · [生态](#生态) · [文档](docs/)

</div>

---

## 亮点

- **全源覆盖** — 86 HF orgs、50 GitHub orgs、71 博客、125 X 账户、5 Reddit 社区、arXiv 5 领域
- **智能体原生** — MCP 16 工具 + REST API + Function Calling + Claude Code 7 Skills
- **高性能异步** — aiohttp + asyncio.gather 全链路并发，500+ 请求同时执行
- **竞品分析** — 竞品矩阵、数据集谱系、组织关系图谱三维交叉分析
- **可视化仪表盘** — 11 Tab 面板 + Chart.js 趋势图 + 全局搜索
- **双格式输出** — JSON (Agent) + Markdown (人类) + AI 分析报告 (决策层)
- **一键 Recipe** — `--recipe` 自动评分选 Top N 数据集，调用 DataRecipe 深度分析

---

## 架构

```mermaid
flowchart TD
    subgraph S[" 6 数据源"]
        direction LR
        S1["HuggingFace 86 orgs"] ~~~ S2["GitHub 50 orgs"] ~~~ S3["博客 71 源"]
        S4["论文 arXiv+HF"] ~~~ S5["X 125 账户"] ~~~ S6["Reddit 5 社区"]
    end

    S --> T["Trackers — aiohttp 异步并发采集"]
    T --> A["Analyzers — 分类 · 趋势 · 竞品矩阵 · 谱系 · 组织图谱"]

    subgraph O[" 输出"]
        direction LR
        O1["JSON 结构化"] ~~~ O2["Markdown 报告"] ~~~ O3["AI Insights"]
    end

    A --> O

    subgraph I[" Agent 接口"]
        direction LR
        I1["REST API 18 端点"] ~~~ I2["MCP 16 工具"] ~~~ I3["Skills 7 命令"] ~~~ I4["Dashboard 11 Tab"]
    end

    O --> I
```

---

## 快速开始

```bash
git clone https://github.com/liuxiaotong/ai-dataset-radar.git
cd ai-dataset-radar
pip install -r requirements.txt && playwright install chromium
cp .env.example .env  # 编辑填入 Token（GITHUB_TOKEN / ANTHROPIC_API_KEY 等）

# 基础扫描（自动生成 AI 分析报告）
python src/main_intel.py --days 7

# 扫描 + DataRecipe 深度分析
python src/main_intel.py --days 7 --recipe

# Docker
docker compose run scan
```

**产出文件（按日期子目录）：**

```
data/reports/2026-02-08/
├── intel_report_*.json                # 结构化数据 (Agent)
├── intel_report_*.md                  # 原始报告 (人类)
├── intel_report_*_insights_prompt.md  # 分析提示 (LLM 输入)
├── intel_report_*_insights.md         # AI 分析报告 (决策层)
├── intel_report_*_changes.md          # 日报变化追踪
└── recipe/                            # DataRecipe 分析 (--recipe)
```

> 环境变量、RSSHub 配置、Docker 部署、调度设置详见 `.env.example` 和 [系统架构](docs/architecture.md)。

---

## 使用方式

### CLI

```bash
python src/main_intel.py --days 7                  # 基础扫描
python src/main_intel.py --days 7 --recipe          # + DataRecipe
python src/main_intel.py --days 7 --no-insights     # 跳过 AI 分析
python src/main_intel.py --days 7 --api-insights    # 显式调用 LLM API
```

| 环境 | 行为 |
|------|------|
| 默认 | 保存 prompt 文件，由 Claude Code 环境 LLM 分析 |
| `--api-insights` | 调用 LLM API（Anthropic/Kimi/DeepSeek 等）生成 `_insights.md` |
| `--no-insights` | 跳过 insights |

### REST API + Dashboard

```bash
python agent/api.py
# → http://localhost:8080/dashboard（Web 仪表盘）
# → http://localhost:8080/docs（API 文档）
```

<details>
<summary><b>Dashboard 预览（11 Tab 面板）</b></summary>

![Dashboard Overview](docs/images/dashboard-overview.png)

> 启动 `python agent/api.py` 后访问 `http://localhost:8080/dashboard`。包含概览、数据集、GitHub、论文、博客、Reddit、竞品矩阵、谱系、组织图谱、搜索、趋势 11 个面板。

</details>

核心端点：

| 类别 | 端点 |
|------|------|
| 数据查询 | `/datasets` · `/github` · `/papers` · `/blogs` · `/reddit` |
| 分析 | `/matrix` · `/lineage` · `/org-graph` · `/trends` · `/search` |
| 操作 | `/scan` · `/summary` · `/config` · `/schema` · `/tools` |

> 完整端点列表、代码示例（OpenAI / Anthropic / LangChain）见 [Agent 集成文档](docs/agent-integration.md)。

### MCP Server

```json
{
  "mcpServers": {
    "radar": {
      "command": "uv",
      "args": ["--directory", "/path/to/ai-dataset-radar", "run", "python", "mcp_server/server.py"]
    }
  }
}
```

> 16 个工具（scan/search/diff/trend/history/reddit/matrix/lineage/org-graph 等）及配置详情见 [MCP 文档](docs/mcp.md)。

### Claude Code Skills

在 Claude Code 中输入 `/` 即可调用，覆盖完整的竞争情报工作流：

| 命令 | 用途 | 类型 | 是否联网 |
|------|------|------|----------|
| `/scan` | 运行扫描 + 自动生成 AI 分析报告 | 采集 | 是 |
| `/brief` | 快速情报简报（5 条发现 + 行动建议） | 阅读 | 否 |
| `/search 关键词` | 跨 6 源搜索（数据集/GitHub/论文/博客/X/Reddit） | 查询 | 否 |
| `/diff` | 对比两次报告（新增/消失/变化） | 对比 | 否 |
| `/deep-dive 目标` | 组织/数据集/分类深度分析 | 分析 | 否 |
| `/recipe 数据集ID` | DataRecipe 逆向分析（成本/Schema/难度） | 深潜 | 是 |
| `/radar` | 通用情报助手（路由到其他 Skill） | 入口 | — |

**典型工作流：**

```bash
/scan --days 7 --recipe   # 1. 每周采集
/brief                    # 2. 晨会快速浏览
/search RLHF              # 3. 按主题搜索
/deep-dive NVIDIA         # 4. 聚焦某组织
/recipe allenai/Dolci     # 5. 深入某数据集
/diff                     # 6. 周对比变化
```

**设计原则：**

- **环境 LLM 接管**：`ANTHROPIC_API_KEY` 未设置时，`/scan` 让 Claude Code 自身作为分析引擎
- **纯本地读取**：`/brief`、`/search`、`/diff`、`/deep-dive` 不触发网络请求
- **交叉引用**：每个 Skill 的输出中推荐相关的后续 Skill

---

## 数据源

| 来源 | 数量 | 覆盖 |
|------|-----:|------|
| **HuggingFace** | 86 orgs | 67 Labs + 27 供应商（含机器人、欧洲、亚太） |
| **博客** | 71 源 | 实验室 + 研究者 + 独立博客 + 数据供应商 |
| **GitHub** | 50 orgs | AI Labs + 中国开源 + 机器人 + 数据供应商 |
| **论文** | 2 源 | arXiv (cs.CL/AI/LG/CV/RO) + HF Papers |
| **X/Twitter** | 125 账户 | 13 类别，CEO/Leaders + 研究者 + 机器人 |
| **Reddit** | 5 社区 | MachineLearning、LocalLLaMA、dataset、deeplearning、LanguageTechnology |

> 供应商分类、X 账户明细、数据集分类体系见 [数据源文档](docs/data-sources.md)。
> 输出 JSON Schema 见 [输出规范](docs/schema.md)。

---

## 生态

```mermaid
graph LR
    Radar["🔍 Radar<br/>情报发现"] -->|--recipe| Recipe["📋 Recipe<br/>逆向分析"]
    Recipe --> Synth["🔄 Synth<br/>数据合成"]
    Recipe --> Label["🏷️ Label<br/>数据标注"]
    Synth --> Check["✅ Check<br/>数据质检"]
    Label --> Check
    Check --> Audit["🔬 Audit<br/>模型审计"]
    Audit --> Hub["🎯 Hub<br/>编排层"]
    Hub --> Sandbox["📦 Sandbox<br/>执行沙箱"]
    Sandbox --> Recorder["📹 Recorder<br/>轨迹录制"]
    Recorder --> Reward["⭐ Reward<br/>过程打分"]
    style Radar fill:#0969da,color:#fff,stroke:#0969da
```

| 层 | 项目 | PyPI&nbsp;包 | 说明 | 仓库 |
|---|---|---|---|---|
| 情报 | **AI&nbsp;Dataset&nbsp;Radar** | knowlyr-radar | 竞争情报、趋势分析 | You&nbsp;are&nbsp;here |
| 分析 | **DataRecipe** | knowlyr-datarecipe | 逆向分析、Schema提取、成本估算 | [GitHub](https://github.com/liuxiaotong/data-recipe) |
| 生产 | **DataSynth** | knowlyr-datasynth | LLM批量合成、种子数据扩充 | [GitHub](https://github.com/liuxiaotong/data-synth) |
| 生产 | **DataLabel** | knowlyr-datalabel | 轻量标注、多标注员合并 | [GitHub](https://github.com/liuxiaotong/data-label) |
| 质检 | **DataCheck** | knowlyr-datacheck | 规则验证、重复检测、分布分析 | [GitHub](https://github.com/liuxiaotong/data-check) |
| 质检 | **ModelAudit** | knowlyr-modelaudit | 蒸馏检测、模型指纹、身份验证 | [GitHub](https://github.com/liuxiaotong/model-audit) |
| Agent | **knowlyr-agent** | knowlyr-sandbox/recorder/reward/hub | 沙箱+录制+Reward+编排 | [GitHub](https://github.com/liuxiaotong/knowlyr-agent) |

> DataRecipe 联动详情（评分公式、输出结构、MCP 双服务配置）见 [DataRecipe 文档](docs/datarecipe.md)。

---

## 开发路线

| 能力 | 说明 | 解锁场景 |
|------|------|----------|
| **异常检测与告警** | 阈值规则 + 统计异常检测，触发 Webhook/邮件/飞书推送 | 从"手动查看"变为"主动通知"，情报系统的本质闭环 |
| **增量扫描** | 记录上次扫描水位线，仅抓取增量数据 | 扫描频率从日级提升至小时级，API 调用量降一个量级 |
| **时序持久化** | 每日快照写入 SQLite，支持跨月趋势查询 | 长周期趋势分析、季度报告、组织活跃度变化曲线 |
| **推送分发** | 周报/日报自动推送到 Slack、飞书、邮件、Webhook | 团队被动消费情报，无需主动登录查看 |
| **交互式图谱** | D3.js force-directed 组织关系图 + Sankey 谱系图 | 可视化发现隐藏的组织协作模式和数据集派生链 |
| **自定义监控规则** | 用户自建关键词/组织/阈值过滤器，YAML 或 Web UI 配置 | 不同团队关注不同赛道，无需改代码 |

> 已完成里程碑见 [CHANGELOG.md](CHANGELOG.md)。

## 许可证

[MIT](LICENSE)

---

<div align="center">
<sub><a href="https://github.com/liuxiaotong">knowlyr</a> 数据工程生态 · 训练数据竞争情报</sub>
</div>
