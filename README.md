<div align="center">

# AI Dataset Radar

**面向 AI Agent 的训练数据竞争情报系统**

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Agent Ready](https://img.shields.io/badge/Agent-Ready-orange.svg)](#agent-集成)
[![MCP](https://img.shields.io/badge/MCP-7_Tools-purple.svg)](#mcp-server)

[快速开始](#快速开始) · [Agent 集成](#agent-集成) · [数据源](#数据源) · [输出规范](#输出规范) · [配置](#配置)

</div>

---

监控 30+ 机构的训练数据动态，提供结构化输出供智能体消费。支持 Function Calling、MCP、REST API 多种接入方式。

## 系统概述

```
多源采集 → 智能分类 → 结构化输出 → 智能体消费
```

### 设计目标

| 目标 | 实现方式 |
|------|----------|
| **智能体友好** | 标准化 JSON Schema、Function Calling 工具定义 |
| **多框架兼容** | HTTP API (LangChain)、MCP (Claude)、原生 SDK |
| **开箱即用** | 预置 System Prompt、完整类型定义 |
| **人机兼顾** | 同时输出 Markdown (人类) 与 JSON (智能体) |

### 适用场景

| 使用者 | 接入方式 | 应用场景 |
|--------|----------|----------|
| 🤖 **LLM Agent** | Function Calling | 数据集发现、竞品分析自动化 |
| 🦜 **LangChain** | HTTP API | 构建数据情报 Agent |
| 💬 **Claude Desktop** | MCP Server | 自然语言交互式查询 |
| 🔧 **自定义系统** | REST API | 集成至现有工作流 |
| 👔 **决策者** | Markdown 报告 | 周报阅读、趋势把握 |

### 输出产物

| 产物 | 路径 | 消费者 |
|------|------|--------|
| 情报报告 (JSON) | `data/reports/intel_report_*.json` | AI Agent |
| 情报报告 (MD) | `data/reports/intel_report_*.md` | 人类 |
| 工具定义 | `agent/tools.json` | Function Calling |
| 输出规范 | `agent/schema.json` | 数据验证 |
| 系统提示词 | `agent/prompts.md` | Agent 配置 |

---

## 安装部署

```bash
git clone https://github.com/liuxiaotong/ai-dataset-radar.git
cd ai-dataset-radar
pip install -r requirements.txt

# 可选：Agent API 服务
pip install fastapi uvicorn
```

## 快速开始

### 执行扫描

```bash
python src/main_intel.py --days 7
```

### 启动 API 服务

```bash
uvicorn agent.api:app --port 8080
# 接口文档: http://localhost:8080/docs
```

### 智能体调用

```python
import requests
response = requests.get("http://localhost:8080/datasets?category=sft")
datasets = response.json()
```

---

## Agent 集成

### 接入方式

| 方式 | 适用框架 | 配置文件 |
|------|----------|----------|
| **HTTP API** | LangChain, AutoGPT, Dify | `agent/api.py` |
| **Function Calling** | OpenAI GPT, Claude API | `agent/tools.json` |
| **MCP Server** | Claude Desktop | `mcp_server/server.py` |
| **JSON Schema** | 类型生成、数据验证 | `agent/schema.json` |

### HTTP API 端点

```bash
uvicorn agent.api:app --port 8080
```

| 端点 | 方法 | 功能 |
|------|------|------|
| `/summary` | GET | 获取最新报告摘要 |
| `/datasets` | GET | 数据集列表 (支持 category 筛选) |
| `/github` | GET | GitHub 仓库活动 (支持 relevance 筛选) |
| `/papers` | GET | 论文列表 (支持 dataset_only 筛选) |
| `/blogs` | GET | 博客文章 |
| `/scan` | POST | 执行新扫描 |
| `/schema` | GET | 输出规范 |
| `/tools` | GET | 工具定义 |

### OpenAI Function Calling

```python
import json, openai

with open("agent/tools.json") as f:
    tools = json.load(f)["tools"]

response = openai.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "查询最新的偏好学习数据集"}],
    tools=[{"type": "function", "function": t} for t in tools]
)
```

### Anthropic Tool Use

```python
import json, anthropic

with open("agent/tools.json") as f:
    tools = json.load(f)["tools"]

response = anthropic.messages.create(
    model="claude-sonnet-4-20250514",
    tools=[{"name": t["name"], "description": t["description"],
            "input_schema": t["parameters"]} for t in tools],
    messages=[{"role": "user", "content": "查询合成数据集"}]
)
```

### LangChain 集成

```python
from langchain.tools import Tool
import requests

tools = [
    Tool(
        name="radar_datasets",
        func=lambda cat: requests.get(f"http://localhost:8080/datasets?category={cat}").json(),
        description="按类别查询数据集: sft|preference|synthetic|agent|code"
    ),
]
```

### 预置 System Prompt

`agent/prompts.md` 提供四类预置提示词：

| 角色 | 用途 |
|------|------|
| Dataset Intelligence Analyst | 数据集情报分析 |
| Competitive Intelligence Agent | 竞争情报追踪 |
| Dataset Discovery Assistant | 数据集发现与推荐 |
| Research Trend Monitor | 研究趋势监控 |

---

## MCP Server

配置 Claude Desktop (`~/Library/Application Support/Claude/claude_desktop_config.json`)：

```json
{
  "mcpServers": {
    "ai-dataset-radar": {
      "command": "/path/to/.venv/bin/python",
      "args": ["/path/to/mcp_server/server.py"]
    }
  }
}
```

| 工具 | 功能 |
|------|------|
| `radar_scan` | 执行完整扫描 |
| `radar_summary` | 获取报告摘要 |
| `radar_datasets` | 按类别查询数据集 |
| `radar_github` | 查询 GitHub 活动 |
| `radar_papers` | 查询论文 |
| `radar_blogs` | 查询博客文章 |
| `radar_config` | 获取监控配置 |

---

## 数据源

### 监控范围

| 来源 | 覆盖范围 |
|------|----------|
| **HuggingFace** | 30+ 机构：OpenAI, DeepMind, Meta, Anthropic, Qwen, DeepSeek 等 |
| **博客** | 17 来源：OpenAI, Anthropic, Google AI, Mistral, Scale AI, Qwen 等 |
| **GitHub** | 15+ 组织：openai, anthropics, deepseek-ai, argilla-io 等 |
| **论文** | arXiv (cs.CL/AI/LG) + HuggingFace Daily Papers |

### 数据集分类体系

| 类别 | 关键词 | 典型数据集 |
|------|--------|-----------|
| **sft** | instruction, chat | Alpaca, ShareGPT |
| **preference** | rlhf, dpo | UltraFeedback, HelpSteer |
| **synthetic** | synthetic, generated | Magpie, Sera |
| **agent** | tool, function | SWE-bench, WebArena |
| **multimodal** | image, video | LLaVA, Action100M |
| **code** | code, programming | StarCoder |

---

## 输出规范

### JSON Schema

完整规范见 `agent/schema.json`，核心结构：

```json
{
  "generated_at": "2026-02-05T12:59:46",
  "summary": {
    "total_datasets": 15,
    "total_github_repos": 134,
    "total_papers": 23,
    "total_blog_posts": 25
  },
  "datasets": [{
    "id": "allenai/Dolci-Instruct-SFT",
    "category": "sft_instruction",
    "downloads": 2610,
    "languages": ["en", "zh"],
    "license": "odc-by"
  }],
  "github_repos": [{
    "name": "open-instruct",
    "stars": 1500,
    "relevance": "high"
  }],
  "papers": [{
    "title": "...",
    "is_dataset_paper": true
  }],
  "blog_posts": [{
    "source": "OpenAI Blog",
    "articles": [{"title": "...", "url": "..."}]
  }]
}
```

---

## 配置

编辑 `config.yaml`：

```yaml
watched_orgs:
  frontier_labs:
    openai: { hf_ids: ["openai"] }
    google_deepmind: { hf_ids: ["google", "deepmind"] }

watched_vendors:
  blogs:
    - name: "OpenAI Blog"
      url: "https://openai.com/blog"

priority_data_types:
  preference: { keywords: ["rlhf", "dpo"] }
  sft: { keywords: ["instruction", "chat"] }
```

---

## 系统架构

```
ai-dataset-radar/
├── src/                        # 核心模块
│   ├── main_intel.py           # 主入口
│   ├── scrapers/               # 数据采集器 (9 个)
│   ├── analyzers/              # 分类器
│   └── utils/                  # 工具库
├── agent/                      # Agent 集成层
│   ├── api.py                  # REST API
│   ├── tools.json              # 工具定义
│   ├── schema.json             # 输出规范
│   └── prompts.md              # 系统提示词
├── mcp_server/                 # MCP 服务
├── config.yaml                 # 配置文件
└── data/reports/               # 输出目录
```

---

## 与 DataRecipe 协同

```
Radar (情报采集) → DataRecipe (逆向分析) → 复刻生产
```

联合配置实现端到端工作流：

```json
{
  "mcpServers": {
    "ai-dataset-radar": { "command": "..." },
    "datarecipe": { "command": "..." }
  }
}
```

---

## 开发路线

- [x] 多源数据采集 (HuggingFace, GitHub, arXiv, Blogs)
- [x] 双格式输出 (Markdown + JSON)
- [x] Agent 集成层 (HTTP API, Function Calling, Schema)
- [x] MCP Server (7 工具)
- [x] 插件化采集器 (9 个)
- [x] 性能优化 (并行采集、缓存、连接池)
- [x] 测试覆盖 (198 用例)
- [ ] 定时任务与告警
- [ ] Web 可视化界面

---

## 许可证

[MIT](LICENSE)

---

<div align="center">

**面向 AI Agent 的训练数据竞争情报系统**

</div>
