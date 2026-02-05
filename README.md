<div align="center">

# AI Dataset Radar

**AI 训练数据竞争情报系统**

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Agent Ready](https://img.shields.io/badge/Agent-Ready-orange.svg)](#agent-集成)
[![MCP](https://img.shields.io/badge/MCP-7_Tools-purple.svg)](#mcp-server)

[快速开始](#快速开始) · [Agent 集成](#agent-集成) · [数据源](#数据源) · [MCP Server](#mcp-server) · [配置](#配置)

</div>

---

监控 30+ AI 组织的训练数据动态，输出结构化 JSON 供任意 AI Agent 消费。

## 核心价值

```
多源监控 → 智能分类 → 结构化输出 → 任意 Agent 消费
```

### 为什么 Agent Ready？

| 特性 | 说明 |
|------|------|
| **HTTP API** | RESTful 接口，任意语言/框架可调用 |
| **Function Calling** | OpenAI / Anthropic 标准工具定义 |
| **JSON Schema** | 严格的输出格式定义，便于解析验证 |
| **MCP Server** | Claude Desktop 原生集成 |
| **Agent Prompts** | 预置 system prompt，即插即用 |

### 按使用者导航

| 使用者 | 接入方式 | 说明 |
|--------|----------|------|
| 🤖 **GPT/Claude Agent** | Function Calling | 加载 `agent/tools.json` |
| 🦜 **LangChain Agent** | HTTP API | `localhost:8080/datasets` |
| 🔧 **AutoGPT/自定义** | REST API | 标准 HTTP 调用 |
| 💬 **Claude Desktop** | MCP Server | 自然语言交互 |
| 👔 **人类决策者** | Markdown 报告 | `intel_report.md` |

### 输出物

| 文件 | 消费者 | 格式 |
|------|--------|------|
| `intel_report.json` | AI Agent | JSON (有 Schema) |
| `intel_report.md` | 人类 | Markdown |
| `agent/tools.json` | LLM Function Calling | Tool Spec |
| `agent/schema.json` | 数据验证 | JSON Schema |

---

## 安装

```bash
git clone https://github.com/liuxiaotong/ai-dataset-radar.git
cd ai-dataset-radar
pip install -r requirements.txt

# Agent API 服务 (可选)
pip install fastapi uvicorn
```

## 快速开始

### 1. 命令行扫描

```bash
python src/main_intel.py --days 7
# 输出: data/reports/intel_report_2026-02-05.json
```

### 2. 启动 Agent API

```bash
uvicorn agent.api:app --port 8080
# API 文档: http://localhost:8080/docs
```

### 3. Agent 调用

```python
# 任意 HTTP 客户端
import requests
datasets = requests.get("http://localhost:8080/datasets?category=sft").json()
```

---

## Agent 集成

### 集成方式一览

| 方式 | 文件 | 适用场景 |
|------|------|----------|
| **HTTP API** | `agent/api.py` | LangChain, AutoGPT, 自定义 Agent |
| **Function Calling** | `agent/tools.json` | OpenAI GPT, Anthropic Claude |
| **JSON Schema** | `agent/schema.json` | 输出验证, 类型生成 |
| **System Prompts** | `agent/prompts.md` | 快速原型, Agent 配置 |
| **MCP Server** | `mcp_server/server.py` | Claude Desktop |

### HTTP API

```bash
# 启动服务
uvicorn agent.api:app --port 8080
```

| 端点 | 方法 | 说明 |
|------|------|------|
| `GET /summary` | 获取最新报告摘要 |
| `GET /datasets?category=sft` | 按类别筛选数据集 |
| `GET /github?relevance=high` | 高相关 GitHub 仓库 |
| `GET /papers?dataset_only=true` | 数据集论文 |
| `GET /blogs` | 博客文章 |
| `POST /scan` | 运行新扫描 |
| `GET /schema` | JSON Schema |
| `GET /tools` | 工具定义 |

### OpenAI Function Calling

```python
import json
import openai

# 加载工具定义
with open("agent/tools.json") as f:
    tools = json.load(f)["tools"]

response = openai.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "最近有什么新的 SFT 数据集?"}],
    tools=[{"type": "function", "function": t} for t in tools]
)
```

### Anthropic Tool Use

```python
import json
import anthropic

with open("agent/tools.json") as f:
    tools = json.load(f)["tools"]

response = anthropic.messages.create(
    model="claude-sonnet-4-20250514",
    tools=[{
        "name": t["name"],
        "description": t["description"],
        "input_schema": t["parameters"]
    } for t in tools],
    messages=[{"role": "user", "content": "查找偏好训练数据集"}]
)
```

### LangChain

```python
from langchain.tools import Tool
import requests

def query_datasets(category: str) -> dict:
    return requests.get(f"http://localhost:8080/datasets?category={category}").json()

tools = [
    Tool(
        name="radar_datasets",
        func=query_datasets,
        description="Get AI training datasets by category: sft|preference|synthetic|agent|code"
    ),
]
```

### Agent System Prompt

预置 prompt 在 `agent/prompts.md`，包括：

- **Dataset Intelligence Analyst** - 数据集情报分析
- **Competitive Intelligence Agent** - 竞争情报追踪
- **Dataset Discovery Assistant** - 数据集发现助手
- **Research Trend Monitor** - 研究趋势监控

---

## MCP Server (Claude Desktop)

添加到 `~/Library/Application Support/Claude/claude_desktop_config.json`：

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

7 个 MCP 工具：

| 工具 | 功能 |
|------|------|
| `radar_scan` | 运行完整扫描 |
| `radar_summary` | 最新报告摘要 |
| `radar_datasets` | 按类别筛选数据集 |
| `radar_github` | GitHub 仓库活动 |
| `radar_papers` | 最新论文 |
| `radar_blogs` | 博客文章 |
| `radar_config` | 监控配置 |

---

## 数据源

### HuggingFace (30+ 组织)

| 类别 | 组织 |
|------|------|
| **Frontier** | OpenAI, Google/DeepMind, Meta, Anthropic |
| **Emerging** | Mistral, Cohere, AI21, Together |
| **Research** | EleutherAI, Allen AI, HuggingFace, NVIDIA |
| **China** | Qwen, DeepSeek, Baichuan, Yi, InternLM, Zhipu |

### Blogs (17 sources)

OpenAI, Anthropic, Google AI, DeepMind, Meta AI, Mistral, Scale AI, Qwen, Tencent, Zhipu...

### GitHub (15+ 组织)

`openai`, `anthropics`, `deepseek-ai`, `argilla-io`, `scaleapi`, `EleutherAI`...

### Papers

arXiv (cs.CL, cs.AI, cs.LG) + HuggingFace Daily Papers

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

## 输出格式

### JSON Schema

完整 schema 在 `agent/schema.json`，主要结构：

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

## 数据集分类

| 类别 | 关键词 | 示例 |
|------|--------|------|
| **sft** | instruction, chat | Alpaca, ShareGPT |
| **preference** | rlhf, dpo | UltraFeedback, HelpSteer |
| **synthetic** | synthetic, generated | Sera, Magpie |
| **agent** | tool, function | SWE-bench, WebArena |
| **multimodal** | image, video, audio | Action100M |
| **code** | code, programming | StarCoder |

---

## 项目架构

```
ai-dataset-radar/
├── src/                        # 核心逻辑
│   ├── main_intel.py           # 入口
│   ├── scrapers/               # 9 个爬虫
│   ├── analyzers/              # 分类器
│   └── utils/                  # 工具 (cache, http, logging)
├── agent/                      # Agent 集成层
│   ├── api.py                  # HTTP REST API
│   ├── tools.json              # Function Calling 定义
│   ├── schema.json             # JSON Schema
│   └── prompts.md              # System Prompts
├── mcp_server/server.py        # Claude Desktop MCP
├── config.yaml                 # 监控配置
└── data/reports/               # 输出报告
```

---

## 与 DataRecipe 联动

```
Radar (发现) → Recipe (逆向分析) → 复刻生产
```

配置两个 MCP Server 实现 AI Native 工作流：

```json
{
  "mcpServers": {
    "ai-dataset-radar": { "command": "..." },
    "datarecipe": { "command": "..." }
  }
}
```

---

## Roadmap

- [x] 多源聚合 (HF, GitHub, arXiv, Blogs)
- [x] 双格式输出 (Markdown + JSON)
- [x] Agent 集成 (HTTP API + Function Calling + Schema)
- [x] MCP Server (7 工具)
- [x] 插件化爬虫 (9 个)
- [x] 性能优化 (并行、缓存、连接池)
- [x] 198 个测试
- [ ] 定时执行 & 告警
- [ ] Web 仪表盘

---

## License

[MIT](LICENSE)

---

<div align="center">

**Agent Ready** · 为任意 AI Agent 提供训练数据情报

</div>
