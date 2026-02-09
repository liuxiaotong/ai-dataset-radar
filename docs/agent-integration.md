# Agent 集成 / Agent Integrations

> 返回 [README](../README.md)

## 接入方式

| 方式 | 适用框架 | 配置文件 |
|------|----------|----------|
| **HTTP API** | LangChain, AutoGPT, Dify | `agent/api.py` |
| **Function Calling** | OpenAI GPT, Claude API | `agent/tools.json` |
| **MCP Server** | Claude Desktop | `mcp_server/server.py` |
| **JSON Schema** | 类型生成、数据验证 | `agent/schema.json` |

## HTTP API 端点

```bash
uvicorn agent.api:app --port 8080
```

| 端点 | 方法 | 功能 |
|------|------|------|
| `/dashboard` | GET | Web 可视化仪表盘（11 Tab 面板） |
| `/ui` | GET | 重定向至仪表盘 |
| `/health` | GET | 健康检查（认证状态、报告可用性） |
| `/summary` | GET | 获取最新报告摘要 |
| `/datasets` | GET | 数据集列表 (支持 category/org 筛选) |
| `/github` | GET | GitHub 仓库活动 (支持 relevance/org 筛选) |
| `/papers` | GET | 论文列表 (支持 source/dataset_only 筛选) |
| `/blogs` | GET | 博客文章 (支持 category/source 筛选) |
| `/reddit` | GET | Reddit 帖子 (支持 subreddit/min_score 筛选) |
| `/search` | GET | 跨 6 源全文搜索 (支持 q/sources/limit) |
| `/trends` | GET | 历史趋势时序数据 (支持 limit) |
| `/matrix` | GET | 竞品矩阵（组织×数据类型交叉分析） |
| `/lineage` | GET | 数据集谱系（派生/版本链/Fork 树） |
| `/org-graph` | GET | 组织关系图谱（聚类/中心性） |
| `/scan` | POST | 执行新扫描（含 insights 生成） |
| `/config` | GET | 监控配置（敏感信息自动脱敏） |
| `/schema` | GET | 输出规范 |
| `/alerts` | GET | 告警记录 (支持 severity/limit 筛选) |
| `/tools` | GET | 工具定义 |

## OpenAI Function Calling

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

## Anthropic Tool Use

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

## LangChain 集成

```python
from langchain.tools import Tool
import requests

tools = [
    Tool(
        name="radar_datasets",
        func=lambda cat: requests.get(f"http://localhost:8080/datasets?category={cat}").json(),
        description="按类别查询数据集: sft_instruction|reward_model|synthetic|multimodal|code|evaluation"
    ),
]
```

## 预置 System Prompt

`agent/prompts.md` 提供四类预置提示词：

| 角色 | 用途 |
|------|------|
| Dataset Intelligence Analyst | 数据集情报分析 |
| Competitive Intelligence Agent | 竞争情报追踪 |
| Dataset Discovery Assistant | 数据集发现与推荐 |
| Research Trend Monitor | 研究趋势监控 |
