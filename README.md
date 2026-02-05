<div align="center">

# AI Dataset Radar

**AI 训练数据竞争情报系统**

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![MCP](https://img.shields.io/badge/MCP-7_Tools-purple.svg)](#mcp-server)
[![Tests](https://img.shields.io/badge/tests-198_passed-brightgreen.svg)](#tests)

[快速开始](#快速开始) · [数据源](#数据源) · [MCP Server](#mcp-server) · [配置](#配置) · [输出格式](#输出格式)

</div>

---

监控 30+ AI 组织的训练数据动态，聚合 HuggingFace、GitHub、arXiv、公司博客，输出结构化 JSON 供 LLM 消费。

## 核心价值

```
多源监控 → 智能分类 → 结构化输出 → LLM 消费 / 人工阅读
```

### 按角色快速导航

| 角色 | 用法 | 说明 |
|------|------|------|
| 👔 **决策层** | 阅读 `intel_report.md` | 周报摘要，了解行业动态 |
| 🤖 **AI Agent** | 消费 `intel_report.json` | 结构化数据，供 LLM 分析 |
| 🔧 **开发者** | Claude Desktop MCP | 自然语言查询数据集情报 |
| 📊 **分析师** | 配合 DataRecipe | 发现 → 逆向分析完整流程 |

### 输出物一览

| 文件 | 用途 | 格式 |
|------|------|------|
| `intel_report_YYYY-MM-DD.md` | 人类阅读 | Markdown |
| `intel_report_YYYY-MM-DD.json` | LLM/脚本消费 | JSON |

## 安装

```bash
git clone https://github.com/liuxiaotong/ai-dataset-radar.git
cd ai-dataset-radar
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 可选：安装 Playwright 以抓取 JS 渲染的博客
playwright install chromium
```

## 快速开始

### 命令行扫描

```bash
python src/main_intel.py --days 7
```

<details>
<summary>输出示例</summary>

```
2026-02-05 12:59:00 [INFO] Starting AI Dataset Intelligence scan...
2026-02-05 12:59:00 [INFO] Scan period: 2026-01-29 to 2026-02-05

2026-02-05 12:59:15 [INFO] HuggingFace: Found 15 datasets from watched orgs
2026-02-05 12:59:30 [INFO] GitHub: Found 134 repos (85 high relevance)
2026-02-05 12:59:40 [INFO] Blogs: Found 25 articles from 8 active sources
2026-02-05 12:59:45 [INFO] Papers: Found 23 papers (15 arXiv, 8 HF Papers)

2026-02-05 12:59:46 [INFO] Reports saved:
  - data/reports/intel_report_2026-02-05.md
  - data/reports/intel_report_2026-02-05.json
```

</details>

### Claude Desktop (MCP Server)

添加到 `~/Library/Application Support/Claude/claude_desktop_config.json`：

```json
{
  "mcpServers": {
    "ai-dataset-radar": {
      "command": "/path/to/ai-dataset-radar/.venv/bin/python",
      "args": ["/path/to/ai-dataset-radar/mcp_server/server.py"]
    }
  }
}
```

然后用自然语言问 Claude：

```
用户: 扫描最近的数据集动态
Claude: [调用 radar_scan] 发现 15 个数据集...

用户: 有哪些合成数据集？
Claude: [调用 radar_datasets category=synthetic] 找到 3 个...

用户: 看看 OpenAI 的博客更新
Claude: [调用 radar_blogs] OpenAI Blog 有 2 篇新文章...
```

### AI Native 工作流 (配合 DataRecipe)

联合 [DataRecipe](https://github.com/liuxiaotong/data-recipe) 实现完整的数据集情报 + 逆向分析：

```json
{
  "mcpServers": {
    "ai-dataset-radar": {
      "command": "/path/to/ai-dataset-radar/.venv/bin/python",
      "args": ["/path/to/ai-dataset-radar/mcp_server/server.py"]
    },
    "datarecipe": {
      "command": "uv",
      "args": ["--directory", "/path/to/data-recipe", "run", "datarecipe-mcp"]
    }
  }
}
```

<details>
<summary>工作流示例</summary>

```
用户: 扫描这周的数据集，找一个 SFT 类型的深度分析

Claude 自动执行:
  1. [radar_scan] → 获取 15 个数据集
  2. [radar_datasets category=sft] → allenai/Dolci-Instruct-SFT
  3. [datarecipe deep_analyze] → 生成逆向分析报告
  4. 返回：构造方法、成本估算、复刻指南
```

</details>

---

## 数据源

### HuggingFace Datasets (30+ 组织)

| 类别 | 组织 |
|------|------|
| **Frontier Labs** | OpenAI, Google/DeepMind, Meta, Anthropic |
| **Emerging Labs** | Mistral, Cohere, AI21, Together |
| **Research Labs** | EleutherAI, Allen AI, HuggingFace, NVIDIA |
| **China Labs** | Qwen, DeepSeek, Baichuan, Yi, InternLM, Zhipu |

### Blogs (17 sources)

| 类别 | 博客 |
|------|------|
| **US Frontier** | OpenAI, Google AI, DeepMind, Meta AI |
| **US Emerging** | Mistral AI, Scale AI, Together AI, AI21 |
| **Research** | Stanford HAI, Berkeley BAIR, Anthropic |
| **China** | Qwen, Tencent Hunyuan, Zhipu AI, 01.AI, Baidu |
| **Data Vendors** | Argilla, Scale AI |

### GitHub (15+ 组织)

监控: `openai`, `anthropics`, `deepseek-ai`, `argilla-io`, `scaleapi`, `EleutherAI`...

### Papers

- arXiv (cs.CL, cs.AI, cs.LG) 关键词过滤
- HuggingFace Daily Papers

---

## MCP Server

7 个工具供 Claude 调用：

| 工具 | 功能 | 参数 |
|------|------|------|
| `radar_scan` | 运行完整扫描 | `days` (默认 7) |
| `radar_summary` | 获取最新报告摘要 | - |
| `radar_datasets` | 按类别筛选数据集 | `category` (sft/preference/synthetic/...) |
| `radar_github` | 查看 GitHub 活动 | `relevance` (high/low/all) |
| `radar_papers` | 查看最新论文 | `source` (arxiv/hf/all) |
| `radar_blogs` | 查看博客文章 | `source` (可选) |
| `radar_config` | 显示监控配置 | - |

---

## 配置

编辑 `config.yaml`：

```yaml
# HuggingFace 监控组织
watched_orgs:
  frontier_labs:
    openai: { hf_ids: ["openai"], keywords: ["gpt"] }
    google_deepmind: { hf_ids: ["google", "deepmind"] }
  china_opensource:
    qwen: { hf_ids: ["Qwen"], keywords: ["qwen"] }
    deepseek: { hf_ids: ["deepseek-ai"] }

# 博客源 (支持 RSS、爬虫、Playwright)
watched_vendors:
  blogs:
    - name: "OpenAI Blog"
      url: "https://openai.com/blog"
      type: "auto"
    - name: "Tencent Hunyuan"
      url: "https://hy.tencent.com/research"
      type: "browser"  # JS 渲染页面
      selector: ".blog-item"

# 数据集分类关键词
priority_data_types:
  preference: { keywords: ["rlhf", "dpo", "preference"] }
  sft: { keywords: ["instruction", "chat", "sft"] }
  synthetic: { keywords: ["synthetic", "generated"] }
```

设置 `GITHUB_TOKEN` 环境变量以提高 API 限额。

---

## 输出格式

### JSON (供 LLM 消费)

```json
{
  "generated_at": "2026-02-05T12:59:46",
  "summary": {
    "total_datasets": 15,
    "total_github_repos": 134,
    "total_papers": 23,
    "total_blog_posts": 25
  },
  "datasets": [
    {
      "id": "allenai/Dolci-Instruct-SFT",
      "category": "sft_instruction",
      "downloads": 2610,
      "languages": ["en", "zh", "ja", "..."],
      "license": "odc-by"
    }
  ],
  "blog_posts": [
    {
      "source": "OpenAI Blog",
      "articles": [
        {"title": "Introducing Codex", "url": "https://..."}
      ]
    }
  ]
}
```

### Markdown (供人类阅读)

<details>
<summary>示例</summary>

```markdown
# AI Dataset Intelligence Report
> Period: 2026-01-29 to 2026-02-05

## Summary
- 15 new datasets from watched organizations
- 134 GitHub repos (85 high relevance)
- 25 blog articles from 8 sources
- 23 papers (15 arXiv, 8 HF Papers)

## High-Value Datasets

### SFT / Instruction
| Dataset | Publisher | Downloads |
|---------|-----------|-----------|
| Dolci-Instruct-SFT | allenai | 2,610 |

## Blog Updates

### OpenAI Blog
- [Introducing Codex](https://openai.com/...)
- [Inside our data agent](https://openai.com/...)
```

</details>

---

## 数据集分类

| 类别 | 示例 | 说明 |
|------|------|------|
| **SFT** | Alpaca, ShareGPT | 指令微调 |
| **Preference** | UltraFeedback, HelpSteer | RLHF/DPO 训练 |
| **Synthetic** | Sera, Magpie | AI 生成数据 |
| **Agent** | SWE-bench, WebArena | 工具使用 |
| **Multimodal** | Action100M, VoxPopuli | 图/音/视频 |
| **Multilingual** | WaxalNLP | 多语言 |
| **Code** | StarCoder | 编程数据 |

---

## 性能优化

| 特性 | 说明 |
|------|------|
| **并行抓取** | ThreadPoolExecutor 并发 API 调用 |
| **API 缓存** | 文件缓存，HuggingFace README 24h TTL |
| **连接池** | 线程本地 SQLite 连接 |
| **HTTP 重试** | 指数退避，可配置重试次数 |
| **统一日志** | 结构化日志，可配置级别 |

---

## 项目架构

```
ai-dataset-radar/
├── src/
│   ├── main_intel.py           # 入口 (并行抓取)
│   ├── scrapers/               # 9 个爬虫
│   │   ├── base.py             # BaseScraper 抽象类
│   │   ├── registry.py         # 插件注册系统
│   │   ├── huggingface.py      # HuggingFace 数据集
│   │   ├── github.py           # GitHub 仓库
│   │   ├── github_org.py       # GitHub 组织监控
│   │   ├── arxiv.py            # arXiv 论文
│   │   ├── hf_papers.py        # HuggingFace 论文
│   │   ├── blog_rss.py         # RSS 博客
│   │   └── ...
│   ├── trackers/               # 博客追踪 (RSS + Playwright)
│   ├── analyzers/              # 数据集分类
│   ├── utils/                  # 工具模块
│   │   ├── cache.py            # 文件缓存 (TTL)
│   │   ├── http.py             # HTTP 重试
│   │   ├── keywords.py         # 关键词匹配
│   │   └── logging_config.py   # 日志配置
│   ├── db.py                   # SQLite (连接池)
│   └── output_formatter.py     # Markdown + JSON 输出
├── mcp_server/server.py        # MCP Server (7 工具)
├── tests/                      # 198 个测试
├── config.yaml                 # 监控配置
└── data/reports/               # 生成的报告
```

---

## 与 DataRecipe 联动

```
Radar (发现数据集) → Recipe (逆向分析) → 复刻生产
```

| Radar 产出 | Recipe 消费 |
|-----------|-------------|
| `intel_report.json` | `batch-from-radar` 批量分析 |
| 数据集 ID | `deep-analyze` 深度分析 |
| 分类标签 | 按类型筛选分析目标 |

---

## Roadmap

- [x] 多源聚合 (HF, GitHub, arXiv, Blogs)
- [x] 双格式输出 (Markdown + JSON)
- [x] MCP Server (7 工具)
- [x] Playwright 支持 (JS 渲染页面)
- [x] 17 个博客源 (US/China/Research)
- [x] AI Native 工作流 (DataRecipe 联动)
- [x] 插件化爬虫架构 (9 个爬虫)
- [x] 性能优化 (并行、缓存、连接池)
- [x] 完整测试覆盖 (198 个测试)
- [ ] 定时执行 & 告警
- [ ] Web 仪表盘

---

## Contributing

欢迎 PR！需要帮助的领域：

- 新博客源 (尤其是中国闭源实验室)
- 复杂 SPA 的爬虫选择器
- Web UI 仪表盘
- 更多语言支持

---

## License

[MIT](LICENSE)

---

<div align="center">
<sub>为 AI 研究者、数据团队和所有关注训练数据动态的人而建</sub>
</div>
