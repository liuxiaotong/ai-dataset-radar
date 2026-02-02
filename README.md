<p align="center">
  <h1 align="center">🛰️ AI Dataset Radar</h1>
  <p align="center">
    <strong>Track AI training datasets across HuggingFace, GitHub, arXiv & blogs</strong><br>
    <strong>追踪 HuggingFace、GitHub、arXiv 和博客上的 AI 训练数据集</strong>
  </p>
  <p align="center">
    <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
    <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python 3.10+"></a>
    <a href="#mcp-server"><img src="https://img.shields.io/badge/MCP-Server-purple.svg" alt="MCP Server"></a>
  </p>
  <p align="center">
    <a href="#-quick-start">English</a> | <a href="#-快速开始">中文</a>
  </p>
</p>

---

Monitor 30+ AI labs and data vendors. Get structured reports on new datasets, GitHub repos, papers, and blog posts — delivered as Markdown for humans or JSON for LLMs.

监控 30+ AI 实验室和数据供应商。获取新数据集、GitHub 仓库、论文和博客文章的结构化报告 — 支持 Markdown（人类可读）和 JSON（供 LLM 使用）双格式输出。

## ✨ What You Get / 功能概览

```
┌────────────────────────────────────────────────────────────────┐
│  12 datasets │ 138 repos │ 28 papers │ 4 blog posts            │
│  12 个数据集 │ 138 个仓库 │ 28 篇论文 │ 4 篇博客                │
├────────────────────────────────────────────────────────────────┤
│  • OpenAI, Anthropic, Google, Meta, DeepSeek, Qwen...          │
│  • Scale AI, Argilla, Snorkel, Labelbox...                     │
│  • RLHF, SFT, Synthetic, Agent, Evaluation datasets            │
└────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Option 1: Command Line

```bash
git clone https://github.com/liuxiaotong/ai-dataset-radar.git
cd ai-dataset-radar
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run scan
python src/main_intel.py --days 7
```

Reports saved to `data/reports/`:
- `intel_report_YYYY-MM-DD.md` — Human-readable
- `intel_report_YYYY-MM-DD.json` — For LLMs/scripts

### Option 2: Claude Desktop (MCP)

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

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

Then ask Claude: *"Scan for new AI datasets"* or *"What's new from OpenAI?"*

### Option 3: Claude Code

```bash
/radar    # Get project context
/scan     # Run intelligence scan
```

---

## 🚀 快速开始

### 方式一：命令行

```bash
git clone https://github.com/liuxiaotong/ai-dataset-radar.git
cd ai-dataset-radar
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 运行扫描
python src/main_intel.py --days 7
```

报告保存在 `data/reports/`:
- `intel_report_YYYY-MM-DD.md` — 人类可读
- `intel_report_YYYY-MM-DD.json` — 供 LLM/脚本使用

### 方式二：Claude Desktop (MCP)

编辑 `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "ai-dataset-radar": {
      "command": "/你的路径/ai-dataset-radar/.venv/bin/python",
      "args": ["/你的路径/ai-dataset-radar/mcp_server/server.py"]
    }
  }
}
```

然后在 Claude 中说：*"扫描新的 AI 数据集"* 或 *"OpenAI 最近有什么新动态？"*

### 方式三：Claude Code

```bash
/radar    # 获取项目上下文
/scan     # 运行情报扫描
```

---

## 📊 Output Example / 输出示例

### JSON (for LLMs / 供 LLM 使用)

```json
{
  "summary": {
    "total_datasets": 12,
    "total_github_repos": 138,
    "total_github_repos_high_relevance": 2,
    "total_papers": 28
  },
  "datasets": [
    {
      "id": "google/WaxalNLP",
      "category": "multilingual",
      "downloads": 1539,
      "license": "cc-by-4.0",
      "signals": ["multilingual", "audio"]
    }
  ],
  "github_activity": [
    {
      "org": "argilla-io",
      "repos_updated": [
        {"name": "argilla", "relevance": "high", "relevance_signals": ["annotation", "rlhf"]}
      ]
    }
  ]
}
```

### Markdown (for humans / 人类可读)

```markdown
## AI Labs Activity / AI 实验室动态

### google_deepmind
- **WaxalNLP** (1.5K downloads) - ASR/TTS for African languages

## GitHub Activity / GitHub 活动
### argilla-io
- **argilla** ⭐ 8.2K [HIGH] - Data curation for LLMs
```

---

## ⚙️ Configuration / 配置

Edit `config.yaml` to customize / 编辑 `config.yaml` 自定义配置:

```yaml
# Organizations to monitor / 监控的组织
watched_orgs:
  frontier_labs:                    # 一线实验室
    openai: { hf_ids: ["openai"], keywords: ["gpt"] }
    anthropic: { hf_ids: ["anthropic"], keywords: ["claude"] }
  china_opensource:                 # 中国开源大模型
    qwen: { hf_ids: ["Qwen"], keywords: ["qwen"] }
    deepseek: { hf_ids: ["deepseek-ai"], keywords: ["deepseek"] }

# Data types to track / 关注的数据类型
priority_data_types:
  preference: { keywords: ["rlhf", "dpo", "preference"] }
  sft: { keywords: ["instruction", "chat", "alpaca"] }
  agent: { keywords: ["tool use", "function calling"] }

# GitHub relevance keywords / GitHub 相关性关键词
sources:
  github:
    relevance_keywords: [dataset, annotation, benchmark, rlhf]
```

**Optional / 可选:** Set `GITHUB_TOKEN` for higher API rate limits / 设置 `GITHUB_TOKEN` 获得更高的 API 速率限制。

---

## 🏗️ Architecture / 架构

```
ai-dataset-radar/
├── src/
│   ├── main_intel.py        # Entry point / 入口
│   ├── scrapers/            # HuggingFace, GitHub, arXiv, RSS
│   ├── trackers/            # Org & blog monitors / 组织和博客监控
│   ├── analyzers/           # Dataset classification / 数据集分类
│   └── output_formatter.py  # Dual output / 双格式输出
├── mcp_server/              # Claude Desktop integration / Claude Desktop 集成
│   └── server.py
├── .claude/commands/        # Claude Code skills / Claude Code 技能
│   ├── radar.md
│   └── scan.md
├── config.yaml              # Watchlist configuration / 监控配置
└── data/reports/            # Generated reports / 生成的报告
```

---

## 🔌 MCP Server Tools / MCP 服务器工具

When using Claude Desktop / 在 Claude Desktop 中使用:

| Tool / 工具 | Description / 描述 |
|-------------|-------------------|
| `radar_scan` | Run full scan / 运行完整扫描 |
| `radar_summary` | Get report summary / 获取报告摘要 |
| `radar_datasets` | List datasets (filter by category) / 列出数据集（按类型过滤） |
| `radar_github` | View GitHub activity (filter by relevance) / 查看 GitHub 活动（按相关性过滤） |
| `radar_papers` | View recent papers / 查看最新论文 |
| `radar_config` | Show current watchlist / 显示当前监控配置 |

---

## 📦 Dataset Categories / 数据集类型

| Category / 类型 | Examples / 示例 | Description / 描述 |
|----------------|-----------------|-------------------|
| **SFT** | Alpaca, ShareGPT, OpenOrca | Instruction-following / 指令跟随 |
| **Preference** | UltraFeedback, HelpSteer, HH-RLHF | RLHF/DPO training / RLHF/DPO 训练 |
| **Synthetic** | Sera, Magpie | AI-generated / AI 生成 |
| **Agent** | SWE-bench, WebArena, ToolBench | Tool use / 工具使用 |
| **Evaluation** | MMLU, HumanEval, GPQA | Benchmarks / 基准测试 |
| **Multimodal** | Action100M, VoxPopuli | Image/Audio/Video / 多模态 |
| **Code** | StarCoder, CodeParrot | Programming / 编程 |

---

## 🎯 Organizations Tracked / 监控的组织

| Category / 类别 | Organizations / 组织 |
|----------------|---------------------|
| **Frontier Labs / 一线实验室** | OpenAI, Anthropic, Google/DeepMind, Meta, xAI |
| **Emerging Labs / 新兴实验室** | Mistral, Cohere, AI21, Together, Databricks |
| **Research Labs / 研究机构** | EleutherAI, HuggingFace, Allen AI, LMSys, NVIDIA |
| **China Open Source / 中国开源** | Qwen, DeepSeek, ChatGLM, Baichuan, Yi, InternLM |
| **China Closed Source / 中国闭源** | Baidu ERNIE, ByteDance Doubao, Tencent Hunyuan, Moonshot Kimi |
| **Data Vendors / 数据供应商** | Scale AI, Surge AI, Argilla, Snorkel, Labelbox |

---

## 🧪 Development / 开发

```bash
# Run tests / 运行测试
python -m pytest tests/ -v

# Add a new scraper / 添加新爬虫
# 1. Create src/scrapers/my_source.py / 创建文件
# 2. Inherit from BaseScraper / 继承 BaseScraper
# 3. Register with @register_scraper("my_source") / 注册
```

<details>
<summary>Example: Custom Scraper / 示例：自定义爬虫</summary>

```python
from src.scrapers.base import BaseScraper
from src.scrapers.registry import register_scraper

@register_scraper("my_source")
class MySourceScraper(BaseScraper):
    name = "my_source"
    source_type = "dataset_registry"

    def scrape(self, config=None) -> list[dict]:
        return [{"source": "my_source", "id": "dataset-1"}]
```

</details>

---

## 🗺️ Roadmap / 路线图

- [x] Multi-source aggregation / 多源聚合 (HF, GitHub, arXiv, blogs)
- [x] Dual output / 双格式输出 (Markdown + JSON)
- [x] MCP Server for Claude Desktop / Claude Desktop MCP 服务器
- [x] Claude Code skills / Claude Code 技能
- [ ] Scheduled execution & alerts / 定时执行和告警
- [ ] Web dashboard / Web 控制台
- [ ] LLM-powered summarization / LLM 驱动的摘要

---

## 🤝 Contributing / 贡献

PRs welcome! Areas where help is needed / 欢迎 PR！需要帮助的领域:

- New data sources / 新数据源 (e.g., Twitter/X, Discord)
- Improved classification heuristics / 改进分类算法
- Web UI / Web 界面
- Documentation translations / 文档翻译

---

## 📄 License / 许可证

MIT — see [LICENSE](LICENSE)

---

<p align="center">
  <sub>Built for the AI data community. Star ⭐ if useful!</sub><br>
  <sub>为 AI 数据社区而建。如果有用请点个星 ⭐</sub>
</p>
