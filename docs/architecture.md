# 系统架构 / System Architecture

> 返回 [README](../README.md)

```
ai-dataset-radar/
├── src/                        # 核心模块
│   ├── main_intel.py           # 主入口（async 编排 + 进度指示 + 趋势注入 + insights + --recipe）
│   ├── _version.py             # 版本号单一来源 (__version__)
│   ├── trackers/               # 数据追踪器（全异步 aiohttp）
│   │   ├── org_tracker.py      # HuggingFace 组织追踪
│   │   ├── blog_tracker.py     # 博客监控（RSS/HTML/Playwright async）
│   │   ├── github_tracker.py   # GitHub 组织活动
│   │   ├── x_tracker.py        # X/Twitter 账户监控（RSSHub / API）
│   │   └── paper_tracker.py    # arXiv + HF Papers
│   ├── scrapers/               # 数据采集器
│   ├── analyzers/              # 分类器 + 趋势分析 + change_tracker 日报变化追踪
│   └── utils/                  # 工具库
│       ├── async_http.py       # AsyncHTTPClient（连接池 + 重试 + 限速）
│       ├── llm_client.py       # LLM 调用（Anthropic API insights 生成）
│       └── cache.py            # FileCache（TTL + LRU 驱逐）
├── agent/                      # Agent 集成层
│   ├── api.py                  # REST API（认证 + 限速 + 健康检查）
│   ├── static/index.html       # Web 仪表盘（单文件，Tailwind + Chart.js）
│   ├── tools.json              # 工具定义
│   ├── schema.json             # 输出规范
│   └── prompts.md              # 系统提示词
├── .claude/commands/            # Claude Code Skills（7 个）
│   ├── scan.md                # /scan — 扫描 + 自动分析
│   ├── brief.md               # /brief — 快速情报简报
│   ├── search.md              # /search — 跨源智能搜索
│   ├── diff.md                # /diff — 报告对比
│   ├── deep-dive.md           # /deep-dive — 深度分析
│   ├── recipe.md              # /recipe — DataRecipe 逆向分析
│   └── radar.md               # /radar — 通用情报助手
├── mcp_server/                 # MCP 服务
├── .github/workflows/ci.yml    # CI：ruff lint + pytest
├── Dockerfile                  # 容器镜像（含 Playwright）
├── docker-compose.yml          # scan + api 服务编排
├── config.yaml                 # 监控配置（组织/供应商/博客/关键词）
├── .env.example                # 环境变量模板
└── data/reports/               # 输出目录（按日期子目录）
    └── YYYY-MM-DD/             # 每日报告 + recipe/ 分析结果
```
