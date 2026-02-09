# Changelog

AI Dataset Radar 已完成的里程碑。

## 2026-02-09 — Current State

### 核心功能

- 多源数据采集 (HuggingFace, GitHub, arXiv, Blogs)
- 双格式输出 (Markdown + JSON)
- 全链路异步 I/O (aiohttp + asyncio.gather 替代 requests + ThreadPoolExecutor，~2x 提速)
- 插件化采集器 (9 个)
- 时间信息全链路贯通 (HF camelCase→snake_case 归一化, HF Papers `<time>` 提取, insights 数据集/模型/论文均带日期, 新增时间线章节)
- 趋势数据写入报告 (每个 dataset 注入 growth_7d/growth_30d, Markdown 增加「📈 数据集增长趋势」节, JSON 增加 featured_trends)
- 自动日报变化追踪 (每次扫描后对比前日报告生成 `_changes.md`)
- 扫描进度指示 (`[1/N]...[N/N]` 步骤编号)
- 报告按日期子目录组织 (`data/reports/YYYY-MM-DD/`)
- stdout 清理 (insights prompt 不再 dump 到终端)

### Agent & MCP

- Agent 集成层 (HTTP API, Function Calling, Schema)
- MCP Server (16 工具: scan/summary/datasets/github/papers/blogs/reddit/config/search/diff/trend/trends/history/matrix/lineage/org-graph)
- Claude Code Skills (7 个: scan/brief/search/diff/deep-dive/recipe/radar)
- radar_search 全文搜索 (跨 6 类数据源, 支持正则, 按来源过滤)
- radar_reddit Reddit 社区动态 (5 子版块, 信号关键词过滤)
- radar_trends 历史趋势数据 (时序图数据输出)
- radar_matrix 竞品矩阵 (组织×数据类型交叉分析)
- radar_lineage 数据集谱系 (派生/版本链/Fork 树)
- radar_org_graph 组织关系图谱 (聚类/中心性)
- radar_diff 报告对比 (自动识别新增/消失的数据集、仓库、论文、博客)
- 工具参数扩展 (radar_scan sources 过滤, radar_datasets/github org 过滤)
- 趋势分析集成 (radar_trend 增长/上升/突破查询)
- 历史时间线 (radar_history 跨期报告统计对比)
- MCP/Schema 数据管道修复 (X/Twitter 数据写入 JSON, 博客搜索字段名修正)
- 数据集分类对齐 (Dashboard 下拉菜单 + API 文档 + schema.json 枚举统一)
- Dashboard 筛选增强 (论文「仅数据集」复选框 + 博客分类下拉)

### 数据源

- X/Twitter 监控 (125 账户, 13 类别, RSSHub + 多实例 fallback + 连续失败阈值保护)
- 中国数据供应商监控 (海天瑞声、整数智能、数据堂、智源 BAAI)
- Reddit 社区监控 (MachineLearning, LocalLLaMA, dataset, deeplearning, LanguageTechnology)
- 监控源大扩展 (HF 86 orgs, GitHub 50 orgs, arXiv +cs.CV/cs.RO, X 125 账户, 博客 71 源, Reddit 5 社区)
- 研究者博客监控 (Lil'Log, fast.ai, Interconnects, LessWrong, Alignment Forum, The Gradient, Epoch AI)
- 博客分类标注 (config.yaml 62 个博客源添加 category 字段)
- X 账号自动修正 (5 个改名/格式错误账号修复)

### 情报分析

- Insights 分析提示生成 (`--insights` 模式)
- 异常报告独立输出 (`_anomalies.md` 与 `_insights.md` 分离)
- DataRecipe 自动衔接 (`--recipe` 智能评分选 Top N 数据集)
- 竞品矩阵 (CompetitorMatrix: 组织×数据类型交叉统计, rankings, top_orgs)
- 数据集谱系 (DatasetLineageTracker: 派生关系, 版本链, Fork 树检测)
- 组织关系图谱 (OrgRelationshipGraph: 协作边, BFS 聚类, 度中心性)
- Recipe 评分公式优化 (新增 likes 维度, 渐进式新鲜度衰减)
- Insights API 集成 (run_intel_scan API 路径复用 LLM insights 生成)
- 多 LLM 提供商 (Kimi/DeepSeek/Qwen/Zhipu/OpenAI 通过 OpenAI 兼容协议接入)

### 质量 & 健壮性

- 分类器增强 (覆盖率 37%→84%：新增机器人/具身、文档理解、语音、形式化验证等)
- 博客抓取多策略降级 (RSS → HTML → Playwright, networkidle → domcontentloaded)
- 博客抓取修复 (移除过度激进的信号关键词过滤)
- 博客噪声过滤 (nav/sidebar/footer 自动排除, 浏览器每 15 页重启)
- 全链路指数退避重试 (HF/GitHub/RSSHub 5xx 自动恢复)
- 数据质量校验 (各源 0 结果自动告警, JSON 输出 data_quality_warnings)
- datetime 全面修复 (21 处 utcnow() 替换为 timezone-aware)
- GitHub 加权相关性评分 (keyword×10 + stars/100 + 近 3 天活跃加成)
- 健壮性加固 (asyncio.get_running_loop 替代已弃用 API, UTF-8 编码, JSON 异常处理)
- 全链路性能优化 (OrgTracker 并行化, feedparser→线程池, 并发调优, 超时/重试优化)

### 基础设施

- CI 流水线 (GitHub Actions: ruff lint + pytest, push/PR 触发)
- Docker 容器化 (Dockerfile + docker-compose: scan 扫描 + api 服务)
- 测试覆盖 (855 用例)
- API 安全加固 v1+v2 (Bearer Token 认证 + 速率限制 + XSS 防护 + 非 root Docker)
- 启动配置校验 (validate_config: 必需配置段 + 类型检查)
- 缓存大小限制 (FileCache LRU 驱逐, max_entries=1000)
- 版本号统一管理 (`src/_version.py` 单一来源 + git pre-commit hook)
- Web 可视化仪表盘 (`/dashboard`: 11 Tab 面板, Chart.js 趋势图, 全局搜索, 深色主题)
- dotenv 环境变量支持 (python-dotenv 自动加载 .env)
- API 扫描 X/Twitter 补全 + Markdown 报告 X/Twitter 章节
