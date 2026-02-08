# 跨源情报搜索

在最新 Radar 报告中搜索指定关键词，跨数据集、GitHub、论文、博客、X/Twitter 五大数据源。

## 参数

$ARGUMENTS - 搜索关键词，如 `RLHF`、`NVIDIA`、`机器人`、`reward model`、`代码生成`

## 执行步骤

### 第一步：定位最新报告

```bash
ls -t /Users/liukai/ai-dataset-radar/data/reports/*/intel_report_*.json 2>/dev/null | head -1
```

如果没有报告，提示用户先运行 `/scan`。

### 第二步：读取 JSON 报告并搜索

读取完整 JSON 报告，对 `$ARGUMENTS` 关键词（不区分大小写）在以下字段中搜索：

**数据集** (`datasets` 数组)：
- `id` — 数据集名称
- `description` / `readme_snippet` — 描述
- `tags` — 标签列表
- `signals` — 分类信号
- `category` — 数据类型

**GitHub** (`github_activity` → 各 org → `repos_updated`)：
- `name` / `full_name` — 仓库名
- `description` — 描述
- `topics` — 主题标签
- `relevance_signals` — 相关性信号

**论文** (`papers` 数组)：
- `title` — 标题
- `summary` / `abstract` — 摘要
- `paper_type` — 论文类型

**博客** (`blog_posts` 数组 → 各 source → `articles`)：
- `title` — 标题
- `summary` — 摘要

**X/Twitter** (`x_activity` → `accounts` → `relevant_tweets`)：
- `text` — 推文内容

### 第三步：按源分组展示结果

按以下格式输出，只展示匹配的结果：

```
## 搜索结果："{关键词}"

### 数据集 (N 条匹配)
| 数据集 | 类别 | 下载量 | Likes | 发布日期 | 匹配字段 |
|--------|------|--------|-------|----------|----------|
| org/name | sft | 2,472 | 45 | 2025-11-18 | description |

### GitHub (N 条匹配)
| 仓库 | Stars | 相关性 | 匹配字段 |
|------|-------|--------|----------|
| org/repo | 12,345 | high | description, signals |

### 论文 (N 条匹配)
| 标题 | 日期 | 来源 |
|------|------|------|
| Paper Title | 2026-02-05 | arxiv |

### 博客 (N 条匹配)
| 标题 | 来源 | 日期 |
|------|------|------|
| Blog Title | OpenAI Blog | 2026-02-05 |
```

如果某个数据源无匹配结果，省略该章节。

### 第四步：分析总结

在结果末尾提供 2-3 句总结：
- 该关键词在各数据源中的活跃度
- 与当前 AI 训练数据趋势的关联
- 建议的后续行动（如 `/deep-dive` 某个组织、`/recipe` 某个数据集）

## 注意

- 搜索不区分大小写
- 支持中英文关键词
- 如果关键词是组织名（如 NVIDIA、Qwen），建议用户使用 `/deep-dive` 获得更深入的分析
