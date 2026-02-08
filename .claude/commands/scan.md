# 运行 AI 数据集扫描

执行 AI Dataset Radar 竞争情报扫描，并自动生成 AI 分析报告。

## 参数

$ARGUMENTS - 可选参数，如 `--days 14` 或 `--no-blogs`

## 执行步骤

### 第一步：运行扫描

```bash
cd /Users/liukai/ai-dataset-radar && .venv/bin/python src/main_intel.py $ARGUMENTS
```

### 第二步：读取报告并展示关键发现

找到最新的 JSON 报告：

```bash
ls -t data/reports/intel_report_*.json | head -1
```

解析 JSON 报告，用简洁的表格向用户展示：
- 新数据集数量和类型分布
- GitHub 高相关性仓库（Top 10）
- 重要论文
- X/Twitter 活跃账号和推文数
- 博客信号

### 第三步：自动生成 AI 分析报告（关键步骤）

检查是否需要生成 insights 分析：

1. 找到最新的 `_insights_prompt.md` 文件
2. 检查对应的 `_insights.md` 是否已存在（如果程序通过 API 已生成，则跳过）
3. 如果 `_insights.md` 不存在：
   - 读取 `_insights_prompt.md` 的全部内容
   - 作为环境中的 AI，根据 prompt 中的分析要求生成竞争情报分析
   - 将分析结果写入对应的 `_insights.md` 文件
   - 这就是"环境 LLM 接管"——你（Claude Code）就是环境中的 AI 能力

### 第四步：展示分析结论

读取 `_insights.md`，向用户展示：
- 关键发现（Top 3-5 事件）
- 行动建议
- 异常项

## 注意

- 如果设置了 `ANTHROPIC_API_KEY`，程序会自动通过 API 生成 insights，第三步可跳过
- 如果没有 API Key，程序只保存 prompt 文件，第三步由你（环境 AI）完成分析
- 分析报告应保存为 `data/reports/intel_report_{日期}_insights.md`
