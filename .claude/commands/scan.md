# 运行 AI 数据集扫描

执行 AI Dataset Radar 竞争情报扫描。

## 参数

$ARGUMENTS - 可选参数，如 `--days 14` 或 `--no-blogs`

## 执行步骤

1. 切换到项目目录并运行扫描：

```bash
cd /Users/liukai/ai-dataset-radar && .venv/bin/python src/main_intel.py $ARGUMENTS
```

2. 读取生成的报告，找到最新的文件：

```bash
ls -t data/reports/intel_report_*.json | head -1
```

3. 解析 JSON 报告并向用户展示关键发现：
   - 新数据集数量和类型分布
   - GitHub 高相关性仓库
   - 重要论文
   - 博客信号

4. 如果用户需要详细信息，读取完整的 Markdown 报告。

## 输出建议

用简洁的表格展示：
- 按类型分组的数据集
- 高相关性 GitHub 仓库
- 关键论文标题

最后提供 JSON 文件路径供用户进一步分析。
