# DataRecipe 协同 / DataRecipe Integration

> 返回 [README](../README.md)

## 一键联动

`--recipe` 参数让 Radar 扫描完成后**自动**挑选高价值数据集，调用 DataRecipe 深度分析：

```bash
# 扫描 → 智能评分 → 自动分析 Top 5 数据集
python src/main_intel.py --days 7 --recipe

# 指定分析数量
python src/main_intel.py --days 7 --recipe --recipe-limit 3

# 前置：安装 DataRecipe（软依赖，未安装时自动跳过）
pip install -e /path/to/data-recipe
```

## 智能评分公式（0-100）

| 维度 | 权重 | 说明 |
|------|------|------|
| 下载量 | max 25 | log10 缩放，覆盖 10~100k+ 量级 |
| 社区认可 | max 10 | sqrt(likes) 缩放，社区 star 越多分越高 |
| 信号强度 | max 18 | 有意义分类信号越多越优先 |
| 分类优先级 | max 20 | preference > reward > sft > code/agent > synthetic > ... |
| 新鲜度 | max 12 | ≤7 天 +12，≤14 天 +8，≤30 天 +4（渐进衰减） |
| 低下载惩罚 | ×0.5 | <50 次下载的数据集总分减半，过滤噪声 |

## 输出结构

输出位于同一日期目录下：

```
data/reports/2026-02-08/
├── intel_report_2026-02-08.json    # Radar 报告
└── recipe/                         # DataRecipe 分析
    ├── recipe_analysis_summary.md  # 人类摘要
    ├── aggregate_summary.json      # 机器摘要（总复刻成本、难度分布）
    └── Anthropic__hh-rlhf/         # 每个数据集 23+ 分析文件
```

## MCP 双服务配置

Claude Desktop 中同时配置两个 MCP Server，可自然语言驱动端到端工作流：

```json
{
  "mcpServers": {
    "ai-dataset-radar": { "command": "..." },
    "datarecipe": { "command": "..." }
  }
}
```

详细 MCP 配置见 [mcp.md](mcp.md)。
