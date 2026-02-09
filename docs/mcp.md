# MCP Server

> 返回 [README](../README.md)

## 配置 Claude Desktop

编辑 `~/Library/Application Support/Claude/claude_desktop_config.json`：

### 方式一：uv 启动（推荐）

```json
{
  "mcpServers": {
    "radar": {
      "command": "uv",
      "args": ["--directory", "/path/to/ai-dataset-radar", "run", "python", "mcp_server/server.py"],
      "env": {
        "RADAR_DATA_SOURCES": "github,huggingface",
        "RADAR_REPORT_DAYS": "7"
      }
    }
  }
}
```

### 方式二：直接 Python

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

## 可用工具（16 个）

| 工具 | 功能 | 参数 |
|------|------|------|
| `radar_scan` | 执行完整扫描 | `sources`, `days` |
| `radar_summary` | 报告摘要 | |
| `radar_datasets` | 按类别查询数据集 | `category`, `org`, `limit` |
| `radar_github` | GitHub 活动 | `relevance`, `org` |
| `radar_papers` | 论文列表 | `source`, `dataset_only`, `limit` |
| `radar_blogs` | 博客文章 | `source`, `category`, `limit` |
| `radar_reddit` | Reddit 社区动态 | `subreddit`, `min_score`, `limit` |
| `radar_config` | 监控配置 | |
| `radar_search` | 全文搜索（跨 6 源，支持正则） | `query`, `sources`, `limit` |
| `radar_diff` | 报告对比（新增/消失项） | `date_a`, `date_b` |
| `radar_trend` | 趋势分析（增长/突破） | `mode`, `dataset_id`, `days` |
| `radar_trends` | 历史趋势数据（时序图） | `limit` |
| `radar_history` | 历史时间线 | `limit` |
| `radar_matrix` | 竞品矩阵（组织×数据类型） | `top_n` |
| `radar_lineage` | 数据集谱系（派生/版本/Fork） | `dataset_id` |
| `radar_org_graph` | 组织关系图谱（聚类/中心性） | `org` |

## 常见问题

- `Tool invocation timed out` → 增大 `MCP_TIMEOUT` 或减小 `--days`。
- `No insights model configured` → `.env` 中未设置 `INSIGHTS_MODEL` 或 `ANTHROPIC_API_KEY`。
- `Permission denied writing data/reports` → 确保在项目根目录运行或设置 `RADAR_OUTPUT_DIR`。
