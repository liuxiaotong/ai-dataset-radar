# AI Dataset Radar MCP Server

让 Claude Desktop 能够调用 AI Dataset Radar 进行竞争情报分析。

## 安装

### 1. 安装依赖

```bash
cd /Users/liukai/ai-dataset-radar
.venv/bin/pip install mcp pyyaml
```

### 2. 配置 Claude Desktop

编辑 Claude Desktop 配置文件：

**macOS:**
```bash
code ~/Library/Application\ Support/Claude/claude_desktop_config.json
```

**Windows:**
```bash
code %APPDATA%\Claude\claude_desktop_config.json
```

添加以下配置：

```json
{
  "mcpServers": {
    "ai-dataset-radar": {
      "command": "/Users/liukai/ai-dataset-radar/.venv/bin/python",
      "args": ["/Users/liukai/ai-dataset-radar/mcp_server/server.py"]
    }
  }
}
```

### 3. 重启 Claude Desktop

关闭并重新打开 Claude Desktop。

## 可用工具

| 工具 | 描述 |
|------|------|
| `radar_scan` | 运行竞争情报扫描 |
| `radar_summary` | 获取报告摘要 |
| `radar_datasets` | 查看数据集列表 |
| `radar_github` | 查看 GitHub 活动 |
| `radar_papers` | 查看相关论文 |
| `radar_config` | 查看监控配置 |

## 使用示例

在 Claude Desktop 中说：

- "运行一次 AI 数据集扫描"
- "最近有什么新的数据集？"
- "查看 GitHub 上的高相关性仓库"
- "显示最新的论文"
- "当前监控了哪些组织？"

Claude 会自动调用相应的工具。

## 测试

```bash
# 测试 MCP server 是否能启动
cd /Users/liukai/ai-dataset-radar
.venv/bin/python mcp_server/server.py
```

按 Ctrl+C 退出。
