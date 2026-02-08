# DataRecipe 逆向分析

对指定数据集运行 DataRecipe 深度分析，逆向推导其构建方法、复刻成本和技术难度。

## 参数

$ARGUMENTS - 数据集 ID（HuggingFace 格式），如 `Qwen/RationaleRM`、`allenai/Dolci-Instruct-SFT`

## 执行步骤

### 第一步：检查 DataRecipe 是否安装

```bash
cd /Users/liukai/ai-dataset-radar && .venv/bin/python -c "from datarecipe.core.deep_analyzer import DeepAnalyzerCore; print('OK')"
```

如果报错 `ModuleNotFoundError`，提示用户安装：
```bash
pip install -e /path/to/data-recipe
```

### 第二步：解析数据集 ID

从 `$ARGUMENTS` 中提取数据集 ID。确保格式为 `org/name`（如 `Qwen/RationaleRM`）。

如果用户只提供了名称而非完整 ID，尝试在最新 JSON 报告中查找匹配。

### 第三步：运行深度分析

准备输出目录，然后运行分析：

```bash
cd /Users/liukai/ai-dataset-radar && .venv/bin/python -c "
from datarecipe.core.deep_analyzer import DeepAnalyzerCore
from datetime import datetime

date_str = datetime.now().strftime('%Y-%m-%d')
dataset_id = '$DATASET_ID'
output_dir = f'data/reports/{date_str}/recipe'

analyzer = DeepAnalyzerCore()
result = analyzer.analyze(
    dataset_id=dataset_id,
    sample_size=300,
    output_dir=output_dir
)
print(f'Analysis complete: {output_dir}')
"
```

注意：分析过程会下载数据集样本并进行多维度分析，可能需要几分钟。

### 第四步：读取分析结果

分析完成后，在输出目录中查找结果文件：

1. 首先读取 `EXECUTIVE_SUMMARY.md`（在 `01_决策参考/` 子目录下）
2. 读取 `recipe_summary.json` 获取结构化数据

### 第五步：向用户展示分析结果

用以下格式输出：

```
## DataRecipe 分析结果：{数据集ID}

### 复刻成本估算
| 维度 | 估算 |
|------|------|
| 总成本 | $X,XXX |
| 人工成本 | $X,XXX (XX%) |
| API 成本 | $XXX (XX%) |
| 难度 | easy/medium/hard |

### 数据集概要
- 样本数量：XXX
- 文件数量：XX
- Schema 字段：[列出主要字段]

### 关键发现
[从 EXECUTIVE_SUMMARY.md 中提取核心发现]

### 建议
[基于分析结果的行动建议：是否值得复刻、复刻策略、替代方案]
```

## 高级用法

如果用户想批量分析，建议使用 Radar 的 `--recipe` 参数：
```bash
/scan --days 7 --recipe --recipe-limit 5
```

这会自动评分并选择最有价值的数据集进行分析。

## 注意

- DataRecipe 是软依赖，需要单独安装
- 分析会从 HuggingFace 下载 300 条样本数据
- 每次分析会在输出目录生成 23+ 分析文件
- 如果想了解数据集是否值得分析，先使用 `/deep-dive {数据集ID}` 评估
