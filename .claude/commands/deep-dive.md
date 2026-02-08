# 深度分析

针对指定的组织、数据集或数据类别，从最新报告中提取全部相关信息并生成竞争情报分析。

## 参数

$ARGUMENTS - 分析目标，支持三种类型：
- **组织名**：如 `NVIDIA`、`Qwen`、`Meta`、`Allen AI`、`BAAI`
- **数据集 ID**：如 `Qwen/RationaleRM`、`nvidia/Nemotron-Safety-Guard-Dataset-v3`
- **数据类别**：如 `reward_model`、`sft_instruction`、`rl_environment`、`multimodal`

## 执行步骤

### 第一步：定位最新报告

```bash
ls -t /Users/liukai/ai-dataset-radar/data/reports/*/intel_report_*.json 2>/dev/null | head -1
```

读取完整 JSON 报告。

### 第二步：判断目标类型并提取数据

**判断逻辑**：
- 如果包含 `/`（如 `Qwen/RationaleRM`）→ 数据集 ID
- 如果是已知分类名（`sft_instruction`, `reward_model`, `rl_environment`, `code`, `agent_tool`, `synthetic`, `multimodal`, `multilingual`, `evaluation`, `rlhf_preference`）→ 数据类别
- 否则 → 组织名

### 对于组织名

从所有数据源中提取该组织的活动：

1. **数据集**：过滤 `id` 中包含组织名（不区分大小写）的数据集
2. **模型**：如果 JSON 中有 models 信息，同样过滤
3. **GitHub**：在 `github_activity` 中查找匹配的 org
4. **论文**：在论文标题/摘要中搜索组织名
5. **博客**：在博客标题/来源中搜索

输出格式：
```
## 组织深度分析：{组织名}

### 数据集发布
[表格：ID、类别、下载量、Likes、发布日期]

### 模型发布
[表格：ID、Pipeline、下载量、发布日期]

### GitHub 活动
[表格：仓库、Stars、描述、信号]

### 相关论文
[列表：标题、日期、摘要]

### 数据策略分析
[竞争情报分析：该组织的数据战略、训练重点、对数据服务商的启示]
```

### 对于数据集 ID

从报告中提取该数据集的完整信息：

1. 数据集详情（id、category、downloads、likes、signals、tags、license、created_at、description）
2. 同类数据集（相同 category 的其他数据集，作为竞品对比）
3. 关联论文（标题中包含该数据集名称或关键词的论文）
4. 所属组织的其他活动

输出格式：
```
## 数据集深度分析：{数据集ID}

### 基本信息
[详细字段表]

### 同类竞品
[表格：同 category 的数据集对比]

### 关联研究
[匹配的论文列表]

### 评估
- 市场定位：[分析]
- 数据服务机会：[分析]
- 是否值得 DataRecipe 分析：[建议，如值得可使用 `/recipe {ID}`]
```

### 对于数据类别

汇总该类别下的所有数据集：

1. 该类别的数据集列表（按下载量排序）
2. 相关论文（按类别关键词搜索）
3. GitHub 中相关仓库
4. 趋势分析

输出格式：
```
## 类别深度分析：{类别名}

### 数据集列表
[表格：按下载量排序]

### 市场趋势
[分析该类别的供需状况、增长方向]

### 相关研究
[论文列表]

### 机会与建议
[数据服务商的切入点]
```

## 注意

- 也可以读取 `config.yaml` 获取组织的监控配置（HF IDs、GitHub orgs 等）
- 分析应该对数据服务公司有实际参考价值
- 如果某数据集值得更深入分析，建议使用 `/recipe {数据集ID}`
- 如果用户想搜索更广泛的关键词，建议使用 `/search`
