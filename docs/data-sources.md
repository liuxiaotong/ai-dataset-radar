# 数据源 / Data Sources

> 返回 [README](../README.md)

## 监控范围

| 来源 | 数量 | 覆盖 |
|------|-----:|------|
| **HuggingFace** | 86 orgs | 67 Labs + 27 供应商（含机器人、欧洲、亚太） |
| **博客** | 71 源 | 实验室 + 研究者 + 独立博客 + 数据供应商 |
| **GitHub** | 50 orgs | AI Labs + 中国开源 + 机器人 + 数据供应商 |
| **论文** | 2 源 | arXiv (cs.CL/AI/LG/CV/RO) + HF Papers |
| **X/Twitter** | 125 账户 | 13 类别，CEO/Leaders + 研究者 + 机器人 |
| **Reddit** | 5 社区 | MachineLearning、LocalLLaMA、dataset、deeplearning、LanguageTechnology |

## 数据供应商分类

| 类别 | 覆盖 |
|------|------|
| **Premium（海外）** | Scale AI, Appen, Mercor, Invisible Technologies, TELUS Digital |
| **Specialized（海外）** | Surge AI, Snorkel AI, Labelbox, Turing, Prolific, Cohere for AI |
| **China Premium（中国）** | 海天瑞声, 整数智能 MolarData, 云测数据 Testin |
| **China Specialized（中国）** | 标贝科技 DataBaker, 数据堂 Datatang |
| **China Research（中国）** | 智源研究院 BAAI |

## X/Twitter 监控账户

通过自托管 RSSHub（推荐）或 X API v2 监控 125 个账户。多 RSSHub 实例自动 fallback + 连续失败阈值保护。

| 类别 | 数量 | 代表账户 |
|------|-----:|----------|
| CEO/Leaders | 4 | sama, DarioAmodei, demaborishassabis |
| 前沿实验室 | 8 | OpenAI, AnthropicAI, GoogleDeepMind, MetaAI, NVIDIAAI |
| 新兴/开源 | 12 | MistralAI, CohereForAI, StabilityAI, NousResearch |
| 研究/开源 | 5 | AiEleuther, huggingface, allen_ai, lmsysorg |
| 中国实验室 | 14 | Alibaba_Qwen, deepseek_ai, BaichuanAI, Kimi_Moonshot |
| 亚太/欧洲 | 11 | SakanaAILabs, NAVER_AI_Lab, laion_ai, StanfordHAI |
| 机器人公司 | 10 | Figure_robot, physical_int, UnitreeRobotics, AgiBot_zhiyuan |
| 机器人研究者 | 10 | pabbeel, svlevine, chelseabfinn, LerrelPinto |
| 数据供应商 | 9 | scale_AI, HelloSurgeAI, argilla_io, LabelBox |
| 基准/MLOps | 7 | lmarena_ai, ArtificialAnlys, kaggle, modal_labs |
| 安全/对齐 | 4 | ai_risks, JaredKaplan |
| 研究者 | 31 | karpathy, ylecun, jimfan, emollick, Hesamation |

信号关键词过滤：dataset, training data, benchmark, RLHF, synthetic data, fine-tuning 等。完整列表见 `config.yaml`。

## 数据集分类体系

多维评分分类：关键词(+1) + 名称模式(+2) + 字段模式(+2) + 标签(+3)，阈值 ≥ 2 分。

| 类别 | 关键词示例 | 典型数据集 |
|------|-----------|-----------|
| **sft** | instruction, chat, dialogue | Alpaca, ShareGPT |
| **preference** | rlhf, dpo, chosen/rejected | UltraFeedback, HelpSteer |
| **reward_model** | reward, ppo | RationaleRM |
| **synthetic** | synthetic, distillation | Magpie, Sera |
| **agent** | tool use, function calling | SWE-bench, WebArena |
| **multimodal** | image, video, audio, speech, OCR, document, CLIP | LLaVA, Numb3rs, doc_split |
| **multilingual** | multilingual, translation | WaxalNLP, EuroLLM |
| **rl_environment** | robot, embodied, haptic, simulation | RoboCasa, ToucHD, LIBERO |
| **code** | programming, verification, proof | StarCoder, Verus |
| **evaluation** | benchmark, safety guard, control task | Nemotron-Safety |
