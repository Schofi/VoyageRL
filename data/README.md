# 数据说明

公开仓库只附带 [examples/](../examples/) 中为展示而编写的合成输入。示例不包含真实用户记录、模型生成答案或评测结果，也不足以代表训练集。

| 位置 | 用途 | 是否随仓库发布 |
| --- | --- | --- |
| `examples/questions.jsonl` | `qid`、`question` 格式的问题示例 | 是 |
| `examples/inference.jsonl` | `id`、`messages` 格式的天气推理输入 | 是 |
| `data/train_seed.jsonl` | 问题扩充种子，本地创建 | 否 |
| `data/train_expansion_50.jsonl` | 模型扩充结果 | 否 |
| `data/clean/` | 清洗输入、输出与报告 | 否 |
| `data/final/` | 用户自行准备的 SFT/RL 数据 | 否 |
| `data/local/` | 本地保留的数据、旧结果及参考记录 | 否 |

`.gitignore` 排除了本地数据与训练产物。已有本地 `data/local/final/` 时，可以检查来源及格式后复制到 `data/final/` 供本机使用；全新克隆没有这些文件。历史结果不代表本次整理重新运行所得。

蒸馏保存逐样本 JSON，主要轨迹字段为 `messages`；清洗脚本要求每行含 `id` 与 `conversations`；推理输入使用 `messages`；独立评估使用 `query` 与完整 `messages`。这些格式不能不经转换就直接串联，具体步骤见 [复现指南](../docs/reproduction.md)。

默认训练路径为 `sft_train.jsonl`、`sft_val.jsonl` 和 `rl.jsonl`，均位于 `data/final/`。划分数据时按原始问题和近似改写检查泄漏，记录生成模型、提示词、时间与清洗规则。发布新数据前确认使用与分发权限，并移除密钥和个人信息。
