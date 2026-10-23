# 示例数据

这些是为仓库展示编写的合成输入，不含原始用户记录，也不是模型评测结果。

| 文件 | 用途 | 字段 |
| --- | --- | --- |
| `questions.jsonl` | 展示蒸馏/问题扩充输入 | `qid`、`question` |
| `inference.jsonl` | 单条天气工具推理输入 | `id`、`messages` |

`inference.jsonl` 包含工具定义和调用协议，供 `scripts/infer.sh` 使用。执行它需要模型权重、运行依赖及高德 Web 服务密钥；查看文件和运行离线测试不需要这些资源。

训练集请单独放入 `data/final/`。不要用这几条演示数据报告训练效果。
