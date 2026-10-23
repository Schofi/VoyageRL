# VoyageRL｜旅行规划智能体与强化学习实践

**Stack:** Python 3.11 / Qwen-Agent / Transformers / vLLM / ms-swift
**Version:** 未发布版本；原训练环境 ms-swift 4.2.0.dev0。
**Port:** 无应用端口；训练 rollout 默认 `127.0.0.1:8000`。

## What

旅行规划 Agent 的问题扩充、教师轨迹蒸馏、清洗、多轮推理、SFT/GRPO 接入与评估。
航班和火车票由模型模拟，不是实时库存。GPU 训练与模型效果尚未在此发布版本验证。

## Quick Start

Windows PowerShell，项目根目录：

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements/dev.txt
.\.venv\Scripts\python.exe -m pytest -q
```

Linux/WSL，项目根目录：

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements/dev.txt
.venv/bin/python -m pytest -q
```

## Commands

下面使用已配置运行环境的 Python；真实调用需要模型、数据和服务凭据。

```bash
python -m pip install -r requirements.txt
cp .env.example .env
python data_distill.py --mode batch --filename ../examples/questions.jsonl --save-dir saved/demo-trajectories
python -m data_clean.format_filter --help
python -m data_clean.quality_filter --help
python llm_judge.py --help
```

Linux/WSL Bash 启动入口：`scripts/infer.sh`、`train_sft.sh`、`merge_lora.sh`、
`rollout.sh`、`train_rl.sh`，均需要 `MODEL_PATH`；合并还需 `ADAPTER_PATH`。
详细命令与前提见 `docs/reproduction.md`。没有构建、开发服务器或统一 lint 命令。

## Architecture

```text
data_clean/        格式归一与模型质量筛选
tools/             地图/天气/网页/模拟票务工具
training/plugins/  ms-swift 多轮调度与六项课程奖励
scripts/           读取 .env 的 Bash 启动器
utils/             文本处理、日志与请求级遥测
tests/             无 API/GPU 的离线测试
examples/          合成问题和天气推理输入
docs/              架构、复现、评估与面试说明
```

教师蒸馏产生工具轨迹，清洗后自行整理训练集，SFT/GRPO 由 ms-swift 执行。
训练调度器和奖励复用推理解析器。推理结果可进行双向模型评估，并附加统计日志。
逐样本蒸馏 JSON、清洗 conversations JSONL、推理 messages JSONL 不能直接混用。

## Key Files

- `data_distill.py`：轨迹生成、完整性校验、续跑和用量统计。
- `test_expand_50.py`：有界扩充脚本，不是 pytest 测试；会调用 API。
- `run_tool_loop_infer.py`：推理循环、解析器、工具调用与终止控制。
- `prompt.py`：系统、工具协议与评审提示词。
- `llm_judge.py`：按 query 交集匹配，双向打分；需完整 messages。
- `training/plugins/tooluse_multi_turn_scheduler.py`：rollout 调度。
- `training/plugins/tooluse_reward_parser_aligned.py`：六项加权课程奖励。
- `utils/inference_telemetry.py`：无正文遥测；token 为估算，TTFT/TPOT 为空。
- `scripts/common.sh`：加载 .env 并切换项目根目录。

## Configuration

配置参考 `.env.example`；模型、数据和 GPU 参数也通过启动器环境变量传入。

| 变量 | 何时需要 | 用途 |
| --- | --- | --- |
| `AGENT_API_KEY`、`AGENT_BASE_URL`、`AGENT_MODEL_ID` | 教师蒸馏 | 教师服务连接 |
| `TOKENIZER_PATH` | 蒸馏 | 本地 tokenizer 或模型 ID |
| `OPENAI_API_KEY`、`OPENAI_BASE_URL`、`LLM_MODEL_ID` | 提取/模拟 | 通用模型服务 |
| `JUDGE_MODEL_ID` | 评估/筛选 | 评审模型，另需服务凭据 |
| `AMAP_MAPS_API_KEY` | 地图/天气 | 高德 Web 服务 |
| `FIRECRAWL_API_KEY` | 默认搜索 | 搜索提供方 |
| `INFERENCE_METRICS_PATH` | 可选 | 请求级 JSONL 日志 |

## Contributing

见 `CONTRIBUTING.md`。保留来源说明，不提交 .env、本地数据或模型权重。
依赖范围不等于 GPU 锁定环境；不把 mock 测试、旧指标或合成输入描述为训练效果。
项目尚无整体许可证，第三方声明见 `THIRD_PARTY_NOTICES.md`。
