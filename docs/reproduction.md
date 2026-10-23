# 复现指南

所有命令从项目根目录执行。Python 基线为 3.11；离线测试可以在 Windows 运行，训练启动脚本需要 Linux / WSL Bash。模型、数据、凭据和 GPU 环境不随仓库提供。

## 1. 最小离线环境

Windows PowerShell：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements/dev.txt
.\.venv\Scripts\python.exe -m pytest -q
```

Linux / WSL：

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements/dev.txt
.venv/bin/python -m pytest -q
```

`requirements/dev.txt` 只安装离线测试所需依赖；通过这些测试不代表已安装模型推理或训练环境。

## 2. 配置真实调用

建议另建运行环境，使用其 Python 安装 `requirements.txt`。下文的 `python` 指该环境的解释器。

```bash
python -m pip install -r requirements.txt
```

复制 `.env.example` 为 `.env`（PowerShell 使用 `Copy-Item .env.example .env`；Bash 使用 `cp .env.example .env`），并编辑需要的变量：

| 用途 | 配置 |
| --- | --- |
| 教师轨迹蒸馏 | `AGENT_API_KEY`、`AGENT_BASE_URL`、`AGENT_MODEL_ID`、`TOKENIZER_PATH` |
| 通用模型提取与交通模拟 | `OPENAI_API_KEY`、`OPENAI_BASE_URL`、`LLM_MODEL_ID` |
| 评估与质量筛选 | `JUDGE_MODEL_ID` 及通用服务凭据；独立评估支持 `JUDGE_API_KEY` / `JUDGE_BASE_URL` |
| 问题扩充 | `EXPANSION_MODEL`；`EXPANSION_API_KEY` / `EXPANSION_BASE_URL` 未设置时回退到通用凭据 |
| 高德地图与天气 | `AMAP_MAPS_API_KEY` |
| 默认网页搜索 | `FIRECRAWL_API_KEY`；替代模块使用各自提供方密钥 |

`TOKENIZER_PATH` 可为本地 tokenizer 路径或模型 ID；默认模型 ID 首次使用可能下载文件。外部模型调用、搜索和地图服务可能计费。

## 3. 数据扩充与蒸馏

先复制合成问题到扩充脚本使用的种子路径。Bash 示例：

```bash
cp examples/questions.jsonl data/train_seed.jsonl
python test_expand_50.py
python data_distill.py --mode batch --filename train_expansion_50.jsonl
```

PowerShell 使用 `Copy-Item examples/questions.jsonl data/train_seed.jsonl`，随后 Python 命令相同。扩充会写入 `data/train_expansion_50.jsonl`，最多新增 50 条；在请求耗尽或连续无进展时可能不足 50 条。重新执行扩充会重新写输出，保存已有结果后再运行。

只生成三条示例的教师轨迹时，可直接运行：

```bash
python data_distill.py --mode batch --filename ../examples/questions.jsonl --save-dir saved/demo-trajectories
```

`--filename` 相对于 `data/` 解析。蒸馏按样本保存 JSON；批量模式默认续跑，校验通过的结果跳过，未完成或损坏结果重试。`--no-resume` 会重新生成。公开示例只有输入，没有预先编造的模型输出。

## 4. 清洗与训练集准备

清洗脚本的输入每行为 `{"id": "...", "conversations": [...]}`。蒸馏结果包含 `messages`；应先完成字段映射、逐样本 JSON 合并和数据检查，再执行清洗。仓库目前没有一个命令自动完成这一步；不要直接把推理示例当作清洗输入。

准备好 `data/clean/raw.jsonl` 后：

```bash
python -m data_clean.format_filter --input data/clean/raw.jsonl --output data/clean/formatted.jsonl --report data/clean/format_report.json
python -m data_clean.quality_filter --input data/clean/formatted.jsonl --output data/clean/judged.jsonl --summary-output data/clean/quality_summary.json --filtered-output data/clean/filtered.jsonl
```

格式清洗不依赖模型调用；质量筛选会调用评审模型。质量筛选输出仍需转换/检查为训练框架支持的 schema，并划分互不泄漏的训练、验证和测试集。

训练脚本默认读取 `data/final/sft_train.jsonl`、`data/final/sft_val.jsonl`、`data/final/rl.jsonl`。GitHub 克隆不含这些数据，需要自行准备。整理前的本地数据若存在于 `data/local/final/`，可在检查来源和格式后复制到 `data/final/`；不要把它们误认为公开示例或此次重新生成的结果。

Notebook 属于辅助分析入口；如需运行，另外安装 `jupyter pandas matplotlib`。它们依赖本地数据，不能替代上述数据检查。

## 5. 多轮推理

Linux / WSL，环境已激活、模型路径已准备：

```bash
export MODEL_PATH=/absolute/path/to/your/model
bash scripts/infer.sh
```

`MODEL_PATH` 是待替换值。默认使用 Transformers、处理一条天气示例，将完整结果写入 `saved/inference/results.jsonl`。遥测路径由 `INFERENCE_METRICS_PATH` 控制；复制示例配置后为 `outputs/inference_metrics.jsonl`。

Windows 可以绕过 Bash 使用 Python 入口（`C:/models/your-model` 同样需替换）：

```powershell
python run_tool_loop_infer.py --infer_backend transformers --model_dir C:/models/your-model --dataset_path examples/inference.jsonl --tools_dir tools --num_samples 1 --max_turns 13 --tool_first_enforce --save_full_messages --output_path saved/inference/results.jsonl --metrics_output outputs/inference_metrics.jsonl
```

这条命令是代码支持的入口，不代表 Windows 模型运行已验证；硬件和模型格式必须满足推理后端要求。

## 6. SFT 与 GRPO

训练环境尚未完整验证。先检查 CUDA、PyTorch、vLLM、DeepSpeed 与 ms-swift 的兼容性；`requirements/training.txt` 包含候选依赖，原环境为 `ms-swift 4.2.0.dev0`，不是已验证的可复现版本组合。

```bash
python -m pip install -r requirements/training.txt
export MODEL_PATH=/absolute/path/to/base-model
bash scripts/train_sft.sh
export ADAPTER_PATH=/absolute/path/to/lora-checkpoint
bash scripts/merge_lora.sh
export MODEL_PATH=/absolute/path/to/merged-model
```

准备合并后的模型与 RL 数据后，在一个终端运行 `bash scripts/rollout.sh`，另一个终端运行 `bash scripts/train_rl.sh`；两个终端都应配置相同的 `MODEL_PATH` 和环境。rollout 默认监听 `127.0.0.1:8000`。

SFT 默认 4 个进程；RL 默认 rollout 使用 GPU `0,1`，训练使用 `2,3,4,5,6,7`。需按机器修改脚本支持的环境变量；减少 GPU 数量还要检查张量并行度、生成数量、批大小及显存约束。这里没有经过验证的单卡训练配方。

## 7. 比较两组输出

两个输入文件必须包含相同问题的 `query` 和完整 `messages`，可由推理的 `--save_full_messages` 生成：

```bash
python llm_judge.py --reference saved/inference/baseline.jsonl --prediction saved/inference/candidate.jsonl
```

文件需由你实际运行产生。评估按问题文本取交集，重复问题会覆盖；先检查样本唯一性和两组问题集合。脚本进行交换顺序的双向打分并打印均值，细节与局限见 [评估说明](evaluation.md)。
