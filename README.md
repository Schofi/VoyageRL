# VoyageRL｜旅行规划智能体与强化学习实践

围绕旅行规划任务，整理问题扩充、教师轨迹蒸馏、数据清洗、多轮工具推理、SFT 与 GRPO 训练接口及评估流程。重点是把模型输出转化为可执行的工具交互，并让训练奖励与实际推理协议保持一致。

这是一个代码与实验流程仓库，没有 Web 界面。公开版本包含合成输入示例和离线测试，不包含模型权重、完整训练数据或已复现的效果排名。

## 能看到什么

| 能力 | 实现入口 | 关注的问题 |
| --- | --- | --- |
| 有界问题扩充 | [test_expand_50.py](test_expand_50.py) | 去重、请求预算、失败退出与增量落盘 |
| 教师轨迹蒸馏 | [data_distill.py](data_distill.py) | 多轮工具交互、按样本续跑、token 用量记录 |
| 数据清洗 | [data_clean/](data_clean/) | 工具调用格式归一、异常轨迹过滤、模型质量判断 |
| 多轮推理 | [run_tool_loop_infer.py](run_tool_loop_infer.py) | Transformers/vLLM 后端、工具执行、重复调用与轮次限制 |
| 训练接入 | [training/plugins/](training/plugins/) | 复用推理解析器的调度器、六项课程奖励、GRPO |
| 评估与观测 | [llm_judge.py](llm_judge.py)、[utils/inference_telemetry.py](utils/inference_telemetry.py) | 双向比较、请求耗时、工具错误与估算 token |

工具涵盖天气、POI、周边搜索、路线规划、网页搜索、网页访问、航班与火车票八类。**航班和火车票由模型模拟生成，不连接实时库存，也不能用于订票。** 搜索与地图类工具需要各自的服务凭据。

## 先运行离线测试

需要 Python 3.11。下载仓库后，在项目根目录执行；不需要 API 密钥、GPU 或模型权重。

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

新建 Python 3.11 环境已验证 **71 项离线测试通过**，覆盖扩充边界、蒸馏续跑、票务参数处理与推理遥测等行为。它们使用替身隔离外部依赖，不能证明真实 API 可用性或模型规划能力。验证范围见 [评估说明](docs/evaluation.md)。

## 运行模型与工具

在独立环境中安装 [requirements.txt](requirements.txt)，复制 [.env.example](.env.example) 为 `.env`，填写所需凭据。Linux / WSL Bash 示例：

```bash
python -m pip install -r requirements.txt
cp .env.example .env
# 编辑 .env，至少配置天气示例需要的 AMAP_MAPS_API_KEY。
export MODEL_PATH=/absolute/path/to/your/model
bash scripts/infer.sh
```

该命令默认读取 [天气示例](examples/inference.jsonl)，需要可用的模型和运行环境，会调用外部天气服务。模型目录为待替换路径；Windows 原生环境请参考 [复现指南](docs/reproduction.md) 中的 Python 命令。不要将 `.env` 提交到仓库。

SFT、LoRA 合并、rollout 服务与 GRPO 的启动入口位于 [scripts/](scripts/)。**完整 GPU 训练尚未在此发布版本中验证。** 原环境使用 `ms-swift 4.2.0.dev0`；[训练依赖](requirements/training.txt) 是候选范围，不是可复现锁文件。训练默认使用多张 GPU，不能直接当作单卡配置。

## 文档导航

- [架构与数据流](docs/architecture.md)：工具协议、调度器、奖励组成及模块边界。
- [复现指南](docs/reproduction.md)：配置、数据准备、运行命令和训练前提。
- [评估说明](docs/evaluation.md)：验证范围、指标口径与建议实验设计。
- [面试讲解](docs/interview-guide.md)：两分钟介绍、代码证据与技术追问。
- [数据说明](data/README.md)：公开示例、本地数据和格式区别。

## 贡献与来源

修改建议和问题报告见 [CONTRIBUTING.md](CONTRIBUTING.md)。本仓库基于已有实现进行整理、迁移和工程化，第三方框架与组件的来源单独说明。

仓库尚未选择项目整体许可证，当前没有新增整体授权声明。第三方组件保留各自许可证；ms-swift 的 Apache-2.0 文本及来源说明见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
