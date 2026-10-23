# 第三方组件与来源说明

本仓库由已有实现整理、迁移和工程化而来。仓库名称与新的 Git 历史不改变既有代码的来源或权利归属，也不构成对全部实现的原创声明。

## ms-swift

训练脚本和 `training/plugins/` 对接 ms-swift 的多轮 rollout 与奖励扩展接口。整理前的训练环境标识为 `4.2.0.dev0`；公开仓库只保留任务相关插件，不包含整个框架副本。

ms-swift 上游项目为 `modelscope/ms-swift`，其 Apache License 2.0 文本保留在 [third_party/ms-swift-LICENSE](third_party/ms-swift-LICENSE)。该文本适用于相应上游组件，不自动覆盖本项目其他文件。后续修改或分发上游材料时应保留对应声明。

## 其他依赖

依赖清单位于 [requirements.txt](requirements.txt)、[requirements/dev.txt](requirements/dev.txt) 和 [requirements/training.txt](requirements/training.txt)。Qwen-Agent、Transformers、vLLM、DeepSpeed、OpenAI SDK 等依赖保留各自许可证；其模型权重、外部 API、地图和搜索数据也可能有单独条款。

本仓库未包含模型权重、外部服务凭据或实时票务数据库。航班与火车票工具是模型模拟实现，不代表与任何票务服务存在集成或合作。

## 项目整体许可状态

目前未选择项目整体许可证，也未新增整体版权授权。公开展示和提供源代码不应被解读为这里已经授予一套新的统一许可。待项目权利和授权范围确认后，再由维护者加入适当的根目录许可证；不要将保留的上游 Apache-2.0 文本直接解释为全仓库许可。
