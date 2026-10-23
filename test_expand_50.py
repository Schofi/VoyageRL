#!/usr/bin/env python3
"""GLM-5.3-Flash 数据扩充测试：新增 50 条后立即结束，无后台请求。"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from openai import APIError, OpenAI
from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")
INPUT_PATH = BASE_DIR / "data" / "train_seed.jsonl"
OUTPUT_PATH = BASE_DIR / "data" / "train_expansion_50.jsonl"
# 配置来自环境变量或项目根目录的 .env。
API_KEY = os.getenv("EXPANSION_API_KEY", os.getenv("OPENAI_API_KEY", ""))
BASE_URL = os.getenv("EXPANSION_BASE_URL", os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))
MODEL = os.getenv("EXPANSION_MODEL", "ZHIPU/GLM-5.3-Flash")
MAX_NEW_SAMPLES = 50       # 最终新增条数，不是种子条数
BATCH_SIZE = 5
MAX_API_CALLS = 30         # 含失败请求，达到上限也退出
MAX_NO_PROGRESS = 3        # 连续 3 次失败/无新增数据时退出
REQUEST_TIMEOUT = 60.0    # 单次请求超时，单位：秒

QUERY_EXPANSION_PROMPT = """
你是一个经验丰富的旅游问题改写专家，熟悉中国的旅游城市、交通和景点路线。
请参考下面的用户问题，扩充生成{count}个全新的旅游问题。

要求：
1. 原问题仅供参考，不要复述原问题，也不要重复生成的问题。
2. 地名和景点必须真实存在，路线、交通和预算等条件必须符合逻辑。
3. 可以替换城市、地点，或增加价格、距离、评分、交通方式、预算、途经点等约束。
4. 每行只输出一个完整的问题，不要输出编号、解释、标题、代码块或其他内容。

用户问题：{question}
"""


def build_client() -> OpenAI:
    """从环境配置构建客户端；密钥不会保存在源代码中。"""
    if not API_KEY:
        raise ValueError("请配置 EXPANSION_API_KEY 或 OPENAI_API_KEY。")
    return OpenAI(
        api_key=API_KEY,
        base_url=BASE_URL,
        timeout=REQUEST_TIMEOUT,
        max_retries=0,  # 关闭 SDK 的隐式重试，确保请求次数有明确上限
    )


def load_seed_questions(path: Path) -> list[str]:
    questions = []
    with path.open(encoding="utf-8-sig") as source:
        for line_no, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"种子文件第 {line_no} 行不是合法 JSON。") from exc
            question = item.get("question") if isinstance(item, dict) else None
            if not isinstance(question, str) or not question.strip():
                raise ValueError(f"种子文件第 {line_no} 行缺少非空 question 字符串。")
            questions.append(" ".join(question.split()))
    if not questions:
        raise ValueError("种子文件没有可用问题，测试结束。")
    return questions


def parse_questions(content: str) -> list[str]:
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
    questions = []
    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith("```"):
            continue
        line = re.sub(r"^(?:[-*•]\s*|\d+\s*[.)、．）:：]\s*)", "", line)
        question = " ".join(line.strip(' \t\"“”').split())
        if question:
            questions.append(question)
    return questions


def expand_to_50(
    client: OpenAI,
    input_path: Path = INPUT_PATH,
    output_path: Path = OUTPUT_PATH,
    *,
    model: str = MODEL,
    max_api_calls: int = MAX_API_CALLS,
) -> int:
    """逐次调用；仅写入新增问题，达到 50 条立刻停止发起请求。"""
    input_path = Path(input_path).resolve()
    output_path = Path(output_path).resolve()
    if input_path == output_path:
        raise ValueError("输入文件和输出文件必须不同。")
    if max_api_calls < 1:
        raise ValueError("max_api_calls 必须大于 0。")

    seeds = load_seed_questions(input_path)
    seen = set(seeds)  # 排除全部原始种子，不把原问题计为新增数据
    count = 0
    no_progress = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"模型：{model}；目标：新增 {MAX_NEW_SAMPLES} 条；最多 {max_api_calls} 次请求。")
    print(f"输出：{output_path}")

    # 不使用线程池，也不提前提交请求：停止时没有后台任务继续调用 API。
    # 每次运行覆盖本测试输出，不追加到已有的大批量扩充文件。
    with output_path.open("w", encoding="utf-8") as output:
        for call_no in range(1, max_api_calls + 1):
            remaining = MAX_NEW_SAMPLES - count
            question = seeds[(call_no - 1) % len(seeds)]
            prompt = QUERY_EXPANSION_PROMPT.format(
                count=min(BATCH_SIZE, remaining), question=question
            )
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "你是有用的人工智能助手。"},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=4096,
                )
                content = response.choices[0].message.content if response.choices else None
                new_questions = parse_questions(content) if isinstance(content, str) else []
            except APIError as exc:
                status = getattr(exc, "status_code", None)
                print(f"请求 {call_no} 失败：{type(exc).__name__}，HTTP 状态：{status}。")
                if status in {400, 401, 403, 404}:
                    print("请检查 API Key、接口地址、模型名称及开通权限；停止测试。")
                    break
                new_questions = []

            before = count
            for new_question in new_questions:
                if new_question in seen:
                    continue
                seen.add(new_question)
                output.write(json.dumps({"question": new_question}, ensure_ascii=False) + "\n")
                count += 1
                if count == MAX_NEW_SAMPLES:
                    break
            output.flush()  # 每批保存；出错或 Ctrl+C 时保留已经生成的数据
            print(f"请求 {call_no}/{max_api_calls}：已保存 {count}/{MAX_NEW_SAMPLES} 条。", flush=True)
            if count == MAX_NEW_SAMPLES:
                print("已新增 50 条，立即停止，不再发送任何请求。")
                break

            no_progress = no_progress + 1 if count == before else 0
            if no_progress >= MAX_NO_PROGRESS:
                print(f"连续 {MAX_NO_PROGRESS} 次没有新增数据，停止测试。")
                break

    if count < MAX_NEW_SAMPLES:
        print(f"测试已结束，只生成 {count}/50 条；已保存现有结果，不会继续循环。")
    return count


def main() -> int:
    try:
        with build_client() as client:
            count = expand_to_50(client, model=MODEL)
    except KeyboardInterrupt:
        print("测试已手动停止，已经写入的数据会保留。")
        return 130
    except (ValueError, OSError, APIError) as exc:
        print(f"测试未完成：{exc}")
        return 1
    return 0 if count == MAX_NEW_SAMPLES else 1


if __name__ == "__main__":
    raise SystemExit(main())
