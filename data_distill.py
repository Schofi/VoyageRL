# -*- coding: utf-8 -*-
# --------------------------------------------
# 文件描述: 蒸馏推理路径和答案
# --------------------------------------------

import json
import re
import os
import ast
import sys
import time
import json5
import hashlib
import asyncio
import argparse
import datetime
import tempfile
import json_repair
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import urlparse
from dotenv import load_dotenv
from pydantic import BaseModel
from qwen_agent.agents.fncall_agent import FnCallAgent
from qwen_agent.llm.schema import Message

from transformers import AutoTokenizer
from transformers import PreTrainedTokenizerFast
from openai import AsyncOpenAI

from utils import logging as agent_logging
from tools.tool_web_search import WebSearch
from tools.tool_visit import Visit
from tools.tool_weather import WeatherSearch
from tools.tool_transport import FlightsSearch 
from tools.tool_train_ticket import TrainTicketsSearch
from tools.tool_route_planning import RoutePlanning
from tools.tool_poi_search import POISearch
from tools.tool_around_search import AroundSearch
from prompt import AGENTIC_SYSTEM_PROMPT


# 初始化logger
logger = agent_logging.get_logger(__name__)

# 加载环境变量
load_dotenv()


def today_date():
    return datetime.date.today().strftime("%Y-%m-%d")


def extract_token_usage(completion):
    """读取服务端实际返回的用量；缺失的缓存字段不当作零命中。"""
    counts = {"requests": 1, "usage_reported_requests": 0, "cache_reported_requests": 0}
    usage = getattr(completion, "usage", None)
    if isinstance(usage, BaseModel):
        usage = usage.model_dump()
    if not isinstance(usage, dict):
        return counts
    counts["usage_reported_requests"] = 1
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = usage.get(key)
        if type(value) is int and value >= 0:
            counts[key] = value

    hit = usage.get("prompt_cache_hit_tokens")
    if hit is None:
        details = usage.get("prompt_tokens_details") or {}
        hit = details.get("cached_tokens")
    miss = usage.get("prompt_cache_miss_tokens")
    if miss is None and type(hit) is int and "prompt_tokens" in counts:
        miss = counts["prompt_tokens"] - hit
    if type(hit) is int and type(miss) is int and hit >= 0 and miss >= 0:
        counts.update(
            cache_reported_requests=1,
            prompt_cache_hit_tokens=hit,
            prompt_cache_miss_tokens=miss,
        )
    return counts


@contextmanager
def batch_logging(log_dir: str = "saved/logs"):
    """每次批量运行创建独立的 UTF-8 日志，并在退出时关闭文件。"""
    output_dir = Path(log_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    with tempfile.NamedTemporaryFile(
        dir=output_dir, prefix=f"batch_{timestamp}_", suffix=".log", delete=False
    ) as fd:
        log_path = Path(fd.name)
    handler = agent_logging.FileHandler(log_path, encoding="utf-8")
    handler.setLevel("DEBUG")
    handler.setFormatter(agent_logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    ))
    root = agent_logging.getLogger()
    previous_level = logger.level
    root.addHandler(handler)
    logger.setLevel("DEBUG")
    try:
        logger.info("Batch log file: %s", log_path)
        yield log_path
    except KeyboardInterrupt:
        logger.warning("Batch interrupted; rerun the same command to resume.")
        raise
    except Exception:
        logger.exception("Batch failed.")
        raise
    finally:
        logger.setLevel(previous_level)
        root.removeHandler(handler)
        handler.close()


def is_completed_result(result, data):
    """兼容旧结果文件，只把当前问题的有效答案视为已完成。"""
    if not isinstance(result, dict) or not isinstance(result.get("messages"), list):
        return False
    messages = [
        message.model_dump() if isinstance(message, BaseModel) else message
        for message in result["messages"]
    ]
    return (
        result.get("question") == data["question"]
        and result.get("termination") in (
            "answer", "generate an answer as token limit reached"
        )
        and isinstance(result.get("prediction"), str)
        and bool(result["prediction"].strip())
        and bool(messages)
        and all(
            isinstance(message, dict)
            and message.get("role") in ("system", "user", "assistant", "tool")
            for message in messages
        )
        and any(
            message.get("role") == "assistant"
            and isinstance(message.get("content"), str)
            and "<answer>" in message["content"]
            and "</answer>" in message["content"]
            for message in messages
        )
    )


def load_completed_result(data, save_path):
    if not save_path:
        return None
    result_path = Path(save_path) / f"{data['qid']}.json"
    if not result_path.is_file():
        return None
    try:
        with result_path.open(encoding="utf-8") as fd:
            result = json.load(fd)
    except (OSError, ValueError, UnicodeError) as exc:
        logger.warning(f"Result {result_path} is unreadable; retrying: {exc}")
        return None
    return result if is_completed_result(result, data) else None


def save_episode_result(data, result, save_path):
    """先写临时文件，再原子替换，避免中断破坏已保存的结果。"""
    if not save_path:
        return
    output_dir = Path(save_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / f"{data['qid']}.json"
    to_dump = {key: value for key, value in result.items() if key != "completions"}
    to_dump["messages"] = [
        message.model_dump() if isinstance(message, BaseModel) else message
        for message in result["messages"]
    ]
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=output_dir,
            prefix=f".{result_path.name}.", suffix=".tmp", delete=False,
        ) as fd:
            temp_path = Path(fd.name)
            json.dump(to_dump, fd, ensure_ascii=False, indent=4)
            fd.flush()
            os.fsync(fd.fileno())
        os.replace(temp_path, result_path)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
    logger.debug(f"Result dumped to {result_path}")


class MultiTurnReactAgent(FnCallAgent):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerFast | str = None,
        max_tokens_per_turn: int = 10000,
        max_llm_calls_per_run: int = 50,
        max_total_tokens: int = 204800,
    ):

        if not tokenizer:
            llm_model_path = os.getenv("TOKENIZER_PATH", "Qwen/Qwen3-4B-Instruct-2507")
            self.tokenizer = AutoTokenizer.from_pretrained(llm_model_path)
        elif isinstance(tokenizer, str):
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        else:
            self.tokenizer =  tokenizer

        self.max_tokens_per_turn = max_tokens_per_turn
        self.max_llm_calls_per_run = max_llm_calls_per_run
        self.max_total_tokens = max_total_tokens
        self.max_total_tokens_before_finishing = int(max_total_tokens * 0.8)
        self.usage_totals = {}
        self.tool_class = [
            Visit(),
            WebSearch(),
            WeatherSearch(),
            FlightsSearch(),
            TrainTicketsSearch(),
            RoutePlanning(),
            POISearch(),
            AroundSearch()
        ]
        self.tool_map = {tool.name: tool for tool in self.tool_class}
        self.tools = []
        for tool_name in self.tool_map:
            tool = self.tool_map[tool_name]
            self.tools.append({
                "type": "function",
                "function": tool.function
            })

    def count_tokens(self, messages):
        message_strs = []
        for msg in messages:
            if isinstance(msg, BaseModel):
                msg = msg.model_dump()
                assert "role" in msg and "content" in msg
            message_strs.append(
                f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
            )
        message_strs.append("<|im_start|>assistant\n")
        prompt_token_ids = self.tokenizer.encode("".join(message_strs))
        return len(prompt_token_ids)

    async def call_server(
        self, client: AsyncOpenAI, messages: list[dict], max_attempts: int = 2
    ) -> tuple:
        if urlparse(str(client.base_url)).hostname == "api.deepseek.com":
            # DeepSeek 自动管理上下文缓存，无需发送 cache 开关或缓存 ID。
            request_options = {
                "max_tokens": self.max_tokens_per_turn,
                "extra_body": {"thinking": {"type": "enabled"}},
            }
        else:
            request_options = {
                "max_completion_tokens": self.max_tokens_per_turn,
                "extra_body": {"reasoning_split": True, "enable_thinking": True},
            }
        for attempts in range(1, max_attempts + 1):
            try:
                completion = await client.chat.completions.create(
                    model=os.environ["AGENT_MODEL_ID"],
                    messages=messages,
                    tools=self.tools,
                    **request_options,
                )
                message = completion.choices[0].message
                assert message, "Error: LLM response is empty."
                return completion, message
            except RuntimeError as e:
                logger.warning(
                    f"RuntimeError during LLM call_server at attempt {attempts}: {e}"
                )
                continue
        raise RuntimeError(
            f"Failed to get response from LLM after {max_attempts} attempts."
        )

    def record_token_usage(self, stats, completion, qid):
        counts = extract_token_usage(completion)
        if not hasattr(self, "usage_totals"):
            self.usage_totals = {}
        for totals in (stats.setdefault("usage", {}), self.usage_totals):
            for key, value in counts.items():
                totals[key] = totals.get(key, 0) + value
        hit = counts.get("prompt_cache_hit_tokens")
        miss = counts.get("prompt_cache_miss_tokens")
        rate = f"{100 * hit / (hit + miss):.2f}%" if hit is not None and hit + miss > 0 else "unreported"
        logger.info(
            "QID %s usage: prompt_tokens=%s cache_hit_tokens=%s cache_miss_tokens=%s "
            "output_tokens=%s cache_hit_rate=%s",
            qid, counts.get("prompt_tokens", "unreported"),
            hit if hit is not None else "unreported",
            miss if miss is not None else "unreported",
            counts.get("completion_tokens", "unreported"), rate,
        )

    async def run_agent(
        self, data, client: AsyncOpenAI, save_path: str | None = None
    ) -> dict:
        start_time = time.time()
        qid = data["qid"]
        question = data["question"]
        answer = data.get("answer", "")
        self.user_prompt = question
        system_prompt = AGENTIC_SYSTEM_PROMPT 
        cur_date = today_date()
        system_prompt = system_prompt.replace("__CURRENT_DATE__", str(cur_date))
        system_prompt = system_prompt.replace("__MAX_TOOL_CALL__", str(self.max_llm_calls_per_run))
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]
        stats = dict(
            turns=0
        )
        num_llm_calls_available = self.max_llm_calls_per_run
        completions = []
        round = 0
        while num_llm_calls_available > 0:
            # Check whether time is reached
            if time.time() - start_time > 10 * 60:  # 10 minutes in seconds
                prediction = "No answer found after 10mins"
                termination = "No answer found after 10mins"
                result = {
                    "question": question,
                    "answer": answer,
                    "messages": messages,
                    "prediction": prediction,
                    "termination": termination,
                    "stats": stats,
                }
                save_episode_result(data, result, save_path)
                return result

            round += 1
            stats["turns"] += 1
            num_llm_calls_available -= 1
            completion, message = await self.call_server(client, messages)
            self.record_token_usage(stats, completion, qid)
            content = message.content
            completions.append(completion)
            messages.append(message)

            # 打印response
            if content:
                logger.info(f"[ASSITANT] LLM response: {content}")

            if message.tool_calls:
                try:
                    tool_calls = [
                        {
                            "id": item.id,
                            "name": item.function.name,
                            "arguments": json.loads(item.function.arguments)
                        } for item in message.tool_calls
                    ]
                except Exception:
                    try:
                        tool_calls = [
                            {
                                "id": item.id,
                                "name": item.function.name,
                                "arguments": json_repair.loads(item.function.arguments)
                            } for item in message.tool_calls
                        ]
                    except:
                        tool_calls = []
                        logger.warn(f"Parse tool_calls {matches} failed")
                tasks = [self.custom_call_tool(tool_call) for tool_call in tool_calls]
                results = await asyncio.gather(*tasks, return_exceptions=True)

                # 处理异常情况
                for tool_call, result in zip(tool_calls, results):
                    if isinstance(result, Exception):
                        result = "<tool_response>\n" + f"Fetch {tool_call} response failed." + "\n</tool_response>"

                    logger.info(f"[Tool] execute function call: {tool_call}")
                    logger.debug(
                        "[Tool] response id=%s name=%s result=%s",
                        tool_call["id"], tool_call["name"], result,
                    )
                    messages.append({"role": "tool", "tool_call_id": tool_call["id"], "content": result})
                    print(tool_call, result)

                    tool_name = tool_call["name"]
                    if tool_name not in stats:
                        stats[tool_name] = 0
                    stats[tool_name] += 1

            if content and "<answer>" in content and "</answer>" in content:
                termination = "answer"
                break

            if num_llm_calls_available <= 0 and "<answer>" not in content:
                messages.append(
                    {
                        "role": "user",
                        "content": "Sorry, the number of llm calls exceeds the limit. You should stop making tool calls and, "
                        "based on all the information above, think again and provide what you consider the most likely answer "
                        "in the following format:<think>your final thinking</think>\n<answer>your answer</answer>",
                    }
                )

            max_tokens = self.max_total_tokens_before_finishing
            token_count = self.count_tokens(messages)
            logger.debug(
                f"QID {data['qid']} Round: {round}, token count: {token_count}"
            )

            if token_count > max_tokens:
                logger.debug(
                    f"QID {data['qid']} Token quantity exceeds the limit: {token_count} > {max_tokens}"
                )
                messages.append(
                    {
                        "role": "user",
                        "content": "You have now reached the maximum context length you can handle. "
                        "You should stop making tool calls and, based on all the information above, "
                        "think again and provide what you consider the most likely answer in the following format:"
                        "<think>your final thinking</think>\n<answer>your answer</answer>",
                    }
                )
                completion, message = await self.call_server(client, messages)
                self.record_token_usage(stats, completion, qid)
                completions.append(completion)
                content = message.content
                messages.append(message)
                if "<answer>" in content and "</answer>" in content:
                    prediction = content.split("<answer>")[1].split("</answer>")[0]
                    termination = "generate an answer as token limit reached"
                else:
                    prediction = content
                    termination = (
                        "format error: generate an answer as token limit reached"
                    )
                result = {
                    "question": question,
                    "answer": answer,
                    "messages": messages,
                    "prediction": prediction,
                    "termination": termination,
                    "completions": completions,
                    "stats": stats,
                }
                token_count = self.count_tokens(messages)
                if token_count > self.max_total_tokens:
                    logger.warning(
                        f"Warning: total token count {token_count} exceeds the hard limit {self.max_total_tokens}."
                    )
                save_episode_result(data, result, save_path)
                return result

        if "<answer>" in content:
            prediction = content.split("<answer>")[1].split("</answer>")[0]
            termination = "answer"
        else:
            prediction = "No answer found."
            termination = "answer not found"
            if num_llm_calls_available == 0:
                termination = "exceed available llm calls"
        result = {
            "question": question,
            "answer": answer,
            "messages": [
                m.model_dump() if isinstance(m, BaseModel) else m for m in messages
            ],
            "prediction": prediction,
            "termination": termination,
            "completions": completions,  # final completion
            "stats": stats,
        }
        save_episode_result(data, result, save_path)
        return result

    async def custom_call_tool(self, tool_call: dict, **kwargs):
        tool_name = tool_call["name"]
        tool_args = tool_call.get("arguments", {})
        if tool_name in self.tool_map:
            tool_args["params"] = tool_args
            raw_result = await self.tool_map[tool_name].call(tool_args, **kwargs)
            result = raw_result
            return result
        else:
            return f"Error: Tool {tool_name} not found"


    async def make_trajectory(
        self,
        data: dict[str, str] | list[dict],
        client: AsyncOpenAI,
        save_path: str | None = None,
        resume: bool = False,
    ) -> dict:
        if isinstance(data, dict):
            if resume:
                saved = load_completed_result(data, save_path)
                if saved is not None:
                    logger.info(f"Skip completed QID {data['qid']}")
                    return saved
            return await self.run_agent(data, client, save_path=save_path)

        before_usage = getattr(self, "usage_totals", {}).copy()
        summary = {"total": len(data), "completed": 0, "skipped": 0, "failed": 0}
        for index, item in enumerate(data, start=1):
            qid = item.get("qid", "<missing>") if isinstance(item, dict) else "<invalid>"
            try:
                if resume and load_completed_result(item, save_path) is not None:
                    summary["skipped"] += 1
                    logger.info(f"[{index}/{len(data)}] Skip completed QID {qid}")
                    continue
                logger.info(f"[{index}/{len(data)}] Run QID {qid}")
                result = await self.run_agent(item, client, save_path=save_path)
                if is_completed_result(result, item):
                    summary["completed"] += 1
                    logger.info(f"[{index}/{len(data)}] Completed QID {qid}; progress: {summary}")
                else:
                    summary["failed"] += 1
                    logger.warning(f"QID {qid} has no valid answer; retry on next run.")
            except Exception:
                summary["failed"] += 1
                logger.exception(f"Rollout QID {qid} failed; retry on next run.")
        dataset_usage = {
            key: value - before_usage.get(key, 0)
            for key, value in getattr(self, "usage_totals", {}).items()
        }
        logger.info("Dataset token usage for %s: %s", save_path, dataset_usage)
        return summary


async def arun_episode(data, save_path: str | None = None, resume: bool = False):
    agent = MultiTurnReactAgent()
    async with AsyncOpenAI(
        base_url=os.environ.get("AGENT_BASE_URL"),
        api_key=os.environ.get("AGENT_API_KEY"),
    ) as client:
        return await agent.make_trajectory(
            data=data, client=client, save_path=save_path, resume=resume
        )


def main():
    parser = argparse.ArgumentParser(description="单样本测试或可断点续跑的批量蒸馏")
    parser.add_argument("--mode", choices=("single", "batch"), default="single")
    parser.add_argument(
        "--filename", nargs="+", default=["test.jsonl", "train_expansion.jsonl"],
        help="批量模式下 data/ 目录中的 JSONL 文件名",
    )
    parser.add_argument(
        "--save-dir", type=Path,
        help="批量模式下单个输入文件的轨迹输出目录；默认 saved/<文件名去扩展名>",
    )
    parser.add_argument("--no-resume", action="store_true", help="重新生成所有样本")
    args = parser.parse_args()

    if args.save_dir is not None and (args.mode != "batch" or len(args.filename) != 1):
        parser.error("--save-dir 仅支持批量模式且 --filename 只指定一个文件")

    if args.mode == "single":
        data = {
            "qid": "123",
            "question": "去重庆旅游时，如何在预算有限的情况下规划一条包含洪崖洞和长江索道的经济路线？",
        }
        result = asyncio.run(arun_episode(data, "saved/tmp"))
        print(result)
        return

    with batch_logging():
        logger.info("Batch started: files=%s; resume=%s", args.filename, not args.no_resume)
        for filename in args.filename:
            start = time.time()
            data = []
            save_path = str(args.save_dir or Path("saved") / Path(filename).stem)
            logger.info("Loading dataset %s; output=%s", filename, save_path)
            with (Path("data") / filename).open(encoding="utf-8") as fd:
                for line in fd:
                    if not line.strip():
                        continue
                    info = json.loads(line)
                    info.setdefault("qid", hashlib.md5(info["question"].encode("utf-8")).hexdigest())
                    data.append(info)
            logger.info("Loaded %s samples from %s", len(data), filename)
            summary = asyncio.run(arun_episode(data, save_path, resume=not args.no_resume))
            logger.info("%s: %s, rollout cost: %.2fs", filename, summary, time.time() - start)
        logger.info("Batch finished.")


if __name__ == "__main__":
    main()
