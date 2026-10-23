# -*- coding: utf-8 -*-
# --------------------------------------------
# 文件描述: 火车票查询工具 
# --------------------------------------------


import asyncio
import os
import json
import logging
import httpx
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

from prompt import TRAIN_TICKET_SYSTEM_PROMPT 
from qwen_agent.tools.base import BaseTool, register_tool


logger = logging.getLogger(__name__)


@register_tool("train_tickets_search", allow_overwrite=True)
class TrainTicketsSearch(BaseTool):
    name = "train_tickets_search"
    description = "根据日期查询城市间火车/动车/高铁票信息。"
    parameters = {
        "type": "object",
        "properties": {
            "date": {
                "type": "string",
                "description": "日期，格式 YYYY-MM-DD"
            },
            "from_city": {
                "type": "string",
                "description": "出发城市中文名"
            },
            "to_city": {
                "type": "string",
                "description": "到达城市中文名"
            },
        },
        "required": ["date", "from_city", "to_city"]
    }

    def __init__(self, cfg: dict | None = None):
        super().__init__(cfg)

    async def call(self, params: dict, **kwargs) -> str:  # type: ignore[override]
        try:
            date = params["date"]
            from_city = params["from_city"]
            to_city = params["to_city"]
        except Exception:
            return "[train_tickets_search] Invalid request format: Input must be a JSON object containing 'date' and 'from_city' and 'to_city' field"

        kwargs = {
            "date": date,
            "from_city": from_city,
            "to_city": to_city,
        }
        query = json.dumps(kwargs, ensure_ascii=False)
        messages = [
            {"role": "system", "content": TRAIN_TICKET_SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ]

        result = None
        async with AsyncOpenAI(
            api_key=os.environ["OPENAI_API_KEY"],
            base_url=os.environ["OPENAI_BASE_URL"],
            timeout=60,
            max_retries=2
        ) as client:
            try:
                logger.info(
                    "[train_tickets_search] Requesting model %s; waiting for response",
                    os.environ.get("LLM_MODEL_ID"),
                )
                response = await client.chat.completions.create(
                    messages=messages, model=os.environ["LLM_MODEL_ID"]
                )
                result = response.choices[0].message.content.strip()
                logger.info("[train_tickets_search] Model response received")
            except Exception as e:
                error = f"{type(e).__name__}: {e}"
                logger.error("[train_tickets_search] Failed to get response: %s", error)
                result = f"[train_tickets_search] Failed to get response: {error}"
            finally:
                await client.close()

        if not result:
            result = "两地无直达火车票"

        return result


if __name__ == "__main__":
    import time
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    t1 = time.time()
    search = TrainTicketsSearch()
    params = {"date": "2026-03-30", "from_city": "成都", "to_city": "北京"}
    res = asyncio.run(search.call(params))
    t2 = time.time()
    print(res)
    print(t2 - t1)
