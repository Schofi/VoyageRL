# -*- coding: utf-8 -*-
"""从命令行抓取网页，并根据目标提取证据和摘要。"""

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--url", nargs="+", required=True, metavar="URL", help="一个或多个网页地址"
    )
    parser.add_argument("--goal", required=True, help="希望从网页中提取的信息")
    args = parser.parse_args()

    try:
        from dotenv import load_dotenv
    except ModuleNotFoundError:
        print("缺少 python-dotenv，请先安装运行依赖。", file=sys.stderr)
        return 1

    load_dotenv(Path(__file__).resolve().with_name(".env"))
    missing = [
        name for name in ("OPENAI_API_KEY", "LLM_MODEL_ID") if not os.getenv(name)
    ]
    if missing:
        parser.error("请在项目根目录 .env 中配置：" + ", ".join(missing))

    try:
        from tools.tool_visit import Visit
    except ModuleNotFoundError as exc:
        print(f"缺少依赖 {exc.name}，请先安装运行依赖。", file=sys.stderr)
        return 1

    params = {
        "url": args.url[0] if len(args.url) == 1 else args.url,
        "goal": args.goal,
    }
    started = time.perf_counter()
    result = asyncio.run(Visit().call(params))
    print(result)
    print(f"\n耗时：{time.perf_counter() - started:.2f} 秒")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
