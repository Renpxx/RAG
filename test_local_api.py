"""
Lightweight script to test the local FastAPI endpoint at
http://localhost:8012/v1/chat/completions without relying on external tools.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict

import requests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Send a test request to the local RAG completion endpoint."
    )
    parser.add_argument(
        "--question",
        default="DeepSeek-R1模型在奖励机制中具体设计了哪些显性激励方法？",
        help="User question to submit to the /v1/chat/completions endpoint.",
    )
    parser.add_argument(
        "--url",
        default="http://localhost:8012/v1/chat/completions",
        help="Target endpoint URL.",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Request a streaming response and print chunks as they arrive.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="Request timeout in seconds.",
    )
    return parser.parse_args()


def pretty_print(data: Dict[str, Any]) -> None:
    print(json.dumps(data, ensure_ascii=False, indent=2))


def main() -> None:
    args = parse_args()
    payload = {
        "messages": [{"role": "user", "content": args.question}],
        "stream": args.stream,
    }
    try:
        if args.stream:
            with requests.post(
                args.url, json=payload, timeout=args.timeout, stream=True
            ) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line:
                        continue
                    print(line.decode("utf-8"))
        else:
            resp = requests.post(args.url, json=payload, timeout=args.timeout)
            resp.raise_for_status()
            pretty_print(resp.json())
    except requests.HTTPError as exc:
        print(f"Request failed with status {exc.response.status_code}: {exc}", file=sys.stderr)
        if exc.response is not None:
            print(exc.response.text, file=sys.stderr)
        sys.exit(1)
    except Exception as exc:  # pragma: no cover - best-effort CLI
        print(f"Request failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

