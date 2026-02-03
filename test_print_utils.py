import json
from typing import Any


def shorten(text: str, limit: int = 800) -> str:
    t = text.strip()
    if len(t) <= limit:
        return t
    return t[:limit] + " ... (truncated)"


def pretty_json(text: str, limit: int = 800) -> str:
    try:
        data: Any = json.loads(text)
    except json.JSONDecodeError:
        return shorten(text, limit=limit)
    pretty = json.dumps(data, indent=2, ensure_ascii=False)
    return shorten(pretty, limit=limit)


def print_block(title: str, body: str) -> None:
    print(f"\n=== {title} ===")
    print(body)
