import argparse
import json
import sys
from typing import List

import httpx

from test_print_utils import pretty_json, print_block, shorten

def _print(title: str, ok: bool, detail: str = "") -> None:
    status = "OK" if ok else "FAIL"
    line = f"[{status}] {title}"
    if detail:
        line += f" - {detail}"
    print(line)


def _post_json(client: httpx.Client, url: str, payload: dict) -> httpx.Response:
    return client.post(url, json=payload, headers={"Content-Type": "application/json"})


def _read_sse_text(resp: httpx.Response, max_chunks: int = 50) -> str:
    text_parts: List[str] = []
    chunk_count = 0
    for line in resp.iter_lines():
        if not line:
            continue
        if line.startswith("data: "):
            data = line[6:]
            if data == "[DONE]":
                break
            try:
                payload = json.loads(data)
            except json.JSONDecodeError:
                continue
            # OpenAI-style: choices[0].delta.content or choices[0].text
            choices = payload.get("choices") or []
            if choices:
                delta = choices[0].get("delta") or {}
                content = delta.get("content")
                if content:
                    text_parts.append(content)
                elif choices[0].get("text"):
                    text_parts.append(choices[0]["text"])
        elif line.startswith("event: "):
            # Anthropic style events; data lines will carry payload
            continue
        elif line.startswith("data: "):
            continue
        else:
            # Some servers emit raw data without "data:"; ignore.
            continue

        chunk_count += 1
        if chunk_count >= max_chunks:
            break
    return "".join(text_parts)


def _read_anthropic_sse_text(resp: httpx.Response, max_chunks: int = 50) -> str:
    text_parts: List[str] = []
    chunk_count = 0
    for line in resp.iter_lines():
        if not line:
            continue
        if not line.startswith("data: "):
            continue
        data = line[6:]
        try:
            payload = json.loads(data)
        except json.JSONDecodeError:
            continue
        if payload.get("type") == "content_block_delta":
            delta = payload.get("delta") or {}
            if delta.get("type") == "text_delta":
                text_parts.append(delta.get("text", ""))

        chunk_count += 1
        if chunk_count >= max_chunks:
            break
    return "".join(text_parts)


def main() -> int:
    parser = argparse.ArgumentParser(description="Test llama-cpp API compatibility endpoints.")
    parser.add_argument("--base-url", default="http://localhost:8000", help="Base URL for the API server")
    parser.add_argument("--model", default="llama-local", help="Model name to send in requests")
    parser.add_argument("--timeout", type=float, default=60.0, help="HTTP timeout seconds")
    parser.add_argument("--skip-openai-lib", action="store_true", help="Skip OpenAI SDK test")
    parser.add_argument("--skip-anthropic-lib", action="store_true", help="Skip Anthropic SDK test")
    parser.add_argument("--print-responses", action="store_true", help="Print non-stream responses")
    args = parser.parse_args()

    base = args.base_url.rstrip("/")
    client = httpx.Client(timeout=args.timeout)

    try:
        r = client.get(f"{base}/health")
        _print("GET /health", r.status_code == 200)
    except Exception as exc:
        _print("GET /health", False, str(exc))
        return 1

    openai_payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": "Say hello in one short sentence."}],
        "max_tokens": 200,
    }
    r = _post_json(client, f"{base}/v1/chat/completions", openai_payload)
    _print("POST /v1/chat/completions", r.status_code == 200, shorten(r.text) if r.status_code != 200 else "")
    if args.print_responses and r.status_code == 200:
        print_block("OPENAI /v1/chat/completions", pretty_json(r.text))

    r = _post_json(client, f"{base}/v1/chat/completions/think", openai_payload)
    _print("POST /v1/chat/completions/think", r.status_code == 200, shorten(r.text) if r.status_code != 200 else "")
    if args.print_responses and r.status_code == 200:
        print_block("OPENAI /v1/chat/completions/think", pretty_json(r.text))

    openai_stream_payload = dict(openai_payload)
    openai_stream_payload["stream"] = True
    r = _post_json(client, f"{base}/v1/chat/completions", openai_stream_payload)
    text = _read_sse_text(r)
    _print(
        "POST /v1/chat/completions (stream)",
        r.status_code == 200 and len(text) > 0,
        shorten(r.text) if r.status_code != 200 else "",
    )

    r = _post_json(client, f"{base}/v1/chat/completions/think", openai_stream_payload)
    text = _read_sse_text(r)
    _print(
        "POST /v1/chat/completions/think (stream)",
        r.status_code == 200 and len(text) > 0,
        shorten(r.text) if r.status_code != 200 else "",
    )

    anthropic_payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": "Reply with just the word OK."}],
        "max_tokens": 200,
    }
    r = _post_json(client, f"{base}/v1/messages", anthropic_payload)
    _print("POST /v1/messages", r.status_code == 200, shorten(r.text) if r.status_code != 200 else "")
    if args.print_responses and r.status_code == 200:
        print_block("ANTHROPIC /v1/messages", pretty_json(r.text))

    r = _post_json(client, f"{base}/v1/messages/think", anthropic_payload)
    _print("POST /v1/messages/think", r.status_code == 200, shorten(r.text) if r.status_code != 200 else "")
    if args.print_responses and r.status_code == 200:
        print_block("ANTHROPIC /v1/messages/think", pretty_json(r.text))

    anthropic_stream_payload = dict(anthropic_payload)
    anthropic_stream_payload["stream"] = True
    r = _post_json(client, f"{base}/v1/messages", anthropic_stream_payload)
    text = _read_anthropic_sse_text(r)
    _print(
        "POST /v1/messages (stream)",
        r.status_code == 200 and len(text) > 0,
        shorten(r.text) if r.status_code != 200 else "",
    )

    r = _post_json(client, f"{base}/v1/messages/think", anthropic_stream_payload)
    text = _read_anthropic_sse_text(r)
    _print(
        "POST /v1/messages/think (stream)",
        r.status_code == 200 and len(text) > 0,
        shorten(r.text) if r.status_code != 200 else "",
    )

    if not args.skip_openai_lib:
        try:
            from openai import OpenAI  # type: ignore

            sdk_client = OpenAI(base_url=f"{base}/v1", api_key="local")
            resp = sdk_client.chat.completions.create(
                model=args.model,
                messages=[{"role": "user", "content": "Say hello in one short sentence."}],
                max_tokens=64,
            )
            ok = bool(resp.choices and resp.choices[0].message.content)
            _print("OpenAI SDK chat.completions.create", ok)
        except Exception as exc:
            _print("OpenAI SDK chat.completions.create", False, str(exc))

    if not args.skip_anthropic_lib:
        try:
            from anthropic import Anthropic  # type: ignore

            anth_client = Anthropic(base_url=base, api_key="local")
            resp = anth_client.messages.create(
                model=args.model,
                messages=[{"role": "user", "content": "Reply with just the word OK."}],
                max_tokens=32,
            )
            text = ""
            if resp and resp.content:
                first = resp.content[0]
                text = getattr(first, "text", "") or ""
            _print("Anthropic SDK messages.create", bool(text))
        except Exception as exc:
            _print("Anthropic SDK messages.create", False, str(exc))

    return 0


if __name__ == "__main__":
    sys.exit(main())
