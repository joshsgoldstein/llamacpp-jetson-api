import json
import re
from openai import OpenAI

MAX_TOKENS = 200
FINAL_MAX_TOKENS = 500


def remove_thinking(text: str) -> str:
    return re.sub(r"(?s)^\s*(?:<think>(.*?)</think>)?\s*(.*)$", r"\2", text)


def needs_retry(text: str) -> bool:
    return "<think>" in text and "</think>" not in text


def chat_completion(
    client: OpenAI,
    model: str,
    messages: list[dict],
    tools: list[dict] | None = None,
    max_tokens: int = 128,
) -> tuple[str, list[dict] | None]:
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools or None,
        max_tokens=max_tokens,
    )
    message = resp.choices[0].message
    content = message.content or ""
    return remove_thinking(content), message.tool_calls or []


def chat_completion_think(
    client: OpenAI,
    model: str,
    messages: list[dict],
    tools: list[dict] | None = None,
    max_tokens: int = 128,
) -> tuple[str, list[dict] | None]:
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools or None,
        max_tokens=max_tokens,
    )
    message = resp.choices[0].message
    return message.content or "", message.tool_calls or []


def chat_completion_with_retry(
    client: OpenAI,
    model: str,
    messages: list[dict],
    tools: list[dict] | None = None,
    max_tokens: int = MAX_TOKENS,
    retry_max_tokens: int = FINAL_MAX_TOKENS,
) -> tuple[str, list[dict] | None]:
    text, tool_calls = chat_completion(
        client=client,
        model=model,
        messages=messages,
        tools=tools,
        max_tokens=max_tokens,
    )
    if needs_retry(text):
        text, tool_calls = chat_completion(
            client=client,
            model=model,
            messages=messages,
            tools=tools,
            max_tokens=retry_max_tokens,
        )
    return text, tool_calls


def execute_tool_calls(tool_calls: list, tool_fn_map: dict[str, callable]) -> list[dict]:
    tool_messages: list[dict] = []
    for call in tool_calls:
        name = call.function.name
        raw_args = call.function.arguments or "{}"
        try:
            args = json.loads(raw_args)
        except json.JSONDecodeError:
            args = {"_raw": raw_args}
        fn = tool_fn_map.get(name)
        if not fn:
            result = {"error": f"Tool not found: {name}"}
        else:
            result = fn(**args)
        tool_messages.append(
            {
                "role": "tool",
                "tool_call_id": call.id,
                "content": json.dumps(result),
            }
        )
    return tool_messages


def print_tool_calls(tool_calls: list) -> None:
    if not tool_calls:
        return
    print("\n--- Tool Calls ---")
    for call in tool_calls:
        name = call.function.name
        args = call.function.arguments or ""
        print(f"- {name}({args})")


def run_turn(
    turn_num: int,
    user_text: str,
    client: OpenAI,
    model: str,
    messages: list[dict],
    tools: list[dict] | None,
    tool_fn_map: dict[str, callable],
) -> None:
    messages.append({"role": "user", "content": user_text})
    print(f"\n--- User {turn_num} ---\n", user_text)

    assistant, tool_calls = chat_completion_with_retry(
        client=client,
        model=model,
        messages=messages,
        tools=tools,
        max_tokens=MAX_TOKENS,
        retry_max_tokens=FINAL_MAX_TOKENS,
    )
    print(f"\n--- Turn {turn_num} ---\n", assistant)
    messages.append({"role": "assistant", "content": assistant})

    if tool_calls:
        print_tool_calls(tool_calls)
        tool_messages = execute_tool_calls(tool_calls, tool_fn_map)
        messages.extend(tool_messages)

        # Follow-up call to get final assistant response after tools.
        final_text, _ = chat_completion_with_retry(
            client=client,
            model=model,
            messages=messages,
            tools=None,
            max_tokens=FINAL_MAX_TOKENS,
            retry_max_tokens=FINAL_MAX_TOKENS,
        )
        print(f"\n--- Turn {turn_num} (final) ---\n", final_text)
        messages.append({"role": "assistant", "content": final_text})


def main() -> None:
    base = "http://192.168.1.177:8000"
    model = "llama-local"
    use_tools = False

    client = OpenAI(base_url=f"{base}/v1", api_key="local")

    messages: list[dict] = []

    tools = [
        {
            "type": "function",
            "function": {
                "name": "search_vector_db",
                "description": "Search a vector database and return the most relevant documents.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query text."},
                        "top_k": {"type": "integer", "description": "Number of results to return.", "default": 5},
                        "namespace": {"type": "string", "description": "Optional namespace/filter."},
                    },
                    "required": ["query"],
                },
            },
        }
    ]

    def search_vector_db(query: str, top_k: int = 5, namespace: str | None = None) -> dict:
        return {
            "query": query,
            "top_k": top_k,
            "namespace": namespace,
            "results": [],
        }

    tool_fn_map = {
        "search_vector_db": search_vector_db,
    }

    run_turn(
        turn_num=1,
        user_text="Hi! Please say hello and ask my name in a question.",
        client=client,
        model=model,
        messages=messages,
        tools=tools if use_tools else None,
        tool_fn_map=tool_fn_map,
    )
    run_turn(
        turn_num=2,
        user_text="My name is Josh. Ask me what I do.",
        client=client,
        model=model,
        messages=messages,
        tools=tools if use_tools else None,
        tool_fn_map=tool_fn_map,
    )
    run_turn(
        turn_num=3,
        user_text="I build apps for Jetson. Summarize our chat in one sentence.",
        client=client,
        model=model,
        messages=messages,
        tools=tools if use_tools else None,
        tool_fn_map=tool_fn_map,
    )


if __name__ == "__main__":
    main()
