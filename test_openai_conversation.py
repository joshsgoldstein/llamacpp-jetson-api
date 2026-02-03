from openai import OpenAI


def main() -> None:
    base = "http://192.168.1.177:8000"
    model = "llama-local"

    client = OpenAI(base_url=f"{base}/v1", api_key="local")

    messages = [
        {"role": "user", "content": "Hi! Please say hello and ask my name in a question."},
    ]

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

    # Turn 1
    print("\n--- User 1 ---\n", messages[-1]["content"])
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        # tools=tools,
        max_tokens=128,
    )
    assistant_1 = resp.choices[0].message.content or ""
    print("\n--- Turn 1 ---\n", assistant_1)
    messages.append({"role": "assistant", "content": assistant_1})

    # Turn 2
    messages.append({"role": "user", "content": "My name is Josh. Ask me what I do."})
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        # tools=tools,
        max_tokens=128,
    )
    assistant_2 = resp.choices[0].message.content or ""
    print("\n--- Turn 2 ---\n", assistant_2)
    messages.append({"role": "assistant", "content": assistant_2})

    # Turn 3
    messages.append({"role": "user", "content": "I build apps for Jetson. Summarize our chat in one sentence."})
    print("\n--- User 3 ---\n", messages[-1]["content"])
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        # tools=tools,
        max_tokens=128,
    )
    assistant_3 = resp.choices[0].message.content or ""
    print("\n--- Turn 3 ---\n", assistant_3)


if __name__ == "__main__":
    main()
