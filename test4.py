import time
from pathlib import Path

from llama_cpp import Llama
import re

model_path = next(Path("models/enacimie/Qwen3-0.6B-Q8_0-GGUF").glob("*.gguf"))

N_GPU_LAYERS = 40
N_CTX = 4096
N_THREADS = 8
MAX_TOKENS = 200
SUMMARY_MAX_TOKENS = 900


llm = Llama(
    model_path=str(model_path),
    n_gpu_layers=N_GPU_LAYERS,
    n_ctx=N_CTX,
    n_threads=N_THREADS,
    verbose=False,
)


def run_chat_completion(messages, max_tokens=MAX_TOKENS):
    return llm.create_chat_completion(
        messages=messages,
        max_tokens=max_tokens,
    )

def remove_thinking(text):
    return re.sub(r"(?s)^\s*(?:<think>(.*?)</think>)?\s*(.*)$", r"\2", text)


def needs_retry(text):
    return "<think>" in text and "</think>" not in text

# Retry once with a higher max_tokens if thinking tag is incomplete.
def run_chat_completion_with_retry(messages, max_tokens=MAX_TOKENS, retry_max_tokens=SUMMARY_MAX_TOKENS):
    resp = run_chat_completion(messages, max_tokens=max_tokens)
    content = resp["choices"][0]["message"]["content"]
    if needs_retry(content):
        resp = run_chat_completion(messages, max_tokens=retry_max_tokens)
    return resp

# print(remove_thinking(assistant_1)


messages = [
    {"role": "user", "content": "Hi! Please say hello and ask my name in a question."}
]

start_time = time.time()

print("\n--- User 1 ---\n")
print(messages[-1]["content"])

resp = run_chat_completion(messages)
assistant_1 = resp["choices"][0]["message"]["content"]
print("\n--- Turn 1 ---\n")
print(remove_thinking(assistant_1))
messages.append({"role": "assistant", "content": assistant_1})

messages.append({"role": "user", "content": "My name is Josh. Ask me what I do."})
print("\n--- User 2 ---\n")
print(messages[-1]["content"])

resp = run_chat_completion(messages)
assistant_2 = resp["choices"][0]["message"]["content"]
print("\n--- Turn 2 ---\n")
print(assistant_2)
messages.append({"role": "assistant", "content": assistant_2})

messages.append({"role": "user", "content": "I build apps for Jetson. Summarize our chat"})
print("\n--- User 3 ---\n")
print(messages[-1]["content"])

resp = run_chat_completion_with_retry(messages)
assistant_3 = resp["choices"][0]["message"]["content"]
print("\n--- Turn 3 ---\n")
print(assistant_3)

elapsed = time.time() - start_time
print("\n--- Performance ---")
print(f"Total time: {elapsed:.2f} sec")
