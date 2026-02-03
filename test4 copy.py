import time
from pathlib import Path
from llama_cpp import Llama

model_path = next(Path("models/enacimie/Qwen3-0.6B-Q8_0-GGUF").glob("*.gguf"))

llm = Llama(
    model_path=str(model_path),
    n_gpu_layers=40,
    n_ctx=4096,
    n_threads=8,
)

messages = [
    {"role": "user", "content": "Explain GPUs briefly in simple language."}
]

start_time = time.time()
token_count = 0

print("\n--- Model Output ---\n")

response = llm.create_chat_completion(
    messages=messages,
    max_tokens=200,
)
print(response["choices"][0]["message"]["content"])

end_time = time.time()

total_time = end_time - start_time
tokens_per_sec = token_count / total_time if total_time > 0 else 0

print("\n\n--- Performance ---")
print(f"Total time:      {total_time:.2f} sec")
print(f"Tokens generated: {token_count}")
print(f"Tokens/sec:       {tokens_per_sec:.2f}")


messages.append({"role": "assistant", "content": response["choices"][0]["message"]["content"]})

messages.append({"role": "user", "content": "My name is Josh. Ask me what I do."})
response = llm.create_chat_completion(
    messages=messages,
    max_tokens=128,
)
print(response["choices"][0]["message"]["content"])

messages.append({"role": "assistant", "content": response["choices"][0]["message"]["content"]})

messages.append({"role": "user", "content": "I build apps for Jetson. Summarize our chat in one sentence."})
response = llm.create_chat_completion(
    messages=messages,
    max_tokens=128,
)
print(response["choices"][0]["message"]["content"])