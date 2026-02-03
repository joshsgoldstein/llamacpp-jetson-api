import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from llama_cpp import Llama

@contextmanager
def suppress_stderr():
    old_stderr = sys.stderr
    try:
        with open(os.devnull, "w") as devnull:
            sys.stderr = devnull
            yield
    finally:
        sys.stderr = old_stderr

model_path = next(Path("models/enacimie/Qwen3-0.6B-Q8_0-GGUF").glob("*.gguf"))

llm = Llama(
    model_path=str(model_path),
    n_gpu_layers=80,
    n_ctx=4096,
    n_threads=8,
    verbose=False,
)

messages = [{"role": "user", "content": "Explain what a GPU is in 1-2 simple sentences."}]

start = time.time()
output_text = []

print("\n--- Model Output ---\n")

with suppress_stderr():
    for chunk in llm.create_chat_completion(
        messages=messages,
        stream=True,
        max_tokens=200,
        temperature=0.7,
    ):
        delta = chunk["choices"][0]["delta"].get("content")
        if delta:
            output_text.append(delta)
            print(delta, end="", flush=True)

elapsed = time.time() - start
final_text = "".join(output_text)

# True token count using the model's tokenizer
token_count = len(llm.tokenize(final_text.encode("utf-8")))
tps = token_count / elapsed if elapsed > 0 else 0.0
mspt = (elapsed / token_count) * 1000 if token_count > 0 else 0.0

print("\n\n--- Performance ---")
print(f"Total time:        {elapsed:6.2f} sec")
print(f"Tokens generated:  {token_count:6d} (true tokens)")
print(f"Tokens/sec:        {tps:6.2f}")
print(f"ms per token:      {mspt:6.2f}")
