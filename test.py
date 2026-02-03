from pathlib import Path
from llama_cpp import Llama

MODEL_DIR = Path("models/enacimie/Qwen3-0.6B-Q8_0-GGUF")

# Find first GGUF file
gguf_files = list(MODEL_DIR.glob("*.gguf"))

if not gguf_files:
    raise FileNotFoundError("No GGUF files found in directory")

model_path = gguf_files[0]

print(f"Loading model: {model_path}")

llm = Llama(
    model_path=str(model_path),
    n_gpu_layers=40,   # use GPU
    n_ctx=4096,
    n_threads=8,
)

output = llm("Explain GPUs briefly.", max_tokens=100)

print(output["choices"][0]["text"])
