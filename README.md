# Llama.cpp Lab Multi-API Server

FastAPI server that exposes OpenAI- and Anthropic-compatible endpoints on top of
`llama-cpp-python`. Designed for local inference, low-friction testing, and
drop-in compatibility with popular client SDKs.

## Highlights
- OpenAI-style chat completions: `/v1/chat/completions`
- Anthropic-style messages: `/v1/messages`
- Streaming responses (SSE)
- Optional "think" routes that preserve reasoning tokens
- Single-process model lock for safe concurrent requests

## Table of contents
- [Quickstart](#quickstart)
- [Installation](#installation)
- [Configuration](#configuration)
- [Endpoints](#endpoints)
- [Usage](#usage)
- [Streaming](#streaming)
- [Tool calling](#tool-calling)
- [Notes](#notes)
- [Troubleshooting](#troubleshooting)
- [License](#license)

## Quickstart
```bash
export MODEL_PATH=/abs/path/to/model.gguf
python api.py
```

Or with uvicorn:
```bash
export MODEL_PATH=/abs/path/to/model.gguf
uvicorn api:app --host 0.0.0.0 --port 8000
```

Verify:
```bash
curl http://localhost:8000/health
```

## Installation
Create a virtual environment, then install dependencies:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install fastapi uvicorn llama-cpp-python
```

## Configuration
Set one of the following:
- `MODEL_PATH=/abs/path/to/model.gguf`
- `MODEL_DIR=/abs/path/to/dir/with/gguf`

Optional environment variables:

| Name | Default | Description |
| ---- | ------- | ----------- |
| `MODEL_NAME` | `llama-local` | Reported model id in responses |
| `N_CTX` | `4096` | Context window size |
| `N_THREADS` | `8` | CPU threads for inference |
| `N_GPU_LAYERS` | `40` | GPU layers offload |
| `MAX_TOKENS_HARD_CAP` | `1024` | Hard cap for `max_tokens` |
| `PORT` | `8000` | Server port |

## Endpoints
- `GET /health`
- `GET /v1/models`
- `POST /v1/chat/completions`
- `POST /v1/chat/completions/think`
- `POST /v1/messages`
- `POST /v1/messages/clean`
- `POST /v1/messages/think`

## Usage

### OpenAI chat
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama-local",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Write a haiku about Jetson."}
    ],
    "max_tokens": 128
  }'
```

### Anthropic messages
```bash
curl http://localhost:8000/v1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama-local",
    "messages": [
      {"role": "user", "content": "Say hello in three words."}
    ],
    "max_tokens": 64
  }'
```

## Streaming
OpenAI-style streaming uses SSE. Add `"stream": true`:
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama-local",
    "messages": [
      {"role": "user", "content": "Give me a short list of planets."}
    ],
    "stream": true,
    "max_tokens": 128
  }'
```

## Tool calling
Both endpoints support function tools. Example for OpenAI:
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama-local",
    "messages": [
      {"role": "user", "content": "Get the weather in Paris."}
    ],
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "get_weather",
          "description": "Return current weather for a city",
          "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"]
          }
        }
      }
    ],
    "tool_choice": "auto",
    "max_tokens": 128
  }'
```

## Notes
- `/v1/chat/completions` and `/v1/messages` strip `<think>`-style reasoning by default.
- Use `/v1/chat/completions/think` or `/v1/messages/think` to preserve reasoning tokens.
- A single model instance is protected by a lock to avoid concurrent access issues.

## Troubleshooting
- **Model not loaded**: set `MODEL_PATH` or `MODEL_DIR` and restart.
- **Slow generation**: reduce `N_CTX` or `N_GPU_LAYERS`, or increase `N_THREADS`.
- **Out of memory**: reduce `N_CTX` and `N_GPU_LAYERS`, or use a smaller model.

## License
Add a license file if you plan to publish this publicly.
