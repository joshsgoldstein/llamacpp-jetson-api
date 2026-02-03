import os
import json
import time
import uuid
import asyncio
import re
from enum import Enum
from pathlib import Path
from typing import Optional, List, Literal, Dict, Any, Iterator, Union

from fastapi import FastAPI, HTTPException, Header
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from llama_cpp import Llama
import uvicorn


# =========================
# Config
# =========================

APP_TITLE = "Llama.cpp Lab Multi-API Server"
MODEL_NAME = os.getenv("MODEL_NAME", "llama-local")

# Either:
#   MODEL_PATH=/abs/path/to/model.gguf
# or:
#   MODEL_DIR=/abs/path/to/dir/with/gguf
MODEL_PATH = os.getenv("MODEL_PATH")
MODEL_DIR = os.getenv("MODEL_DIR")

# Jetson AGX Orin sensible defaults
N_CTX = int(os.getenv("N_CTX", "4096"))
N_THREADS = int(os.getenv("N_THREADS", "8"))
# Use a conservative default for a server. You can override:
#   GPU_LAYERS=999 for tiny models like 0.6B
N_GPU_LAYERS = int(os.getenv("N_GPU_LAYERS", "40"))

# Safety: cap max tokens per request unless explicitly overridden
MAX_TOKENS_HARD_CAP = int(os.getenv("MAX_TOKENS_HARD_CAP", "1024"))

# Single-model instance is not concurrency-safe -> lock around generation
MODEL_LOCK = asyncio.Lock()

app = FastAPI(title=APP_TITLE)

llm: Optional[Llama] = None


# =========================
# Shared Models
# =========================

class MessageRole(str, Enum):
    system = "system"
    user = "user"
    assistant = "assistant"

class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class BaseCompletionRequest(BaseModel):
    model: str = MODEL_NAME
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=0.95, ge=0.0, le=1.0)
    max_tokens: int = Field(default=512, gt=0)
    stream: bool = False


# =========================
# OpenAI API Models
# =========================

class OpenAIToolFunction(BaseModel):
    name: str
    description: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None


class OpenAITool(BaseModel):
    type: Literal["function"] = "function"
    function: OpenAIToolFunction


class OpenAIToolCallFunction(BaseModel):
    name: str
    arguments: str


class OpenAIToolCall(BaseModel):
    id: str
    type: Literal["function"] = "function"
    function: OpenAIToolCallFunction


class OpenAIMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: Optional[str] = None
    name: Optional[str] = None
    tool_call_id: Optional[str] = None
    tool_calls: Optional[List[OpenAIToolCall]] = None

class OpenAIChatRequest(BaseCompletionRequest):
    messages: List[OpenAIMessage]
    stop: Optional[List[str]] = None
    tools: Optional[List[OpenAITool]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None

class OpenAIChoice(BaseModel):
    index: int
    message: OpenAIMessage
    finish_reason: Optional[str]

class OpenAIChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[OpenAIChoice]
    usage: Usage


# =========================
# Anthropic API Models
# =========================

class AnthropicTool(BaseModel):
    name: str
    description: Optional[str] = None
    input_schema: Dict[str, Any]


class AnthropicMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str

class AnthropicRequest(BaseCompletionRequest):
    messages: List[AnthropicMessage]
    system: Optional[str] = None
    stop_sequences: Optional[List[str]] = None
    tools: Optional[List[AnthropicTool]] = None
    tool_choice: Optional[Dict[str, Any]] = None

class AnthropicContent(BaseModel):
    type: str = "text"
    text: Optional[str] = None
    id: Optional[str] = None
    name: Optional[str] = None
    input: Optional[Dict[str, Any]] = None

class AnthropicUsage(BaseModel):
    input_tokens: int
    output_tokens: int

class AnthropicResponse(BaseModel):
    id: str
    type: str = "message"
    role: str = "assistant"
    content: List[AnthropicContent]
    model: str
    stop_reason: Optional[str] = "end_turn"
    usage: AnthropicUsage


# =========================
# Helpers
# =========================

def _pick_model_path() -> Path:
    if MODEL_PATH:
        p = Path(MODEL_PATH).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"MODEL_PATH does not exist: {p}")
        if p.suffix.lower() != ".gguf":
            raise ValueError(f"MODEL_PATH must point to a .gguf file: {p}")
        return p

    if MODEL_DIR:
        d = Path(MODEL_DIR).expanduser().resolve()
        if not d.exists() or not d.is_dir():
            raise FileNotFoundError(f"MODEL_DIR does not exist or is not a directory: {d}")
        ggufs = sorted(d.glob("*.gguf"))
        if not ggufs:
            # try one level deep
            ggufs = sorted(d.glob("**/*.gguf"))
        if not ggufs:
            raise FileNotFoundError(f"No .gguf files found in MODEL_DIR: {d}")
        return ggufs[0]

    # default: look in ./models for a gguf
    d = Path("./models").resolve()
    ggufs = sorted(d.glob("**/*.gguf")) if d.exists() else []
    if not ggufs:
        raise FileNotFoundError(
            "No model configured. Set MODEL_PATH or MODEL_DIR, or put a .gguf under ./models."
        )
    return ggufs[0]


def _now() -> int:
    return int(time.time())


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def _cap_max_tokens(requested: int) -> int:
    return min(requested, MAX_TOKENS_HARD_CAP)


def _openai_messages_to_llama(messages: List[OpenAIMessage]) -> List[Dict[str, str]]:
    # llama-cpp-python expects list[dict(role, content)]
    out: List[Dict[str, Any]] = []
    for m in messages:
        entry: Dict[str, Any] = {"role": m.role, "content": m.content or ""}
        if m.name:
            entry["name"] = m.name
        if m.tool_call_id:
            entry["tool_call_id"] = m.tool_call_id
        out.append(entry)
    return out


def _anthropic_tools_to_openai_tools(req: AnthropicRequest) -> Optional[List[Dict[str, Any]]]:
    if not req.tools:
        return None
    tools: List[Dict[str, Any]] = []
    for tool in req.tools:
        tools.append({
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.input_schema,
            },
        })
    return tools


def _anthropic_tool_choice_to_openai(tool_choice: Optional[Dict[str, Any]]) -> Optional[Union[str, Dict[str, Any]]]:
    if not tool_choice:
        return None
    choice_type = tool_choice.get("type")
    if choice_type == "tool":
        name = tool_choice.get("name")
        if name:
            return {"type": "function", "function": {"name": name}}
    if choice_type == "auto":
        return "auto"
    if choice_type == "any":
        return "required"
    return tool_choice


def _anthropic_to_openai_messages(req: AnthropicRequest) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    if req.system:
        out.append({"role": "system", "content": req.system})
    for m in req.messages:
        out.append({"role": m.role, "content": m.content})
    return out


def _extract_stream_text(chunk: Dict[str, Any]) -> str:
    """
    Streaming chunks differ by API:
      - chat: choices[0].delta.content
      - completion: choices[0].text
    We support both.
    """
    choices = chunk.get("choices") or []
    if not choices:
        return ""
    c0 = choices[0]
    delta = c0.get("delta") or {}
    if isinstance(delta, dict):
        content = delta.get("content")
        if content:
            return content
    text = c0.get("text")
    return text or ""


def _extract_finish_reason(chunk: Dict[str, Any]) -> Optional[str]:
    choices = chunk.get("choices") or []
    if not choices:
        return None
    return choices[0].get("finish_reason")


def _compute_usage_from_text(prompt_messages: List[Dict[str, str]], completion_text: str) -> Usage:
    """
    llama-cpp-python sometimes returns usage on non-stream calls, but streaming often doesn't.
    We'll compute usage using the model tokenizer.
    This is an approximation for chat since it depends on the model's chat template,
    but it's consistent and good enough for lab usage.
    """
    assert llm is not None
    # Simple prompt string to tokenize: roles + content
    prompt_blob = "".join([f"{m['role']}:{m['content']}\n" for m in prompt_messages])
    prompt_tokens = len(llm.tokenize(prompt_blob.encode("utf-8")))
    completion_tokens = len(llm.tokenize(completion_text.encode("utf-8")))
    return Usage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )


def _strip_think_full(text: str) -> str:
    # Remove common think-tag variants used by chat templates.
    patterns = [
        r"<think>.*?</think>",
        r"<\|think\|>.*?<\|/think\|>",
        r"<\|think\|>.*?<\|end\|>",
        r"\[THINK\].*?\[/THINK\]",
    ]
    cleaned = text
    for pat in patterns:
        cleaned = re.sub(pat, "", cleaned, flags=re.DOTALL | re.IGNORECASE)
    # Remove any stray tag markers.
    cleaned = re.sub(r"(<\|/?think\|>|</?think>|\[/?THINK\])", "", cleaned, flags=re.IGNORECASE)
    return cleaned


def _looks_like_reasoning(paragraph: str) -> bool:
    p = paragraph.strip()
    if not p:
        return True
    cues = [
        r"^okay\b",
        r"^first\b",
        r"^let me\b",
        r"^i should\b",
        r"^i need to\b",
        r"^i will\b",
        r"^maybe\b",
        r"^wait\b",
        r"^from what i remember\b",
        r"^the user\b",
        r"^i think\b",
        r"^here'?s how i\b",
    ]
    for cue in cues:
        if re.search(cue, p, flags=re.IGNORECASE):
            return True
    # Paragraphs that explicitly mention "reasoning" or "analysis" are likely meta.
    if re.search(r"\b(reasoning|analysis|chain[- ]of[- ]thought)\b", p, flags=re.IGNORECASE):
        return True
    return False


def _strip_reasoning_prefix(text: str) -> str:
    # Remove leading "reasoning" paragraphs while keeping the final answer.
    cleaned = text.strip()
    if not cleaned:
        return cleaned
    paragraphs = re.split(r"\n\s*\n+", cleaned)
    kept: List[str] = []
    drop_mode = True
    for para in paragraphs:
        if drop_mode and _looks_like_reasoning(para):
            continue
        drop_mode = False
        kept.append(para.strip())
    # If everything looks like reasoning, fall back to the original cleaned text.
    result = "\n\n".join([k for k in kept if k]).strip()
    if not result:
        return cleaned
    return result


def _partial_tag_suffix(text: str) -> int:
    tags = ["<think>", "</think>", "<|think|>", "<|/think|>", "[THINK]", "[/THINK]"]
    max_len = 0
    for tag in tags:
        for i in range(1, len(tag)):
            if text.endswith(tag[:i]):
                max_len = max(max_len, i)
    return max_len


class _ThinkStripper:
    def __init__(self) -> None:
        self.in_think = False
        self.buffer = ""
        self.start_tags = ["<think>", "<|think|>", "[THINK]"]
        self.end_tags = ["</think>", "<|/think|>", "[/THINK]", "<|end|>"]

    def feed(self, text: str) -> str:
        if not text:
            return ""

        data = self.buffer + text
        self.buffer = ""
        out: List[str] = []
        i = 0

        while i < len(data):
            if self.in_think:
                end_idx = -1
                end_tag = ""
                for tag in self.end_tags:
                    idx = data.find(tag, i)
                    if idx != -1 and (end_idx == -1 or idx < end_idx):
                        end_idx = idx
                        end_tag = tag
                if end_idx == -1:
                    # Buffer tail to catch a split closing tag; drop content.
                    tail_len = max(len(t) for t in self.end_tags) - 1
                    if len(data) - i > tail_len:
                        self.buffer = data[-tail_len:]
                    else:
                        self.buffer = data[i:]
                    return "".join(out)
                i = end_idx + len(end_tag)
                self.in_think = False
                continue

            start_idx = -1
            start_tag = ""
            for tag in self.start_tags:
                idx = data.find(tag, i)
                if idx != -1 and (start_idx == -1 or idx < start_idx):
                    start_idx = idx
                    start_tag = tag
            if start_idx == -1:
                segment = data[i:]
                suffix_len = _partial_tag_suffix(segment)
                if suffix_len:
                    self.buffer = segment[-suffix_len:]
                    segment = segment[:-suffix_len]
                out.append(segment)
                return "".join(out)

            segment = data[i:start_idx]
            out.append(segment)
            i = start_idx + len(start_tag)
            self.in_think = True

        return "".join(out)


# =========================
# Startup
# =========================

@app.on_event("startup")
async def load_model():
    global llm
    model_path = _pick_model_path()

    # Important: verbose=False prevents perf lines from polluting stdout/stderr in many builds
    llm = Llama(
        model_path=str(model_path),
        n_gpu_layers=N_GPU_LAYERS,
        n_ctx=N_CTX,
        n_threads=N_THREADS,
        verbose=False,
    )

    # Touch the tokenizer to ensure model is ready
    _ = llm.tokenize(b"ready")

    print(f"[startup] Loaded model: {model_path}")
    print(f"[startup] n_gpu_layers={N_GPU_LAYERS} n_ctx={N_CTX} n_threads={N_THREADS}")


# =========================
# OpenAI: /v1/chat/completions
# =========================

def _openai_sse_stream(
    stream_iter: Iterator[Dict[str, Any]],
    model_name: str,
    strip_think: bool = False,
) -> Iterator[str]:
    """
    OpenAI-compatible SSE:
      data: {chat.completion.chunk json}\n\n
      ...
      data: [DONE]\n\n
    """
    chunk_id = _new_id("chatcmpl")
    created = _now()

    stripper = _ThinkStripper() if strip_think else None

    for raw in stream_iter:
        choices = raw.get("choices") or []
        if not choices:
            continue
        delta = choices[0].get("delta") or {}
        if not isinstance(delta, dict):
            continue

        delta_payload: Dict[str, Any] = {}
        if "content" in delta:
            text = delta.get("content")
            if text:
                if stripper:
                    text = stripper.feed(text)
                if text:
                    delta_payload["content"] = text
        if "tool_calls" in delta:
            delta_payload["tool_calls"] = delta.get("tool_calls")

        if not delta_payload:
            # sometimes first chunks are empty
            continue

        payload = {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_name,
            "choices": [{
                "index": 0,
                "delta": delta_payload,
                "finish_reason": None
            }]
        }
        yield f"data: {json.dumps(payload)}\n\n"

    final_payload = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model_name,
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": "stop"
        }]
    }
    yield f"data: {json.dumps(final_payload)}\n\n"
    yield "data: [DONE]\n\n"


async def _openai_chat_completions_impl(
    request: OpenAIChatRequest,
    strip_think: bool = False,
):
    if llm is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    max_tokens = _cap_max_tokens(request.max_tokens)
    messages = _openai_messages_to_llama(request.messages)
    stop = request.stop or []
    tools = request.tools
    tool_choice = request.tool_choice

    async with MODEL_LOCK:
        if request.stream:
            stream_iter = llm.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                stop=stop,
                tools=tools,
                tool_choice=tool_choice,
                stream=True,
            )

            return StreamingResponse(
                _openai_sse_stream(stream_iter, request.model, strip_think=strip_think),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )

        out = llm.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            stop=stop,
            tools=tools,
            tool_choice=tool_choice,
            stream=False,
        )

    # Non-stream output shape: message content is in choices[0].message.content
    choice0 = out["choices"][0]
    message_obj = choice0.get("message") or {}
    content = message_obj.get("content")
    if strip_think:
        content = _strip_reasoning_prefix(_strip_think_full(content or ""))
    finish_reason = choice0.get("finish_reason")
    tool_calls = message_obj.get("tool_calls")
    if not finish_reason:
        finish_reason = "tool_calls" if tool_calls else "stop"

    usage = out.get("usage")
    if usage:
        usage_obj = Usage(
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
        )
    else:
        usage_obj = _compute_usage_from_text(messages, content)

    return OpenAIChatResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
        created=_now(),
        model=request.model,
        choices=[
            OpenAIChoice(
                index=0,
                message=OpenAIMessage(role="assistant", content=content, tool_calls=tool_calls),
                finish_reason=finish_reason,
            )
        ],
        usage=usage_obj,
    )


@app.post("/v1/chat/completions", response_model=OpenAIChatResponse)
async def openai_chat_completions(request: OpenAIChatRequest):
    return await _openai_chat_completions_impl(request, strip_think=True)


@app.post("/v1/chat/completions/think", response_model=OpenAIChatResponse)
async def openai_chat_completions_think(request: OpenAIChatRequest):
    return await _openai_chat_completions_impl(request, strip_think=False)


# =========================
# Anthropic: /v1/messages
# =========================

def _anthropic_sse_stream(
    stream_iter: Iterator[Dict[str, Any]],
    model_name: str,
    strip_think: bool = False,
) -> Iterator[str]:
    """
    Anthropic-style SSE event stream. Minimal but compatible shape.
    """
    msg_id = _new_id("msg")

    # message_start
    yield "event: message_start\n"
    yield f"data: {json.dumps({'type': 'message_start', 'message': {'id': msg_id, 'type': 'message', 'role': 'assistant', 'model': model_name}})}\n\n"

    # content_block_start
    yield "event: content_block_start\n"
    yield f"data: {json.dumps({'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'text', 'text': ''}})}\n\n"

    # deltas
    stripper = _ThinkStripper() if strip_think else None

    for raw in stream_iter:
        text = _extract_stream_text(raw)
        if not text:
            continue
        if stripper:
            text = stripper.feed(text)
            if not text:
                continue
        chunk = {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": text},
        }
        yield "event: content_block_delta\n"
        yield f"data: {json.dumps(chunk)}\n\n"

    # stops
    yield "event: content_block_stop\n"
    yield f"data: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"

    yield "event: message_delta\n"
    yield f"data: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': 'end_turn'}})}\n\n"

    yield "event: message_stop\n"
    yield f"data: {json.dumps({'type': 'message_stop'})}\n\n"


async def _anthropic_messages_impl(
    request: AnthropicRequest,
    x_api_key: Optional[str] = Header(None),
    strip_think: bool = False,
):
    if llm is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    max_tokens = _cap_max_tokens(request.max_tokens)
    stop = request.stop_sequences or []
    tools = _anthropic_tools_to_openai_tools(request)
    tool_choice = _anthropic_tool_choice_to_openai(request.tool_choice)

    # Convert Anthropic -> OpenAI style messages internally
    messages = _anthropic_to_openai_messages(request)

    async with MODEL_LOCK:
        if request.stream:
            stream_iter = llm.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                stop=stop,
                tools=tools,
                tool_choice=tool_choice,
                stream=True,
            )

            return StreamingResponse(
                _anthropic_sse_stream(stream_iter, request.model, strip_think=strip_think),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )

        out = llm.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            stop=stop,
            tools=tools,
            tool_choice=tool_choice,
            stream=False,
        )

    choice0 = out["choices"][0]
    message_obj = choice0.get("message") or {}
    content = message_obj.get("content")
    if strip_think:
        content = _strip_reasoning_prefix(_strip_think_full(content or ""))
    finish_reason = choice0.get("finish_reason") or "end_turn"
    tool_calls = message_obj.get("tool_calls") or []
    content_blocks: List[AnthropicContent] = []
    if content:
        content_blocks.append(AnthropicContent(type="text", text=content))
    for call in tool_calls:
        func = (call or {}).get("function") or {}
        raw_args = func.get("arguments", "")
        try:
            parsed_args = json.loads(raw_args) if raw_args else {}
        except json.JSONDecodeError:
            parsed_args = {"_raw": raw_args}
        content_blocks.append(
            AnthropicContent(
                type="tool_use",
                id=call.get("id"),
                name=func.get("name"),
                input=parsed_args,
            )
        )

    usage = out.get("usage")
    if usage:
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)
    else:
        usage_obj = _compute_usage_from_text(messages, content)
        input_tokens = usage_obj.prompt_tokens
        output_tokens = usage_obj.completion_tokens

    return AnthropicResponse(
        id=f"msg_{uuid.uuid4().hex}",
        model=request.model,
        content=content_blocks or [AnthropicContent(type="text", text=content or "")],
        stop_reason=finish_reason,
        usage=AnthropicUsage(input_tokens=input_tokens, output_tokens=output_tokens),
    )


@app.post("/v1/messages", response_model=AnthropicResponse)
async def anthropic_messages(
    request: AnthropicRequest,
    x_api_key: Optional[str] = Header(None),
):
    return await _anthropic_messages_impl(request, x_api_key=x_api_key, strip_think=True)


@app.post("/v1/messages/clean", response_model=AnthropicResponse)
async def anthropic_messages_clean(
    request: AnthropicRequest,
    x_api_key: Optional[str] = Header(None),
):
    return await _anthropic_messages_impl(request, x_api_key=x_api_key, strip_think=True)


@app.post("/v1/messages/think", response_model=AnthropicResponse)
async def anthropic_messages_think(
    request: AnthropicRequest,
    x_api_key: Optional[str] = Header(None),
):
    return await _anthropic_messages_impl(request, x_api_key=x_api_key, strip_think=False)


# =========================
# Utility Endpoints
# =========================

@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [{
            "id": MODEL_NAME,
            "object": "model",
            "created": _now(),
            "owned_by": "local",
        }],
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": llm is not None,
        "model_name": MODEL_NAME,
        "apis": ["openai:/v1/chat/completions", "anthropic:/v1/messages"],
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
