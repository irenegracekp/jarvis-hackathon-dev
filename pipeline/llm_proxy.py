"""LLM proxy server for Agora Conversational AI.

Agora calls this as its custom LLM backend. We intercept for OpenClaw
command routing and forward conversation to OpenRouter.

Endpoint: POST /v1/chat/completions (OpenAI-compatible, SSE streaming)
"""

import asyncio
import json
import os
import re
import sys
import time
import uuid

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.openclaw_bridge import OpenClawBridge

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_DIR = os.path.join(_PROJECT_ROOT, "config")

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL = "anthropic/claude-sonnet-4-6"

# Emotion tag pattern: [excited] or [curious] etc at start of response
EMOTION_PATTERN = re.compile(r"^\[(\w+)\]\s*")
# Save memory tag: [save:some fact]
SAVE_PATTERN = re.compile(r"\[save:([^\]]+)\]")

app = FastAPI()

# These get set by init() before the server starts
_state = None
_memory = None
_openclaw = OpenClawBridge()
_system_prompt = ""


def init(state=None, memory=None):
    """Initialize shared state. Call before starting uvicorn."""
    global _state, _memory, _system_prompt
    _state = state
    _memory = memory
    _system_prompt = _load_system_prompt()


def _load_system_prompt():
    path = os.path.join(CONFIG_DIR, "system_prompt_agora.txt")
    if not os.path.exists(path):
        path = os.path.join(CONFIG_DIR, "system_prompt.txt")
    with open(path, "r") as f:
        return f.read().strip()


def _get_memory_context():
    """Get memory context for the current face."""
    if _state is None or _memory is None:
        return ""
    face_id = getattr(_state, "current_face_id", None)
    if not face_id:
        return ""
    return _memory.get_context_string(face_id)


def _inject_system_prompt(messages):
    """Replace or prepend system prompt with ours + memory context."""
    memory_ctx = _get_memory_context()
    system_content = _system_prompt
    if memory_ctx:
        system_content = f"{_system_prompt}\n\n{memory_ctx}"

    # Remove any existing system messages from Agora
    messages = [m for m in messages if m.get("role") != "system"]
    # Clean Agora-specific fields
    for m in messages:
        m.pop("turn_id", None)
        m.pop("timestamp", None)
    # Prepend our system prompt
    messages.insert(0, {"role": "system", "content": system_content})
    return messages


def _make_sse_chunk(content, finish_reason=None):
    """Format a single SSE chunk in OpenAI streaming format."""
    chunk = {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "choices": [{
            "index": 0,
            "delta": {},
        }],
    }
    if content:
        chunk["choices"][0]["delta"]["content"] = content
    if finish_reason:
        chunk["choices"][0]["finish_reason"] = finish_reason
    return f"data: {json.dumps(chunk)}\n\n"


def _post_emotion(emotion):
    """Post robot emotion to the response queue if state is available."""
    if _state is None:
        return
    try:
        _state.response_queue.put_nowait({
            "speech": "",
            "emotion": emotion,
            "head_direction": "toward_speaker",
            "antenna_state": "perked" if emotion == "excited" else "neutral",
        })
    except Exception:
        pass


def _post_save_memory(fact):
    """Save a memory fact for the current face."""
    if _state is None or _memory is None:
        return
    face_id = getattr(_state, "current_face_id", None) or "default"
    if _memory.get_person(face_id) is None:
        if fact.lower().startswith("name is "):
            _memory.create_person(face_id, fact[8:].strip())
        else:
            _memory.create_person(face_id, face_id, facts=[fact])
    else:
        if fact.lower().startswith("name is "):
            _memory.set_name(face_id, fact[8:].strip())
        else:
            _memory.add_fact(face_id, fact)


async def _openclaw_stream(user_text, messages):
    """Execute OpenClaw command and stream LLM commentary about the result."""
    # Execute command synchronously (subprocess)
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, _openclaw.send_command, user_text)
    print(f"[proxy] OpenClaw result: {result}")

    # Ask LLM to comment on the result
    comment_messages = _inject_system_prompt(messages[:-1])  # keep history minus last user msg
    comment_messages.append({
        "role": "user",
        "content": (
            f"I just executed a computer command for the user: '{user_text}'. "
            f"Result: {result}. Comment on it briefly."
        ),
    })

    async for chunk in _openrouter_stream(comment_messages):
        yield chunk


async def _openrouter_stream(messages):
    """Forward to OpenRouter and relay SSE chunks back."""
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        yield _make_sse_chunk("Sorry, my brain isn't connected right now.")
        yield _make_sse_chunk(None, finish_reason="stop")
        yield "data: [DONE]\n\n"
        return

    accumulated_text = ""
    emotion_stripped = False
    buffer = ""  # buffer for emotion tag detection

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            async with client.stream(
                "POST",
                OPENROUTER_BASE_URL,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": OPENROUTER_MODEL,
                    "messages": messages,
                    "stream": True,
                    "max_tokens": 200,
                    "temperature": 0.8,
                },
            ) as resp:
                if resp.status_code != 200:
                    body = await resp.aread()
                    print(f"[proxy] OpenRouter error {resp.status_code}: {body[:200]}")
                    yield _make_sse_chunk("Sorry, my brain is having connection issues.")
                    yield _make_sse_chunk(None, finish_reason="stop")
                    yield "data: [DONE]\n\n"
                    return
                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data = line[6:]
                    if data.strip() == "[DONE]":
                        # Flush any remaining buffer
                        if buffer:
                            yield _make_sse_chunk(buffer)
                            accumulated_text += buffer
                            buffer = ""
                        # Process save tags
                        save_match = SAVE_PATTERN.search(accumulated_text)
                        if save_match:
                            _post_save_memory(save_match.group(1).strip())
                        yield _make_sse_chunk(None, finish_reason="stop")
                        yield "data: [DONE]\n\n"
                        return

                    try:
                        chunk_data = json.loads(data)
                        delta = chunk_data.get("choices", [{}])[0].get("delta", {})
                        content = delta.get("content", "")
                        if not content:
                            continue

                        if not emotion_stripped:
                            # Buffer until we can determine if there's an emotion tag
                            buffer += content
                            # Check if buffer has a complete emotion tag
                            m = EMOTION_PATTERN.match(buffer)
                            if m:
                                emotion = m.group(1)
                                _post_emotion(emotion)
                                remainder = buffer[m.end():]
                                buffer = ""
                                emotion_stripped = True
                                if remainder:
                                    accumulated_text += remainder
                                    yield _make_sse_chunk(remainder)
                            elif "]" in buffer or (len(buffer) > 15) or (buffer and buffer[0] != "["):
                                # No emotion tag — flush buffer as-is
                                emotion_stripped = True
                                accumulated_text += buffer
                                yield _make_sse_chunk(buffer)
                                buffer = ""
                        else:
                            # Strip save tags from output
                            accumulated_text += content
                            clean = SAVE_PATTERN.sub("", content)
                            if clean:
                                yield _make_sse_chunk(clean)

                    except (json.JSONDecodeError, IndexError, KeyError):
                        continue

        except httpx.TimeoutException:
            yield _make_sse_chunk("Sorry, I took too long thinking about that.")
            yield _make_sse_chunk(None, finish_reason="stop")
            yield "data: [DONE]\n\n"
        except Exception as e:
            print(f"[proxy] OpenRouter error: {e}")
            yield _make_sse_chunk("Hmm, my brain hit a snag.")
            yield _make_sse_chunk(None, finish_reason="stop")
            yield "data: [DONE]\n\n"


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    messages = body.get("messages", [])

    # Extract latest user text
    user_text = ""
    for m in reversed(messages):
        if m.get("role") == "user":
            user_text = m.get("content", "")
            break

    print(f"[proxy] Received: '{user_text}'")

    # Inject our system prompt + memory
    messages = _inject_system_prompt(messages)

    # Route: OpenClaw command or conversation
    if _openclaw.is_agent_command(user_text) and _openclaw.is_available():
        print(f"[proxy] OpenClaw command: '{user_text}'")
        return StreamingResponse(
            _openclaw_stream(user_text, messages),
            media_type="text/event-stream",
        )

    return StreamingResponse(
        _openrouter_stream(messages),
        media_type="text/event-stream",
    )


@app.get("/health")
async def health():
    return {"status": "ok"}


def run_server(port=8000, state=None, memory=None):
    """Start the proxy server (blocking). Call from a thread."""
    import uvicorn
    init(state=state, memory=memory)
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="LLM Proxy for Agora")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    # Load .env
    env_path = os.path.join(_PROJECT_ROOT, ".env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, _, v = line.partition("=")
                    k, v = k.strip(), v.strip()
                    if k and v:
                        os.environ.setdefault(k, v)

    init()
    import uvicorn
    print(f"[proxy] Starting LLM proxy on port {args.port}")
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="info")
