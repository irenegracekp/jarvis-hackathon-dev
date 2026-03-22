"""Web server for Agora voice — serves browser RTC client + manages agent lifecycle.

The browser handles all RTC audio (mic input, TTS playback).
This server provides session config, starts/stops the agent,
processes datastream messages (tool calls), and drives Reachy Mini.
"""

import base64
import json
import logging
import os
import threading
import webbrowser
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from pipeline.agent_manager import AgentManager
from pipeline.reachy_bridge import ReachyBridge

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_STATIC_DIR = _PROJECT_ROOT / "static" / "agora"

_agent_manager: AgentManager | None = None
_reachy: ReachyBridge = ReachyBridge()

# Live state for dashboard
_dashboard_state = {
    "conversation_state": "idle",
    "last_user_text": "",
    "last_agent_text": "",
    "current_emotion": "",
    "audio_level": 0.0,
    "reachy_connected": False,
    "agent_running": False,
    "events": [],  # last 20 events
}
_state_lock = threading.Lock()

def _push_event(event_type: str, text: str):
    """Add event to dashboard feed."""
    import time
    with _state_lock:
        _dashboard_state["events"].append({
            "type": event_type,
            "text": text,
            "ts": time.time(),
        })
        # Keep last 30 events
        if len(_dashboard_state["events"]) > 30:
            _dashboard_state["events"] = _dashboard_state["events"][-30:]

def _update_state(**kwargs):
    with _state_lock:
        _dashboard_state.update(kwargs)


class DatastreamPayload(BaseModel):
    text: str = ""
    ts: int | None = None


class AudioChunkPayload(BaseModel):
    pcm_b64: str = ""
    level: float = 0.0
    sample_rate: int = 24000


def _decode_packed(text: str) -> dict[str, Any] | None:
    """Decode Agora packed datastream: parts separated by |, last part is base64 JSON."""
    parts = text.split("|")
    if len(parts) < 4:
        return None
    try:
        decoded = base64.b64decode(parts[-1]).decode("utf-8")
        data = json.loads(decoded)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _dispatch_action(data: dict[str, Any]) -> None:
    """Dispatch robot action from LLM tool call."""
    action_type = str(data.get("action_type", "")).strip().lower()
    if not action_type:
        return

    if action_type in ("display_emotion", "play_emotion", "emotion"):
        emotion = data.get("emotion_type") or data.get("emotion", "")
        if emotion:
            _reachy.play_emotion(str(emotion))
            _update_state(current_emotion=str(emotion))
            _push_event("emotion", str(emotion))

    elif action_type == "move_head":
        direction = data.get("direction", "")
        if direction:
            _reachy.move_head(str(direction))
            _push_event("move", f"head {direction}")

    elif action_type in ("dance",):
        move = data.get("move", "dance1")
        _reachy.play_emotion(str(move))
        _push_event("dance", str(move))

    elif action_type == "wiggle":
        _reachy.wiggle_antennas()
        _push_event("action", "wiggle antennas")

    else:
        logger.info("[action] Unknown action_type: %s", action_type)


def create_app() -> FastAPI:
    app = FastAPI(title="Jarvis Agora Voice Server")
    app.mount("/static/agora", StaticFiles(directory=str(_STATIC_DIR)), name="static")

    @app.on_event("startup")
    def on_startup():
        # Connect to Reachy Mini
        if _reachy.connect():
            _reachy.wiggle_antennas()

    @app.get("/")
    def index():
        return FileResponse(str(_STATIC_DIR / "index.html"))

    @app.get("/api/agora/session")
    def get_session() -> dict[str, Any]:
        app_id = os.environ.get("AGORA_APP_ID", "")
        channel = os.environ.get("AGORA_CHANNEL", "reachy_conversation")
        uid = int(os.environ.get("AGORA_USER_UID", "12345"))

        return {
            "appId": app_id,
            "channel": channel,
            "uid": uid,
            "token": "",
            "deviceKeywords": ["Wireless", "Hollyland", "Shenzhen", "Reachy", "USB", "Pollen"],
            "speakerKeywords": ["HK Onyx", "Bluetooth", "bluez", "A2DP"],
        }

    @app.post("/api/agora/agent/start")
    def start_agent() -> dict[str, Any]:
        global _agent_manager
        if _agent_manager is None:
            _agent_manager = _create_agent_manager()
        if _agent_manager is None:
            return {"ok": False, "error": "Missing Agora credentials"}

        if _agent_manager.is_running():
            return {"ok": True, "started": False, "reason": "already_running"}

        channel = os.environ.get("AGORA_CHANNEL", "reachy_conversation")
        uid = int(os.environ.get("AGORA_USER_UID", "12345"))

        started = _agent_manager.start_agent(channel, uid)
        if started:
            return {"ok": True, "started": True, "agent_id": _agent_manager.agent_id}
        return {"ok": False, "error": "Agent start failed"}

    @app.post("/api/agora/agent/stop")
    def stop_agent() -> dict[str, Any]:
        if _agent_manager and _agent_manager.is_running():
            _agent_manager.stop_agent()
            return {"ok": True}
        return {"ok": True, "reason": "not_running"}

    @app.post("/api/motion/audio-chunk")
    def motion_audio_chunk(payload: AudioChunkPayload) -> dict[str, Any]:
        """Receive TTS audio chunk from browser for head wobble."""
        if payload.level > 0.01:
            _reachy.feed_audio_chunk(payload.level)
            _update_state(audio_level=payload.level)
        return {"ok": True}

    @app.post("/api/motion/session")
    def motion_session(payload: dict[str, Any]) -> dict[str, Any]:
        """Session state from browser."""
        return {"ok": True}

    @app.post("/api/datastream/message")
    async def datastream_message(payload: DatastreamPayload) -> dict[str, Any]:
        """Process datastream messages from the agent."""
        text = payload.text
        if not text:
            return {"ok": True}

        # Try direct JSON
        data = None
        try:
            data = json.loads(text)
        except (json.JSONDecodeError, ValueError):
            data = _decode_packed(text)

        if not data or not isinstance(data, dict):
            return {"ok": True}

        obj_type = data.get("object", "")

        # Conversation state (speaking/listening)
        if obj_type == "message.state":
            state = str(data.get("state", "")).lower()
            _update_state(conversation_state=state)

        # User transcription (live ASR)
        elif obj_type == "user.transcription":
            text_val = data.get("text", "")
            is_final = data.get("final", False)
            if is_final and text_val:
                _update_state(last_user_text=text_val)
                _push_event("user", text_val)

        # User speech transcription or tool action
        elif obj_type == "message.user":
            content = data.get("content", "")
            # Check if content is actually a JSON action from the LLM
            if isinstance(content, str) and "action_type" in content:
                try:
                    action = json.loads(content)
                    if isinstance(action, dict) and action.get("action_type"):
                        _dispatch_action(action)
                except (json.JSONDecodeError, ValueError):
                    pass
            elif isinstance(content, str) and content.strip():
                _update_state(last_user_text=content)
                _push_event("user", content)

        # Agent response text
        elif obj_type == "message.assistant":
            content = data.get("content", "")
            if isinstance(content, str) and content.strip():
                _update_state(last_agent_text=content)
                _push_event("jarvis", content)

        # Any other message — try to extract action
        else:
            action = data
            content = data.get("content")
            if isinstance(content, str):
                try:
                    action = json.loads(content)
                except (json.JSONDecodeError, ValueError):
                    pass
            elif isinstance(content, dict):
                action = content

            if isinstance(action, dict) and action.get("action_type"):
                _dispatch_action(action)

        return {"ok": True}

    @app.get("/api/dashboard/state")
    def dashboard_state() -> dict[str, Any]:
        """Live state for dashboard UI."""
        with _state_lock:
            state = dict(_dashboard_state)
            state["reachy_connected"] = _reachy.connected
            state["agent_running"] = _agent_manager.is_running() if _agent_manager else False
            return state

    @app.get("/api/health")
    def health():
        return {"status": "ok", "reachy": _reachy.connected}

    @app.on_event("shutdown")
    def on_shutdown():
        if _agent_manager and _agent_manager.is_running():
            logger.info("Stopping agent...")
            _agent_manager.stop_agent()
        _reachy.disconnect()

    return app


def _create_agent_manager() -> AgentManager | None:
    app_id = os.environ.get("AGORA_APP_ID", "").strip()
    api_key = os.environ.get("AGORA_CUSTOMER_ID", "").strip()
    api_secret = os.environ.get("AGORA_CUSTOMER_SECRET", "").strip()

    if not all([app_id, api_key, api_secret]):
        logger.error("Missing AGORA_APP_ID, AGORA_CUSTOMER_ID, or AGORA_CUSTOMER_SECRET")
        return None

    config_path = _PROJECT_ROOT / "agent_config.json"
    return AgentManager(
        app_id=app_id,
        api_key=api_key,
        api_secret=api_secret,
        config_file=str(config_path),
    )


def run_server(port=8080, open_browser=True):
    """Start the web server (blocking). Call from a thread or directly."""
    host = "0.0.0.0"
    url = f"http://localhost:{port}"

    if open_browser:
        def _open():
            try:
                webbrowser.open(url, new=1)
                logger.info("Opened browser: %s", url)
            except Exception as e:
                logger.warning("Could not open browser: %s. Open manually: %s", e, url)
        threading.Timer(2.0, _open).start()

    logger.info("Agora voice server at http://%s:%s", host, port)
    uvicorn.run(create_app(), host=host, port=port, log_level="warning", access_log=False)


if __name__ == "__main__":
    import argparse
    import sys
    sys.path.insert(0, str(_PROJECT_ROOT))

    env_path = _PROJECT_ROOT / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, _, v = line.partition("=")
                    k, v = k.strip(), v.strip()
                    if k and v:
                        os.environ.setdefault(k, v)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")

    parser = argparse.ArgumentParser(description="Jarvis Agora Voice Server")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    run_server(port=args.port, open_browser=not args.no_browser)
