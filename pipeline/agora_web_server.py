"""Web server for Agora voice — serves browser RTC client + manages agent lifecycle.

The browser handles all RTC audio (mic input, TTS playback).
This server provides session config, starts/stops the agent, and
processes datastream messages (tool calls from the agent).
"""

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

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_STATIC_DIR = _PROJECT_ROOT / "static" / "agora"

_agent_manager: AgentManager | None = None


class DatastreamPayload(BaseModel):
    text: str = ""
    ts: int | None = None


def create_app() -> FastAPI:
    app = FastAPI(title="Jarvis Agora Voice Server")
    app.mount("/static/agora", StaticFiles(directory=str(_STATIC_DIR)), name="static")

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
            "deviceKeywords": ["Hollyland", "Wireless", "Shenzhen", "USB"],
            "speakerKeywords": ["Bluetooth", "bluez", "A2DP"],
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

    @app.post("/api/datastream/message")
    async def datastream_message(payload: DatastreamPayload) -> dict[str, Any]:
        """Process datastream messages from the agent (tool calls, etc.)."""
        text = payload.text
        if not text:
            return {"ok": True}

        # Try to parse as JSON
        try:
            data = json.loads(text)
        except (json.JSONDecodeError, ValueError):
            # Try packed format: parts separated by |, last part is base64
            parts = text.split("|")
            if len(parts) >= 4:
                try:
                    import base64
                    decoded = base64.b64decode(parts[-1]).decode("utf-8")
                    data = json.loads(decoded)
                except Exception:
                    data = None
            else:
                data = None

        if data and isinstance(data, dict):
            obj_type = data.get("object", "")
            if obj_type == "message.user":
                content = data.get("content", "")
                logger.info("[datastream] User said: %s", content)
            elif "tool" in obj_type or "function" in str(data):
                logger.info("[datastream] Tool call: %s", json.dumps(data)[:200])
            else:
                logger.debug("[datastream] %s", json.dumps(data)[:200])

        return {"ok": True}

    @app.get("/api/health")
    def health():
        return {"status": "ok"}

    @app.on_event("shutdown")
    def on_shutdown():
        if _agent_manager and _agent_manager.is_running():
            logger.info("Stopping agent on server shutdown...")
            _agent_manager.stop_agent()

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

    # Load .env
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
