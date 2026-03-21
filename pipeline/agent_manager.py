"""Agora Conversational AI Agent Manager.

Handles REST API calls to start/stop the conversational AI agent.
Loads config from agent_config.json with {{template}} support.
Based on reachy-mini-agora-web-sdk by Pollen Robotics.
"""

import base64
import copy
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Optional

import requests

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[1]


class AgentManager:
    def __init__(self, app_id: str, api_key: str, api_secret: str,
                 config_file: str = "agent_config.json"):
        self.app_id = app_id
        self.api_key = api_key
        self.api_secret = api_secret
        self.config_file = Path(config_file)
        self.agent_id: Optional[str] = None
        self.agent_config: Optional[dict[str, Any]] = None
        self.base_url = "https://api.agora.io/api/conversational-ai-agent/v2"

        if self.config_file.exists():
            self.load_config()

    def load_config(self) -> bool:
        try:
            with open(self.config_file, "r") as f:
                self.agent_config = json.load(f)
            self._render_placeholders()
            logger.info("Agent config loaded from %s", self.config_file)
            return True
        except Exception as e:
            logger.error("Failed to load config: %s", e)
            return False

    def _render_placeholders(self):
        """Replace {{filename}} placeholders with file contents."""
        if not isinstance(self.agent_config, dict):
            return
        pattern = re.compile(r"\{\{\s*([^{}]+?)\s*\}\}")
        base_dir = self.config_file.parent

        def resolve(file_expr: str) -> str:
            candidate = Path(file_expr.strip())
            if not candidate.is_absolute():
                candidate = base_dir / candidate
            try:
                return candidate.read_text(encoding="utf-8")
            except Exception as e:
                logger.warning("Failed to read placeholder %s: %s", candidate, e)
                return ""

        def walk(node):
            if isinstance(node, dict):
                return {k: walk(v) for k, v in node.items()}
            if isinstance(node, list):
                return [walk(v) for v in node]
            if isinstance(node, str) and "{{" in node and "}}" in node:
                return pattern.sub(lambda m: resolve(m.group(1)), node)
            return node

        self.agent_config = walk(self.agent_config)

    def _auth_header(self) -> str:
        encoded = base64.b64encode(f"{self.api_key}:{self.api_secret}".encode()).decode()
        return f"Basic {encoded}"

    def start_agent(self, channel_name: str, user_uid: int, token: str = "") -> bool:
        if not self.agent_config:
            logger.error("No agent config loaded.")
            return False

        payload = copy.deepcopy(self.agent_config)
        payload["properties"]["channel"] = channel_name
        payload["properties"]["token"] = token
        if user_uid:
            payload["properties"]["remote_rtc_uids"] = [str(user_uid)]

        # Ensure tools are enabled
        advanced = payload["properties"].get("advanced_features", {})
        advanced["enable_tools"] = True
        payload["properties"]["advanced_features"] = advanced

        # Add MCP server config if MCP_PUBLIC_URL is set
        mcp_url = os.environ.get("MCP_PUBLIC_URL", "").strip()
        if mcp_url:
            llm = payload["properties"].get("llm", {})
            llm["mcp_servers"] = [{
                "name": "openclaw",
                "endpoint": f"{mcp_url}/mcp",
                "transport": "streamable_http",
                "allowed_tools": ["execute_desktop_command"],
                "timeout_ms": 30000,
            }]
            payload["properties"]["llm"] = llm
            logger.info("MCP server configured: %s/mcp", mcp_url)

        url = f"{self.base_url}/projects/{self.app_id}/join"
        headers = {
            "Authorization": self._auth_header(),
            "Content-Type": "application/json",
        }

        try:
            logger.info("Starting agent on channel '%s'...", channel_name)
            resp = requests.post(url, headers=headers, json=payload, timeout=30)

            if resp.status_code == 200:
                data = resp.json()
                self.agent_id = data.get("agent_id") or data.get("agentId")
                logger.info("Agent started: %s", self.agent_id)
                return True

            if resp.status_code == 409:
                logger.warning("Agent conflict (409), stopping old agent and retrying...")
                self._handle_conflict(url, headers, payload, resp)
                return self.agent_id is not None

            logger.error("Agent start failed: %s - %s", resp.status_code, resp.text)
            return False

        except Exception as e:
            logger.error("Agent start error: %s", e)
            return False

    def _handle_conflict(self, url, headers, payload, conflict_resp):
        """Stop conflicting agent and retry."""
        try:
            data = conflict_resp.json()
            old_id = data.get("agent_id") or data.get("agentId")
        except Exception:
            old_id = None

        if old_id:
            self.stop_agent_by_id(old_id)
            time.sleep(1.5)

        resp = requests.post(url, headers=headers, json=payload, timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            self.agent_id = data.get("agent_id") or data.get("agentId")
            logger.info("Agent started after retry: %s", self.agent_id)

    def stop_agent(self) -> bool:
        if not self.agent_id:
            return False
        return self.stop_agent_by_id(self.agent_id)

    def stop_agent_by_id(self, agent_id: str) -> bool:
        headers = {
            "Authorization": self._auth_header(),
            "Content-Type": "application/json",
        }
        url = f"{self.base_url}/projects/{self.app_id}/agents/{agent_id}/leave"
        try:
            resp = requests.post(url, headers=headers, timeout=10)
            if resp.status_code in (200, 204, 404):
                logger.info("Agent stopped: %s", agent_id)
                if self.agent_id == agent_id:
                    self.agent_id = None
                return True
            logger.error("Stop failed: %s - %s", resp.status_code, resp.text)
            return False
        except Exception as e:
            logger.error("Stop error: %s", e)
            return False

    def is_running(self) -> bool:
        return self.agent_id is not None
