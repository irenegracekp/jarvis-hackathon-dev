"""MCP server exposing OpenClaw as a tool for Agora Conversational AI.

Runs as a Streamable HTTP MCP server. Agora's LLM calls the
"execute_desktop_command" tool when the user asks to do something
on the computer (open youtube, search google, etc.).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings

# Disable DNS rebinding protection so tunnel (cloudflare/ngrok) can reach us
_security = TransportSecuritySettings(enable_dns_rebinding_protection=False)
mcp = FastMCP("OpenClaw Desktop Control", stateless_http=True, transport_security=_security)


@mcp.tool()
def execute_desktop_command(command: str) -> str:
    """Execute a desktop command on the computer using OpenClaw.

    Use this tool when the user asks you to do something on the computer,
    such as: open a website, search for something, play a video, open an
    application, click something, type text, scroll, navigate tabs, etc.

    Args:
        command: The natural language command to execute, e.g. "open youtube",
                 "search for cats on google", "play lo-fi music".
    """
    import json
    import subprocess

    try:
        result = subprocess.run(
            ["openclaw", "agent", "--agent", "main", "--message", command, "--json"],
            capture_output=True, text=True, timeout=30
        )
        output = result.stdout.strip()
        if not output:
            output = result.stderr.strip()

        # Try to parse JSON response
        try:
            data = json.loads(output)
            return data.get("response", data.get("message", str(data)))
        except (json.JSONDecodeError, ValueError):
            # Filter out banner lines
            lines = output.split("\n")
            content = [l for l in lines if not l.startswith("\U0001f99e") and l.strip()]
            return "\n".join(content) if content else output

    except subprocess.TimeoutExpired:
        return "Command timed out after 30 seconds."
    except FileNotFoundError:
        return "OpenClaw CLI not found. Make sure it's installed."
    except Exception as e:
        return f"Error executing command: {e}"


def run_server(port=8000):
    """Start the MCP server (blocking)."""
    import uvicorn
    app = mcp.streamable_http_app()
    print(f"[mcp] OpenClaw MCP server starting on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="OpenClaw MCP Server")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    run_server(port=args.port)
