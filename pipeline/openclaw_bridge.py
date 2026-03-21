"""Bridge between voice commands and OpenClaw desktop agent.

Routes commands: if the voice input sounds like a computer command, send it to
OpenClaw. Otherwise, it's conversation for the Reachy personality.
"""

import json
import os
import re

try:
    import requests
except ImportError:
    requests = None

DEFAULT_API_URL = "http://127.0.0.1:18789"

# Keywords that indicate a computer/desktop command (vs conversation)
AGENT_KEYWORDS = [
    "open", "close", "search", "click", "type", "go to", "navigate",
    "tab", "window", "minimize", "maximize", "scroll", "download",
    "copy", "paste", "save", "print", "refresh", "reload", "browser",
    "terminal", "file", "folder", "screenshot", "google", "youtube",
    "move", "resize", "switch", "fullscreen", "launch", "run",
    "play", "pause", "next", "previous", "volume", "mute",
]

# Phrases that are clearly conversational despite containing keywords
CONVERSATION_PATTERNS = [
    r"^(what|how|why|who|when|where|do you|can you|tell me|think)",
    r"(your opinion|you think|you feel|about you)",
    r"^(joke|story|sing|dance|hello|hi |hey |goodbye|thanks)",
]


class OpenClawBridge:
    def __init__(self, api_url=DEFAULT_API_URL):
        self.api_url = api_url.rstrip("/")

    def is_agent_command(self, text):
        """Classify whether text is a computer command (True) or conversation (False)."""
        text_lower = text.lower().strip()

        # Check conversation patterns first (higher priority)
        for pattern in CONVERSATION_PATTERNS:
            if re.search(pattern, text_lower):
                return False

        # Check for agent keywords
        for keyword in AGENT_KEYWORDS:
            if keyword in text_lower:
                return True

        return False

    def send_command(self, text):
        """Send a voice command to OpenClaw via CLI. Returns response text."""
        import subprocess

        try:
            result = subprocess.run(
                ["openclaw", "agent", "--agent", "main", "--message", text, "--json"],
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
                # Return raw output, skip the banner line
                lines = output.split("\n")
                # Filter out the OpenClaw banner
                content = [l for l in lines if not l.startswith("🦞") and l.strip()]
                return "\n".join(content) if content else output

        except subprocess.TimeoutExpired:
            return "[error] OpenClaw timed out"
        except FileNotFoundError:
            return "[error] openclaw CLI not found"
        except Exception as e:
            return f"[error] OpenClaw failed: {e}"

    def is_available(self):
        """Check if OpenClaw API is reachable."""
        if requests is None:
            return False
        try:
            resp = requests.get(f"{self.api_url}/health", timeout=3)
            return resp.status_code == 200
        except Exception:
            return False


if __name__ == "__main__":
    bridge = OpenClawBridge()

    # Test classification
    tests = [
        ("open youtube", True),
        ("search for cats on google", True),
        ("go to gmail", True),
        ("close the tab", True),
        ("what do you think about AI?", False),
        ("tell me a joke", False),
        ("hey robot, how are you?", False),
        ("open a terminal", True),
        ("who are you?", False),
        ("click on the search bar", True),
        ("type hello world", True),
        ("do you like music?", False),
    ]

    print("=== Command Classification Tests ===")
    all_pass = True
    for text, expected in tests:
        result = bridge.is_agent_command(text)
        status = "PASS" if result == expected else "FAIL"
        if result != expected:
            all_pass = False
        print(f"  [{status}] '{text}' -> agent={result} (expected {expected})")

    print(f"\n{'All tests passed!' if all_pass else 'Some tests FAILED.'}")

    # Test connection
    print(f"\n=== OpenClaw Connection ===")
    print(f"API URL: {bridge.api_url}")
    print(f"Available: {bridge.is_available()}")
