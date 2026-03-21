#!/usr/bin/env python3
"""Test OpenClaw bridge: classification + live command sending."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.openclaw_bridge import OpenClawBridge


def main():
    bridge = OpenClawBridge()

    print("=== OpenClaw Bridge Test ===")
    print(f"API: {bridge.api_url}")
    print(f"Available: {bridge.is_available()}")
    print()

    if not bridge.is_available():
        print("OpenClaw not running. Classification-only mode.")
        print("Start OpenClaw with: openclaw onboard --install-daemon")
        print()

    print("Type commands to test routing. 'quit' to exit.")
    print("Commands are classified as agent (-> OpenClaw) or conversation (-> Reachy).\n")

    while True:
        try:
            text = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if text.lower() in ("quit", "exit", "q"):
            break
        if not text:
            continue

        is_cmd = bridge.is_agent_command(text)
        print(f"  Classification: {'AGENT (OpenClaw)' if is_cmd else 'CONVERSATION (Reachy)'}")

        if is_cmd and bridge.is_available():
            print(f"  Sending to OpenClaw...")
            response = bridge.send_command(text)
            print(f"  Response: {response}")
        elif is_cmd:
            print(f"  (OpenClaw not running, would send: '{text}')")
        print()


if __name__ == "__main__":
    main()
