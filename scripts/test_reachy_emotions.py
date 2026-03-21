"""Test Reachy Mini emotions - dance, cheerful, surprised, etc."""

import json
import sys
import time
from reachy_mini import ReachyMini
from reachy_mini.motion.recorded_move import RecordedMove

EMOTIONS_DIR = "/home/orin/hacky/models/emotions"

def play(mini, name):
    path = f"{EMOTIONS_DIR}/{name}.json"
    with open(path) as f:
        data = json.load(f)
    move = RecordedMove(data)
    print(f"Playing: {name} ({move.duration:.1f}s)")
    mini.play_move(move, sound=False)
    time.sleep(move.duration + 0.5)

with ReachyMini() as mini:
    mini.enable_motors()

    if len(sys.argv) > 1:
        # Play specific emotion: python3 scripts/test_reachy_emotions.py dance1
        play(mini, sys.argv[1])
    else:
        # Demo sequence
        for name in ["dance1", "cheerful1", "surprised1", "curious1", "sad1"]:
            play(mini, name)
            time.sleep(0.5)

    print("Done!")
