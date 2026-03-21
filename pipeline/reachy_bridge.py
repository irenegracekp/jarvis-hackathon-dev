"""Bridge between Agora voice pipeline and Reachy Mini robot.

Handles:
- Connecting to the robot
- Head wobble during TTS speech (audio-reactive)
- Playing recorded emotions from the HuggingFace emotion library
- Head movement commands
"""

import json
import logging
import os
import threading
import time

import numpy as np

logger = logging.getLogger(__name__)

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_EMOTIONS_DIR = os.path.join(_PROJECT_ROOT, "models", "emotions")


class ReachyBridge:
    def __init__(self):
        self._robot = None
        self._connected = False
        self._wobble_thread = None
        self._wobble_running = False
        self._current_level = 0.0
        self._last_audio_time = 0.0
        self._lock = threading.Lock()
        self._emotions_cache = {}  # name -> RecordedMove
        self._playing_emotion = False

    @property
    def connected(self):
        return self._connected

    def connect(self):
        """Connect to Reachy Mini."""
        try:
            from reachy_mini import ReachyMini
            self._robot = ReachyMini()
            self._robot.__enter__()
            self._robot.enable_motors()
            self._connected = True
            logger.info("Connected to Reachy Mini (motors enabled)")

            # Preload emotion list
            if os.path.isdir(_EMOTIONS_DIR):
                count = len([f for f in os.listdir(_EMOTIONS_DIR) if f.endswith('.json')])
                logger.info("Emotion library: %d emotions available", count)

            # Start wobble thread
            self._wobble_running = True
            self._wobble_thread = threading.Thread(target=self._wobble_loop, daemon=True)
            self._wobble_thread.start()

            return True
        except Exception as e:
            logger.error("Failed to connect to Reachy Mini: %s", e)
            self._connected = False
            return False

    def disconnect(self):
        """Disconnect from Reachy Mini."""
        self._wobble_running = False
        if self._wobble_thread:
            self._wobble_thread.join(timeout=2)
        if self._robot:
            try:
                self._robot.__exit__(None, None, None)
            except Exception:
                pass
            self._robot = None
        self._connected = False
        logger.info("Disconnected from Reachy Mini")

    def feed_audio_chunk(self, level: float):
        """Feed audio energy level (0-1) for head wobble."""
        with self._lock:
            self._current_level = min(max(float(level), 0.0), 1.0)
            self._last_audio_time = time.time()

    def play_emotion(self, emotion_id: str):
        """Play a recorded emotion on the robot."""
        if not self._connected or not self._robot:
            return
        if self._playing_emotion:
            logger.info("Skipping emotion %s (already playing)", emotion_id)
            return

        emotion_id = emotion_id.lower().strip()

        # Load emotion from cache or file
        move = self._load_emotion(emotion_id)
        if move is None:
            logger.warning("Emotion not found: %s", emotion_id)
            return

        # Play in background thread so we don't block
        def _play():
            self._playing_emotion = True
            try:
                logger.info("Playing emotion: %s (%.1fs)", emotion_id, move.duration)
                self._robot.play_move(move, sound=False)
                time.sleep(move.duration + 0.3)
            except Exception as e:
                logger.error("Emotion play error: %s", e)
            finally:
                self._playing_emotion = False

        threading.Thread(target=_play, daemon=True).start()

    def _load_emotion(self, emotion_id):
        """Load a recorded emotion from the library."""
        if emotion_id in self._emotions_cache:
            return self._emotions_cache[emotion_id]

        from reachy_mini.motion.recorded_move import RecordedMove

        path = os.path.join(_EMOTIONS_DIR, f"{emotion_id}.json")
        if not os.path.exists(path):
            return None

        try:
            with open(path) as f:
                data = json.load(f)
            move = RecordedMove(data)
            self._emotions_cache[emotion_id] = move
            return move
        except Exception as e:
            logger.error("Failed to load emotion %s: %s", emotion_id, e)
            return None

    def move_head(self, direction: str):
        """Move head in a direction."""
        if not self._connected or not self._robot:
            return

        from reachy_mini.utils import create_head_pose

        moves = {
            "left": create_head_pose(yaw=20),
            "right": create_head_pose(yaw=-20),
            "up": create_head_pose(pitch=15),
            "down": create_head_pose(pitch=-15),
            "front": create_head_pose(pitch=0, yaw=0),
        }

        try:
            if direction == "nod":
                self._robot.goto_target(head=create_head_pose(pitch=-10), duration=0.3)
                time.sleep(0.35)
                self._robot.goto_target(head=create_head_pose(pitch=5), duration=0.3)
                time.sleep(0.35)
                self._robot.goto_target(head=create_head_pose(pitch=0), duration=0.3)
            elif direction in moves:
                self._robot.goto_target(head=moves[direction], duration=0.5)
            logger.info("Head move: %s", direction)
        except Exception as e:
            logger.error("Head move error: %s", e)

    def wiggle_antennas(self):
        """Quick antenna wiggle."""
        if not self._connected or not self._robot:
            return
        try:
            self._robot.goto_target(antennas=np.deg2rad([40, 40]), duration=0.3)
            time.sleep(0.35)
            self._robot.goto_target(antennas=np.deg2rad([0, 0]), duration=0.3)
        except Exception:
            pass

    def _wobble_loop(self):
        """Background thread: subtle head pitch wobble while speaking."""
        logger.info("Wobble loop started")
        from reachy_mini.utils import create_head_pose
        last_log = 0
        while self._wobble_running:
            with self._lock:
                level = self._current_level
                audio_age = time.time() - self._last_audio_time

            # Wobble if we received audio in the last 0.5s and not playing an emotion
            if audio_age < 0.5 and level > 0.02 and self._connected and self._robot and not self._playing_emotion:
                pitch = np.sin(time.time() * 6) * level * 25
                try:
                    self._robot.set_target(head=create_head_pose(pitch=pitch))
                    now = time.time()
                    if now - last_log > 2:
                        logger.info("Wobbling: pitch=%.1f level=%.3f", pitch, level)
                        last_log = now
                except Exception as e:
                    logger.error("Wobble error: %s", e)
                time.sleep(0.03)
            else:
                time.sleep(0.05)

        logger.info("Wobble loop stopped")
