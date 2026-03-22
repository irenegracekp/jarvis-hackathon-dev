"""Philips Hue integration -- maps robot emotions + state to Hue lights.

Setup:
  1. pip install phue
  2. Set HUE_BRIDGE_IP in .env (e.g. HUE_BRIDGE_IP=192.168.1.100)
  3. Optionally set HUE_LIGHT_IDS as comma-separated light IDs (default: all lights in group 0)
  4. On first run, press the button on the Hue bridge within 30s when prompted.

The bridge IP can be discovered via https://discovery.meethue.com/ or your router's DHCP list.
"""

import os
import threading
import time
import logging

logger = logging.getLogger(__name__)

# Emotion -> Hue color mapping
# Each entry: (hue [0-65535], saturation [0-254], brightness [0-254])
# Hue uses a 0-65535 scale where 0=red, ~10920=yellow, ~21845=green, ~32768=cyan, ~43690=blue, ~54613=magenta
EMOTION_COLORS = {
    "excited":   {"hue": 7600,  "sat": 254, "bri": 254},   # warm yellow
    "curious":   {"hue": 43000, "sat": 254, "bri": 254},   # blue
    "calm":      {"hue": 21845, "sat": 200, "bri": 180},   # soft green
    "surprised": {"hue": 3500,  "sat": 254, "bri": 254},   # orange
    "amused":    {"hue": 56000, "sat": 180, "bri": 254},   # pink
    "skeptical": {"hue": 48000, "sat": 220, "bri": 230},   # purple
}

# State-level presets
STATE_PRESETS = {
    "idle": {"hue": 8000, "sat": 140, "bri": 60, "transitiontime": 30},           # dim warm white
    "ambient": {"hue": 8000, "sat": 140, "bri": 140, "transitiontime": 20},       # medium warm
    "engaged": None,  # Driven by emotion colors instead
}

# Default transition time in units of 100ms (10 = 1 second)
DEFAULT_TRANSITION = 5  # 0.5s


class HueBridge:
    """Controls Philips Hue lights based on robot emotion and state."""

    def __init__(self, bridge_ip=None, light_ids=None):
        self._bridge = None
        self._light_ids = light_ids or []
        self._current_emotion = None
        self._current_state = None
        self._lock = threading.Lock()
        self._breathing = False
        self._breath_thread = None

        bridge_ip = bridge_ip or os.environ.get("HUE_BRIDGE_IP", "").strip()
        if not bridge_ip:
            logger.warning("[hue] HUE_BRIDGE_IP not set. Hue integration disabled.")
            return

        try:
            from phue import Bridge
            self._bridge = Bridge(bridge_ip)
            self._bridge.connect()  # Will prompt button press on first run
            logger.info("[hue] Connected to Hue bridge at %s", bridge_ip)

            # Resolve light IDs
            if not self._light_ids:
                env_ids = os.environ.get("HUE_LIGHT_IDS", "").strip()
                if env_ids:
                    self._light_ids = [int(x.strip()) for x in env_ids.split(",") if x.strip()]
                else:
                    # Use all lights
                    self._light_ids = [l.light_id for l in self._bridge.lights]
                    logger.info("[hue] Using all lights: %s", self._light_ids)

            if self._light_ids:
                logger.info("[hue] Controlling lights: %s", self._light_ids)
            else:
                logger.warning("[hue] No lights found on bridge.")
                self._bridge = None

        except ImportError:
            logger.error("[hue] phue not installed. Run: pip install phue")
            self._bridge = None
        except Exception as e:
            logger.error("[hue] Failed to connect to Hue bridge: %s", e)
            self._bridge = None

    @property
    def is_connected(self):
        return self._bridge is not None

    def set_emotion(self, emotion):
        """Update lights to match the robot's current emotion."""
        if not self._bridge or emotion == self._current_emotion:
            return
        self._current_emotion = emotion
        self._stop_breathing()

        color = EMOTION_COLORS.get(emotion)
        if not color:
            logger.debug("[hue] Unknown emotion '%s', skipping", emotion)
            return

        cmd = {
            "on": True,
            "hue": color["hue"],
            "sat": color["sat"],
            "bri": color["bri"],
            "transitiontime": DEFAULT_TRANSITION,
        }
        self._send_command(cmd)
        logger.info("[hue] Emotion -> %s", emotion)

    def set_state(self, state_name):
        """Update lights for state transition (idle/ambient/engaged)."""
        if not self._bridge or state_name == self._current_state:
            return
        self._current_state = state_name

        if state_name == "ambient":
            self._start_breathing()
            return

        self._stop_breathing()

        preset = STATE_PRESETS.get(state_name)
        if preset is None:
            # "engaged" state — driven by emotion, just ensure lights are on and bright
            return

        cmd = {"on": True, **preset}
        self._send_command(cmd)
        logger.info("[hue] State -> %s", state_name)

    def flash(self, emotion=None, count=2):
        """Quick flash effect for greetings or events."""
        if not self._bridge:
            return

        color = EMOTION_COLORS.get(emotion or "excited")
        if not color:
            return

        def _do_flash():
            for _ in range(count):
                self._send_command({"on": True, "bri": 254, "hue": color["hue"],
                                    "sat": color["sat"], "transitiontime": 1})
                time.sleep(0.2)
                self._send_command({"on": True, "bri": 80, "hue": color["hue"],
                                    "sat": color["sat"], "transitiontime": 1})
                time.sleep(0.2)
            # Restore current emotion
            if self._current_emotion:
                self.set_emotion(self._current_emotion)

        threading.Thread(target=_do_flash, daemon=True, name="hue_flash").start()

    def off(self):
        """Turn off all controlled lights."""
        if not self._bridge:
            return
        self._stop_breathing()
        self._send_command({"on": False, "transitiontime": 20})
        logger.info("[hue] Lights off")

    def _send_command(self, cmd):
        """Send a command to all controlled lights."""
        with self._lock:
            try:
                for light_id in self._light_ids:
                    self._bridge.set_light(light_id, cmd)
            except Exception as e:
                logger.error("[hue] Command failed: %s", e)

    def _start_breathing(self):
        """Start a slow breathing/pulse effect for ambient mode."""
        if self._breathing:
            return
        self._breathing = True
        self._breath_thread = threading.Thread(
            target=self._breathing_loop, daemon=True, name="hue_breathe")
        self._breath_thread.start()

    def _stop_breathing(self):
        """Stop the breathing effect."""
        self._breathing = False
        if self._breath_thread:
            self._breath_thread.join(timeout=3)
            self._breath_thread = None

    def _breathing_loop(self):
        """Slowly pulse brightness up and down."""
        preset = STATE_PRESETS["ambient"]
        base_bri = preset["bri"]
        low_bri = max(40, base_bri - 80)
        high_bri = min(254, base_bri + 40)

        # Set initial color
        self._send_command({
            "on": True, "hue": preset["hue"], "sat": preset["sat"],
            "bri": base_bri, "transitiontime": 10,
        })
        time.sleep(1)

        going_up = True
        while self._breathing:
            target = high_bri if going_up else low_bri
            self._send_command({"bri": target, "transitiontime": 30})  # 3s transition
            going_up = not going_up
            # Wait for transition + pause
            for _ in range(35):  # 3.5s in 0.1s chunks so we can stop quickly
                if not self._breathing:
                    return
                time.sleep(0.1)


class DummyHueBridge:
    """No-op stub when Hue is disabled."""

    is_connected = False

    def set_emotion(self, emotion):
        pass

    def set_state(self, state_name):
        pass

    def flash(self, emotion=None, count=2):
        pass

    def off(self):
        pass
