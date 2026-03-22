"""Smart light integration -- maps robot emotions + state to lights.

Supports multiple backends:
  - hue:   Philips Hue (requires bridge + ethernet)
  - lifx:  LIFX (WiFi direct, no hub, local LAN control)
  - govee: Govee (WiFi, cloud API)

Common interface: set_emotion(), set_state(), flash(), off()

Setup per backend:

  HUE:
    pip install phue
    Set HUE_BRIDGE_IP in .env
    Optionally set HUE_LIGHT_IDS (comma-separated)
    Press bridge button on first run

  LIFX:
    pip install lifxlan
    Connect LIFX light to same WiFi network
    Optionally set LIFX_LIGHT_LABEL in .env to target a specific light

  GOVEE:
    pip install govee-api-laggat
    Get API key from https://developer.govee.com/
    Set GOVEE_API_KEY in .env
    Set GOVEE_DEVICE and GOVEE_MODEL in .env (from Govee app)
"""

import os
import threading
import time
import logging

logger = logging.getLogger(__name__)

# Emotion -> RGB color mapping (shared across all backends)
EMOTION_RGB = {
    "excited":   (255, 200, 0),    # warm yellow
    "curious":   (0, 150, 255),    # blue
    "calm":      (100, 200, 100),  # soft green
    "surprised": (255, 100, 0),    # orange
    "amused":    (255, 150, 200),  # pink
    "skeptical": (200, 100, 255),  # purple
}

# State-level presets (RGB + brightness 0-1)
STATE_RGB = {
    "idle":    {"rgb": (255, 180, 100), "brightness": 0.25},  # dim warm white
    "ambient": {"rgb": (255, 180, 100), "brightness": 0.55},  # medium warm
    "engaged": None,  # Driven by emotion colors
}


def create_light_backend(backend_name):
    """Factory: returns the right light controller based on backend name."""
    if backend_name == "hue":
        return HueBackend()
    elif backend_name == "lifx":
        return LIFXBackend()
    elif backend_name == "govee":
        return GoveeBackend()
    else:
        return DummyLightBackend()


class BaseLightBackend:
    """Base class with shared emotion/state logic. Subclasses implement _set_color/_set_brightness/_turn_off."""

    def __init__(self):
        self._current_emotion = None
        self._current_state = None
        self._lock = threading.Lock()
        self._breathing = False
        self._breath_thread = None

    @property
    def is_connected(self):
        return False

    def set_emotion(self, emotion):
        """Update lights to match the robot's current emotion."""
        if not self.is_connected or emotion == self._current_emotion:
            return
        self._current_emotion = emotion
        self._stop_breathing()

        rgb = EMOTION_RGB.get(emotion)
        if not rgb:
            logger.debug("[lights] Unknown emotion '%s', skipping", emotion)
            return

        self._set_color(*rgb, brightness=1.0, duration_ms=500)
        logger.info("[lights] Emotion -> %s", emotion)

    def set_state(self, state_name):
        """Update lights for state transition (idle/ambient/engaged)."""
        if not self.is_connected or state_name == self._current_state:
            return
        self._current_state = state_name

        if state_name == "ambient":
            self._start_breathing()
            return

        self._stop_breathing()

        preset = STATE_RGB.get(state_name)
        if preset is None:
            return  # "engaged" is emotion-driven

        r, g, b = preset["rgb"]
        self._set_color(r, g, b, brightness=preset["brightness"], duration_ms=3000)
        logger.info("[lights] State -> %s", state_name)

    def flash(self, emotion=None, count=2):
        """Quick flash effect for greetings or events."""
        if not self.is_connected:
            return

        rgb = EMOTION_RGB.get(emotion or "excited", (255, 200, 0))

        def _do_flash():
            for _ in range(count):
                self._set_color(*rgb, brightness=1.0, duration_ms=100)
                time.sleep(0.2)
                self._set_color(*rgb, brightness=0.3, duration_ms=100)
                time.sleep(0.2)
            if self._current_emotion:
                c = EMOTION_RGB.get(self._current_emotion, rgb)
                self._set_color(*c, brightness=1.0, duration_ms=500)

        threading.Thread(target=_do_flash, daemon=True, name="lights_flash").start()

    def off(self):
        """Turn off all controlled lights."""
        if not self.is_connected:
            return
        self._stop_breathing()
        self._turn_off()
        logger.info("[lights] Lights off")

    def _set_color(self, r, g, b, brightness=1.0, duration_ms=500):
        """Override in subclass: set light to RGB color with brightness and transition."""
        raise NotImplementedError

    def _turn_off(self):
        """Override in subclass: turn off lights."""
        raise NotImplementedError

    def _start_breathing(self):
        if self._breathing:
            return
        self._breathing = True
        self._breath_thread = threading.Thread(
            target=self._breathing_loop, daemon=True, name="lights_breathe")
        self._breath_thread.start()

    def _stop_breathing(self):
        self._breathing = False
        if self._breath_thread:
            self._breath_thread.join(timeout=3)
            self._breath_thread = None

    def _breathing_loop(self):
        preset = STATE_RGB["ambient"]
        r, g, b = preset["rgb"]
        self._set_color(r, g, b, brightness=preset["brightness"], duration_ms=1000)
        time.sleep(1)

        going_up = True
        while self._breathing:
            bri = 0.7 if going_up else 0.3
            self._set_color(r, g, b, brightness=bri, duration_ms=3000)
            going_up = not going_up
            for _ in range(35):
                if not self._breathing:
                    return
                time.sleep(0.1)


# ─── LIFX Backend ────────────────────────────────────────────────────────────

class LIFXBackend(BaseLightBackend):
    """Controls LIFX lights via local LAN (no cloud, no hub).

    Setup:
      pip install lifxlan
      Connect LIFX to same WiFi
      Optionally set LIFX_LIGHT_LABEL in .env
    """

    def __init__(self):
        super().__init__()
        self._light = None

        try:
            from lifxlan import LifxLAN
            label = os.environ.get("LIFX_LIGHT_LABEL", "").strip()

            lan = LifxLAN()
            if label:
                lights = lan.get_lights()
                for l in lights:
                    if l.get_label() == label:
                        self._light = l
                        break
                if not self._light:
                    logger.warning("[lifx] Light '%s' not found. Available: %s",
                                   label, [l.get_label() for l in lights])
            else:
                lights = lan.get_lights()
                if lights:
                    self._light = lights[0]

            if self._light:
                self._light.set_power("on")
                logger.info("[lifx] Connected to '%s'", self._light.get_label())
            else:
                logger.warning("[lifx] No LIFX lights found on network.")

        except ImportError:
            logger.error("[lifx] lifxlan not installed. Run: pip install lifxlan")
        except Exception as e:
            logger.error("[lifx] Failed to connect: %s", e)

    @property
    def is_connected(self):
        return self._light is not None

    def _set_color(self, r, g, b, brightness=1.0, duration_ms=500):
        with self._lock:
            try:
                h, s, v = _rgb_to_hsv(r, g, b)
                # LIFX uses HSBK: hue 0-65535, sat 0-65535, bri 0-65535, kelvin 2500-9000
                hue = int(h * 65535)
                sat = int(s * 65535)
                bri = int(v * brightness * 65535)
                bri = max(0, min(65535, bri))
                self._light.set_color([hue, sat, bri, 3500], duration=duration_ms)
            except Exception as e:
                logger.error("[lifx] Color command failed: %s", e)

    def _turn_off(self):
        with self._lock:
            try:
                self._light.set_power("off", duration=2000)
            except Exception as e:
                logger.error("[lifx] Off command failed: %s", e)


# ─── Govee Backend ───────────────────────────────────────────────────────────

class GoveeBackend(BaseLightBackend):
    """Controls Govee lights via cloud API.

    Setup:
      pip install govee-api-laggat
      Get API key from https://developer.govee.com/
      Set GOVEE_API_KEY, GOVEE_DEVICE, GOVEE_MODEL in .env
    """

    def __init__(self):
        super().__init__()
        self._api_key = os.environ.get("GOVEE_API_KEY", "").strip()
        self._device = os.environ.get("GOVEE_DEVICE", "").strip()
        self._model = os.environ.get("GOVEE_MODEL", "").strip()
        self._connected = False

        if not self._api_key:
            logger.warning("[govee] GOVEE_API_KEY not set. Govee integration disabled.")
            return
        if not self._device or not self._model:
            logger.warning("[govee] GOVEE_DEVICE and GOVEE_MODEL required. "
                           "Find them in Govee app or API: GET https://developer-api.govee.com/v1/devices")
            return

        # Test the connection
        try:
            import requests
            resp = requests.get(
                "https://developer-api.govee.com/v1/devices",
                headers={"Govee-API-Key": self._api_key},
                timeout=10,
            )
            if resp.status_code == 200:
                self._connected = True
                devices = resp.json().get("data", {}).get("devices", [])
                names = [(d["device"], d["model"], d.get("deviceName", "")) for d in devices]
                logger.info("[govee] Connected. Devices: %s", names)
            else:
                logger.error("[govee] API error %d: %s", resp.status_code, resp.text)
        except ImportError:
            logger.error("[govee] requests not installed (should be in requirements)")
        except Exception as e:
            logger.error("[govee] Failed to connect: %s", e)

    @property
    def is_connected(self):
        return self._connected

    def _set_color(self, r, g, b, brightness=1.0, duration_ms=500):
        with self._lock:
            try:
                import requests
                # Set color
                requests.put(
                    "https://developer-api.govee.com/v1/devices/control",
                    headers={"Govee-API-Key": self._api_key, "Content-Type": "application/json"},
                    json={
                        "device": self._device,
                        "model": self._model,
                        "cmd": {"name": "color", "value": {"r": r, "g": g, "b": b}},
                    },
                    timeout=5,
                )
                # Set brightness (0-100)
                bri_pct = max(1, min(100, int(brightness * 100)))
                requests.put(
                    "https://developer-api.govee.com/v1/devices/control",
                    headers={"Govee-API-Key": self._api_key, "Content-Type": "application/json"},
                    json={
                        "device": self._device,
                        "model": self._model,
                        "cmd": {"name": "brightness", "value": bri_pct},
                    },
                    timeout=5,
                )
            except Exception as e:
                logger.error("[govee] Command failed: %s", e)

    def _turn_off(self):
        with self._lock:
            try:
                import requests
                requests.put(
                    "https://developer-api.govee.com/v1/devices/control",
                    headers={"Govee-API-Key": self._api_key, "Content-Type": "application/json"},
                    json={
                        "device": self._device,
                        "model": self._model,
                        "cmd": {"name": "turn", "value": "off"},
                    },
                    timeout=5,
                )
            except Exception as e:
                logger.error("[govee] Off command failed: %s", e)


# ─── Hue Backend ─────────────────────────────────────────────────────────────

class HueBackend(BaseLightBackend):
    """Controls Philips Hue lights via bridge.

    Setup:
      pip install phue
      Set HUE_BRIDGE_IP in .env
      Optionally set HUE_LIGHT_IDS (comma-separated)
      Press bridge button on first run
    """

    def __init__(self):
        super().__init__()
        self._bridge = None
        self._light_ids = []

        bridge_ip = os.environ.get("HUE_BRIDGE_IP", "").strip()
        if not bridge_ip:
            logger.warning("[hue] HUE_BRIDGE_IP not set. Hue integration disabled.")
            return

        try:
            from phue import Bridge
            self._bridge = Bridge(bridge_ip)
            self._bridge.connect()
            logger.info("[hue] Connected to Hue bridge at %s", bridge_ip)

            env_ids = os.environ.get("HUE_LIGHT_IDS", "").strip()
            if env_ids:
                self._light_ids = [int(x.strip()) for x in env_ids.split(",") if x.strip()]
            else:
                self._light_ids = [l.light_id for l in self._bridge.lights]

            if self._light_ids:
                logger.info("[hue] Controlling lights: %s", self._light_ids)
            else:
                logger.warning("[hue] No lights found.")
                self._bridge = None

        except ImportError:
            logger.error("[hue] phue not installed. Run: pip install phue")
        except Exception as e:
            logger.error("[hue] Failed to connect: %s", e)
            self._bridge = None

    @property
    def is_connected(self):
        return self._bridge is not None

    def _set_color(self, r, g, b, brightness=1.0, duration_ms=500):
        with self._lock:
            try:
                h, s, v = _rgb_to_hsv(r, g, b)
                cmd = {
                    "on": True,
                    "hue": int(h * 65535),
                    "sat": int(s * 254),
                    "bri": max(1, min(254, int(v * brightness * 254))),
                    "transitiontime": max(1, duration_ms // 100),
                }
                for lid in self._light_ids:
                    self._bridge.set_light(lid, cmd)
            except Exception as e:
                logger.error("[hue] Command failed: %s", e)

    def _turn_off(self):
        with self._lock:
            try:
                for lid in self._light_ids:
                    self._bridge.set_light(lid, {"on": False, "transitiontime": 20})
            except Exception as e:
                logger.error("[hue] Off command failed: %s", e)


# ─── Dummy Backend ───────────────────────────────────────────────────────────

class DummyLightBackend:
    """No-op stub when lights are disabled."""
    is_connected = False

    def set_emotion(self, emotion):
        pass

    def set_state(self, state_name):
        pass

    def flash(self, emotion=None, count=2):
        pass

    def off(self):
        pass


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _rgb_to_hsv(r, g, b):
    """Convert RGB (0-255) to HSV (h: 0-1, s: 0-1, v: 0-1)."""
    r, g, b = r / 255.0, g / 255.0, b / 255.0
    mx = max(r, g, b)
    mn = min(r, g, b)
    diff = mx - mn

    if diff == 0:
        h = 0
    elif mx == r:
        h = (60 * ((g - b) / diff) + 360) % 360
    elif mx == g:
        h = (60 * ((b - r) / diff) + 120) % 360
    else:
        h = (60 * ((r - g) / diff) + 240) % 360

    h = h / 360.0
    s = 0 if mx == 0 else diff / mx
    v = mx
    return h, s, v
