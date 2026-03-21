"""Map detected gestures to keyboard/mouse actions via xdotool.

Gesture mapping:
  - "pan" + motion right  -> Alt+Right (browser forward)
  - "pan" + motion left   -> Alt+Left (browser back)
  - "pan" + motion up     -> scroll up
  - "pan" + motion down   -> scroll down
  - "stop"                -> pause / no action
  - "fist"                -> click at position
  - "fine"                -> Enter / confirm
  - "peace" + motion      -> drag / select
"""

import subprocess
import time


# Minimum interval between the same action (debounce)
DEBOUNCE_SEC = 0.5

# Scroll amount per gesture
SCROLL_CLICKS = 5


class ActionMapper:
    def __init__(self, debounce_sec=DEBOUNCE_SEC):
        self.debounce_sec = debounce_sec
        self._last_action_time = {}  # action_name -> timestamp
        self._xdotool_available = None

    def _check_xdotool(self):
        if self._xdotool_available is None:
            try:
                subprocess.run(["xdotool", "version"], capture_output=True, check=True)
                self._xdotool_available = True
            except (FileNotFoundError, subprocess.CalledProcessError):
                print("[actions] xdotool not found. Install with: apt-get install xdotool")
                self._xdotool_available = False
        return self._xdotool_available

    def _debounced(self, action_name):
        """Returns True if enough time has passed since the last time this action fired."""
        now = time.time()
        last = self._last_action_time.get(action_name, 0)
        if now - last < self.debounce_sec:
            return False
        self._last_action_time[action_name] = now
        return True

    def _xdo(self, *args):
        """Run an xdotool command."""
        if not self._check_xdotool():
            return
        try:
            subprocess.run(["xdotool"] + list(args), capture_output=True, timeout=2)
        except Exception as e:
            print(f"[actions] xdotool error: {e}")

    def execute_gesture(self, gesture, hand_position=None, motion=None):
        """Execute desktop action for detected gesture + motion.

        Args:
            gesture: str - one of fist, pan, stop, fine, peace, no_hand, none
            hand_position: tuple (x, y) or None
            motion: dict with 'direction', 'speed', 'dx', 'dy' or None

        Returns:
            str or None - description of action taken, or None if no action
        """
        if gesture == "no_hand" or gesture == "none":
            return None

        if gesture == "stop":
            # Intentional stop — do nothing
            return None

        if gesture == "pan" and motion:
            direction = motion["direction"]
            if direction == "right" and self._debounced("pan_right"):
                self._xdo("key", "alt+Right")
                return "browser_forward"
            elif direction == "left" and self._debounced("pan_left"):
                self._xdo("key", "alt+Left")
                return "browser_back"
            elif direction == "up" and self._debounced("scroll_up"):
                self._xdo("click", "4", "--repeat", str(SCROLL_CLICKS))
                return "scroll_up"
            elif direction == "down" and self._debounced("scroll_down"):
                self._xdo("click", "5", "--repeat", str(SCROLL_CLICKS))
                return "scroll_down"

        if gesture == "fist" and self._debounced("click"):
            if hand_position:
                # Move mouse to hand position (scaled to screen)
                # For now, just click at current mouse position
                self._xdo("click", "1")
            else:
                self._xdo("click", "1")
            return "click"

        if gesture == "fine" and self._debounced("enter"):
            self._xdo("key", "Return")
            return "enter"

        if gesture == "peace" and motion and self._debounced("peace_scroll"):
            direction = motion["direction"]
            if direction == "up":
                self._xdo("click", "4", "--repeat", str(SCROLL_CLICKS * 2))
                return "fast_scroll_up"
            elif direction == "down":
                self._xdo("click", "5", "--repeat", str(SCROLL_CLICKS * 2))
                return "fast_scroll_down"

        return None


if __name__ == "__main__":
    am = ActionMapper()
    print("[actions] ActionMapper ready.")
    print(f"[actions] xdotool available: {am._check_xdotool()}")
    print("[actions] Run scripts/test_actions.py for live gesture->action test.")
