"""Reachy Mini SDK interface -- stubs for pre-hackathon, real at event"""

import time


class RobotController:
    """
    Stub implementation that logs actions to console.
    At hackathon: replace with real Reachy Mini SDK calls.
    """

    def __init__(self, real_robot=False):
        self.real_robot = real_robot
        self._current_emotion = "calm"
        self._current_head = "scanning"
        self._current_antenna = "neutral"

        if real_robot:
            self._init_reachy()
        else:
            print("[robot] Running in stub mode (no real robot).")

    def _init_reachy(self):
        """Initialize real Reachy Mini SDK connection."""
        try:
            # At hackathon, uncomment and configure:
            # from reachy2_sdk import ReachySDK
            # self.reachy = ReachySDK(host="localhost")
            print("[robot] Real robot init -- TODO at hackathon")
        except Exception as e:
            print(f"[robot] Failed to init real robot: {e}")
            self.real_robot = False

    def set_head_pose(self, direction):
        """
        Set head direction.
        direction: toward_speaker | scanning | tilt_left | tilt_right | nod
        """
        if direction == self._current_head:
            return
        self._current_head = direction

        if self.real_robot:
            self._real_head_pose(direction)
        else:
            print(f"[robot] Head -> {direction}")

    def set_antenna_state(self, state):
        """
        Set antenna state.
        state: perked | wiggle | drooped | neutral
        """
        if state == self._current_antenna:
            return
        self._current_antenna = state

        if self.real_robot:
            self._real_antenna_state(state)
        else:
            print(f"[robot] Antenna -> {state}")

    def set_emotion(self, emotion):
        """
        Set emotion (maps to LED eyes on Reachy Mini).
        emotion: excited | curious | calm | surprised | amused | skeptical
        """
        if emotion == self._current_emotion:
            return
        self._current_emotion = emotion

        if self.real_robot:
            self._real_emotion(emotion)
        else:
            print(f"[robot] Emotion -> {emotion}")

    def play_audio(self, audio_data):
        """Play audio through robot speaker (for when we have the real robot)."""
        if self.real_robot:
            # TODO: route audio to Reachy Mini speaker
            print("[robot] Playing audio on robot speaker")
        else:
            print("[robot] Would play audio on robot speaker")

    def execute_response(self, response_dict):
        """Apply a full brain response (convenience method)."""
        if "emotion" in response_dict:
            self.set_emotion(response_dict["emotion"])
        if "head_direction" in response_dict:
            self.set_head_pose(response_dict["head_direction"])
        if "antenna_state" in response_dict:
            self.set_antenna_state(response_dict["antenna_state"])

    # --- Real robot methods (implement at hackathon) ---

    def _real_head_pose(self, direction):
        """Map direction string to Reachy Mini head angles."""
        # Reachy Mini head has roll, pitch, yaw
        pose_map = {
            "toward_speaker": (0, 0, 0),       # straight ahead
            "scanning": (0, 0, 0),              # would do slow sweep
            "tilt_left": (0, 0, -15),           # tilt left
            "tilt_right": (0, 0, 15),           # tilt right
            "nod": (0, 10, 0),                  # slight nod down
        }
        roll, pitch, yaw = pose_map.get(direction, (0, 0, 0))
        # self.reachy.head.rotate_to(roll, pitch, yaw, duration=0.5)
        print(f"[robot] Real head -> roll={roll} pitch={pitch} yaw={yaw}")

    def _real_antenna_state(self, state):
        """Map antenna state to Reachy Mini antenna servos."""
        # self.reachy.head.l_antenna.goal_position = ...
        # self.reachy.head.r_antenna.goal_position = ...
        print(f"[robot] Real antenna -> {state}")

    def _real_emotion(self, emotion):
        """Map emotion to Reachy Mini LED eyes."""
        # Color mapping
        color_map = {
            "excited": (255, 200, 0),    # warm yellow
            "curious": (0, 150, 255),    # blue
            "calm": (100, 200, 100),     # soft green
            "surprised": (255, 100, 0),  # orange
            "amused": (255, 150, 200),   # pink
            "skeptical": (200, 100, 255),# purple
        }
        color = color_map.get(emotion, (200, 200, 200))
        # self.reachy.head.led.set_color(*color)
        print(f"[robot] Real LED -> {color}")


if __name__ == "__main__":
    rc = RobotController(real_robot=False)
    rc.set_emotion("excited")
    rc.set_head_pose("tilt_left")
    rc.set_antenna_state("wiggle")
    rc.execute_response({
        "emotion": "curious",
        "head_direction": "toward_speaker",
        "antenna_state": "perked",
    })
