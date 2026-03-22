"""Dance to music using Live Audio Processing and Reachy Mini.

Listens for 5 seconds to calculate the exact tempo of the music.
Drives a strict 4-beat sequence:
  Beat 1: Translate head left (x=-0.09, y=0)
  Beat 2: Antennas snap to -1/+1 rad then back to 0; head stays left
  Beat 3: Translate head right (x=0.09, y=0)
  Beat 4: Antennas snap to -1/+1 rad then back to 0; head stays right

Usage:
  python dance_to_music.py                # dance with robot
  python dance_to_music.py --no-robot     # print beats only
"""

import argparse
import threading
import time
import numpy as np
import sounddevice as sd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MOTION_HZ = 30
HEAD_X_OFFSET = 0.09        # meters to translate head left/right
ANTENNA_SNAP = 1.0          # radians to snap antennas out
ANTENNA_SNAP_PHASE = 0.3    # fraction of beat to hold antennas out before snapping back

# Live Audio Settings
SAMPLE_RATE = 44100
BLOCK_SIZE = 1024
COOLDOWN_SEC = 0.25         # Max 240 BPM

# Detection Tuning
MIN_NOISE_FLOOR = 0.01      # ABSOLUTE SILENCE GATE
SENSITIVITY = 1.6           # Multiplier above ambient room noise
LISTEN_SECONDS = 5.0        # How long to listen before dancing
SILENCE_TIMEOUT = 4.0       # How many seconds of silence before stopping


# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------
class DanceState:
    def __init__(self):
        self.lock = threading.Lock()
        self.running = True
        
        # State Machine: "idle", "listening", "dancing"
        self.mode = "idle" 
        
        # Rhythm tracking
        self.listen_start_time = 0.0
        self.beat_timestamps = []
        self.beat_period = 0.5      # Calculated interval between beats (BPM)
        
        # Phase locking
        self.dance_start_time = 0.0
        self.last_beat_time = 0.0
        self.beat_count = 0


# ---------------------------------------------------------------------------
# Motion thread
# ---------------------------------------------------------------------------
def motion_loop(mini, state):
    """Background thread: strict 4-beat sequence based on calculated BPM."""
    if mini is not None:
        from reachy_mini.utils import create_head_pose

    # Current smoothed positions
    curr_head_y = 0.0
    curr_left_antenna = 0.0
    curr_right_antenna = 0.0

    while state.running:
        now = time.time()

        # Read shared state
        with state.lock:
            mode = state.mode
            period = max(state.beat_period, 0.25)
            dance_start = state.dance_start_time

        target_head_y = 0.0
        target_left_antenna = 0.0
        target_right_antenna = 0.0

        if mode == "dancing":
            elapsed = now - dance_start
            beat_index = int(elapsed / period)
            phase = (elapsed % period) / period  # 0.0 to 1.0 over one beat

            # 4-Beat State Machine
            cycle = beat_index % 4

            if cycle == 0:
                # Beat 1: Head moves left
                target_head_y = -HEAD_X_OFFSET
                target_left_antenna = 0.0
                target_right_antenna = 0.0

            elif cycle == 1:
                # Beat 2: Head stays left; antennas snap out then immediately back
                target_head_y = -HEAD_X_OFFSET
                if phase < ANTENNA_SNAP_PHASE:
                    target_left_antenna = -ANTENNA_SNAP
                    target_right_antenna = ANTENNA_SNAP
                else:
                    target_left_antenna = 0.0
                    target_right_antenna = 0.0

            elif cycle == 2:
                # Beat 3: Head moves right
                target_head_y = HEAD_X_OFFSET
                target_left_antenna = 0.0
                target_right_antenna = 0.0

            elif cycle == 3:
                # Beat 4: Head stays right; antennas snap out then immediately back
                target_head_y = HEAD_X_OFFSET
                if phase < ANTENNA_SNAP_PHASE:
                    target_left_antenna = -ANTENNA_SNAP
                    target_right_antenna = ANTENNA_SNAP
                else:
                    target_left_antenna = 0.0
                    target_right_antenna = 0.0

        # Smooth head x transition; antennas snap fast
        curr_head_y += (target_head_y - curr_head_y) * 0.4
        curr_left_antenna += (target_left_antenna - curr_left_antenna) * 0.8
        curr_right_antenna += (target_right_antenna - curr_right_antenna) * 0.8

        # Send to robot
        if mini is not None:
            try:
                mini.set_target(
                    head=create_head_pose(pitch=0.0, roll=0.0, yaw=0.0, x=0.0, y=curr_head_y)
                )
                mini.set_target(antennas=[curr_left_antenna, curr_right_antenna])
            except Exception:
                pass

        time.sleep(1.0 / MOTION_HZ)


# ---------------------------------------------------------------------------
# Live Audio Polling System
# ---------------------------------------------------------------------------
class LiveBeatDetector:
    def __init__(self, state):
        self.state = state
        self.history_len = 43  # ~1 sec at 44100/1024
        self.energy_history = np.zeros(self.history_len)
        self.history_idx = 0
        self.last_audio_beat = 0

    def audio_callback(self, indata, frames, time_info, status):
        """Called automatically by sounddevice for every audio block."""
        if not self.state.running:
            raise sd.CallbackStop()
        
        energy = np.sqrt(np.mean(indata**2))
        now = time.time()
        
        # Always update background noise floor
        self.energy_history[self.history_idx] = energy
        self.history_idx = (self.history_idx + 1) % self.history_len

        # Stop condition: Silence timeout
        with self.state.lock:
            if self.state.mode == "dancing" and (now - self.state.last_beat_time) > SILENCE_TIMEOUT:
                print("\n[dance] Music stopped. Going idle.")
                self.state.mode = "idle"
                self.state.beat_timestamps = []

        # The Noise Gate
        if energy < MIN_NOISE_FLOOR:
            return

        avg_energy = np.mean(self.energy_history)
        threshold = avg_energy * SENSITIVITY

        # Beat Trigger
        if energy > threshold and (now - self.last_audio_beat) > COOLDOWN_SEC:
            self.last_audio_beat = now
            
            with self.state.lock:
                if self.state.mode == "idle":
                    self.state.mode = "listening"
                    self.state.listen_start_time = now
                    self.state.beat_timestamps = [now]
                    print("\n[dance] Heard the music start! Listening for tempo for 5 seconds...")
                
                elif self.state.mode == "listening":
                    self.state.beat_timestamps.append(now)
                    
                    if (now - self.state.listen_start_time) >= LISTEN_SECONDS:
                        if len(self.state.beat_timestamps) >= 4:
                            intervals = np.diff(self.state.beat_timestamps)
                            valid_intervals = [i for i in intervals if 0.25 <= i <= 2.0]
                            
                            if valid_intervals:
                                self.state.beat_period = float(np.median(valid_intervals))
                                self.state.mode = "dancing"
                                self.state.dance_start_time = now
                                bpm = 60.0 / self.state.beat_period
                                print(f"\n[dance] LOCKED IN! Tempo: ~{bpm:.1f} BPM. Beat length: {self.state.beat_period:.2f}s")
                            else:
                                print("[dance] Tempo erratic, restarting listen phase...")
                                self.state.listen_start_time = now
                                self.state.beat_timestamps = [now]
                        else:
                            print("[dance] Not enough beats heard, restarting listen phase...")
                            self.state.listen_start_time = now
                            self.state.beat_timestamps = [now]
                            
                elif self.state.mode == "dancing":
                    # Keep the animation phase locked to the real microphone hits
                    expected_beats = round((now - self.state.dance_start_time) / self.state.beat_period)
                    self.state.dance_start_time = now - (expected_beats * self.state.beat_period)

                self.state.last_beat_time = now
                self.state.beat_count += 1
            
            if self.state.mode == "dancing":
                print(f"[dance] BEAT!")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Strict 4-beat dance sequence")
    parser.add_argument(
        "--no-robot", action="store_true",
        help="Print beat events without connecting to Reachy Mini",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    state = DanceState()

    mini = None
    if not args.no_robot:
        from reachy_mini import ReachyMini
        mini = ReachyMini()
        mini.__enter__()
        print("[dance] Connected to Reachy Mini")

    motion_thread = threading.Thread(target=motion_loop, args=(mini, state), daemon=True)
    motion_thread.start()

    print("[dance] Opening microphone stream...")
    detector = LiveBeatDetector(state)

    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE, 
            channels=1, 
            blocksize=BLOCK_SIZE, 
            callback=detector.audio_callback
        ):
            print("[dance] Room is quiet. Waiting for music... (Ctrl+C to stop)")
            while True:
                time.sleep(0.1)
                
    except KeyboardInterrupt:
        print("\n[dance] Stopping...")
    finally:
        state.running = False
        motion_thread.join(timeout=2.0)
        
        if mini is not None:
            from reachy_mini.utils import create_head_pose
            print("[dance] Returning to neutral...")
            mini.goto_target(head=create_head_pose(pitch=0, roll=0, yaw=0, x=0.0, y=0.0), duration=1.0)
            mini.goto_target(antennas=[0.0, 0.0], duration=1.0)
            time.sleep(1.1)
            mini.__exit__(None, None, None)
            print("[dance] Disconnected.")

if __name__ == "__main__":
    main()