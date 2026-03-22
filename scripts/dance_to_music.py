"""Dance to music using Live Audio Processing and Reachy Mini.

Audio-reactive motion:
  Volume (RMS) → head oscillates on y-axis; amplitude = RMS × HEAD_AMP_SCALE
  Treble energy → antennas spread open

Usage:
  python dance_to_music.py                # react to microphone input
  python dance_to_music.py --loopback     # react to system audio (browser, etc.)
  python dance_to_music.py --no-robot     # print energy levels only
"""

import argparse
import math
import threading
import time
import numpy as np
import sounddevice as sd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MOTION_HZ   = 30
SAMPLE_RATE = 44100
BLOCK_SIZE  = 1024

TREBLE_LOW_HZ  = 2000
TREBLE_HIGH_HZ = 8000

HEAD_Y_MAX     = 0.09   # meters, hard cap on head y translation
HEAD_OSC_HZ    = 0.7    # Hz — head oscillation frequency
HEAD_AMP_SCALE = 6.0    # amp (meters) = smoothed_RMS × this; tune to taste
HEAD_AMP_ATTACK = 1.0   # amplitude rise rate (1.0 = immediate)
HEAD_AMP_DECAY  = 0.02  # amplitude fall rate (lower = slower fade out)

ANTENNA_MAX = 1.0       # radians, max antenna spread

MIN_NOISE_FLOOR   = 0.008   # RMS gate — blocks below this are treated as silence
ENERGY_SMOOTH     = 0.20    # EMA coefficient per audio block (lower = smoother)
TREBLE_PEAK_DECAY = 0.99    # per-block decay of treble adaptive peak
TREBLE_PEAK_FLOOR = 1e-4    # treble peak never decays below this


# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------
class AudioState:
    def __init__(self):
        self.lock    = threading.Lock()
        self.running = True
        self.rms     = 0.0   # smoothed RMS — drives head amplitude
        self.treble  = 0.0   # normalized treble [0, 1] — drives antennas


# ---------------------------------------------------------------------------
# Motion thread
# ---------------------------------------------------------------------------
def motion_loop(mini, state):
    if mini is not None:
        from reachy_mini.utils import create_head_pose

    curr_head_y        = 0.0
    curr_left_antenna  = 0.0
    curr_right_antenna = 0.0
    curr_amp           = 0.0

    while state.running:
        now = time.time()
        with state.lock:
            rms    = state.rms
            treble = state.treble

        target_amp = min(rms * HEAD_AMP_SCALE, HEAD_Y_MAX)
        rate = HEAD_AMP_ATTACK if target_amp > curr_amp else HEAD_AMP_DECAY
        curr_amp  += rate * (target_amp - curr_amp)
        target_head_y        = curr_amp * math.sin(2 * math.pi * HEAD_OSC_HZ * now)
        target_left_antenna  = -ANTENNA_MAX * treble
        target_right_antenna =  ANTENNA_MAX * treble

        curr_head_y        += (target_head_y        - curr_head_y)        * 0.3
        curr_left_antenna  += (target_left_antenna  - curr_left_antenna)  * 0.3
        curr_right_antenna += (target_right_antenna - curr_right_antenna) * 0.3

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
# Audio processing
# ---------------------------------------------------------------------------
class FrequencyReactor:
    def __init__(self, state):
        self.state         = state
        self.rms_smooth    = 0.0
        self.treble_smooth = 0.0
        self.treble_peak   = TREBLE_PEAK_FLOOR
        self.warmup_blocks = 43   # discard first ~1 s while stream settles
        freqs = np.fft.rfftfreq(BLOCK_SIZE, 1.0 / SAMPLE_RATE)
        self.treble_mask = (freqs >= TREBLE_LOW_HZ) & (freqs < TREBLE_HIGH_HZ)

    def audio_callback(self, indata, frames, time_info, status):
        if not self.state.running:
            raise sd.CallbackStop()
        if self.warmup_blocks > 0:
            self.warmup_blocks -= 1
            return

        samples = indata[:, 0].astype(np.float32)
        rms     = float(np.sqrt(np.mean(samples ** 2)))

        if rms < MIN_NOISE_FLOOR:
            self.rms_smooth    = 0.0
            self.treble_smooth = 0.0
            with self.state.lock:
                self.state.rms    = 0.0
                self.state.treble = 0.0
            return

        spectrum   = np.abs(np.fft.rfft(samples))
        raw_treble = float(np.mean(spectrum[self.treble_mask]))

        # Adaptive peak for treble only; floor prevents creep during near-silence
        self.treble_peak = max(self.treble_peak * TREBLE_PEAK_DECAY, raw_treble, TREBLE_PEAK_FLOOR)

        self.rms_smooth    += ENERGY_SMOOTH * (rms                           - self.rms_smooth)
        self.treble_smooth += ENERGY_SMOOTH * (raw_treble / self.treble_peak - self.treble_smooth)

        with self.state.lock:
            self.state.rms    = self.rms_smooth
            self.state.treble = float(np.sqrt(max(self.treble_smooth, 0.0)))


# ---------------------------------------------------------------------------
# Audio device selection
# ---------------------------------------------------------------------------
def find_audio_device(loopback=False):
    if loopback:
        for i, dev in enumerate(sd.query_devices()):
            if "monitor" in dev["name"].lower() and dev["max_input_channels"] > 0:
                return i, dev
        print("[audio] Warning: no monitor device found, falling back to default input.")
    idx = sd.default.device[0]
    return idx, sd.query_devices(idx)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Audio-reactive dance for Reachy Mini")
    parser.add_argument("--no-robot", action="store_true",
                        help="Print energy levels without connecting to Reachy Mini")
    parser.add_argument("--loopback", action="store_true",
                        help="Capture system audio output instead of microphone")
    args = parser.parse_args()

    state = AudioState()

    mini = None
    if not args.no_robot:
        from reachy_mini import ReachyMini
        mini = ReachyMini()
        mini.__enter__()
        print("[dance] Connected to Reachy Mini")

    motion_thread = threading.Thread(target=motion_loop, args=(mini, state), daemon=True)
    motion_thread.start()

    device_idx, device_info = find_audio_device(args.loopback)
    source_label = "system audio (loopback)" if args.loopback else "microphone"
    print(f"[audio] Input: {source_label}")
    print(f"        Device [{device_idx}]: {device_info['name']} "
          f"({int(device_info['default_samplerate'])} Hz default)")

    reactor = FrequencyReactor(state)

    try:
        with sd.InputStream(
            device=device_idx,
            samplerate=SAMPLE_RATE,
            channels=1,
            blocksize=BLOCK_SIZE,
            callback=reactor.audio_callback,
        ):
            print("[dance] Listening... (Ctrl+C to stop)")
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
