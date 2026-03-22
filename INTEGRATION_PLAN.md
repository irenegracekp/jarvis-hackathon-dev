# Unified Main Orchestrator — All Subsystems Working Together

## Context
Currently main.py runs 4 threads (listen, brain, output, gesture) but person tracking and dance are standalone scripts. Camera access, head control, and audio input all conflict between subsystems. The goal is to unify everything into one orchestrator so the robot can: track a person by hand raise, hold conversation, control desktop (YouTube, flights, Amazon, system commands), react with Hue lights, and dance to music — all seamlessly.

## New Thread Architecture (6 threads)

| # | Thread | Hz | Purpose |
|---|--------|----|---------|
| 1 | `camera_loop` | ~30 | **NEW** — single camera reader, publishes frames to shared FrameProvider |
| 2 | `listen_loop` | event | Mic → ASR → audio_queue (mostly unchanged) |
| 3 | `brain_loop` | ~0.3-10 | Face recognition, LLM, state transitions (modified: uses FrameProvider) |
| 4 | `output_loop` | event | TTS + sets robot targets for motion_loop (modified: no longer drives robot directly) |
| 5 | `gesture_loop` | ~30 | Hand gestures → desktop actions (modified: uses FrameProvider) |
| 6 | `tracker_loop` | ~15-30 | **NEW** — YOLOv8-pose person re-ID, writes tracking state |
| 7 | `motion_loop` | 30 | **NEW** — sole robot controller, arbitrates between brain/tracker/skills |

## Extended State Machine

```
Conversation layer:  IDLE ↔ AMBIENT ↔ ENGAGED        (face-driven, unchanged)
Tracking layer:      SCANNING → TRACKING → LOST       (always-on, hand-raise enrolls)
Dance layer:         OFF → LISTENING → DANCING → OFF   (voice-triggered, overrides motion)
```

All three layers run simultaneously. `motion_loop` decides who drives the robot based on priority: **dance > brain emotion > tracker > neutral**.

## New Components

### 1. FrameProvider (new: `pipeline/camera.py`)
Single camera capture thread. Consumers call `get_frame()` for latest frame (non-blocking, returns copy).

```python
class FrameProvider:
    def __init__(self, camera_index=0)
    def capture_loop(self)        # run as thread — grabs frames, handles ZED stereo crop
    def get_frame(self) -> (frame, frame_id)  # thread-safe read of latest frame
```

Replaces: `VisionPipeline.open_camera()`, gesture_loop's own `VideoCapture`, person_follow's own `VideoCapture`.

### 2. motion_loop (in main.py)
The **only** thread that calls `mini.set_target()`. Reads shared state to decide what to do:

```
Priority:
1. Dance mode active → compute dance sway/bop targets (from dance_to_music.py math)
2. Pending brain emotion → play_emotion() + brief head move (nod/tilt), then release
3. Person tracking active → compute follow targets (from person_follow.py math)
4. Default → drift head/body to neutral
```

Port math from:
- `scripts/person_follow.py` control_loop (lines 159-228) → `_compute_follow_targets(state)`
- `scripts/dance_to_music.py` motion_loop (lines 66-151) → `_compute_dance_targets(state)`

Uses `ReachyBridge` for all robot I/O. Disable ReachyBridge's internal wobble_loop (motion_loop handles wobble during TTS via `feed_audio_chunk` integration).

### 3. tracker_loop (in main.py)
Runs `PersonTracker.process_frame()` on each new frame from FrameProvider. Writes results to shared state:
- `state.tracking_target_center` — (px_x, px_y) or None
- `state.tracking_state` — scanning/tracking/lost
- `state.frame_size` — (w, h)

Always-on regardless of conversation state. When hand raise detected in AMBIENT mode, sets `state.hand_raise_event` to signal brain_loop for greeting.

### 4. BeatDetector (new: `pipeline/beat_detect.py`)
Simplified from `dance_to_music.py` `LiveBeatDetector`. Triggered by voice command ("dance", "let's dance"). Opens a separate `sounddevice.InputStream` for beat detection (can use same or different mic device). Writes to `state.dance_mode`, `state.dance_beat_period`, `state.dance_start_time`. Stops on silence timeout (4s) or voice command "stop dancing".

## Modifications to Existing Code

### State class (main.py)
Add fields:
```python
# Tracking (always-on)
self.tracking_target_center = None
self.tracking_state = "scanning"
self.frame_size = (640, 480)
self.hand_raise_event = threading.Event()

# Dance
self.dance_mode = "off"           # off, listening, dancing
self.dance_beat_period = 0.5
self.dance_start_time = 0.0

# Robot arbitration
self.pending_emotion = None       # emotion ID for motion_loop to play
self.pending_head_direction = None
self.pending_antenna = None
```

### brain_loop
- Use `frame_provider.get_frame()` instead of `vision.grab_frame()`
- Pass `FrameProvider` to `VisionPipeline` (or just pass frames directly)
- Add dance trigger in `_handle_speech`:
  ```python
  if re.match(r'(dance|let.s dance|bust a move|play some music)', text, re.I):
      state.dance_mode = "listening"
      # start BeatDetector
      response = {"speech": "Oh yeah! Let me hear the beat!", "emotion": "excited"}
      state.response_queue.put(response)
      return
  if re.match(r'(stop danc|enough danc)', text, re.I):
      state.dance_mode = "off"
      return
  ```
- Hand raise event: if `state.hand_raise_event.is_set()` in ambient mode, treat as arrival trigger

### output_loop
- **No longer calls** `robot.execute_response()` directly
- Sets `state.pending_emotion`, `state.pending_head_direction`, `state.pending_antenna` for motion_loop
- Still handles TTS (blocking) and Hue lights (direct)
- Pauses dance during TTS, resumes after

### gesture_loop
- Use `frame_provider.get_frame()` instead of own `VideoCapture`
- Everything else unchanged

### VisionPipeline (pipeline/vision.py)
- Add `set_frame_provider(fp)` or accept frames directly in `get_face_events(frame)` (already takes frame param)
- Remove `open_camera()` / `release()` — camera lifecycle moves to FrameProvider

### ReachyBridge (pipeline/reachy_bridge.py)
- Add flag to disable internal `_wobble_loop` when motion_loop is active
- motion_loop handles TTS wobble itself using `state.tts_speaking` + audio level

## OpenClaw Capabilities
Already handles via subprocess: open/search/play/navigate/click/type/volume/mute/screenshot/scroll/download/copy/paste. The LLM naturally routes commands like:
- "play a video on YouTube" → OpenClaw
- "book a flight" → OpenClaw (opens browser, navigates)
- "add to Amazon cart" → OpenClaw
- "reduce brightness" / "increase volume" → OpenClaw (keywords already classified)

No code changes needed — OpenClaw bridge already classifies and routes these. The LLM (brain) handles conversational context.

## New CLI Flags
```
--no-tracker       Skip person tracking (saves GPU)
--no-dance         Skip dance/beat detection
--dance-mic N      Audio device for beat detection (default: same as ASR mic)
```

## Files to Modify/Create

| File | Action | What |
|------|--------|------|
| `pipeline/camera.py` | **CREATE** | FrameProvider class |
| `pipeline/beat_detect.py` | **CREATE** | BeatDetector class (port from dance_to_music.py) |
| `main.py` | **MODIFY** | Add State fields, tracker_loop, motion_loop, camera_loop; modify brain_loop, output_loop, gesture_loop; update main() thread startup |
| `pipeline/vision.py` | **MODIFY** | Remove camera management, accept frames from FrameProvider |
| `pipeline/reachy_bridge.py` | **MODIFY** | Add flag to disable wobble_loop when motion_loop controls |

## Implementation Order

1. **FrameProvider** — create `pipeline/camera.py`, wire into main.py as camera_loop
2. **Modify brain_loop + gesture_loop** — use FrameProvider instead of own cameras
3. **Add tracker_loop** — wire PersonTracker into main.py, add State tracking fields
4. **Add motion_loop** — robot arbiter with follow + brain target logic, replace output_loop robot calls
5. **Add dance mode** — BeatDetector, voice trigger, dance targets in motion_loop
6. **Polish** — Hue presets for tracking/dance, CLI flags, test end-to-end

## Verification
1. `python main.py --no-robot --no-tts --no-listen` — verify camera sharing works (brain + gesture + tracker all get frames)
2. `python main.py --no-tts --no-listen` — verify person tracking drives head, brain emotions play briefly then return to tracking
3. Say "dance" → verify beat detection starts, robot dances, conversation pauses
4. Say "open YouTube" → verify OpenClaw routes correctly
5. Full run: person raises hand → tracked → conversation → "dance" → dancing → "stop" → resume tracking
