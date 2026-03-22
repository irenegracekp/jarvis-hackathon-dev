#!/usr/bin/env python3
"""
gesture_ws_server.py — Hand gesture WebSocket server for arch-viewer integration.

Detects gestures via MediaPipe Hands and broadcasts JSON events over WebSocket
to the arch-viewer-full browser client.

Gestures:
  SWIPE_LEFT  → browser: go to previous camera view
  SWIPE_RIGHT → browser: go to next camera view
  PINCH       → browser: zoom (delta > 0 = fingers spreading = zoom in)

Usage:
  source venv/bin/activate
  python gesture_ws_server.py

Then open arch-viewer-full/index.html — it auto-connects to ws://<host>:8765.
"""

import asyncio
import json
import math
import threading
import time
from collections import deque

import os
import cv2
import mediapipe as mp

# ── Config ────────────────────────────────────────────────────────────────────
CAMERA_DEVICE = 10
WS_HOST       = "0.0.0.0"   # bind all interfaces so browser on another device can connect
WS_PORT       = 8765
HEADLESS      = not os.environ.get("DISPLAY")  # skip cv2.imshow if no X11

# ── MediaPipe landmark indices ────────────────────────────────────────────────
WRIST = 0
THUMB_TIP  = 4
INDEX_MCP  = 5;  INDEX_PIP  = 6;  INDEX_TIP  = 8
MIDDLE_MCP = 9;  MIDDLE_PIP = 10; MIDDLE_TIP = 12
RING_MCP   = 13; RING_PIP   = 14; RING_TIP   = 16
PINKY_MCP  = 17; PINKY_PIP  = 18; PINKY_TIP  = 20
THUMB_IP   = 3

mp_hands  = mp.solutions.hands
mp_draw   = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

# ── Gesture helpers ───────────────────────────────────────────────────────────

def finger_is_extended(landmarks, tip, pip):
    return landmarks[tip].y < landmarks[pip].y


def thumb_is_extended(landmarks):
    tip   = landmarks[THUMB_TIP]
    ip    = landmarks[THUMB_IP]
    wrist = landmarks[WRIST]
    return abs(tip.x - wrist.x) > abs(ip.x - wrist.x)


def count_fingers(landmarks):
    return [
        thumb_is_extended(landmarks),
        finger_is_extended(landmarks, INDEX_TIP,  INDEX_PIP),
        finger_is_extended(landmarks, MIDDLE_TIP, MIDDLE_PIP),
        finger_is_extended(landmarks, RING_TIP,   RING_PIP),
        finger_is_extended(landmarks, PINKY_TIP,  PINKY_PIP),
    ]


def pinch_ratio(landmarks):
    """Thumb-tip to index-tip distance normalised by wrist→middle-MCP hand scale."""
    thumb = landmarks[THUMB_TIP]
    index = landmarks[INDEX_TIP]
    wrist = landmarks[WRIST]
    mid   = landmarks[MIDDLE_MCP]
    dist  = math.hypot(thumb.x - index.x, thumb.y - index.y)
    scale = math.hypot(wrist.x - mid.x,   wrist.y - mid.y) + 1e-6
    return dist / scale


# ── Swipe detector ────────────────────────────────────────────────────────────

class SwipeDetector:
    """Fires SWIPE_LEFT / SWIPE_RIGHT from wrist trajectory while open palm is held."""

    def __init__(self, window=20, threshold=0.06, cooldown_s=1.0, grace_frames=5):
        self.positions    = deque(maxlen=window)
        self.threshold    = threshold
        self.cooldown_s   = cooldown_s
        self.grace_frames = grace_frames  # tolerate this many non-palm frames before resetting
        self._last_fire   = 0.0
        self._miss_count  = 0

    def update(self, x, y):
        self._miss_count = 0  # palm detected — reset miss counter
        self.positions.append((x, y))
        if time.time() - self._last_fire < self.cooldown_s:
            return None
        if len(self.positions) < 10:
            return None
        dx = self.positions[-1][0] - self.positions[0][0]
        dy = self.positions[-1][1] - self.positions[0][1]
        # Debug: log swipe tracking every ~0.5s (every 15 frames at ~30fps)
        if len(self.positions) % 15 == 0:
            print(f"[SWIPE-DBG] pts={len(self.positions)} dx={dx:+.3f} dy={dy:+.3f} thresh={self.threshold}", flush=True)
        if abs(dx) > self.threshold and abs(dx) > abs(dy):
            self.positions.clear()
            self._last_fire = time.time()
            # dx > 0 = hand moved right in camera frame = user swiped right
            result = "SWIPE_RIGHT" if dx > 0 else "SWIPE_LEFT"
            print(f"[SWIPE-DBG] FIRED {result} dx={dx:+.3f}", flush=True)
            return result
        return None

    def miss(self):
        """Called when palm is NOT detected. Only clears after grace_frames consecutive misses."""
        self._miss_count += 1
        if self._miss_count >= self.grace_frames:
            self.positions.clear()

    def clear(self):
        self.positions.clear()
        self._miss_count = 0


# ── WebSocket broadcast helper ────────────────────────────────────────────────

async def _broadcast(msg: str, clients: set):
    dead = set()
    for ws in list(clients):
        try:
            await ws.send(msg)
        except Exception:
            dead.add(ws)
    clients.difference_update(dead)


def broadcast(loop, clients, msg: str):
    """Thread-safe: schedule a broadcast coroutine on the asyncio loop."""
    if clients:
        asyncio.run_coroutine_threadsafe(_broadcast(msg, clients), loop)


# ── Gesture detection thread ──────────────────────────────────────────────────

def gesture_thread(loop, connected_clients, stop_event):
    cap = cv2.VideoCapture(CAMERA_DEVICE)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera /dev/video{CAMERA_DEVICE}", flush=True)
        return

    hands_model = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    )

    swipe               = SwipeDetector()
    last_pinch_ratio    = None
    pinch_send_interval = 0.08   # ~12 pinch updates/second max
    last_pinch_send     = 0.0

    print(f"[GESTURE] Camera /dev/video{CAMERA_DEVICE} opened.", flush=True)
    print(f"[GESTURE] Serving on ws://{WS_HOST}:{WS_PORT}", flush=True)
    print(f"[GESTURE] Headless={HEADLESS} (set DISPLAY to enable preview window)", flush=True)

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            break

        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands_model.process(rgb)

        label = ""

        if results.multi_hand_landmarks:
            lm = results.multi_hand_landmarks[0].landmark

            if not HEADLESS:
                mp_draw.draw_landmarks(
                    frame, results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS,
                    mp_styles.get_default_hand_landmarks_style(),
                    mp_styles.get_default_hand_connections_style(),
                )

            # Compute finger state once, then decide which gesture branch to take.
            # Open palm and pinch are mutually exclusive — use elif.
            fingers     = count_fingers(lm)
            num_extended = sum(fingers)

            # Debug: periodic hand state log (~1/sec)
            if int(time.time() * 2) % 2 == 0 and not hasattr(gesture_thread, '_last_dbg') or getattr(gesture_thread, '_last_dbg', 0) < time.time() - 1.0:
                gesture_thread._last_dbg = time.time()
                f_names = ['T','I','M','R','P']
                state = ''.join(f_names[i] if fingers[i] else '.' for i in range(5))
                print(f"[HAND-DBG] fingers=[{state}] ext={num_extended} → {'PALM' if num_extended>=4 else 'PINCH' if not fingers[2] and not fingers[3] and not fingers[4] else 'OTHER'}", flush=True)

            if num_extended >= 4:
                # ── OPEN PALM: track wrist for swipe ──────────────────────
                label = "OPEN PALM"
                last_pinch_ratio = None          # reset pinch baseline
                wrist = lm[WRIST]
                event = swipe.update(wrist.x, wrist.y)
                if event:
                    print(f"[GESTURE] {event}", flush=True)
                    broadcast(loop, connected_clients,
                              json.dumps({"gesture": event}))
                    label = event

            elif (fingers[0] or fingers[1]) and not fingers[2] and not fingers[3] and not fingers[4]:
                # ── PINCH: thumb+index only, middle/ring/pinky curled ─────
                swipe.miss()                     # grace period before clearing swipe
                ratio = pinch_ratio(lm)
                now   = time.time()
                if last_pinch_ratio is not None:
                    delta = ratio - last_pinch_ratio
                    if now - last_pinch_send > 0.5:
                        print(f"[PINCH-DBG] ratio={ratio:.3f} prev={last_pinch_ratio:.3f} delta={delta:+.4f} thresh=0.012", flush=True)
                    if now - last_pinch_send > pinch_send_interval and abs(delta) > 0.012:
                        print(f"[PINCH-DBG] SENDING delta={delta:+.4f}", flush=True)
                        broadcast(loop, connected_clients,
                                  json.dumps({"gesture": "PINCH",
                                              "delta": round(delta, 4)}))
                        last_pinch_send = now
                        label = f"PINCH d={delta:+.3f}"
                else:
                    print(f"[PINCH-DBG] started ratio={ratio:.3f}", flush=True)
                last_pinch_ratio = ratio

            else:
                # ── Transitional / unrecognised hand pose ─────────────────
                swipe.miss()
                last_pinch_ratio = None

        else:
            # No hand detected
            swipe.miss()
            last_pinch_ratio = None

        if not HEADLESS:
            h, w = frame.shape[:2]
            n_clients = len(connected_clients)
            status_color = (0, 220, 100) if n_clients > 0 else (0, 120, 220)
            cv2.putText(frame, f"ws://{WS_HOST}:{WS_PORT}  [{n_clients} client(s)]",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, status_color, 2)
            if label:
                cv2.putText(frame, label, (10, h - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 220, 255), 2)
            cv2.imshow("Gesture Server", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop_event.set()
        elif label:
            print(f"[GESTURE] {label}", flush=True)

    cap.release()
    cv2.destroyAllWindows()
    hands_model.close()


# ── WebSocket server ──────────────────────────────────────────────────────────

async def ws_handler(websocket, connected_clients):
    connected_clients.add(websocket)
    print(f"[WS] Client connected  ({len(connected_clients)} total)", flush=True)
    try:
        async for _ in websocket:
            pass  # server → client only; ignore incoming messages
    except Exception:
        pass
    finally:
        connected_clients.discard(websocket)
        print(f"[WS] Client disconnected ({len(connected_clients)} remaining)", flush=True)


async def main():
    try:
        import websockets
    except ImportError:
        print("[ERROR] 'websockets' not found. Run: pip install websockets", flush=True)
        return

    connected_clients = set()
    stop_event        = threading.Event()
    loop              = asyncio.get_running_loop()

    t = threading.Thread(
        target=gesture_thread,
        args=(loop, connected_clients, stop_event),
        daemon=True,
    )
    t.start()

    async with websockets.serve(
        lambda ws: ws_handler(ws, connected_clients),
        WS_HOST, WS_PORT,
    ):
        print(f"[WS] WebSocket server listening on ws://{WS_HOST}:{WS_PORT}", flush=True)
        # Block until the camera thread signals quit
        await loop.run_in_executor(None, stop_event.wait)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
