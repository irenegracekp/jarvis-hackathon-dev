#!/usr/bin/env python3
"""Reachy stereo camera viewer with MediaPipe hand gesture detection.

Supports:
- Single hand: THUMBS UP/DOWN, OPEN PALM, FIST, single-hand PINCH
- Single hand: SWIPE LEFT/RIGHT/UP/DOWN (open palm, time-based detection)
- Two hands: SCALE UP (spread apart), SCALE DOWN (push together)
"""

import argparse
import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

# Landmark indices
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20


def finger_is_extended(landmarks, tip, pip_idx, mcp):
    return landmarks[tip].y < landmarks[pip_idx].y


def thumb_is_extended(landmarks):
    tip = landmarks[THUMB_TIP]
    ip = landmarks[THUMB_IP]
    wrist = landmarks[WRIST]
    # Use 2D Euclidean distance so thumb pointing up/sideways both work.
    # Require TIP to be meaningfully further than IP (1.2x margin) to avoid
    # fist misclassifying as thumb extended.
    tip_dist = ((tip.x - wrist.x) ** 2 + (tip.y - wrist.y) ** 2) ** 0.5
    ip_dist = ((ip.x - wrist.x) ** 2 + (ip.y - wrist.y) ** 2) ** 0.5
    return tip_dist > ip_dist * 1.2


def count_fingers(landmarks):
    return [
        thumb_is_extended(landmarks),
        finger_is_extended(landmarks, INDEX_TIP, INDEX_PIP, INDEX_MCP),
        finger_is_extended(landmarks, MIDDLE_TIP, MIDDLE_PIP, MIDDLE_MCP),
        finger_is_extended(landmarks, RING_TIP, RING_PIP, RING_MCP),
        finger_is_extended(landmarks, PINKY_TIP, PINKY_PIP, PINKY_MCP),
    ]


def detect_gesture(landmarks):
    fingers = count_fingers(landmarks)
    num_extended = sum(fingers)

    if num_extended >= 4:
        return "OPEN PALM"
    if num_extended == 0:
        return "FIST"
    if fingers[0] and num_extended == 1:
        thumb_tip = landmarks[THUMB_TIP]
        index_mcp = landmarks[INDEX_MCP]
        if thumb_tip.y < index_mcp.y:
            return "THUMBS UP"
        else:
            return "THUMBS DOWN"
    return None


GESTURE_COLORS = {
    "THUMBS UP": (0, 255, 0),
    "THUMBS DOWN": (0, 0, 255),
    "OPEN PALM": (0, 255, 255),
    "FIST": (128, 128, 255),
    "PINCH": (255, 165, 0),
    "PITCH +": (0, 255, 128),
    "PITCH -": (0, 200, 80),
    "YAW +": (255, 0, 200),
    "YAW -": (180, 0, 140),
    "SCALE UP": (0, 255, 128),
    "SCALE DOWN": (255, 0, 128),
    "SWIPE LEFT": (255, 100, 100),
    "SWIPE RIGHT": (100, 255, 100),
    "SWIPE UP": (100, 200, 255),
    "SWIPE DOWN": (255, 200, 100),
}


class TwoHandScaleTracker:
    """Tracks distance between two hands' index fingers for scale gestures."""

    def __init__(self, smooth_window=6, scale_threshold=0.04):
        self.dist_history = deque(maxlen=smooth_window)
        self.scale_threshold = scale_threshold
        self.tracking = False
        self.baseline_dist = None
        self.current_gesture = None
        self.gesture_time = 0

    def update(self, hand1_landmarks, hand2_landmarks):
        """Update with two hand landmarks. Returns (gesture, dist, pt1, pt2)."""
        idx1 = hand1_landmarks[INDEX_TIP]
        idx2 = hand2_landmarks[INDEX_TIP]
        dist = ((idx1.x - idx2.x) ** 2 + (idx1.y - idx2.y) ** 2) ** 0.5

        self.dist_history.append(dist)

        if not self.tracking:
            # Start tracking once we have enough history
            self.tracking = True
            self.baseline_dist = dist
            return None, dist

        # Smooth distance
        smooth_dist = sum(self.dist_history) / len(self.dist_history)

        if self.baseline_dist is None:
            self.baseline_dist = smooth_dist
            return None, dist

        delta = smooth_dist - self.baseline_dist
        now = time.time()

        gesture = None
        if delta > self.scale_threshold:
            gesture = "SCALE UP"
        elif delta < -self.scale_threshold:
            gesture = "SCALE DOWN"

        # Update baseline slowly to allow continuous scaling
        self.baseline_dist = self.baseline_dist * 0.95 + smooth_dist * 0.05

        if gesture and gesture != self.current_gesture:
            self.current_gesture = gesture
            self.gesture_time = now
            print(f"[TWO-HAND] {gesture}", flush=True)
        elif gesture is None and now - self.gesture_time > 0.5:
            self.current_gesture = None

        return gesture, dist

    def reset(self):
        self.dist_history.clear()
        self.tracking = False
        self.baseline_dist = None
        self.current_gesture = None


class SingleHandPinchTracker:
    """Tracks single-hand pinch + drag to detect rotation gestures.

    While pinching, horizontal movement → PINCH ROTATE PITCH
                    vertical movement   → PINCH ROTATE YAW
    Axis is determined by whichever direction dominates movement from the
    pinch-start anchor.
    """

    def __init__(self, pinch_threshold=0.06, move_threshold=0.05, smooth_window=6):
        self.pinch_threshold = pinch_threshold
        self.move_threshold = move_threshold
        self.pinching = False
        self.was_pinching = False
        self.dist_history = deque(maxlen=smooth_window)
        self.rotate_gesture = None
        self.rotate_gesture_time = 0
        # Anchor position (normalised coords) when pinch started
        self._anchor_x = None
        self._anchor_y = None

    def _pinch_center(self, landmarks):
        thumb = landmarks[THUMB_TIP]
        index = landmarks[INDEX_TIP]
        return (thumb.x + index.x) / 2, (thumb.y + index.y) / 2

    def update(self, landmarks):
        thumb = landmarks[THUMB_TIP]
        index = landmarks[INDEX_TIP]
        dist = ((thumb.x - index.x) ** 2 + (thumb.y - index.y) ** 2) ** 0.5

        self.dist_history.append(dist)

        self.pinching = dist < self.pinch_threshold
        now = time.time()
        cx, cy = self._pinch_center(landmarks)

        if self.pinching and not self.was_pinching:
            # Just started pinching — record anchor
            self._anchor_x = cx
            self._anchor_y = cy
            self.rotate_gesture = None

        if self.pinching and self._anchor_x is not None:
            dx = cx - self._anchor_x
            dy = cy - self._anchor_y
            if abs(dx) > self.move_threshold or abs(dy) > self.move_threshold:
                if abs(dx) >= abs(dy):
                    new_gesture = "PITCH +" if dx > 0 else "PITCH -"
                else:
                    new_gesture = "YAW +" if dy < 0 else "YAW -"
                if self.rotate_gesture != new_gesture:
                    self.rotate_gesture = new_gesture
                    self.rotate_gesture_time = now
                    print(f"[PINCH] {new_gesture} (dx={dx:.3f} dy={dy:.3f})", flush=True)
                # Slide anchor so gesture fires continuously while dragging
                self._anchor_x = self._anchor_x * 0.85 + cx * 0.15
                self._anchor_y = self._anchor_y * 0.85 + cy * 0.15

        if not self.pinching:
            if self.rotate_gesture and now - self.rotate_gesture_time > 1.5:
                self.rotate_gesture = None
            self._anchor_x = None
            self._anchor_y = None

        self.was_pinching = self.pinching
        return self.pinching, dist


class SwipeTracker:
    """Time-based swipe detection using a state machine.

    States: IDLE -> MOVING -> FIRED -> COOLDOWN -> IDLE

    All thresholds are time-based (not frame-based) so behavior is
    consistent regardless of FPS. Tuned for ~10 FPS on Jetson.

    Only triggers on open palm (>=3 fingers extended) to avoid
    false positives from hand repositioning during other gestures.
    """

    def __init__(self,
                 min_displacement_frac=0.15,
                 axis_ratio=1.5,
                 rest_speed_threshold=150,
                 rest_frames=2,
                 cooldown_sec=0.5,
                 min_fingers=2,
                 max_fingers=2):
        self.min_displacement_frac = min_displacement_frac
        self.axis_ratio = axis_ratio
        self.rest_speed = rest_speed_threshold
        self.rest_frames_needed = rest_frames
        self.cooldown_sec = cooldown_sec
        self.min_fingers = min_fingers
        self.max_fingers = max_fingers

        # Anchor: position where hand was last at rest
        self._anchor = None          # (time, x, y) or None
        self._anchor_locked = False  # True once anchor is confirmed
        self._rest_count = 0         # consecutive low-speed frames

        # Speed tracking
        self._prev_pos = None
        self._prev_time = None

        # State
        self.last_swipe = None
        self.last_swipe_time = 0
        self.trail = deque(maxlen=20)
        self.state = "no_anchor"
        self._debug_speed = 0
        self._debug_disp = 0
        self.swiping = False

    def _compute_speed(self, x, y, now):
        speed = 0
        if self._prev_pos is not None and self._prev_time is not None:
            dt = now - self._prev_time
            if 0 < dt < 0.5:
                dx = x - self._prev_pos[0]
                dy = y - self._prev_pos[1]
                speed = np.sqrt(dx**2 + dy**2) / dt
        self._prev_pos = (x, y)
        self._prev_time = now
        return speed

    def update(self, x_px, y_px, frame_w, frame_h, num_fingers):
        """Anchor-based swipe detection.

        1. Wait for hand to be at rest (low speed) → set anchor
        2. Measure displacement from anchor
        3. Fire swipe when displacement + speed thresholds met
        4. Invalidate anchor → must come to rest again before next swipe

        Return motion never fires because the hand never rests at the
        swipe endpoint — it immediately starts moving back.
        """
        now = time.time()
        self.trail.append((int(x_px), int(y_px), now))
        speed = self._compute_speed(x_px, y_px, now)
        self._debug_speed = speed
        self.swiping = speed > self.rest_speed * 2

        # Require exactly 2 fingers — rejects fist, open palm, and pinch (0-1 fingers)
        if not (self.min_fingers <= num_fingers <= self.max_fingers):
            self.state = "no_hand"
            return None

        # Cooldown after firing
        if self.last_swipe_time and now - self.last_swipe_time < self.cooldown_sec:
            self.state = "cooldown"
            return None

        # --- Anchor management ---
        if speed < self.rest_speed:
            self._rest_count += 1
            if self._rest_count >= self.rest_frames_needed:
                # Hand is at rest — set/update anchor
                self._anchor = (now, x_px, y_px)
                self._anchor_locked = True
                self.state = "anchored"
        else:
            self._rest_count = 0

        # No anchor yet — need hand to be still first
        if not self._anchor_locked:
            self.state = "no_anchor"
            self._debug_disp = 0
            return None

        # --- Swipe detection from anchor ---
        _, ax, ay = self._anchor
        dx = x_px - ax
        dy = y_px - ay
        displacement = np.sqrt(dx**2 + dy**2)
        self._debug_disp = displacement

        abs_dx = abs(dx)
        abs_dy = abs(dy)
        if abs_dx < 1 and abs_dy < 1:
            return None

        dominant = max(abs_dx, abs_dy)
        secondary = min(abs_dx, abs_dy)
        ratio = dominant / max(secondary, 1)

        # Use axis-specific thresholds so vertical swipes aren't penalised
        # by a frame_w-based distance requirement.
        min_disp_h = self.min_displacement_frac * frame_w
        min_disp_v = self.min_displacement_frac * frame_h

        is_vertical = abs_dy > abs_dx
        min_disp = min_disp_v if is_vertical else min_disp_h
        ratio_thresh = self.axis_ratio * 0.7 if is_vertical else self.axis_ratio

        # Fire purely on displacement + direction ratio — no speed gate.
        # The anchor requirement already prevents accidental triggers.
        if displacement > min_disp and ratio > ratio_thresh:
            direction = ("SWIPE DOWN" if dy > 0 else "SWIPE UP") if is_vertical else \
                        ("SWIPE RIGHT" if dx > 0 else "SWIPE LEFT")

            self.last_swipe = direction
            self.last_swipe_time = now
            self._anchor_locked = False
            self._anchor = None
            self._rest_count = 0
            self.state = "fired"

            print(f"[SWIPE] {direction} (disp={displacement:.0f}px, ratio={ratio:.1f})", flush=True)
            return direction

        min_disp = min_disp_h if abs_dx > abs_dy else min_disp_v
        self.state = "tracking" if displacement > min_disp * 0.3 else "anchored"
        return None

    def draw_debug(self, frame):
        """Draw swipe debug info."""
        h, w = frame.shape[:2]
        x_off = w - 280
        state_colors = {
            "no_anchor": (150, 150, 150),
            "no_hand": (100, 100, 100),
            "cooldown": (0, 0, 255),
            "anchored": (0, 200, 200),
            "tracking": (0, 255, 255),
            "fired": (0, 255, 0),
        }
        color = state_colors.get(self.state, (200, 200, 200))
        cv2.putText(frame, f"Swipe: {self.state}", (x_off, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        cv2.putText(frame, f"Spd:{self._debug_speed:.0f} Dsp:{self._debug_disp:.0f}",
                    (x_off, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
        # Draw anchor point
        if self._anchor_locked and self._anchor:
            _, ax, ay = self._anchor
            cv2.circle(frame, (int(ax), int(ay)), 8, (0, 200, 200), 2)
            # Line from anchor to current trail position
            if self.trail:
                cx, cy, _ = self.trail[-1]
                cv2.line(frame, (int(ax), int(ay)), (cx, cy), (0, 200, 200), 1)

    def draw_trail(self, frame):
        """Draw the hand motion trail on frame."""
        now = time.time()
        pts = [(x, y) for x, y, t in self.trail if now - t < 0.5]
        for i in range(1, len(pts)):
            alpha = i / len(pts)
            color = (int(100 * alpha), int(200 * alpha), int(255 * alpha))
            cv2.line(frame, pts[i - 1], pts[i], color, 2)

    def draw_swipe_indicator(self, frame):
        """Draw a large swipe arrow overlay if recently fired."""
        now = time.time()
        if not self.last_swipe or now - self.last_swipe_time > 1.0:
            return
        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2
        color = GESTURE_COLORS.get(self.last_swipe, (255, 255, 255))
        arrow_len = 100
        thickness = 4
        if self.last_swipe == "SWIPE LEFT":
            cv2.arrowedLine(frame, (cx + arrow_len, cy), (cx - arrow_len, cy), color, thickness, tipLength=0.3)
        elif self.last_swipe == "SWIPE RIGHT":
            cv2.arrowedLine(frame, (cx - arrow_len, cy), (cx + arrow_len, cy), color, thickness, tipLength=0.3)
        elif self.last_swipe == "SWIPE UP":
            cv2.arrowedLine(frame, (cx, cy + arrow_len), (cx, cy - arrow_len), color, thickness, tipLength=0.3)
        elif self.last_swipe == "SWIPE DOWN":
            cv2.arrowedLine(frame, (cx, cy - arrow_len), (cx, cy + arrow_len), color, thickness, tipLength=0.3)


class GestureStabilizer:
    def __init__(self, required_frames=5):
        self.required_frames = required_frames
        self.history = deque(maxlen=required_frames)
        self.last_fired = None
        self.last_fired_time = 0

    def update(self, gesture):
        self.history.append(gesture)
        now = time.time()
        if len(self.history) < self.required_frames:
            return None
        if len(set(self.history)) != 1:
            return None
        result = self.history[0]
        if result is None:
            return None
        if result == self.last_fired and now - self.last_fired_time < 1.5:
            return None
        self.last_fired = result
        self.last_fired_time = now
        return result


def process_frame(frame, hands_detector, stabilizer, pinch_tracker, scale_tracker, swipe_tracker):
    """Process a frame with 1 or 2 hand detection."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands_detector.process(rgb)

    h, w = frame.shape[:2]
    raw_gesture = None
    scale_gesture = None
    num_hands = 0

    if results.multi_hand_landmarks:
        num_hands = len(results.multi_hand_landmarks)

        # Draw all hand skeletons
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_styles.get_default_hand_landmarks_style(),
                mp_styles.get_default_hand_connections_style(),
            )

        if num_hands >= 2:
            # Two-hand scale mode
            lm1 = results.multi_hand_landmarks[0].landmark
            lm2 = results.multi_hand_landmarks[1].landmark

            scale_gesture, dist = scale_tracker.update(lm1, lm2)

            # Draw line between the two index fingers
            idx1 = lm1[INDEX_TIP]
            idx2 = lm2[INDEX_TIP]
            pt1 = (int(idx1.x * w), int(idx1.y * h))
            pt2 = (int(idx2.x * w), int(idx2.y * h))
            mid = ((pt1[0] + pt2[0]) // 2, (pt1[1] + pt2[1]) // 2)

            line_color = (200, 200, 200)
            if scale_gesture == "SCALE UP":
                line_color = GESTURE_COLORS["SCALE UP"]
            elif scale_gesture == "SCALE DOWN":
                line_color = GESTURE_COLORS["SCALE DOWN"]

            cv2.line(frame, pt1, pt2, line_color, 3)
            cv2.circle(frame, pt1, 8, line_color, -1)
            cv2.circle(frame, pt2, 8, line_color, -1)
            cv2.circle(frame, mid, 6, (255, 255, 255), -1)

            # Show distance
            cv2.putText(frame, f"Dist: {dist:.3f}", (10, 94),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 1)

            if scale_gesture:
                raw_gesture = scale_gesture
            else:
                raw_gesture = "TWO HANDS"

        else:
            # Single hand mode
            scale_tracker.reset()
            lm = results.multi_hand_landmarks[0].landmark

            # Check single-hand pinch (suppress during swipe motion)
            if swipe_tracker.swiping:
                is_pinching = False
                pinch_dist = 1.0
            else:
                is_pinching, pinch_dist = pinch_tracker.update(lm)

            # Draw pinch indicator
            thumb = lm[THUMB_TIP]
            index = lm[INDEX_TIP]
            pt1 = (int(thumb.x * w), int(thumb.y * h))
            pt2 = (int(index.x * w), int(index.y * h))

            if pinch_tracker.rotate_gesture:
                # Active pinch-rotate
                color = GESTURE_COLORS.get(pinch_tracker.rotate_gesture, (0, 165, 255))
                cv2.line(frame, pt1, pt2, color, 3)
                mid_pt = ((pt1[0] + pt2[0]) // 2, (pt1[1] + pt2[1]) // 2)
                cv2.circle(frame, mid_pt, 10, color, -1)
                raw_gesture = pinch_tracker.rotate_gesture

                # Draw rotation arrow near the pinch point
                cx, cy = mid_pt
                g = pinch_tracker.rotate_gesture
                if g in ("PITCH +", "PITCH -"):
                    cv2.arrowedLine(frame, (cx - 5, cy), (cx - 40, cy), color, 2)
                    cv2.arrowedLine(frame, (cx + 5, cy), (cx + 40, cy), color, 2)
                else:
                    cv2.arrowedLine(frame, (cx, cy - 5), (cx, cy - 40), color, 2)
                    cv2.arrowedLine(frame, (cx, cy + 5), (cx, cy + 40), color, 2)
            elif is_pinching:
                cv2.line(frame, pt1, pt2, (0, 165, 255), 3)
                mid_pt = ((pt1[0] + pt2[0]) // 2, (pt1[1] + pt2[1]) // 2)
                cv2.circle(frame, mid_pt, 8, (0, 165, 255), -1)
                raw_gesture = "PINCH"
            else:
                cv2.line(frame, pt1, pt2, (200, 200, 200), 1)
                raw_gesture = detect_gesture(lm)

            cv2.putText(frame, f"Pinch: {pinch_dist:.3f}", (10, 94),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 1)

            # Swipe detection (uses hand center for stability)
            wrist = lm[WRIST]
            mid_mcp = lm[MIDDLE_MCP]
            hand_cx = (wrist.x + mid_mcp.x) / 2 * w
            hand_cy = (wrist.y + mid_mcp.y) / 2 * h
            fingers = count_fingers(lm)
            # Count only non-thumb fingers so a natural peace sign with thumb
            # slightly out doesn't exceed max_fingers and get rejected.
            swipe = swipe_tracker.update(hand_cx, hand_cy, w, h, sum(fingers[1:]))
            if swipe:
                raw_gesture = swipe
    else:
        scale_tracker.reset()

    fired = stabilizer.update(raw_gesture)
    now = time.time()

    # Swipes fire exactly once so they never accumulate the frames the
    # stabilizer needs. Bypass it and record them directly.
    if raw_gesture in ("SWIPE LEFT", "SWIPE RIGHT", "SWIPE UP", "SWIPE DOWN"):
        stabilizer.last_fired = raw_gesture
        stabilizer.last_fired_time = now
        fired = raw_gesture

    # HUD
    cv2.putText(frame, f"Hands: {num_hands}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    if num_hands > 0 and results.multi_hand_landmarks:
        fingers = count_fingers(results.multi_hand_landmarks[0].landmark)
        cv2.putText(frame, f"Fingers: {sum(fingers)}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        if raw_gesture:
            cv2.putText(frame, raw_gesture, (10, 72),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
    else:
        cv2.putText(frame, "No hand", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    if fired:
        print(f"[GESTURE] {fired}", flush=True)

    # Show stabilized gesture at bottom
    if stabilizer.last_fired and now - stabilizer.last_fired_time < 2.0:
        color = GESTURE_COLORS.get(stabilizer.last_fired, (255, 255, 255))
        cv2.putText(frame, stabilizer.last_fired, (10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # Swipe debug, trail, and indicator
    swipe_tracker.draw_debug(frame)
    swipe_tracker.draw_trail(frame)
    swipe_tracker.draw_swipe_indicator(frame)

    # Scale arrows overlay
    if scale_tracker.current_gesture and now - scale_tracker.gesture_time < 1.5:
        arrow_color = GESTURE_COLORS.get(scale_tracker.current_gesture, (255, 255, 255))
        cx, cy = w // 2, h // 2
        if scale_tracker.current_gesture == "SCALE UP":
            cv2.arrowedLine(frame, (cx - 10, cy), (cx - 60, cy), arrow_color, 3)
            cv2.arrowedLine(frame, (cx + 10, cy), (cx + 60, cy), arrow_color, 3)
        else:
            cv2.arrowedLine(frame, (cx - 60, cy), (cx - 10, cy), arrow_color, 3)
            cv2.arrowedLine(frame, (cx + 60, cy), (cx + 10, cy), arrow_color, 3)

    return frame


def main():
    parser = argparse.ArgumentParser(description="Reachy Hand Gesture Viewer")
    parser.add_argument("--device", "-d", type=int, default=0,
                        help="Video device index (default: 0)")
    parser.add_argument("--no-stereo", action="store_true",
                        help="Treat as single camera (no stereo split)")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.device)
    if not cap.isOpened():
        print(f"Error: Cannot open /dev/video{args.device}")
        return

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    is_stereo = not args.no_stereo and w > h * 2
    print(f"Camera: /dev/video{args.device} at {w}x{h} ({'stereo' if is_stereo else 'mono'})")
    print("Gestures: THUMBS UP/DOWN, OPEN PALM, FIST, PINCH, SWIPE (1 hand), SCALE UP/DOWN (2 hands)")
    print("Press 'q' or close the window to quit")

    display_w, display_h = 1920, 1080

    if is_stereo:
        win_name = "Reachy Stereo - Hand Gestures"
    else:
        win_name = "Reachy Camera - Hand Gestures"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, display_w, display_h)

    # Detect up to 2 hands
    hands_detector = mp_hands.Hands(
        static_image_mode=False, max_num_hands=2,
        min_detection_confidence=0.7, min_tracking_confidence=0.5,
    )

    stabilizer = GestureStabilizer(required_frames=4)
    pinch_tracker = SingleHandPinchTracker()
    scale_tracker = TwoHandScaleTracker()
    swipe_tracker = SwipeTracker()

    # For stereo, separate detectors per eye
    if is_stereo:
        hands_right = mp_hands.Hands(
            static_image_mode=False, max_num_hands=2,
            min_detection_confidence=0.7, min_tracking_confidence=0.5,
        )
        stab_right = GestureStabilizer(required_frames=4)
        pinch_right = SingleHandPinchTracker()
        scale_right = TwoHandScaleTracker()
        swipe_right = SwipeTracker()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame")
            break

        if is_stereo:
            mid = frame.shape[1] // 2
            left = frame[:, :mid].copy()
            right = frame[:, mid:].copy()

            left = process_frame(left, hands_detector, stabilizer, pinch_tracker, scale_tracker, swipe_tracker)
            right = process_frame(right, hands_right, stab_right, pinch_right, scale_right, swipe_right)

            combined = np.hstack([left, right])
            cv2.imshow(win_name, combined)
        else:
            frame = process_frame(frame, hands_detector, stabilizer, pinch_tracker, scale_tracker, swipe_tracker)
            cv2.imshow(win_name, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()
    hands_detector.close()
    if is_stereo:
        hands_right.close()


if __name__ == "__main__":
    main()
