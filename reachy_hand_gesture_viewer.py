#!/usr/bin/env python3
"""Reachy stereo camera viewer with MediaPipe hand gesture detection.

Supports:
- Single hand: THUMBS UP/DOWN, OPEN PALM, FIST, single-hand PINCH
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
    return abs(tip.x - wrist.x) > abs(ip.x - wrist.x)


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
    "SCALE UP": (0, 255, 128),
    "SCALE DOWN": (255, 0, 128),
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
    """Tracks single-hand pinch (thumb + index)."""

    def __init__(self, pinch_threshold=0.06):
        self.pinch_threshold = pinch_threshold
        self.pinching = False

    def update(self, landmarks):
        thumb = landmarks[THUMB_TIP]
        index = landmarks[INDEX_TIP]
        dist = ((thumb.x - index.x) ** 2 + (thumb.y - index.y) ** 2) ** 0.5
        self.pinching = dist < self.pinch_threshold
        return self.pinching, dist


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


def process_frame(frame, hands_detector, stabilizer, pinch_tracker, scale_tracker):
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

            # Check single-hand pinch
            is_pinching, pinch_dist = pinch_tracker.update(lm)

            # Draw pinch indicator
            thumb = lm[THUMB_TIP]
            index = lm[INDEX_TIP]
            pt1 = (int(thumb.x * w), int(thumb.y * h))
            pt2 = (int(index.x * w), int(index.y * h))

            if is_pinching:
                cv2.line(frame, pt1, pt2, (0, 165, 255), 3)
                mid_pt = ((pt1[0] + pt2[0]) // 2, (pt1[1] + pt2[1]) // 2)
                cv2.circle(frame, mid_pt, 8, (0, 165, 255), -1)
                raw_gesture = "PINCH"
            else:
                cv2.line(frame, pt1, pt2, (200, 200, 200), 1)
                raw_gesture = detect_gesture(lm)

            cv2.putText(frame, f"Pinch: {pinch_dist:.3f}", (10, 94),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 1)
    else:
        scale_tracker.reset()

    fired = stabilizer.update(raw_gesture)
    now = time.time()

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

    # Show stabilized gesture
    if stabilizer.last_fired and now - stabilizer.last_fired_time < 2.0:
        color = GESTURE_COLORS.get(stabilizer.last_fired, (255, 255, 255))
        cv2.putText(frame, stabilizer.last_fired, (10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

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
    print("Gestures: THUMBS UP/DOWN, OPEN PALM, FIST, PINCH (1 hand), SCALE UP/DOWN (2 hands)")
    print("Press 'q' or close the window to quit")

    display_w, display_h = 960, 540

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

    # For stereo, separate detectors per eye
    if is_stereo:
        hands_right = mp_hands.Hands(
            static_image_mode=False, max_num_hands=2,
            min_detection_confidence=0.7, min_tracking_confidence=0.5,
        )
        stab_right = GestureStabilizer(required_frames=4)
        pinch_right = SingleHandPinchTracker()
        scale_right = TwoHandScaleTracker()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame")
            break

        if is_stereo:
            mid = frame.shape[1] // 2
            left = frame[:, :mid].copy()
            right = frame[:, mid:].copy()

            left = process_frame(left, hands_detector, stabilizer, pinch_tracker, scale_tracker)
            right = process_frame(right, hands_right, stab_right, pinch_right, scale_right)

            combined = np.hstack([left, right])
            cv2.imshow(win_name, combined)
        else:
            frame = process_frame(frame, hands_detector, stabilizer, pinch_tracker, scale_tracker)
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
