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


def finger_is_extended(landmarks, tip, pip, mcp):
    """A finger is extended if tip is farther from wrist than PIP joint."""
    wrist = landmarks[WRIST]
    t = landmarks[tip]
    p = landmarks[pip]
    # Use y-coordinate: tip above pip means extended (y decreases upward)
    return t.y < p.y


def thumb_is_extended(landmarks):
    """Thumb: check if tip is farther from palm center than IP joint."""
    tip = landmarks[THUMB_TIP]
    ip = landmarks[THUMB_IP]
    mcp = landmarks[THUMB_MCP]
    # Thumb extends sideways, so check x distance from palm
    wrist = landmarks[WRIST]
    middle_mcp = landmarks[MIDDLE_MCP]
    # Palm direction
    return abs(tip.x - wrist.x) > abs(ip.x - wrist.x)


def count_fingers(landmarks):
    """Count extended fingers. Returns list of booleans [thumb, index, middle, ring, pinky]."""
    fingers = [
        thumb_is_extended(landmarks),
        finger_is_extended(landmarks, INDEX_TIP, INDEX_PIP, INDEX_MCP),
        finger_is_extended(landmarks, MIDDLE_TIP, MIDDLE_PIP, MIDDLE_MCP),
        finger_is_extended(landmarks, RING_TIP, RING_PIP, RING_MCP),
        finger_is_extended(landmarks, PINKY_TIP, PINKY_PIP, PINKY_MCP),
    ]
    return fingers


def detect_gesture(landmarks):
    """Detect: THUMBS UP, THUMBS DOWN, OPEN PALM, FIST."""
    fingers = count_fingers(landmarks)
    num_extended = sum(fingers)

    # OPEN PALM: 4-5 fingers extended
    if num_extended >= 4:
        return "OPEN PALM"

    # FIST: 0 fingers
    if num_extended == 0:
        return "FIST"

    # THUMBS UP / DOWN: only thumb extended
    if fingers[0] and num_extended == 1:
        thumb_tip = landmarks[THUMB_TIP]
        wrist = landmarks[WRIST]
        index_mcp = landmarks[INDEX_MCP]

        # Check if thumb is above or below the index finger base
        if thumb_tip.y < index_mcp.y:
            return "THUMBS UP"
        else:
            return "THUMBS DOWN"

    return None


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


GESTURE_COLORS = {
    "THUMBS UP": (0, 255, 0),
    "THUMBS DOWN": (0, 0, 255),
    "OPEN PALM": (0, 255, 255),
    "FIST": (128, 128, 255),
}

CAMERA_DEVICE = 10
cap = cv2.VideoCapture(CAMERA_DEVICE)
if not cap.isOpened():
    print(f"Error: Cannot open camera /dev/video{CAMERA_DEVICE}", flush=True)
    exit(1)

print("MediaPipe Hand Pose - Gestures: THUMBS UP/DOWN, OPEN PALM, FIST", flush=True)
print("Press 'q' to quit", flush=True)

stabilizer = GestureStabilizer(required_frames=5)
gesture_display = ""
gesture_display_until = 0

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5,
)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    t0 = time.time()

    # MediaPipe needs RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    dt = time.time() - t0

    raw_gesture = None
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw skeleton
            mp_draw.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_styles.get_default_hand_landmarks_style(),
                mp_styles.get_default_hand_connections_style(),
            )
            raw_gesture = detect_gesture(hand_landmarks.landmark)

    fired = stabilizer.update(raw_gesture)
    now = time.time()
    if fired:
        gesture_display = fired
        gesture_display_until = now + 2.0
        print(f"[GESTURE] {fired}", flush=True)

    # HUD
    fps = 1.0 / dt if dt > 0 else 0
    h, w = frame.shape[:2]
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    if results.multi_hand_landmarks:
        fingers = count_fingers(results.multi_hand_landmarks[0].landmark)
        num = sum(fingers)
        cv2.putText(frame, f"Fingers: {num}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        if raw_gesture:
            cv2.putText(frame, f"raw: {raw_gesture}", (10, 85),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
    else:
        cv2.putText(frame, "No hand", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    if now < gesture_display_until:
        color = GESTURE_COLORS.get(gesture_display, (255, 255, 255))
        cv2.putText(frame, gesture_display, (10, h - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

    cv2.imshow("Hand Pose - MediaPipe", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()
