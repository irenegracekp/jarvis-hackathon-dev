#!/usr/bin/env python3
"""Live gesture recognition viewer.

Opens camera, shows hand keypoints + detected gesture on screen in real-time.
Press 'q' to quit.
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np


def draw_keypoints(frame, keypoints, color=(0, 255, 0)):
    """Draw hand keypoints and connections on frame."""
    # Draw keypoint dots
    for idx, (x, y) in keypoints.items():
        cv2.circle(frame, (int(x), int(y)), 5, color, -1)
        cv2.putText(frame, str(idx), (int(x) + 5, int(y) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)

    # Draw finger connections
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),      # thumb
        (0, 5), (5, 6), (6, 7), (7, 8),      # index
        (0, 9), (9, 10), (10, 11), (11, 12),  # middle
        (0, 13), (13, 14), (14, 15), (15, 16),  # ring
        (0, 17), (17, 18), (18, 19), (19, 20),  # pinky
    ]
    for a, b in connections:
        if a in keypoints and b in keypoints:
            pt1 = (int(keypoints[a][0]), int(keypoints[a][1]))
            pt2 = (int(keypoints[b][0]), int(keypoints[b][1]))
            cv2.line(frame, pt1, pt2, color, 2)


def main():
    from pipeline.gestures import GestureRecognizer

    gr = GestureRecognizer()
    if not gr.load_model():
        print("Failed to load gesture model. Exiting.")
        print("  For TRT: python scripts/setup_hand_pose.py (inside Docker)")
        print("  For MediaPipe: pip install mediapipe")
        return

    print(f"Backend: {gr.backend_name}")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera.")
        return

    print("Gesture viewer running. Press 'q' to quit.")
    fps_time = time.time()
    fps = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = gr.process_frame(frame)

        # Draw keypoints
        if result["keypoints"]:
            draw_keypoints(frame, result["keypoints"])

        # Draw gesture label
        gesture = result["gesture"]
        color = (0, 255, 0) if gesture != "no_hand" else (128, 128, 128)
        cv2.putText(frame, f"Gesture: {gesture}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

        # Draw motion
        if result["motion"]:
            motion = result["motion"]
            cv2.putText(frame, f"Motion: {motion['direction']} ({motion['speed']:.0f}px/s)",
                        (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 0), 2)

        # FPS
        now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(now - fps_time, 0.001))
        fps_time = now
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.imshow("Gesture Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
