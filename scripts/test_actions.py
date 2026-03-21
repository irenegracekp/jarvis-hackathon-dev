#!/usr/bin/env python3
"""Live gesture -> desktop action test.

Opens camera, detects gestures, executes corresponding desktop actions.
Open a browser window first to see navigation/scrolling effects.
Press 'q' to quit.
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2


def main():
    from pipeline.gestures import GestureRecognizer
    from pipeline.actions import ActionMapper

    gr = GestureRecognizer()
    if not gr.load_model():
        print("Failed to load gesture model. Exiting.")
        return

    am = ActionMapper()
    if not am._check_xdotool():
        print("WARNING: xdotool not available. Actions will be logged but not executed.")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera.")
        return

    print("Gesture -> Action test running. Press 'q' to quit.")
    print("Open a browser window to see navigation effects.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = gr.process_frame(frame)
        gesture = result["gesture"]
        motion = result["motion"]
        hand_pos = result["hand_position"]

        # Execute action
        action = am.execute_gesture(gesture, hand_pos, motion)

        # Display
        color = (0, 255, 0) if gesture not in ("no_hand", "none") else (128, 128, 128)
        cv2.putText(frame, f"Gesture: {gesture}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

        if action:
            cv2.putText(frame, f"Action: {action}", (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 255), 2)
            print(f"[action] {gesture} -> {action}")

        if motion:
            cv2.putText(frame, f"Motion: {motion['direction']}", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)

        cv2.imshow("Gesture Actions", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
