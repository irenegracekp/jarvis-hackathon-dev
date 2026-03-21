#!/usr/bin/env python3
"""Live camera feed with face detection + recognition overlays.

Shows bounding boxes, names, and confidence scores in real-time.
Press 'q' to quit.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import time
from pipeline.faces import FaceRecognizer
from pipeline.memory import MemoryStore


def main():
    recognizer = FaceRecognizer()
    if not recognizer.load_models():
        print("Failed to load face models.")
        return

    memory = MemoryStore()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera.")
        return

    print("Live face viewer running. Press 'q' to quit.")
    fps_time = time.time()
    fps = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ZED stereo: left half
        if frame.shape[1] > 2000:
            frame = frame[:, :frame.shape[1] // 2]

        faces = recognizer.detect_and_identify(frame)

        for face in faces:
            x, y, w, h = face["bbox"]
            is_known = face["is_known"]
            face_id = face["face_id"]
            conf = face["confidence"]

            # Color: green for known, red for unknown
            color = (0, 255, 0) if is_known else (0, 0, 255)

            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            # Build label
            if is_known and face_id:
                person = memory.get_person(face_id)
                if person:
                    name = person["name"]
                    facts = person.get("facts", [])
                    label = f"{name} ({conf:.2f})"
                    # Show facts below
                    for i, fact in enumerate(facts[:3]):
                        cv2.putText(frame, fact, (x, y + h + 20 + i * 18),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                else:
                    label = f"{face_id} ({conf:.2f})"
            else:
                label = f"Unknown ({conf:.2f})"

            # Draw label background
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(frame, (x, y - th - 10), (x + tw, y), color, -1)
            cv2.putText(frame, label, (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # FPS
        now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(now - fps_time, 0.001))
        fps_time = now
        cv2.putText(frame, f"FPS: {fps:.1f} | Faces: {len(faces)}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
