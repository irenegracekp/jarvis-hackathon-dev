#!/usr/bin/env python3
"""Enroll a face from the live camera with interactive prompts.

Usage:
    python3 scripts/enroll_live.py

Takes a photo, detects your face, asks for your name and facts,
then saves the embedding + memory entry.
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
from pipeline.faces import FaceRecognizer
from pipeline.memory import MemoryStore


def capture_face(cap, recognizer, max_attempts=30):
    """Capture a frame with exactly one prominent face."""
    print("Look at the camera...")
    for i in range(max_attempts):
        ret, frame = cap.read()
        if not ret:
            continue

        # ZED stereo: take left half
        if frame.shape[1] > 2000:
            frame = frame[:, :frame.shape[1] // 2]

        faces = recognizer.detect_and_identify(frame)
        if faces:
            # Pick the largest face
            faces.sort(key=lambda f: f["bbox"][2] * f["bbox"][3], reverse=True)
            face = faces[0]

            # Draw bbox on preview
            x, y, w, h = face["bbox"]
            preview = frame.copy()
            cv2.rectangle(preview, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.imshow("Enroll Face", preview)
            cv2.waitKey(500)

            print(f"  Face detected! bbox={face['bbox']}, confidence={face['confidence']:.2f}")
            return frame, face

        cv2.imshow("Enroll Face", frame)
        cv2.waitKey(100)

    return None, None


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

    print("=" * 40)
    print("  Live Face Enrollment")
    print("=" * 40)

    while True:
        print("\nReady to enroll a new person.")
        input("Press Enter when the person is facing the camera...")

        frame, face = capture_face(cap, recognizer)
        if face is None:
            print("No face detected. Try again.")
            continue

        # Check if this face is already known
        if face["is_known"] and face["face_id"]:
            person = memory.get_person(face["face_id"])
            if person:
                print(f"\nThis person is already enrolled as: {person['name']}")
                print(f"  Facts: {person.get('facts', [])}")
                update = input("Update their info? (y/n): ").strip().lower()
                if update != "y":
                    continue
                face_id = face["face_id"]
            else:
                face_id = face["face_id"]
        else:
            face_id = input("Enter a face ID (e.g., irene, bf_marco): ").strip()
            if not face_id:
                print("Skipped.")
                continue

        name = input("Enter their name: ").strip()
        if not name:
            print("Skipped.")
            continue

        print("Enter facts about them (one per line, empty line to finish):")
        facts = []
        while True:
            fact = input("  > ").strip()
            if not fact:
                break
            facts.append(fact)

        # Save embedding
        recognizer.enroll(face_id, face["embedding"])

        # Save photo
        photo_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                 "data", "known_faces", "photos")
        os.makedirs(photo_dir, exist_ok=True)
        photo_path = os.path.join(photo_dir, f"{face_id}.jpg")
        cv2.imwrite(photo_path, frame)
        print(f"  Photo saved: {photo_path}")

        # Save memory
        existing = memory.get_person(face_id)
        if existing:
            memory.set_name(face_id, name)
            for fact in facts:
                memory.add_fact(face_id, fact)
        else:
            memory.create_person(face_id, name, facts=facts)

        print(f"\nEnrolled: {name} ({face_id})")
        print(f"  Facts: {facts}")
        print(f"  Memory: {memory.get_context_string(face_id)}")

        another = input("\nEnroll another person? (y/n): ").strip().lower()
        if another != "y":
            break

    cap.release()
    cv2.destroyAllWindows()

    # Summary
    print("\n" + "=" * 40)
    print("  Enrolled People")
    print("=" * 40)
    for fid, person in memory.list_people().items():
        print(f"  {fid}: {person['name']} — {person.get('facts', [])}")


if __name__ == "__main__":
    main()
