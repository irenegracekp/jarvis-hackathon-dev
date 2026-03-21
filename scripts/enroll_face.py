#!/usr/bin/env python3
"""Pre-register a face from a photo for The Witness face recognition.

Usage:
    python scripts/enroll_face.py --photo judges/alice.jpg --name "Alice" --face-id judge_alice \
        --facts "Hackathon judge" "Works at NVIDIA"
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
from pipeline.faces import FaceRecognizer
from pipeline.memory import MemoryStore


def main():
    parser = argparse.ArgumentParser(description="Enroll a face from a photo")
    parser.add_argument("--photo", required=True, help="Path to photo containing the face")
    parser.add_argument("--name", required=True, help="Person's name")
    parser.add_argument("--face-id", required=True, help="Unique face ID (e.g. judge_alice)")
    parser.add_argument("--facts", nargs="*", default=[], help="Known facts about this person")
    args = parser.parse_args()

    if not os.path.exists(args.photo):
        print(f"Error: Photo not found: {args.photo}")
        sys.exit(1)

    frame = cv2.imread(args.photo)
    if frame is None:
        print(f"Error: Could not read image: {args.photo}")
        sys.exit(1)

    recognizer = FaceRecognizer()
    if not recognizer.load_models():
        print("Error: Could not load face detection/recognition models.")
        sys.exit(1)

    faces = recognizer.detect_and_identify(frame)
    if not faces:
        print("Error: No face detected in the photo.")
        sys.exit(1)

    if len(faces) > 1:
        print(f"Warning: {len(faces)} faces detected, using the largest one.")
        # Pick the largest face by area
        faces.sort(key=lambda f: f["bbox"][2] * f["bbox"][3], reverse=True)

    face = faces[0]
    print(f"Face detected: bbox={face['bbox']}, confidence={face['confidence']:.3f}")

    # Enroll the embedding
    recognizer.enroll(args.face_id, face["embedding"])
    print(f"Embedding saved for face_id={args.face_id}")

    # Create memory entry
    memory = MemoryStore()
    memory.create_person(args.face_id, args.name, facts=args.facts, pre_loaded=True)
    print(f"Memory entry created: {args.name} ({args.face_id})")

    if args.facts:
        print(f"  Facts: {', '.join(args.facts)}")

    print("Done! This person will be recognized on camera.")


if __name__ == "__main__":
    main()
