#!/usr/bin/env python3
"""Simple GUI to stream video frames from the Reachy robot's USB camera."""

import argparse
import cv2
import sys


def main():
    parser = argparse.ArgumentParser(description="Reachy Camera Viewer")
    parser.add_argument(
        "--device", "-d", type=int, default=0,
        help="Video device index (default: 0)"
    )
    parser.add_argument(
        "--width", "-W", type=int, default=None,
        help="Capture width (default: camera native)"
    )
    parser.add_argument(
        "--height", "-H", type=int, default=None,
        help="Capture height (default: camera native)"
    )
    parser.add_argument(
        "--split-stereo", "-s", action="store_true",
        help="Split stereo frame into left/right views"
    )
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.device)
    if not cap.isOpened():
        print(f"Error: Cannot open /dev/video{args.device}")
        sys.exit(1)

    if args.width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    if args.height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Streaming /dev/video{args.device} at {w}x{h}")
    print("Press 'q' to quit")

    window_name = f"Reachy Camera (video{args.device})"

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame")
            break

        if args.split_stereo:
            mid = frame.shape[1] // 2
            left = frame[:, :mid]
            right = frame[:, mid:]
            cv2.imshow("Reachy Left Eye", left)
            cv2.imshow("Reachy Right Eye", right)
        else:
            cv2.imshow(window_name, frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
