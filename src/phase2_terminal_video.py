#!/usr/bin/env python3
"""Render live camera feed as ASCII art in the terminal."""

from __future__ import annotations

import argparse
import os
import sys
import time

import cv2  # type: ignore
import numpy as np

ASCII_GRADIENT = "@%#*+=-:. "
CHAR_ASPECT_RATIO = 0.55
CLEAR_COMMAND = "cls" if os.name == "nt" else "clear"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Display live video as ASCII art.")
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="Camera device index passed to OpenCV (default: %(default)s)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=80,
        help="Width of ASCII art in characters (default: %(default)s)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=15.0,
        help="Target refresh rate for ASCII output (default: %(default)s)",
    )
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Disable the OpenCV preview window (terminal-only experience).",
    )
    return parser.parse_args()


def resize_frame(frame: np.ndarray, target_width: int) -> np.ndarray:
    if target_width <= 0:
        raise ValueError("Width must be greater than zero.")

    height, width = frame.shape[:2]
    scale = target_width / float(width)
    target_height = max(1, int(height * scale * CHAR_ASPECT_RATIO))
    return cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)


def frame_to_ascii(frame: np.ndarray, width: int) -> str:
    resized = resize_frame(frame, width)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    normalized = gray / 255.0
    num_chars = len(ASCII_GRADIENT)
    indices = (normalized * (num_chars - 1)).astype(int)
    rows = ["".join(ASCII_GRADIENT[pixel] for pixel in row) for row in indices]
    return "\n".join(rows)


def clear_terminal() -> None:
    os.system(CLEAR_COMMAND)


def main() -> int:
    args = parse_args()

    if args.fps <= 0:
        print("Error: FPS must be greater than zero.", file=sys.stderr)
        return 2

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        print(f"Error: Unable to open camera index {args.camera_index}.", file=sys.stderr)
        return 1

    frame_interval = 1.0 / args.fps
    try:
        while True:
            loop_start = time.perf_counter()

            success, frame = cap.read()
            if not success:
                print("Warning: Unable to read frame from camera.", file=sys.stderr)
                break

            ascii_art = frame_to_ascii(frame, args.width)
            clear_terminal()
            print(ascii_art)

            if not args.no_preview:
                cv2.imshow("Camera Preview (press 'q' to exit)", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            elapsed = time.perf_counter() - loop_start
            delay = frame_interval - elapsed
            if delay > 0:
                time.sleep(delay)
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        if not args.no_preview:
            cv2.destroyAllWindows()
        clear_terminal()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
