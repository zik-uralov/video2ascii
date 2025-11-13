#!/usr/bin/env python3
"""Render live camera feed as ASCII art inside a Pygame window."""

from __future__ import annotations

import argparse
import sys
from typing import List, Tuple

import cv2  # type: ignore
import numpy as np
import pygame

ASCII_GRADIENT = "@%#*+=-:. "
CHAR_ASPECT_RATIO = 0.55  # compensate for taller terminal font cells


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Display live ASCII art using Pygame.")
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
        default=20.0,
        help="Target frame rate for rendering (default: %(default)s)",
    )
    parser.add_argument(
        "--font-size",
        type=int,
        default=16,
        help="Font size (in px) for the ASCII characters (default: %(default)s)",
    )
    return parser.parse_args()


def resize_frame(frame: np.ndarray, target_width: int) -> np.ndarray:
    if target_width <= 0:
        raise ValueError("Width must be greater than zero.")

    height, width = frame.shape[:2]
    scale = target_width / float(width)
    target_height = max(1, int(height * scale * CHAR_ASPECT_RATIO))
    return cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)


def frame_to_ascii_lines(frame: np.ndarray, width: int) -> List[str]:
    resized = resize_frame(frame, width)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    normalized = gray / 255.0
    num_chars = len(ASCII_GRADIENT)
    indices = (normalized * (num_chars - 1)).astype(int)
    return ["".join(ASCII_GRADIENT[pixel] for pixel in row) for row in indices]


def ensure_screen_size(
    screen: pygame.Surface | None,
    line_count: int,
    col_count: int,
    font: pygame.font.Font,
) -> Tuple[pygame.Surface, Tuple[int, int]]:
    char_width, char_height = font.size("A")
    width = max(1, char_width * col_count)
    height = max(1, font.get_linesize() * line_count)

    if screen is None or screen.get_width() != width or screen.get_height() != height:
        screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("video2ascii – Pygame ASCII feed")

    return screen, (width, height)


def handle_events() -> bool:
    """Return False if the user requested exit."""
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return False
        if event.type == pygame.KEYDOWN and event.key in (pygame.K_q, pygame.K_ESCAPE):
            return False
    return True


def main() -> int:
    args = parse_args()
    if args.fps <= 0:
        print("Error: FPS must be greater than zero.", file=sys.stderr)
        return 2

    pygame.init()
    pygame.font.init()
    font = pygame.font.SysFont("Courier New", args.font_size)
    if font is None:
        font = pygame.font.Font(pygame.font.get_default_font(), args.font_size)

    screen: pygame.Surface | None = None
    clock = pygame.time.Clock()

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        print(f"Error: Unable to open camera index {args.camera_index}.", file=sys.stderr)
        pygame.quit()
        return 1

    try:
        while True:
            if not handle_events():
                break

            success, frame = cap.read()
            if not success:
                print("Warning: Unable to read frame from camera.", file=sys.stderr)
                break

            ascii_lines = frame_to_ascii_lines(frame, args.width)
            if not ascii_lines:
                continue

            screen, _ = ensure_screen_size(screen, len(ascii_lines), len(ascii_lines[0]), font)
            screen.fill((0, 0, 0))

            line_height = font.get_linesize()
            for row_index, line in enumerate(ascii_lines):
                text_surface = font.render(line, True, (255, 255, 255))
                screen.blit(text_surface, (0, row_index * line_height))

            pygame.display.flip()
            clock.tick(args.fps)
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        pygame.quit()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
