#!/usr/bin/env python3
"""Convert an image into ASCII art and store the result in a text file."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2  # type: ignore
import numpy as np

ASCII_GRADIENT = "@%#*+=-:. "
# Terminal characters are taller than they are wide, so compensate when scaling.
CHAR_ASPECT_RATIO = 0.55


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert an image to ASCII art.")
    parser.add_argument(
        "image_path",
        nargs="?",
        default="assets/test.jpg",
        help="Path to the source image (default: %(default)s)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=80,
        help="Target width for the ASCII art in characters (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/frame.txt"),
        help="Where to store the ASCII art text output (default: %(default)s)",
    )
    return parser.parse_args()


def load_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"Unable to load image at '{path}'.")
    return image


def resize_image(image: np.ndarray, target_width: int) -> np.ndarray:
    height, width = image.shape[:2]
    if target_width <= 0:
        raise ValueError("Target width must be greater than zero.")

    scale = target_width / float(width)
    target_height = max(1, int(height * scale * CHAR_ASPECT_RATIO))
    return cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)


def to_ascii(image: np.ndarray) -> str:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    normalized = gray / 255.0
    num_chars = len(ASCII_GRADIENT)
    indices = (normalized * (num_chars - 1)).astype(int)

    rows = ["".join(ASCII_GRADIENT[pixel] for pixel in row) for row in indices]
    return "\n".join(rows)


def main() -> int:
    args = parse_args()
    image_path = Path(args.image_path)
    output_path = args.output

    try:
        image = load_image(image_path)
        resized = resize_image(image, args.width)
        ascii_art = to_ascii(resized)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # pragma: no cover - catch unexpected OpenCV errors
        print(f"Unexpected error: {exc}", file=sys.stderr)
        return 3

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(ascii_art)
    print(f"ASCII art saved to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
