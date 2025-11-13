#!/usr/bin/env python3
"""Feature-rich Pygame ASCII art viewer with live camera input."""

from __future__ import annotations

import argparse
import datetime as _dt
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import cv2  # type: ignore
import numpy as np
import pygame

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MIN_WIDTH = 40
MAX_WIDTH = 220
WIDTH_STEP = 10
MIN_WINDOW_DIMENSION = 200
STATUS_DURATION = 2.0  # seconds
CAMERA_SCAN_LIMIT = 5  # inclusive max index probed for fallbacks


@dataclass(frozen=True)
class DensitySet:
    name: str
    characters: str


DENSITY_SETS: Sequence[DensitySet] = (
    DensitySet("Classic", "@%#*+=-:. "),
    DensitySet("Blocks", "█▓▒░ ."),
    DensitySet("Geometric Bold", "█▛▜▐▌▞▚▖▗▝▘ "),
    DensitySet("Serif Whisper", "@#MW&8%BQ0OCL|!lI;:,.` "),
    DensitySet("Box Lines", "█▓▒░╬╫╪╩╨╧╤╥═║─│· "),
    DensitySet("Braille Dots", "⣿⣷⣶⣤⣠⣀⡀⠄⠂⠁ "),
    DensitySet(
        "Detailed",
        "$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\\|()1{}[]?-_+~<>i!lI;:,\"^`'. ",
    ),
    DensitySet("Emoji Pop", " .'`^,:;~-_=+*ox%#@🙂😀😃😄😁😆😎🤩🔥"),
)


ASCII_GRADIENT_DEFAULT = DENSITY_SETS[0]
CHAR_ASPECT_RATIO = 0.55


@dataclass(frozen=True)
class ColorScheme:
    name: str
    gradient: Sequence[tuple[float, Tuple[int, int, int]]] | None = None
    use_source_colors: bool = False


COLOR_SCHEMES: Sequence[ColorScheme] = (
    ColorScheme("Camera", use_source_colors=True),
    ColorScheme(
        "Sunburst",
        (
            (0.0, (10, 5, 40)),
            (0.2, (90, 20, 120)),
            (0.45, (255, 215, 80)),  # central tones pushed to yellow
            (0.7, (255, 140, 60)),
            (1.0, (255, 255, 255)),
        ),
    ),
    ColorScheme(
        "Aurora",
        (
            (0.0, (5, 20, 60)),
            (0.35, (0, 200, 150)),
            (0.5, (255, 255, 120)),  # bright yellowish core
            (0.75, (120, 180, 255)),
            (1.0, (255, 255, 255)),
        ),
    ),
    ColorScheme(
        "Vaporwave",
        (
            (0.0, (20, 30, 90)),
            (0.25, (40, 200, 200)),
            (0.55, (255, 110, 200)),
            (0.85, (255, 180, 255)),
            (1.0, (255, 240, 255)),
        ),
    ),
    ColorScheme(
        "CRT",
        (
            (0.0, (0, 10, 0)),
            (0.3, (0, 80, 0)),
            (0.6, (40, 180, 40)),
            (0.85, (180, 255, 180)),
            (1.0, (225, 255, 225)),
        ),
    ),
    ColorScheme(
        "Thermal",
        (
            (0.0, (10, 5, 60)),
            (0.25, (0, 90, 255)),
            (0.5, (255, 255, 0)),
            (0.75, (255, 120, 0)),
            (1.0, (255, 255, 255)),
        ),
    ),
)


@dataclass
class AppSettings:
    width: int
    fps_target: float
    density_index: int = 0
    color_mode: bool = False
    show_fps: bool = True
    paused: bool = False
    color_scheme_index: int = 0
    mirrored: bool = False
    glitch_jitter: bool = False
    noise_sparkle: bool = False
    drop_shadow: bool = False

    @property
    def density(self) -> DensitySet:
        return DENSITY_SETS[self.density_index]

    @property
    def color_scheme(self) -> ColorScheme:
        return COLOR_SCHEMES[self.color_scheme_index]


@dataclass
class InputConfig:
    use_video: bool = False
    camera_index: int = 0
    video_path: Path | None = None
    loop_video: bool = False


@dataclass
class AsciiFrame:
    lines: List[str]
    colors: np.ndarray | None  # RGB color matrix matching lines, None for grayscale


class CharSurfaceCache:
    """Cache rendered glyphs for fast blitting."""

    def __init__(self, font: pygame.font.Font):
        self.font = font
        self.cache: dict[str, pygame.Surface] = {}
        self.char_width = max(1, font.size("A")[0])
        self.line_height = font.get_linesize()

    def get(self, char: str) -> pygame.Surface:
        surface = self.cache.get(char)
        if surface is None:
            surface = self.font.render(char, True, (255, 255, 255))
            self.cache[char] = surface.convert_alpha()
        return surface

    def preload(self, chars: Sequence[str], progress_callback=None) -> None:
        total = len(chars)
        for index, char in enumerate(chars, start=1):
            self.get(char)
            if progress_callback:
                progress_callback(index, total)


class AsciiRenderer:
    def __init__(self, char_cache: CharSurfaceCache):
        self.cache = char_cache

    def convert(
        self,
        frame: np.ndarray,
        width: int,
        density: DensitySet,
        color_mode: bool,
        color_scheme: ColorScheme,
    ) -> AsciiFrame:
        resized = self._resize(frame, width)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

        normalized = gray / 255.0
        chars = density.characters
        num_chars = len(chars)
        indices = np.clip((normalized * (num_chars - 1)).astype(int), 0, num_chars - 1)
        lines = ["".join(chars[idx] for idx in row) for row in indices]

        colors = None
        if color_mode:
            if color_scheme.use_source_colors:
                colors = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            elif color_scheme.gradient:
                colors = gradient_colors(normalized, color_scheme)

        return AsciiFrame(lines=lines, colors=colors)

    def _resize(self, frame: np.ndarray, target_width: int) -> np.ndarray:
        target_width = int(max(MIN_WIDTH, min(MAX_WIDTH, target_width)))
        height, width = frame.shape[:2]
        scale = target_width / float(width)
        target_height = max(1, int(height * scale * CHAR_ASPECT_RATIO))
        return cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)


RNG = np.random.default_rng()


def _roll_line(text: str, shift: int) -> str:
    if not text or shift == 0:
        return text
    length = len(text)
    shift = shift % length
    if shift == 0:
        return text
    return text[-shift:] + text[:-shift]


def _apply_line_jitter(
    lines: Sequence[str],
    colors: np.ndarray | None,
    max_offset: int = 2,
) -> tuple[list[str], np.ndarray | None]:
    if not lines or max_offset <= 0:
        return list(lines), colors
    jittered_lines: list[str] = []
    new_colors = np.copy(colors) if colors is not None else None
    for row_idx, line in enumerate(lines):
        shift = int(RNG.integers(-max_offset, max_offset + 1))
        if shift == 0 or not line:
            jittered_lines.append(line)
            continue
        jittered_lines.append(_roll_line(line, shift))
        if new_colors is not None:
            new_colors[row_idx] = np.roll(new_colors[row_idx], shift, axis=0)
    return jittered_lines, new_colors


def _apply_noise_swaps(
    lines: Sequence[str],
    palette: str,
    swap_probability: float = 0.035,
) -> list[str]:
    if swap_probability <= 0 or not palette:
        return list(lines)
    palette_array = np.array(list(palette))
    noisy_lines: list[str] = []
    for line in lines:
        if not line:
            noisy_lines.append(line)
            continue
        chars = np.array(list(line))
        mask = RNG.random(len(chars)) < swap_probability
        if mask.any():
            replacements = RNG.choice(palette_array, size=mask.sum())
            chars[mask] = replacements
        noisy_lines.append("".join(chars.tolist()))
    return noisy_lines


def apply_style_passes(frame: AsciiFrame, settings: AppSettings) -> AsciiFrame:
    if not (settings.glitch_jitter or settings.noise_sparkle):
        return frame

    lines: Sequence[str] = frame.lines
    colors = frame.colors
    if settings.glitch_jitter:
        lines, colors = _apply_line_jitter(lines, colors)
    else:
        lines = list(lines)
    if settings.noise_sparkle:
        lines = _apply_noise_swaps(lines, settings.density.characters)
    return AsciiFrame(lines=list(lines), colors=colors)


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------
def clamp_width(value: int) -> int:
    return max(MIN_WIDTH, min(MAX_WIDTH, value))


def cycle_density(settings: AppSettings, direction: int) -> None:
    settings.density_index = (settings.density_index + direction) % len(DENSITY_SETS)


def cycle_color_scheme(settings: AppSettings, direction: int) -> None:
    settings.color_scheme_index = (settings.color_scheme_index + direction) % len(COLOR_SCHEMES)


def save_ascii_frame(lines: Sequence[str], output_dir: Path = Path("output")) -> Path | None:
    if not lines:
        return None
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = output_dir / f"ascii_frame_{timestamp}.txt"
    path.write_text("\n".join(lines))
    return path


def clamp_window_dimension(value: int) -> int:
    return max(MIN_WINDOW_DIMENSION, value)


def probe_camera_indices(max_index: int = CAMERA_SCAN_LIMIT) -> list[int]:
    """Return a list of camera indexes that we can open successfully."""
    available: list[int] = []
    for idx in range(max(0, max_index) + 1):
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            available.append(idx)
        cap.release()
    return available


def open_camera(preferred_index: int, max_fallback_index: int = CAMERA_SCAN_LIMIT) -> tuple[cv2.VideoCapture | None, int | None]:
    """Try to open the preferred camera, falling back to other detected devices."""

    def try_index(index: int) -> cv2.VideoCapture | None:
        cap_candidate = cv2.VideoCapture(index)
        if cap_candidate.isOpened():
            return cap_candidate
        cap_candidate.release()
        return None

    attempted: set[int] = set()
    cap = try_index(preferred_index)
    if cap:
        return cap, preferred_index
    attempted.add(preferred_index)

    for idx in probe_camera_indices(max_fallback_index):
        if idx in attempted:
            continue
        cap = try_index(idx)
        if cap:
            return cap, idx

    return None, None


def _build_gradient_lut(gradient: Sequence[tuple[float, Tuple[int, int, int]]]) -> np.ndarray:
    points = sorted((max(0.0, min(1.0, float(pos))), tuple(max(0, min(255, int(c))) for c in color)) for pos, color in gradient)
    if not points:
        raise ValueError("Gradient definition must contain at least one point.")
    if points[0][0] > 0.0:
        points.insert(0, (0.0, points[0][1]))
    if points[-1][0] < 1.0:
        points.append((1.0, points[-1][1]))

    lut = np.zeros((256, 3), dtype=np.uint8)
    prev_idx = int(points[0][0] * 255)
    prev_color = np.array(points[0][1], dtype=np.float32)
    lut[: prev_idx + 1] = prev_color

    for pos, color in points[1:]:
        idx = int(pos * 255)
        idx = max(prev_idx + 1, idx)
        span = idx - prev_idx
        start = prev_idx
        end = min(255, idx)
        target = np.array(color, dtype=np.float32)
        for i in range(start, end + 1):
            t = (i - start) / max(1, span)
            lut[i] = np.clip(prev_color + (target - prev_color) * t, 0, 255)
        prev_idx = end
        prev_color = target

    if prev_idx < 255:
        lut[prev_idx:] = prev_color
    return lut


GRADIENT_LUT_CACHE: dict[str, np.ndarray] = {}


def gradient_colors(normalized: np.ndarray, scheme: ColorScheme) -> np.ndarray:
    if not scheme.gradient:
        raise ValueError("Gradient colors requested but scheme has no gradient.")
    lut = GRADIENT_LUT_CACHE.get(scheme.name)
    if lut is None:
        lut = _build_gradient_lut(scheme.gradient)
        GRADIENT_LUT_CACHE[scheme.name] = lut
    indices = np.clip((normalized * 255).astype(np.uint8), 0, 255)
    return lut[indices]


def show_loading_message(screen: pygame.Surface, font: pygame.font.Font, message: str) -> None:
    screen.fill((0, 0, 0))
    text_surface = font.render(message, True, (200, 200, 200))
    text_rect = text_surface.get_rect(center=(screen.get_width() // 2, screen.get_height() // 2))
    screen.blit(text_surface, text_rect)
    pygame.display.flip()


def render_ascii_surface(
    screen: pygame.Surface,
    ascii_frame: AsciiFrame,
    cache: CharSurfaceCache,
    settings: AppSettings,
) -> None:
    if not ascii_frame.lines:
        return

    char_width = cache.char_width
    line_height = cache.line_height
    colors = ascii_frame.colors
    color_mode = settings.color_mode
    drop_shadow = settings.drop_shadow

    for row_idx, line in enumerate(ascii_frame.lines):
        y = row_idx * line_height
        for col_idx, char in enumerate(line):
            glyph = cache.get(char)
            x = col_idx * char_width
            if drop_shadow:
                shadow = glyph.copy()
                shadow.fill((0, 0, 0, 220), special_flags=pygame.BLEND_RGBA_MULT)
                screen.blit(shadow, (x + 2, y + 2))
            if color_mode and colors is not None:
                color = tuple(int(c) for c in colors[row_idx, col_idx])
                tinted = glyph.copy()
                tinted.fill(color + (255,), special_flags=pygame.BLEND_RGBA_MULT)
                screen.blit(tinted, (x, y))
            else:
                screen.blit(glyph, (x, y))


def render_overlay(
    screen: pygame.Surface,
    font: pygame.font.Font,
    settings: AppSettings,
    fps_value: float,
    status_message: str | None,
    input_label: str,
) -> None:
    color_label = (
        f"Color – {settings.color_scheme.name}"
        if settings.color_mode
        else "Mono"
    )
    style_flags: list[str] = []
    if settings.glitch_jitter:
        style_flags.append("Jitter")
    if settings.noise_sparkle:
        style_flags.append("Noise")
    if settings.drop_shadow:
        style_flags.append("Shadow")
    lines = [
        f"Input: {input_label}",
        f"Resolution: {settings.width} cols",
        f"Density: {settings.density.name}",
        f"Colors: {color_label}",
        f"State: {'Paused' if settings.paused else 'Live'}",
        f"Mirror: {'On' if settings.mirrored else 'Off'}",
        f"Style: {', '.join(style_flags) if style_flags else 'Clean'}",
    ]
    if settings.show_fps:
        lines.append(f"FPS: {fps_value:5.1f}")
    if status_message:
        lines.append(status_message)

    overlay_height = font.get_linesize() * len(lines) + 12
    overlay_width = max(font.size(line)[0] for line in lines) + 20
    overlay = pygame.Surface((overlay_width, overlay_height), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 150))

    for idx, line in enumerate(lines):
        text = font.render(line, True, (255, 255, 255))
        overlay.blit(text, (10, 6 + idx * font.get_linesize()))

    screen.blit(overlay, (10, 10))


def adjust_width_from_resize(event_width: int, cache: CharSurfaceCache) -> int:
    char_width = max(1, cache.char_width)
    new_width = max(MIN_WIDTH, event_width // char_width)
    return clamp_width(new_width)


def update_screen_size(
    screen: pygame.Surface,
    cache: CharSurfaceCache,
    ascii_frame: AsciiFrame,
) -> pygame.Surface:
    if not ascii_frame.lines:
        return screen
    target_width = cache.char_width * len(ascii_frame.lines[0])
    target_height = cache.line_height * len(ascii_frame.lines)
    if (
        screen.get_width() != target_width
        or screen.get_height() != target_height
    ):
        screen = pygame.display.set_mode((target_width, target_height), pygame.RESIZABLE)
    return screen


# ---------------------------------------------------------------------------
# Main application loop
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Polished ASCII art camera viewer.")
    parser.add_argument("--camera-index", type=int, default=0, help="Camera device index.")
    parser.add_argument("--width", type=int, default=100, help="Initial ASCII width (columns).")
    parser.add_argument("--fps", type=float, default=24.0, help="Target frames per second.")
    parser.add_argument("--font-size", type=int, default=18, help="Primary font size.")
    parser.add_argument("--font-name", type=str, default="Courier New", help="Preferred monospaced font.")
    parser.add_argument(
        "--window-width",
        type=int,
        default=800,
        help="Initial window width (pixels).",
    )
    parser.add_argument(
        "--window-height",
        type=int,
        default=600,
        help="Initial window height (pixels).",
    )
    parser.add_argument("--color", action="store_true", help="Start in color mode.")
    parser.add_argument(
        "--use-video",
        action="store_true",
        help="Use a video file instead of a live camera feed (defaults to assets/test.mp4).",
    )
    parser.add_argument(
        "--video-path",
        type=str,
        default=None,
        help="Path to a video file used when --use-video is provided.",
    )
    parser.add_argument(
        "--loop-video",
        action="store_true",
        help="Loop the video file when reaching the end (only applies to --use-video).",
    )
    parser.add_argument(
        "--color-scheme",
        type=str,
        default=COLOR_SCHEMES[0].name,
        choices=[scheme.name for scheme in COLOR_SCHEMES],
        help="Color scheme used when color mode is enabled.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Prompt for configuration values before starting the viewer.",
    )
    return parser.parse_args()


def init_font(name: str, size: int) -> pygame.font.Font:
    font = pygame.font.SysFont(name, size)
    if font is None:
        font = pygame.font.Font(pygame.font.get_default_font(), size)
    return font


def prompt_user_configuration(
    settings: AppSettings,
    window_size: tuple[int, int],
    input_config: InputConfig,
) -> tuple[AppSettings, tuple[int, int], InputConfig]:
    try:
        width = input(f"ASCII width in columns [{settings.width}]: ").strip()
        if width:
            settings.width = clamp_width(int(width))

        fps = input(f"Target FPS [{settings.fps_target}]: ").strip()
        if fps:
            fps_value = float(fps)
            if fps_value <= 0:
                raise ValueError
            settings.fps_target = fps_value

        color = input(f"Start in color mode (y/n) [{'y' if settings.color_mode else 'n'}]: ").strip().lower()
        if color in {"y", "yes", "true", "1"}:
            settings.color_mode = True
        elif color in {"n", "no", "false", "0"}:
            settings.color_mode = False

        window_w = input(f"Window width (px) [{window_size[0]}]: ").strip()
        if window_w:
            window_size = (clamp_window_dimension(int(window_w)), window_size[1])

        window_h = input(f"Window height (px) [{window_size[1]}]: ").strip()
        if window_h:
            window_size = (window_size[0], clamp_window_dimension(int(window_h)))

        video_choice = input(
            f"Use video file instead of camera (y/n) [{'y' if input_config.use_video else 'n'}]: "
        ).strip().lower()
        if video_choice in {"y", "yes", "true", "1"}:
            input_config.use_video = True
        elif video_choice in {"n", "no", "false", "0"}:
            input_config.use_video = False

        if input_config.use_video:
            default_video = input_config.video_path or Path("assets/test.mp4")
            video_prompt = input(f"Video path [{default_video}]: ").strip()
            if video_prompt:
                input_config.video_path = Path(video_prompt).expanduser()
            else:
                input_config.video_path = default_video
            loop_choice = input(
                f"Loop video playback (y/n) [{'y' if input_config.loop_video else 'n'}]: "
            ).strip().lower()
            if loop_choice in {"y", "yes", "true", "1"}:
                input_config.loop_video = True
            elif loop_choice in {"n", "no", "false", "0"}:
                input_config.loop_video = False
        else:
            detected = probe_camera_indices()
            if detected:
                print(f"Detected cameras: {', '.join(str(idx) for idx in detected)}")
            else:
                print("No cameras detected automatically. You can still choose a camera index to try.")
            camera = input(f"Camera index [{input_config.camera_index}]: ").strip()
            if camera:
                input_config.camera_index = int(camera)

        scheme_names = ", ".join(scheme.name for scheme in COLOR_SCHEMES)
        scheme = input(f"Color scheme [{settings.color_scheme.name}] ({scheme_names}): ").strip()
        if scheme:
            for idx, option in enumerate(COLOR_SCHEMES):
                if option.name.lower() == scheme.lower():
                    settings.color_scheme_index = idx
                    break

    except ValueError:
        print("Invalid input provided. Keeping previous configuration.")
    return settings, window_size, input_config


def main() -> int:
    args = parse_args()
    if args.fps <= 0:
        print("Error: FPS must be greater than zero.", file=sys.stderr)
        return 2

    color_scheme_index = next(
        (idx for idx, scheme in enumerate(COLOR_SCHEMES) if scheme.name == args.color_scheme),
        0,
    )
    settings = AppSettings(
        width=clamp_width(args.width),
        fps_target=args.fps,
        color_mode=args.color,
        color_scheme_index=color_scheme_index,
    )
    window_size = (
        clamp_window_dimension(args.window_width),
        clamp_window_dimension(args.window_height),
    )
    use_video_flag = args.use_video or bool(args.video_path)
    video_path = Path(args.video_path).expanduser() if args.video_path else None
    if use_video_flag and video_path is None:
        video_path = Path("assets/test.mp4")
    input_config = InputConfig(
        use_video=use_video_flag,
        camera_index=args.camera_index,
        video_path=video_path,
        loop_video=args.loop_video,
    )
    if args.interactive:
        settings, window_size, input_config = prompt_user_configuration(settings, window_size, input_config)

    pygame.init()
    pygame.font.init()
    primary_font = init_font(args.font_name, args.font_size)
    overlay_font = init_font("Courier New", max(14, args.font_size - 2))

    screen = pygame.display.set_mode(window_size, pygame.RESIZABLE)
    pygame.display.set_caption("video2ascii – polished app")

    show_loading_message(screen, overlay_font, "Loading glyph cache...")
    cache = CharSurfaceCache(primary_font)
    unique_chars = sorted({char for density in DENSITY_SETS for char in density.characters})
    cache.preload(unique_chars, lambda idx, total: show_loading_message(
        screen, overlay_font, f"Loading glyphs {idx * 100 // total}%"
    ))

    renderer = AsciiRenderer(cache)
    clock = pygame.time.Clock()

    show_loading_message(screen, overlay_font, "Opening capture source...")
    input_label = ""
    if input_config.use_video:
        if input_config.video_path is None:
            input_config.video_path = Path("assets/test.mp4")
        video_path = input_config.video_path
        if not video_path.exists():
            print(f"Error: Video path '{video_path}' does not exist.", file=sys.stderr)
            pygame.quit()
            return 1
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"Error: Unable to open video file '{video_path}'.", file=sys.stderr)
            pygame.quit()
            return 1
        input_label = f"Video: {video_path.name}"
    else:
        cap, actual_index = open_camera(input_config.camera_index)
        if cap is None or actual_index is None:
            print(
                f"Error: Unable to open camera index {input_config.camera_index} and no fallback cameras were detected.",
                file=sys.stderr,
            )
            pygame.quit()
            return 1
        if actual_index != input_config.camera_index:
            print(
                f"Warning: Unable to open camera index {input_config.camera_index}; using camera index {actual_index} instead.",
                file=sys.stderr,
            )
            input_config.camera_index = actual_index
        input_label = f"Camera #{input_config.camera_index}"

    last_ascii_frame: AsciiFrame | None = None
    status_message: str | None = None
    status_expires: float = 0.0

    running = True
    try:
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.VIDEORESIZE:
                    settings.width = adjust_width_from_resize(event.w, cache)
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_ESCAPE, pygame.K_q):
                        running = False
                    elif event.key in (pygame.K_PLUS, pygame.K_EQUALS):
                        settings.width = clamp_width(settings.width + WIDTH_STEP)
                        status_message = f"Resolution: {settings.width} cols"
                        status_expires = time.time() + STATUS_DURATION
                    elif event.key == pygame.K_MINUS:
                        settings.width = clamp_width(settings.width - WIDTH_STEP)
                        status_message = f"Resolution: {settings.width} cols"
                        status_expires = time.time() + STATUS_DURATION
                    elif event.key == pygame.K_c:
                        settings.color_mode = not settings.color_mode
                        label = f"Color – {settings.color_scheme.name}" if settings.color_mode else "Mono"
                        status_message = f"Colors: {label}"
                        status_expires = time.time() + STATUS_DURATION
                    elif event.key == pygame.K_d:
                        cycle_density(settings, 1)
                        status_message = f"Density: {settings.density.name}"
                        status_expires = time.time() + STATUS_DURATION
                    elif event.key == pygame.K_g:
                        cycle_color_scheme(settings, 1)
                        label = f"Color – {settings.color_scheme.name}" if settings.color_mode else "(inactive)"
                        status_message = f"Scheme: {label}"
                        status_expires = time.time() + STATUS_DURATION
                    elif event.key == pygame.K_j:
                        settings.glitch_jitter = not settings.glitch_jitter
                        status_message = f"Glitch: {'On' if settings.glitch_jitter else 'Off'}"
                        status_expires = time.time() + STATUS_DURATION
                    elif event.key == pygame.K_n:
                        settings.noise_sparkle = not settings.noise_sparkle
                        status_message = f"Noise: {'On' if settings.noise_sparkle else 'Off'}"
                        status_expires = time.time() + STATUS_DURATION
                    elif event.key == pygame.K_h:
                        settings.drop_shadow = not settings.drop_shadow
                        status_message = f"Shadow: {'On' if settings.drop_shadow else 'Off'}"
                        status_expires = time.time() + STATUS_DURATION
                    elif event.key == pygame.K_x and event.mod & pygame.KMOD_CTRL:
                        settings.mirrored = not settings.mirrored
                        status_message = f"Mirror: {'On' if settings.mirrored else 'Off'}"
                        status_expires = time.time() + STATUS_DURATION
                    elif event.key == pygame.K_f:
                        settings.show_fps = not settings.show_fps
                    elif event.key == pygame.K_p:
                        settings.paused = not settings.paused
                        status_message = "Paused" if settings.paused else "Live"
                        status_expires = time.time() + STATUS_DURATION
                    elif event.key == pygame.K_s and last_ascii_frame:
                        path = save_ascii_frame(last_ascii_frame.lines)
                        if path:
                            status_message = f"Saved {path.name}"
                            status_expires = time.time() + STATUS_DURATION

            if settings.paused:
                ascii_frame = last_ascii_frame
            else:
                success, frame = cap.read()
                if not success:
                    if input_config.use_video and input_config.loop_video:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    source_label = "video file" if input_config.use_video else "camera"
                    print(f"Warning: Unable to read frame from {source_label}.", file=sys.stderr)
                    break
                if settings.mirrored:
                    frame = cv2.flip(frame, 1)
                ascii_frame = renderer.convert(
                    frame,
                    settings.width,
                    settings.density,
                    settings.color_mode,
                    settings.color_scheme,
                )
                ascii_frame = apply_style_passes(ascii_frame, settings)
                last_ascii_frame = ascii_frame

            if ascii_frame is None or not ascii_frame.lines:
                clock.tick(settings.fps_target)
                continue

            screen = update_screen_size(screen, cache, ascii_frame)
            screen.fill((0, 0, 0))
            render_ascii_surface(screen, ascii_frame, cache, settings)

            now = time.time()
            if status_message and now > status_expires:
                status_message = None

            render_overlay(screen, overlay_font, settings, clock.get_fps(), status_message, input_label)

            pygame.display.flip()
            clock.tick(settings.fps_target)
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        pygame.quit()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
