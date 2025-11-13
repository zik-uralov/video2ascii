# video2ascii

`video2ascii` is a small playground for exploring how live camera or video frames can be translated into readable ASCII art. It started as a command-line experiment and now includes progressively richer viewers: a basic image converter, a terminal stream, an early Pygame renderer, and a polished desktop app with colors, shaders, and quality-of-life touches such as saving frames.

## Features
- Convert any still image into ASCII art and save it to `output/frame.txt` or another text file.
- Stream your webcam straight into the terminal as ASCII characters, optionally alongside the OpenCV preview window.
- Render a live feed inside a resizable Pygame window with smooth font rendering.
- Launch a full-featured viewer (`src/phase4_polished_app.py`) with:
  - Multiple density presets, gradients, and optional full-color rendering.
  - Live camera input or pluggable video files with looped playback.
  - Interactive controls for resolution, glitch/noise effects, mirroring, FPS overlay, pause, and saving snapshots.
  - Interactive startup wizard (`--interactive`) that walks you through all configuration choices.

## Requirements
- Python 3.10+ (tested on CPython)
- pip and a working compiler toolchain for `opencv-python`
- Dependencies listed in `requirements.txt`:
  - `opencv-python` – video/image capture and preprocessing
  - `numpy` – vectorized ASCII mapping and extra effects
  - `pygame` – rich-window rendering

## Setup
1. (Recommended) Create a virtual environment:
   ```bash
   python -m venv .venv && source .venv/bin/activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Optional: place your own media in `assets/` or point the scripts to alternate paths via CLI flags.

## Usage

### 1. Convert a single image
```
python image_to_ascii.py assets/test.jpg --width 120 --output output/test_frame.txt
```
Arguments:
- `image_path` (positional) – file to convert (defaults to `assets/test.jpg`).
- `--width` – ASCII output width; the script auto-calculates height using the character aspect ratio.
- `--output` – path to the `.txt` file created (folders are created automatically).

### 2. Stream ASCII to the terminal (Phase 2)
```
python -m src.phase2_terminal_video --camera-index 0 --width 100 --fps 18 --no-preview
```
Key options:
- `--camera-index` – OpenCV capture index (start at 0, bump to try other devices).
- `--width` – number of columns for ASCII output.
- `--fps` – target refresh rate; the loop sleeps to maintain this pace.
- `--no-preview` – hide the OpenCV preview window for a terminal-only vibe.

Press `Ctrl+C` to stop; the terminal is cleared between frames for readability.

### 3. Basic Pygame viewer (Phase 3)
```
python -m src.phase3_pygame_display --camera-index 0 --width 110 --fps 20 --font-size 18
```
This version renders ASCII art with a monospace font inside a dedicated Pygame window. Resize is automatic and you can quit with `Esc` or `Q`.

### 4. Polished desktop app (Phase 4)
```
python -m src.phase4_polished_app \
  --camera-index 0 \
  --width 140 \
  --fps 24 \
  --font-size 24 \
  --color \
  --color-scheme "Sunburst"
```
Highlights:
- Start with live camera input or use `--use-video`/`--video-path myclip.mp4` to play a file instead (defaults to `assets/test.mp4`).
- `--loop-video` keeps playback rolling when the file ends.
- `--window-width`/`--window-height` determine the initial resizable window; resizing automatically tweaks ASCII resolution.
- `--interactive` launches a quick questionnaire before the window opens, making it easy to mix-and-match settings without memorizing flags.

#### Keyboard controls (Phase 4)
| Key | Action |
| --- | --- |
| `Esc` or `Q` | Quit the viewer |
| `+` / `=` | Increase ASCII resolution (more columns) |
| `-` | Decrease ASCII resolution |
| `C` | Toggle color mode on/off |
| `G` | Cycle color schemes (when color mode is active) |
| `D` | Cycle ASCII density sets (changes character palette) |
| `F` | Toggle FPS overlay |
| `P` | Pause/resume the live feed |
| `Ctrl+X` | Mirror the feed horizontally |
| `J` | Toggle glitch/jitter effect |
| `N` | Toggle sparkle/noise substitution |
| `H` | Toggle drop shadow for ASCII glyphs |
| `S` | Save the current ASCII frame (`output/ascii_frame_YYYYMMDD_HHMMSS.txt`) |

Status messages appear briefly in the HUD whenever you change a setting so you know what just happened.

## Directory Layout
- `assets/` – sample media (replace with your own files if desired).
- `output/` – generated ASCII files (ignored by git); automatically created.
- `src/phase2_terminal_video.py` – terminal streaming prototype.
- `src/phase3_pygame_display.py` – first Pygame renderer.
- `src/phase4_polished_app.py` – full-featured application.
- `image_to_ascii.py` – single-image converter for quick tests.
- `requirements.txt` – dependency pin list.

## Next steps
- Drop in your own fonts by passing `--font-name` or editing the code to pre-load multiple choices.
- Expand the density sets or color gradients in `src/phase4_polished_app.py` to explore different looks.
- Integrate this renderer into OBS/streaming workflows by capturing the Pygame window.

Have fun turning the world into ASCII!
