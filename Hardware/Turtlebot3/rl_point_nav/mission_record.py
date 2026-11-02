"""Per-mission record of a hardware bridge run: log, driven path, planned routes, map image.

Everything is in the mission frame (metres from the start pose, +x ahead, +y left) and
needs only NumPy and the standard library, so it runs on the Pi. One folder per run:

    log.txt     every bridge log line
    path.csv    robot position after every action
    routes.json the first route to each checkpoint and every changed route
    map.png     what the LiDAR saw and what the robot did (see MAP_LEGEND)
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import struct
import time
import zlib
from typing import Optional, Sequence

import numpy as np

MAP_LEGEND = ("map.png: up = straight ahead at the start; grey grid = 0.5 m; black = obstacles seen; "
              "blue = planned routes (first and changed); red = driven path; green = start; orange = checkpoints")
METRES_PER_PIXEL = 0.005
MARGIN_M = 0.4
MAX_EXTENT_M = 8.0

WHITE, GRID, BLACK = (255, 255, 255), (225, 225, 225), (30, 30, 30)
BLUE, RED, GREEN, ORANGE = (70, 130, 230), (220, 40, 40), (30, 170, 60), (245, 150, 20)

Point = tuple[float, float]


class MissionRecorder:
    def __init__(self, folder: Path) -> None:
        self.folder = folder
        folder.mkdir(parents=True, exist_ok=True)
        self._log = open(folder / "log.txt", "a", buffering=1)
        self.path: list[Point] = []
        self.routes: list[dict] = []
        self.targets: list[Point] = []

    def log(self, message: str) -> None:
        self._log.write(f"{time.strftime('%H:%M:%S')} {message}\n")

    def save(self, obstacles: Sequence[Point], obstacle_size_m: float) -> None:
        """Write path.csv, routes.json and map.png (called after every command, so a
        stopped run still leaves its record)."""
        with open(self.folder / "path.csv", "w") as handle:
            handle.write("step,x_m,y_m\n")
            handle.writelines(f"{i},{x:.3f},{y:.3f}\n" for i, (x, y) in enumerate(self.path))
        (self.folder / "routes.json").write_text(json.dumps(
            {"targets": self.targets, "routes": self.routes}, indent=1))
        write_png(self.folder / "map.png", render(obstacles, obstacle_size_m, self.path,
                                                  [r["points"] for r in self.routes], self.targets))

    def close(self) -> None:
        self._log.close()


def render(obstacles: Sequence[Point], obstacle_size_m: float, path: Sequence[Point],
           routes: Sequence[Sequence[Point]], targets: Sequence[Point]) -> np.ndarray:
    points = [(0.0, 0.0), *path, *targets, *(p for r in routes for p in r)]
    near = [p for p in obstacles if min(math.dist(p, q) for q in points) < 1.0]
    xs = [p[0] for p in points + near]
    ys = [p[1] for p in points + near]
    x_lo, x_hi = min(xs) - MARGIN_M, max(xs) + MARGIN_M
    y_lo, y_hi = min(ys) - MARGIN_M, max(ys) + MARGIN_M
    x_lo, y_lo = max(x_lo, x_hi - MAX_EXTENT_M), max(y_lo, y_hi - MAX_EXTENT_M)
    rows = int((x_hi - x_lo) / METRES_PER_PIXEL) + 1
    cols = int((y_hi - y_lo) / METRES_PER_PIXEL) + 1
    image = np.full((rows, cols, 3), WHITE, dtype=np.uint8)

    def pixel(p: Point) -> tuple[int, int]:   # up = +x (ahead), left = +y (left)
        return int((x_hi - p[0]) / METRES_PER_PIXEL), int((y_hi - p[1]) / METRES_PER_PIXEL)

    for k in range(math.ceil(x_lo / 0.5), math.floor(x_hi / 0.5) + 1):
        image[pixel((k * 0.5, 0.0))[0], :] = GRID
    for k in range(math.ceil(y_lo / 0.5), math.floor(y_hi / 0.5) + 1):
        image[:, pixel((0.0, k * 0.5))[1]] = GRID
    half = max(1, int(obstacle_size_m / METRES_PER_PIXEL / 2))
    for p in near:
        _dot(image, pixel(p), half, BLACK)
    for route in routes:
        _polyline(image, [pixel(p) for p in route], BLUE, 1)
    _polyline(image, [pixel(p) for p in path], RED, 2)
    _dot(image, pixel((0.0, 0.0)), 6, GREEN)
    for target in targets:
        _dot(image, pixel(target), 6, ORANGE)
    return image


def _dot(image: np.ndarray, center: tuple[int, int], half: int, color) -> None:
    r, c = center
    image[max(0, r - half):r + half + 1, max(0, c - half):c + half + 1] = color


def _polyline(image: np.ndarray, pixels: Sequence[tuple[int, int]], color, half: int) -> None:
    for (r0, c0), (r1, c1) in zip(pixels, pixels[1:]):
        steps = max(abs(r1 - r0), abs(c1 - c0), 1)
        for t in range(steps + 1):
            _dot(image, (r0 + (r1 - r0) * t // steps, c0 + (c1 - c0) * t // steps), half, color)


def write_png(filename: Path, image: np.ndarray) -> None:
    """Minimal RGB PNG writer (no Pillow/matplotlib on the Pi)."""
    height, width, _ = image.shape
    raw = b"".join(b"\x00" + image[row].tobytes() for row in range(height))

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    with open(filename, "wb") as handle:
        handle.write(b"\x89PNG\r\n\x1a\n")
        handle.write(chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)))
        handle.write(chunk(b"IDAT", zlib.compress(raw, 6)))
        handle.write(chunk(b"IEND", b""))


def new_folder(root: Path, plan: Optional[Path]) -> Path:
    name = time.strftime("%Y%m%d_%H%M%S") + ("_" + plan.stem if plan is not None else "_server")
    return root / name
