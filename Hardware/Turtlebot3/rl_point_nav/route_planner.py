"""Obstacle map from LiDAR scans + A* routes for the RL policy (pure NumPy, no ROS).

The V3 policy only reacts to three sector distances and has no memory, so it cannot
plan around obstacles. This module is the planner above it: every scan is drawn into an
occupancy grid in the odometry frame, A* finds a route to the checkpoint that keeps a
margin from everything seen so far, and the policy drives short legs along that route.
Re-planning after every leg makes the route change as soon as a new obstacle is seen.

Grid rules:
* each LiDAR return adds evidence to its cell; cells a beam passes through lose evidence,
  so obstacles that move away fade out;
* a cell is an obstacle once its evidence reaches OCCUPIED_AT;
* obstacles are inflated by the robot radius plus a margin, and cells near an obstacle
  cost more, so routes keep away from obstacles when there is room.
"""

from __future__ import annotations

import heapq
import math
from typing import Optional, Sequence

import numpy as np

LIDAR_MOUNT_M = (-0.032, 0.0)   # TurtleBot3 Burger base_scan in base_link
RESOLUTION_M = 0.05
HALF_SIZE_M = 4.0               # map covers +-4 m around where it was created
MAX_RANGE_M = 2.5               # ignore far returns (noisy, irrelevant for short legs)
HIT, MISS, CAP = 2, 1, 6        # evidence per return / per pass-through, and its ceiling
OCCUPIED_AT = 2
ROBOT_RADIUS_M = 0.105
MARGIN_M = 0.10                 # hard clearance kept on top of the robot radius
SOFT_MARGIN_M = 0.20            # extra band around that with a higher step cost
SOFT_COST = 4.0


class RoutePlanner:
    def __init__(self, center_xy: tuple[float, float]) -> None:
        self.cx, self.cy = center_xy
        self.n = int(round(2 * HALF_SIZE_M / RESOLUTION_M))
        self.evidence = np.zeros((self.n, self.n), dtype=np.int8)
        self._inflate_cells = int(math.ceil((ROBOT_RADIUS_M + MARGIN_M) / RESOLUTION_M))
        self._soft_cells = self._inflate_cells + int(math.ceil(SOFT_MARGIN_M / RESOLUTION_M))

    # ------------------------------------------------------------ frames
    def to_cell(self, x: float, y: float) -> tuple[int, int]:
        return (int(math.floor((x - self.cx + HALF_SIZE_M) / RESOLUTION_M)),
                int(math.floor((y - self.cy + HALF_SIZE_M) / RESOLUTION_M)))

    def to_xy(self, cell: tuple[int, int]) -> tuple[float, float]:
        return (self.cx - HALF_SIZE_M + (cell[0] + 0.5) * RESOLUTION_M,
                self.cy - HALF_SIZE_M + (cell[1] + 0.5) * RESOLUTION_M)

    def inside(self, cell: tuple[int, int]) -> bool:
        return 0 <= cell[0] < self.n and 0 <= cell[1] < self.n

    # ------------------------------------------------------------ mapping
    def add_scan(self, pose: tuple[float, float, float], ranges: Sequence[float], angle_min: float,
                 angle_increment: float, range_min: float, range_max: float) -> None:
        """Draw one LaserScan, taken at robot pose (x, y, yaw) in the odometry frame."""
        x, y, yaw = pose
        c, s = math.cos(yaw), math.sin(yaw)
        sx, sy = x + c * LIDAR_MOUNT_M[0] - s * LIDAR_MOUNT_M[1], y + s * LIDAR_MOUNT_M[0] + c * LIDAR_MOUNT_M[1]
        origin = self.to_cell(sx, sy)
        for i, r in enumerate(ranges):
            if math.isnan(r) or r <= 0.0 or r < range_min:
                continue  # no return: says nothing about free space
            hit = r <= min(range_max, MAX_RANGE_M)
            length = min(r, MAX_RANGE_M)
            a = yaw + angle_min + i * angle_increment
            end = self.to_cell(sx + length * math.cos(a), sy + length * math.sin(a))
            cells = _line(origin, end)
            for cell in cells[:-1] if hit else cells:
                if self.inside(cell) and self.evidence[cell] > 0:
                    self.evidence[cell] -= MISS
            if hit and self.inside(end):
                self.evidence[end] = min(CAP, self.evidence[end] + HIT)

    def obstacles(self) -> np.ndarray:
        return self.evidence >= OCCUPIED_AT

    # ------------------------------------------------------------ planning
    def _cost_map(self) -> tuple[np.ndarray, np.ndarray]:
        occupied = self.obstacles()
        blocked = _dilate(occupied, self._inflate_cells)
        near = _dilate(occupied, self._soft_cells) & ~blocked
        return blocked, near

    def plan(self, start_xy: tuple[float, float], goal_xy: tuple[float, float]) -> Optional[list[tuple[float, float]]]:
        """A* route (list of (x, y), start first) or None when no route exists."""
        blocked, near = self._cost_map()
        start, goal = self.to_cell(*start_xy), self.to_cell(*goal_xy)
        if not (self.inside(start) and self.inside(goal)):
            return None
        # The robot may already sit inside the inflated band (it drove close): let it leave.
        blocked[_disc(start, self._inflate_cells, self.n)] = False
        if blocked[goal]:
            goal = self._nearest_free(goal, blocked)
            if goal is None:
                return None
        came: dict = {start: None}
        cost = {start: 0.0}
        queue = [(_dist(start, goal), start)]
        steps = ((1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
                 (1, 1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (-1, -1, 1.414))
        while queue:
            _, cell = heapq.heappop(queue)
            if cell == goal:
                break
            for dx, dy, length in steps:
                nxt = (cell[0] + dx, cell[1] + dy)
                if not self.inside(nxt) or blocked[nxt]:
                    continue
                new = cost[cell] + length * (SOFT_COST if near[nxt] else 1.0)
                if new < cost.get(nxt, math.inf):
                    cost[nxt], came[nxt] = new, cell
                    heapq.heappush(queue, (new + _dist(nxt, goal), nxt))
        if goal not in came:
            return None
        path, cell = [], goal
        while cell is not None:
            path.append(self.to_xy(cell))
            cell = came[cell]
        path.reverse()
        path[0] = start_xy
        return path

    def _nearest_free(self, cell, blocked) -> Optional[tuple[int, int]]:
        free = np.argwhere(~blocked)
        if free.size == 0:
            return None
        best = free[np.argmin(((free - np.array(cell)) ** 2).sum(axis=1))]
        return int(best[0]), int(best[1])


def lookahead(path: Sequence[tuple[float, float]], distance_m: float) -> tuple[float, float]:
    """The point ``distance_m`` along the route (or its end)."""
    travelled = 0.0
    for (ax, ay), (bx, by) in zip(path, path[1:]):
        segment = math.hypot(bx - ax, by - ay)
        if travelled + segment >= distance_m and segment > 0.0:
            t = (distance_m - travelled) / segment
            return ax + t * (bx - ax), ay + t * (by - ay)
        travelled += segment
    return path[-1]


def route_length(path: Sequence[tuple[float, float]]) -> float:
    return sum(math.hypot(bx - ax, by - ay) for (ax, ay), (bx, by) in zip(path, path[1:]))


def _dist(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _line(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    """Bresenham cells from a to b, both included."""
    (x0, y0), (x1, y1) = a, b
    dx, dy = abs(x1 - x0), -abs(y1 - y0)
    sx, sy = (1 if x0 < x1 else -1), (1 if y0 < y1 else -1)
    err, cells = dx + dy, []
    while True:
        cells.append((x0, y0))
        if (x0, y0) == (x1, y1):
            return cells
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy


def _disc(center: tuple[int, int], radius: int, n: int) -> tuple[slice, slice]:
    return (slice(max(0, center[0] - radius), min(n, center[0] + radius + 1)),
            slice(max(0, center[1] - radius), min(n, center[1] + radius + 1)))


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    """Grow True cells by a disc of ``radius`` cells."""
    out = mask.copy()
    if not mask.any():
        return out
    padded = np.pad(mask, radius)
    n0, n1 = mask.shape
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            if dx * dx + dy * dy <= radius * radius:
                out |= padded[radius + dx:radius + dx + n0, radius + dy:radius + dy + n1]
    return out
