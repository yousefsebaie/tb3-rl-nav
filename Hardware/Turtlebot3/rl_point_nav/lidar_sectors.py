"""Convert a real TurtleBot3 /scan into the V2 policy's three LiDAR sector values.

Pure Python (no ROS, no NumPy) so it can be unit-tested anywhere.

In Isaac Sim the LiDAR sat +0.10 m AHEAD of base_link. On the real Burger the
LDS sits -0.032 m BEHIND it (URDF base_scan xyz = -0.032 0 0.172). Each real
ray is therefore converted to a point in base_link, then re-measured from a
virtual sensor at the simulated mount, so a front obstacle reads the same
distance the policy saw during training (otherwise ~13 cm too far).

Sectors, angles in the (virtual) sensor frame, 0 deg = forward, CCW positive:
    front [-30, 30]   left [30, 120]   right [-120, -30]   (rear rays unused)
clearance = clip(min range / 1.0 m, 0, 1)   (anything beyond 1 m reads 1.0)

Edge cases, matching the simulator's rules as closely as the real sensor allows:
* Point closer than the sim near range (0.10 m) to the virtual sensor, or
  behind it while the real ray points forward -> sector is BLOCKED (0.0).
  The sim could not measure inside 0.10 m and reported such sectors as 0.0.
* No usable rays in a sector:
    - some ray reported inf / > range_max  -> open space (1.0)
    - only 0 / < range_min (LDS "no return") -> ambiguous: 0.0 if that sector's
      previous clearance was already close (< 0.25 m), else open (1.0).
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

REAL_MOUNT_M = (-0.032, 0.0)   # TurtleBot3 Burger base_scan in base_link
SIM_MOUNT_M = (0.10, 0.0)      # Isaac V2 training LiDAR offset from base_link
SIM_NEAR_RANGE_M = 0.10
CLEARANCE_NORM_M = 1.0
AMBIGUOUS_CLOSE_M = 0.25

SECTOR_NAMES = ("front", "left", "right")
SECTORS_DEG = {"front": (-30.0, 30.0), "left": (30.0, 120.0), "right": (-120.0, -30.0)}


def _sector_of(angle_deg: float) -> Optional[str]:
    for name in SECTOR_NAMES:
        lo, hi = SECTORS_DEG[name]
        if lo <= angle_deg <= hi:
            return name
    return None


def _wrap_deg(angle_deg: float) -> float:
    return (angle_deg + 180.0) % 360.0 - 180.0


def sim_matched_sectors(
    ranges: Sequence[float],
    angle_min: float,
    angle_increment: float,
    range_min: float,
    range_max: float,
    previous_m: Optional[dict] = None,
) -> tuple[list[float], dict]:
    """Return ([front, left, right] normalized to 0..1, diagnostics in meters).

    Also reports ``nearest_body_m``: the closest valid point to the robot centre
    (base_link) within the forward half, used by the hardware env as a contact guard.
    """
    best = {name: math.inf for name in SECTOR_NAMES}
    blocked = {name: False for name in SECTOR_NAMES}
    saw_open = {name: False for name in SECTOR_NAMES}
    valid_rays = {name: 0 for name in SECTOR_NAMES}
    nearest_body_m = math.inf

    for i, r in enumerate(ranges):
        raw_angle = angle_min + i * angle_increment
        raw_sector = _sector_of(_wrap_deg(math.degrees(raw_angle)))
        if math.isnan(r):
            continue
        if math.isinf(r) or r > range_max:
            if raw_sector is not None:
                saw_open[raw_sector] = True
            continue
        if r <= 0.0 or r < range_min:
            continue  # LDS "no return": handled by the empty-sector rule below

        # Point in base_link, then relative to the virtual (simulated) sensor.
        bx = REAL_MOUNT_M[0] + r * math.cos(raw_angle)
        by = REAL_MOUNT_M[1] + r * math.sin(raw_angle)
        if bx > 0.0:
            nearest_body_m = min(nearest_body_m, math.hypot(bx, by))
        px, py = bx - SIM_MOUNT_M[0], by - SIM_MOUNT_M[1]
        rng = math.hypot(px, py)
        sector = _sector_of(math.degrees(math.atan2(py, px)))

        too_close = rng < SIM_NEAR_RANGE_M or (px <= 0.0 and raw_sector == "front")
        if too_close:
            target = raw_sector if raw_sector is not None else sector
            if target is not None:
                blocked[target] = True
            continue
        if sector is not None:
            valid_rays[sector] += 1
            best[sector] = min(best[sector], rng)

    previous_m = previous_m or {}
    clearance_m = {}
    for name in SECTOR_NAMES:
        if blocked[name]:
            clearance_m[name] = 0.0
        elif valid_rays[name] > 0:
            clearance_m[name] = best[name]
        elif saw_open[name]:
            clearance_m[name] = math.inf
        else:
            prev = previous_m.get(name, math.inf)
            clearance_m[name] = 0.0 if prev < AMBIGUOUS_CLOSE_M else math.inf

    normalized = [min(max(clearance_m[n] / CLEARANCE_NORM_M, 0.0), 1.0) for n in SECTOR_NAMES]
    diagnostics = {
        "clearance_m": clearance_m,
        "valid_rays": valid_rays,
        "blocked": blocked,
        "nearest_body_m": nearest_body_m,
    }
    return normalized, diagnostics
