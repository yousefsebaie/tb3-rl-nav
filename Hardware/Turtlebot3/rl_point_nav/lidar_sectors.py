"""Convert a real TurtleBot3 /scan into the policy's LiDAR sector values.

Pure Python (no ROS, no NumPy) so it can be unit-tested anywhere.

Each real ray is converted to a point in base_link, then re-measured from a virtual
sensor at the mount the policy was trained with, so an obstacle reads the distance
the policy saw in training.

* V2 (profile "v2") trained with the LiDAR +0.10 m AHEAD of base_link.
* V3 / V4 / MAPPO V5 (profiles "v3", "v5") trained with the stock Burger mount,
  base_scan -0.032 m BEHIND base_link (URDF xyz = -0.032 0 0.172).
* Our two Burgers drive caster-first (docs/STATUS.md), so on them the LDS sits
  ~0.032 m AHEAD of the wheel axle: REAL_MOUNT_M. Measured once with a ruler.

Sectors, angles in the (virtual) sensor frame, 0 deg = forward, CCW positive,
inclusive at both ends. They may overlap (V5's narrow_front lies inside front): each
sector takes the minimum over ALL of its own rays, like the simulator's sector masks.
    v2: front [-30, 30]  left [30, 120]  right [-120, -30]
    v3: front [-18, 18]  left [18, 120]  right [-120, -18]
    v5: the v3 three, then narrow_front [-6, 6], front_left [18, 45], front_right [-45, -18],
        left_side [45, 100], right_side [-100, -45]   (docs/MAPPO_HARDWARE_HANDOFF.md 4.2)
clearance = clip(min range / 1.0 m, 0, 1)   (anything beyond 1 m reads 1.0)

Edge cases, matching the simulator's rules as closely as the real sensor allows:
* Point closer than the sim near range (0.10 m) to the virtual sensor, or
  behind it while the real ray points into the front sector -> sector is BLOCKED (0.0).
  The sim could not measure inside 0.10 m and reported such sectors as 0.0.
* No usable rays in a sector:
    - some ray reported inf / > range_max  -> open space (1.0)
    - only 0 / < range_min (LDS "no return") -> ambiguous: 0.0 if that sector's
      previous clearance was already close (< 0.25 m), else open (1.0).
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

REAL_MOUNT_M = (0.032, 0.0)       # our caster-first Burgers: LDS centre ahead of the wheel axle
STOCK_MOUNT_M = (-0.032, 0.0)     # stock Burger base_scan in base_link = V3/V4/V5 training mount
SIM_MOUNT_M = (0.10, 0.0)         # Isaac V2 training LiDAR offset from base_link
SIM_NEAR_RANGE_M = 0.10
CLEARANCE_NORM_M = 1.0
AMBIGUOUS_CLOSE_M = 0.25

V3_SECTORS_DEG = {"front": (-18.0, 18.0), "left": (18.0, 120.0), "right": (-120.0, -18.0)}
V5_EXTRA_SECTORS_DEG = {
    "narrow_front": (-6.0, 6.0), "front_left": (18.0, 45.0), "front_right": (-45.0, -18.0),
    "left_side": (45.0, 100.0), "right_side": (-100.0, -45.0),
}

SECTOR_NAMES = ("front", "left", "right")
SECTORS_DEG = {"front": (-30.0, 30.0), "left": (30.0, 120.0), "right": (-120.0, -30.0)}

# Per training profile (profile.txt next to the model). Sector order = policy input order.
PROFILES = {
    "v2": {"sim_mount_m": SIM_MOUNT_M, "sectors_deg": SECTORS_DEG},
    "v3": {"sim_mount_m": STOCK_MOUNT_M, "sectors_deg": V3_SECTORS_DEG},
    "v5": {"sim_mount_m": STOCK_MOUNT_M, "sectors_deg": {**V3_SECTORS_DEG, **V5_EXTRA_SECTORS_DEG}},
}


def sector_names(profile: str) -> tuple[str, ...]:
    return tuple(PROFILES[profile]["sectors_deg"])


def _sectors_of(angle_deg: float, sectors_deg: dict) -> list[str]:
    return [name for name, (lo, hi) in sectors_deg.items() if lo <= angle_deg <= hi]


def _wrap_deg(angle_deg: float) -> float:
    return (angle_deg + 180.0) % 360.0 - 180.0


def sim_matched_sectors(
    ranges: Sequence[float],
    angle_min: float,
    angle_increment: float,
    range_min: float,
    range_max: float,
    previous_m: Optional[dict] = None,
    profile: str = "v2",
) -> tuple[list[float], dict]:
    """Return (sector values normalized to 0..1 in the profile's order, diagnostics in meters).

    ``profile`` selects the training geometry (see PROFILES): virtual sensor mount
    and sector bounds. Also reports ``nearest_body_m``: the closest valid point to
    the robot centre (base_link) within the forward half, used by the hardware env
    as a contact guard.
    """
    if profile not in PROFILES:
        raise ValueError(f"unknown LiDAR profile {profile!r}; expected one of {sorted(PROFILES)}")
    sim_mount_m = PROFILES[profile]["sim_mount_m"]
    sectors_deg = PROFILES[profile]["sectors_deg"]
    names = tuple(sectors_deg)
    best = {name: math.inf for name in names}
    blocked = {name: False for name in names}
    saw_open = {name: False for name in names}
    valid_rays = {name: 0 for name in names}
    nearest_body_m = math.inf

    for i, r in enumerate(ranges):
        raw_angle = angle_min + i * angle_increment
        raw_sectors = _sectors_of(_wrap_deg(math.degrees(raw_angle)), sectors_deg)
        if math.isnan(r):
            continue
        if math.isinf(r) or r > range_max:
            for name in raw_sectors:
                saw_open[name] = True
            continue
        if r <= 0.0 or r < range_min:
            continue  # LDS "no return": handled by the empty-sector rule below

        # Point in base_link, then relative to the virtual (simulated) sensor.
        bx = REAL_MOUNT_M[0] + r * math.cos(raw_angle)
        by = REAL_MOUNT_M[1] + r * math.sin(raw_angle)
        if bx > 0.0:
            nearest_body_m = min(nearest_body_m, math.hypot(bx, by))
        px, py = bx - sim_mount_m[0], by - sim_mount_m[1]
        rng = math.hypot(px, py)
        sectors = _sectors_of(math.degrees(math.atan2(py, px)), sectors_deg)

        too_close = rng < SIM_NEAR_RANGE_M or (px <= 0.0 and "front" in raw_sectors)
        if too_close:
            for name in raw_sectors or sectors:
                blocked[name] = True
            continue
        for name in sectors:
            valid_rays[name] += 1
            best[name] = min(best[name], rng)

    previous_m = previous_m or {}
    clearance_m = {}
    for name in names:
        if blocked[name]:
            clearance_m[name] = 0.0
        elif valid_rays[name] > 0:
            clearance_m[name] = best[name]
        elif saw_open[name]:
            clearance_m[name] = math.inf
        else:
            prev = previous_m.get(name, math.inf)
            clearance_m[name] = 0.0 if prev < AMBIGUOUS_CLOSE_M else math.inf

    normalized = [min(max(clearance_m[n] / CLEARANCE_NORM_M, 0.0), 1.0) for n in names]
    diagnostics = {
        "clearance_m": clearance_m,
        "valid_rays": valid_rays,
        "blocked": blocked,
        "nearest_body_m": nearest_body_m,
    }
    return normalized, diagnostics
