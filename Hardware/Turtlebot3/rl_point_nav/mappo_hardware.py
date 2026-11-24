"""MAPPO V5 on the real Burgers: the teammate half of the 15 inputs (no ROS, no NumPy).

What the simulation gave each robot for free, rebuilt for two separate Pis
(docs/MAPPO_HARDWARE_HANDOFF.md sections 4 and 6.2):

* SharedFrame: odometry starts at (0, 0, 0) wherever a Burger was switched on. Both
  robots' start poses are measured on the floor in one mission frame, and each robot's
  odometry is placed there, so both report positions in the same frame.
* PoseLink: each robot sends its shared-frame (x, y, yaw, active) to the other ~10x a
  second as a small UDP JSON packet and keeps the newest one it received.
* teammate_inputs: role + the other robot's position in my own frame + its "active" flag,
  inputs 11..14 of V5, exactly as train_mappo.actor_input() builds them.
* merge_teammate_circle: the simulator hid robot bodies from the LiDAR and drew the
  teammate as a 0.105 m circle. The policy learned that shape, so it is merged into the
  real scan (minimum of real and virtual) before the sectors are computed.

Inputs 0..10 (goal distance and bearing, 8 LiDAR sectors) come from
RosPointNavLidarEnv with lidar_profile="v5" (lidar_sectors.py).
"""

from __future__ import annotations

import json
import math
import socket
import threading
import time
from typing import Callable, Optional, Sequence

from lidar_sectors import REAL_MOUNT_M, SIM_NEAR_RANGE_M, sector_names

TEAMMATE_RADIUS_M = 0.105        # sim: teammate footprint as seen by the LiDAR
TEAMMATE_SCALE_M = 2.0           # sim: relative position normalization
ROLE = {"TURTLEBOT3_1": 1.0, "TURTLEBOT3_2": 0.0}   # Burger 1 / Burger 2
FAR_TEAMMATE = (1.0, 1.0, 0.0)   # handoff step 3: teammate far ahead-left and idle -> single-robot behaviour
V5_INPUT_NAMES = (("d_norm", "sin_bearing", "cos_bearing") + sector_names("v5")
                  + ("role", "teammate_x", "teammate_y", "teammate_active"))
POSE_PORT = 47110
POSE_RATE_HZ = 10.0
POSE_MAX_AGE_S = 0.5             # older teammate poses are not trusted

Pose = tuple[float, float, float]   # x, y, yaw (m, m, rad)


def wrap_to_pi(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


class SharedFrame:
    """Odometry frame <-> the mission frame both robots share.

    ``start_shared`` is where the robot stands at mission start, measured on the floor in
    the mission frame; ``start_odom`` is its odometry pose at that moment.
    """

    def __init__(self, start_shared: Pose, start_odom: Pose) -> None:
        self.start_shared = start_shared
        self.start_odom = start_odom
        self._turn = wrap_to_pi(start_shared[2] - start_odom[2])

    def to_shared(self, pose: Pose) -> Pose:
        dx, dy = pose[0] - self.start_odom[0], pose[1] - self.start_odom[1]
        c, s = math.cos(self._turn), math.sin(self._turn)
        return (self.start_shared[0] + c * dx - s * dy,
                self.start_shared[1] + s * dx + c * dy,
                wrap_to_pi(pose[2] + self._turn))

    def to_odom(self, x: float, y: float) -> tuple[float, float]:
        dx, dy = x - self.start_shared[0], y - self.start_shared[1]
        c, s = math.cos(self._turn), math.sin(self._turn)
        return self.start_odom[0] + c * dx + s * dy, self.start_odom[1] - s * dx + c * dy


def teammate_local(me: Pose, other_xy: tuple[float, float]) -> tuple[float, float]:
    """The other robot's position in my frame, in metres: (ahead of me, to my left)."""
    dx, dy = other_xy[0] - me[0], other_xy[1] - me[1]
    c, s = math.cos(me[2]), math.sin(me[2])
    return c * dx + s * dy, -s * dx + c * dy


def teammate_inputs(me: Pose, other_xy: Optional[tuple[float, float]], other_active: bool,
                    role: float) -> list[float]:
    """V5 inputs 11..14: [role, teammate_x, teammate_y, teammate_active].

    ``other_xy=None`` gives FAR_TEAMMATE (handoff step 3: V5 should then drive like V4).
    """
    if other_xy is None:
        return [role, *FAR_TEAMMATE]
    ahead, left = teammate_local(me, other_xy)
    clip = lambda v: min(max(v / TEAMMATE_SCALE_M, -1.0), 1.0)   # noqa: E731
    return [role, clip(ahead), clip(left), 1.0 if other_active else 0.0]


def lidar_pose(robot: Pose, mount_m: tuple[float, float] = REAL_MOUNT_M) -> Pose:
    c, s = math.cos(robot[2]), math.sin(robot[2])
    return (robot[0] + c * mount_m[0] - s * mount_m[1], robot[1] + s * mount_m[0] + c * mount_m[1], robot[2])


def merge_teammate_circle(ranges: Sequence[float], angle_min: float, angle_increment: float,
                          range_min: float, range_max: float, sensor: Pose,
                          teammate_xy: tuple[float, float],
                          radius_m: float = TEAMMATE_RADIUS_M) -> list[float]:
    """Scan ranges with the teammate drawn in as a circle (sim _teammate_ranges()).

    ``sensor`` is the LiDAR pose and ``teammate_xy`` the other robot's centre, both in the
    shared frame. A ray keeps its real range when that is valid and closer.
    """
    merged = list(ranges)
    dx, dy = teammate_xy[0] - sensor[0], teammate_xy[1] - sensor[1]
    d_sq, r_sq = dx * dx + dy * dy, radius_m * radius_m
    if d_sq <= r_sq:
        return merged            # sensor inside the circle: nothing sensible to draw
    for i, real in enumerate(ranges):
        a = sensor[2] + angle_min + i * angle_increment
        along = math.cos(a) * dx + math.sin(a) * dy
        perp_sq = d_sq - along * along
        if along <= 0.0 or perp_sq > r_sq:
            continue
        hit = along - math.sqrt(r_sq - perp_sq)
        if hit < SIM_NEAR_RANGE_M:
            continue             # the sim LiDAR returned nothing inside its near range
        real_valid = math.isfinite(real) and range_min <= real <= range_max and real > 0.0
        merged[i] = min(real, hit) if real_valid else hit
    return merged


BODY_RADIUS_AT_SCAN_M = 0.035   # what a LiDAR hits of another Burger at its scan plane: the LDS turret


def nearest_ahead(ranges: Sequence[float], angle_min: float, angle_increment: float,
                  range_min: float, range_max: float, cone_deg: float = 30.0) -> Optional[tuple[float, float]]:
    """(range m, bearing deg) of the nearest object within +-cone_deg of ahead, None if nothing.

    The bearing is the middle of the rays within 3 cm of the nearest range, so a round
    object (the other Burger) gives the direction to its centre.
    """
    hits = []
    for i, r in enumerate(ranges):
        a = (math.degrees(angle_min + i * angle_increment) + 180.0) % 360.0 - 180.0
        if abs(a) <= cone_deg and math.isfinite(r) and range_min <= r <= range_max and r > 0.0:
            hits.append((r, a))
    if not hits:
        return None
    r_min = min(r for r, _ in hits)
    close = [a for r, a in hits if r <= r_min + 0.03]
    return r_min, sum(close) / len(close)


def head_on_starts(seen_by_1: tuple[float, float], seen_by_2: tuple[float, float],
                   mount_m: float = REAL_MOUNT_M[0]) -> tuple[Pose, Pose]:
    """Start poses of two Burgers that face each other, from what each LiDAR sees of the other.

    ``seen_by_k`` = (range m, bearing deg) from nearest_ahead() on robot k. Robot 1 defines the
    shared frame: its centre at (0, 0), facing +x. Returns (start of robot 1, start of robot 2).
    """
    (r1, b1), (r2, b2) = seen_by_1, seen_by_2
    t1, t2 = math.radians(b1), math.radians(b2)
    reach = 0.5 * (r1 + r2) + BODY_RADIUS_AT_SCAN_M   # my LDS centre -> the other's LDS centre
    l1 = (mount_m + reach * math.cos(t1), reach * math.sin(t1))   # robot 2's LDS centre, robot 1 frame
    yaw2 = wrap_to_pi(t1 + math.pi - t2)                               # robot 2 sees robot 1 at b2
    x2, y2 = l1[0] - mount_m * math.cos(yaw2), l1[1] - mount_m * math.sin(yaw2)
    return (0.0, 0.0, 0.0), (x2, y2, yaw2)


FORWARD, TURN_LEFT, TURN_RIGHT, STAY = 0, 1, 2, 3   # config.PointNavAction

# Follow-the-gap hand-over, train_mappo.gap_assist() (handoff 5.3). Ray counts are converted
# to degrees: the sim scan had 0.34 deg per ray, the LDS-02 has ~1.55.
GAP_STUCK_STEPS = 18
GAP_STUCK_MOVE_M = 0.06
GAP_MIN_GOAL_M = 0.3
GAP_COOLDOWN_STEPS = 25
GAP_HANDBACK_M = 1.2
GAP_HANDBACK_STEPS = 150
GAP_FACE_GOAL_DEG = 60.0
GAP_CONE_DEG = 75.0
GAP_FREE_BEYOND_M = 0.35
GAP_NO_RETURN_M = 5.0
GAP_MEDIAN_DEG = 1.7            # sim: 5 rays
GAP_MIN_WIDTH_DEG = 2.0         # sim: 6 rays
GAP_ON_TARGET_DEG = 8.0
GAP_EDGE_MARGIN_DEG = 10.0    # aim at least this far inside a free gap's edge
ALIGN_UNTIL_DEG = 15.0          # Goal Interface: turn in place until the goal is this close to ahead


def free_gap_heading(ranges: Sequence[float], angle_min: float, angle_increment: float,
                     goal_bearing_deg: float) -> Optional[float]:
    """Heading (deg, 0 = ahead, CCW +) through the free gap nearest the goal, None if there is none.

    sim free_gap_heading(): within +-75 deg, no return = far, median filter, a ray is free if it
    reaches 0.35 m beyond the closest ray in the cone, gaps narrower than ~2 deg are ignored.
    Unlike the sim, the heading is the goal bearing clamped into the gap (see below), not its centre.
    """
    inc_deg = abs(math.degrees(angle_increment))
    rays = []
    for i, r in enumerate(ranges):
        az = (math.degrees(angle_min + i * angle_increment) + 180.0) % 360.0 - 180.0
        rays.append((az, r if math.isfinite(r) and r > 0.0 else GAP_NO_RETURN_M))
    rays.sort()
    az = [a for a, _ in rays]
    r = [v for _, v in rays]
    k = max(3, 2 * int(round(GAP_MEDIAN_DEG / inc_deg / 2)) + 1)   # odd window, >= 3 to drop LDS dropouts
    half = k // 2
    r = [sorted(r[max(0, j - half):j + half + 1])[len(r[max(0, j - half):j + half + 1]) // 2] for j in range(len(r))]
    cone = [abs(a) <= GAP_CONE_DEG for a in az]
    if not any(cone):
        return None
    floor = min(v for v, c in zip(r, cone) if c)
    free = [c and v > floor + GAP_FREE_BEYOND_M for v, c in zip(r, cone)]
    min_rays = max(1, math.ceil(GAP_MIN_WIDTH_DEG / inc_deg))
    best, best_cost, j = None, math.inf, 0
    while j < len(az):
        if free[j]:
            start = j
            while j < len(az) and free[j]:
                j += 1
            if j - start >= min_rays:
                # Hardware change (2026-11-24): aim at the goal bearing clamped into the gap, not at the gap's
                # centre. With one wide free arc the centre can sit ~40 deg off a clear goal direction, and the
                # chosen arc flips sides after each turn: Burger 2 turned +-42 deg in place for 65 s.
                lo, hi = az[start], az[j - 1]
                margin = min(GAP_EDGE_MARGIN_DEG, 0.5 * (hi - lo))
                aim = min(max(goal_bearing_deg, lo + margin), hi - margin)
                cost = abs(aim - goal_bearing_deg)
                if cost < best_cost:
                    best, best_cost = aim, cost
        j += 1
    return best


class GapAssist:
    """One robot's follow-the-gap hand-over state (sim gap_assist(), per robot)."""

    def __init__(self) -> None:
        self.history: list[tuple[float, float]] = []
        self.mode: Optional[tuple[float, float, int]] = None   # (entry x, entry y, steps) while handed over
        self.cooldown = 0
        self.handovers = 0

    def reset(self) -> None:
        self.history.clear()
        self.mode = None

    def decide(self, policy_action: int, xy: tuple[float, float], goal_dist_m: float,
               goal_bearing_deg: float, gap_heading: Callable[[], Optional[float]]) -> tuple[int, str]:
        """Final action and why ("policy", "gap ...")."""
        self.history = (self.history + [xy])[-GAP_STUCK_STEPS:]
        self.cooldown = max(0, self.cooldown - 1)
        if self.mode is None:
            h = self.history
            stuck = (len(h) == GAP_STUCK_STEPS and math.dist(h[0], h[-1]) < GAP_STUCK_MOVE_M
                     and goal_dist_m > GAP_MIN_GOAL_M and self.cooldown == 0)
            if not stuck:
                return policy_action, "policy"
            self.mode = (xy[0], xy[1], 0)
            self.handovers += 1
        ex, ey, n = self.mode
        if n >= GAP_HANDBACK_STEPS or math.dist((ex, ey), xy) >= GAP_HANDBACK_M:
            self.mode, self.cooldown = None, GAP_COOLDOWN_STEPS
            self.history.clear()
            return policy_action, "gap handback"
        self.mode = (ex, ey, n + 1)
        if abs(goal_bearing_deg) > GAP_FACE_GOAL_DEG:
            return (TURN_LEFT if goal_bearing_deg > 0 else TURN_RIGHT), "gap face goal"
        target = gap_heading()
        if target is None:
            return policy_action, "gap none"
        if abs(target) > GAP_ON_TARGET_DEG:
            return (TURN_LEFT if target > 0 else TURN_RIGHT), f"gap turn {target:+.0f}"
        return FORWARD, f"gap forward {target:+.0f}"


class PoseLink:
    """Shared-frame pose exchange between the two Pis over UDP.

    Each robot listens on ``listen_port`` and sends ``own()`` (x, y, yaw, active) to
    ``peer_host:peer_port`` at ``rate_hz``. ``own`` may return None while the pose is
    not known yet; nothing is sent then.
    """

    def __init__(self, name: str, peer_host: str, own: Callable[[], Optional[tuple]],
                 listen_port: int = POSE_PORT, peer_port: int = POSE_PORT,
                 rate_hz: float = POSE_RATE_HZ) -> None:
        self.name = name
        self.peer = (peer_host, peer_port)
        self.own = own
        self.period_s = 1.0 / rate_hz
        self._lock = threading.Lock()
        self._latest: Optional[dict] = None
        self._arrival = 0.0
        self._stop = threading.Event()
        self.sent = self.received = 0
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.bind(("0.0.0.0", listen_port))
        self._sock.settimeout(0.2)
        self._threads = [threading.Thread(target=self._send_loop, daemon=True, name="pose-send"),
                         threading.Thread(target=self._recv_loop, daemon=True, name="pose-recv")]
        for t in self._threads:
            t.start()

    def _send_loop(self) -> None:
        seq = 0
        while not self._stop.is_set():
            pose = self.own()
            if pose is not None:
                x, y, yaw, active = pose
                seq += 1
                msg = {"robot": self.name, "seq": seq, "x": x, "y": y, "yaw": yaw, "active": bool(active)}
                try:
                    self._sock.sendto(json.dumps(msg).encode(), self.peer)
                    self.sent += 1
                except OSError:
                    pass         # peer not reachable yet; keep trying
            self._stop.wait(self.period_s)

    def _recv_loop(self) -> None:
        while not self._stop.is_set():
            try:
                data, _ = self._sock.recvfrom(1024)
                msg = json.loads(data)
            except (socket.timeout, OSError, ValueError):
                continue
            if msg.get("robot") == self.name:
                continue
            with self._lock:
                self._latest, self._arrival = msg, time.monotonic()
                self.received += 1

    def teammate(self, max_age_s: float = POSE_MAX_AGE_S) -> Optional[dict]:
        """Newest pose from the other robot, with ``age_s``; None if none or too old."""
        with self._lock:
            if self._latest is None:
                return None
            age = time.monotonic() - self._arrival
            if age > max_age_s:
                return None
            return {**self._latest, "age_s": age}

    def close(self) -> None:
        self._stop.set()
        for t in self._threads:
            t.join(timeout=1.0)
        self._sock.close()
