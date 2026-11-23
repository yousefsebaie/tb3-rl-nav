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
