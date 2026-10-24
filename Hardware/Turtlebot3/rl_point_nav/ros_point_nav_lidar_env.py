#!/usr/bin/env python3
"""Real-TurtleBot3 environment matching the Isaac training loop (V1B or V2).

Extends the odometry env (ros_point_nav_env.py) with what the simulator did
but the plain ROS env does not:

* /scan -> the same 3 LiDAR sector values as V2 training (lidar_sectors.py,
  including the sim-vs-real LiDAR mount correction);
* a zero-velocity settle (~0.13 s = the sim's 8 settle frames at 60 Hz) after
  every action, and the observation is read only AFTER the settle, from fresh
  odometry and a LiDAR scan that arrived after the robot stopped;
* a contact guard: the sim had geometric collision; on hardware the episode
  ends as COLLISION if anything in front is within CONTACT_GUARD_M of the
  robot centre (the Burger's radius is 0.105 m).

Actions, timings and reward constants come unchanged from config.py.
Commands go to /cmd_vel_raw, so safety_filter_node.py must be running.
"""

from __future__ import annotations

import math
import threading
import time
from typing import Optional

from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import LaserScan

try:
    from .config import PointNavAction, PointNavConfig
    from .lidar_sectors import SECTOR_NAMES, sim_matched_sectors
    from .ros_point_nav_env import RosPointNavEnv, OdomState
except ImportError:  # pragma: no cover - direct-script convenience only
    from config import PointNavAction, PointNavConfig
    from lidar_sectors import SECTOR_NAMES, sim_matched_sectors
    from ros_point_nav_env import RosPointNavEnv, OdomState


SETTLE_S = 8 / 60.0            # sim: 8 zero-velocity frames at 60 Hz physics
SCAN_TOPIC = "/scan"
SCAN_WAIT_S = 1.0              # LDS publishes ~5 Hz
CONTACT_GUARD_M = 0.13         # robot radius 0.105 m + small margin


def v1b_config() -> PointNavConfig:
    return PointNavConfig()  # 150 steps / 60 s, as trained


def v2_config() -> PointNavConfig:
    return PointNavConfig(max_episode_steps=250, max_episode_sim_time_s=120.0)


class RosPointNavLidarEnv(RosPointNavEnv):
    """``use_lidar=False`` -> 3D V1B observation; ``True`` -> 6D V2 observation."""

    def __init__(self, goal_x: float, goal_y: float, *, use_lidar: bool,
                 config: Optional[PointNavConfig] = None, settle_s: float = SETTLE_S) -> None:
        self.use_lidar = use_lidar
        self.settle_s = settle_s
        self._scan_lock = threading.Lock()
        self._scan: Optional[LaserScan] = None
        self._scan_arrival = 0.0
        self._previous_clearance_m: dict = {}
        self.last_lidar: dict = {}
        super().__init__(goal_x, goal_y, config=config or (v2_config() if use_lidar else v1b_config()))
        self._scan_sub = self.create_subscription(
            LaserScan,
            SCAN_TOPIC,
            self._scan_callback,
            QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=5),
        )

    # ------------------------------------------------------------ LiDAR

    def _scan_callback(self, msg: LaserScan) -> None:
        with self._scan_lock:
            self._scan = msg
            self._scan_arrival = time.monotonic()

    def wait_for_scan_after(self, t: float, timeout_s: float = SCAN_WAIT_S) -> LaserScan:
        deadline = time.monotonic() + timeout_s
        while True:
            with self._scan_lock:
                if self._scan is not None and self._scan_arrival > t:
                    return self._scan
            if time.monotonic() > deadline:
                raise RuntimeError(f"No fresh LiDAR scan on {SCAN_TOPIC} within {timeout_s:.1f} s "
                                   "(is the LiDAR running? check `ros2 topic hz /scan`).")
            time.sleep(0.02)

    def _lidar_values(self) -> list[float]:
        with self._scan_lock:
            scan = self._scan
        if scan is None:
            raise RuntimeError(f"No LiDAR scan received on {SCAN_TOPIC} yet.")
        values, diag = sim_matched_sectors(
            scan.ranges, scan.angle_min, scan.angle_increment, scan.range_min, scan.range_max,
            self._previous_clearance_m,
        )
        self._previous_clearance_m = diag["clearance_m"]
        self.last_lidar = diag
        return values

    def _observation(self, state: OdomState) -> list[float]:
        base = super()._observation(state)
        return base + self._lidar_values() if self.use_lidar else base

    def contact(self) -> bool:
        return self.use_lidar and self.last_lidar.get("nearest_body_m", math.inf) < CONTACT_GUARD_M

    # ------------------------------------------------------------ gym-like API

    def reset(self, goal=None):
        if self.use_lidar:
            self.wait_for_scan_after(time.monotonic() - 10.0, timeout_s=5.0)
        observation, info = super().reset(goal)
        info["lidar"] = self.last_lidar
        return observation, info

    def step(self, action):
        _, reward, _, truncated, info = super().step(action)   # drives, then sends zero
        settle_start = time.monotonic()
        while time.monotonic() - settle_start < self.settle_s:
            self.stop()
            time.sleep(0.02)
        if self.use_lidar:
            self.wait_for_scan_after(settle_start)
        state = self._current_state()
        observation = self._observation(state)
        distance, bearing = self._distance_and_bearing(state)
        terminated = distance <= self.config.goal_tolerance_m
        collision = (not terminated) and self.contact()
        if terminated:
            truncated = False
        if collision:
            terminated, truncated = True, False
        info.update({
            "pose": {"x": state.x, "y": state.y, "yaw": state.yaw},
            "distance_to_goal_m": distance,
            "relative_goal_bearing_rad": bearing,
            "collision": collision,
            "lidar": self.last_lidar,
        })
        return observation, reward, terminated, truncated, info


__all__ = ["RosPointNavLidarEnv", "PointNavAction", "SECTOR_NAMES", "v1b_config", "v2_config"]
