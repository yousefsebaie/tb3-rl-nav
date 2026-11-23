#!/usr/bin/env python3
"""Real-Burger environment with MAPPO V5's 15 inputs (handoff section 4.2).

RosPointNavLidarEnv with the "v5" LiDAR profile gives inputs 0..10 (goal distance and
bearing, 8 sectors). This adds the teammate (mappo_hardware.py):

* the robot's pose in the shared mission frame (SharedFrame, from its measured start pose);
* the other robot's pose over UDP (PoseLink), drawn into the scan as the 0.105 m circle
  the policy was trained with, and inputs 11..14 = role, teammate_x, teammate_y, active.

Goals are given in the shared frame (set_goal_shared). Without a peer the teammate reads
"far and idle" (handoff step 3). With a peer and no fresh teammate pose, the observation
raises TeammateLost: the caller must not drive on a guessed teammate position.
"""

from __future__ import annotations

from typing import Optional

from mappo_hardware import (POSE_MAX_AGE_S, POSE_PORT, ROLE, Pose, PoseLink, SharedFrame, lidar_pose,
                            merge_teammate_circle, teammate_inputs)
from ros_point_nav_lidar_env import RosPointNavLidarEnv
from ros_point_nav_env import OdomState


class TeammateLost(RuntimeError):
    """No fresh pose from the other robot (POSE_MAX_AGE_S)."""


class MappoRosEnv(RosPointNavLidarEnv):
    def __init__(self, robot: str, start_shared: Pose, *, config, peer: Optional[str] = None,
                 listen_port: int = POSE_PORT, peer_port: int = POSE_PORT) -> None:
        if robot not in ROLE:
            raise ValueError(f"robot must be one of {sorted(ROLE)}")
        super().__init__(0.0, 0.0, use_lidar=True, config=config, lidar_profile="v5")
        self.robot = robot
        self.role = ROLE[robot]
        odom = self._current_state()
        self.frame = SharedFrame(start_shared, (odom.x, odom.y, odom.yaw))
        self.active = False          # sent to the teammate: am I driving to a goal this phase?
        self.last_teammate: Optional[dict] = None
        self.link = None if peer is None else PoseLink(robot, peer, self._own_pose, listen_port, peer_port)

    # ------------------------------------------------------------ poses

    def latest_shared(self) -> Optional[Pose]:
        """Newest odometry in the shared frame, without waiting (None before the first message)."""
        with self._state_lock:
            state = self._state
        return None if state is None else self.frame.to_shared((state.x, state.y, state.yaw))

    def _own_pose(self):
        pose = self.latest_shared()
        return None if pose is None else (*pose, self.active)

    def set_goal_shared(self, x: float, y: float) -> None:
        self.set_goal(*self.frame.to_odom(x, y))

    def teammate(self) -> Optional[dict]:
        return None if self.link is None else self.link.teammate()

    # ------------------------------------------------------------ observation

    def _sector_ranges(self, scan):
        self.last_teammate = self.teammate()
        me = self.latest_shared()
        if self.last_teammate is None or me is None:
            return scan.ranges
        mate = (self.last_teammate["x"], self.last_teammate["y"])
        return merge_teammate_circle(scan.ranges, scan.angle_min, scan.angle_increment,
                                     scan.range_min, scan.range_max, lidar_pose(me), mate)

    def _observation(self, state: OdomState) -> list[float]:
        inputs = super()._observation(state)          # 11: goal + v5 sectors (calls _sector_ranges)
        me = self.frame.to_shared((state.x, state.y, state.yaw))
        if self.link is None:
            return inputs + teammate_inputs(me, None, False, self.role)
        mate = self.last_teammate
        if mate is None:
            raise TeammateLost(f"{self.robot}: no pose from the other robot for over {POSE_MAX_AGE_S} s "
                               f"(sent {self.link.sent}, received {self.link.received})")
        return inputs + teammate_inputs(me, (mate["x"], mate["y"]), mate["active"], self.role)

    def close(self) -> None:
        if self.link is not None:
            self.link.close()
        super().close()
