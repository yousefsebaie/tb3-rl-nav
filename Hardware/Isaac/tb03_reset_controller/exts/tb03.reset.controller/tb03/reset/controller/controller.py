"""Isaac-native shared state and articulation reset operations.

This module deliberately has no ROS Python imports.  Requests arrive through
custom OmniGraph nodes after the built-in ROS2 Bridge decodes them.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from typing import Optional

import carb
import numpy as np
import omni.timeline
import omni.usd
from pxr import UsdGeom

from isaacsim.core.prims import SingleArticulation

ARTICULATION_ROOT = "/World/turtlebot3_burger_02"
BASE_LINK = "/World/turtlebot3_burger_02/a__namespace_base_link"
SETTLE_UPDATE_FRAMES = 8


@dataclass(frozen=True)
class ResetRequest:
    x: float
    y: float
    yaw: float


def _world_matrix(stage, path: str) -> np.ndarray:
    matrix = UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath(path))
    return np.array(matrix, dtype=np.float64).reshape(4, 4)


def _quat_wxyz_from_rotation(rotation: np.ndarray) -> np.ndarray:
    trace = float(np.trace(rotation))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w, x, y, z = 0.25 * s, (rotation[2, 1] - rotation[1, 2]) / s, (rotation[0, 2] - rotation[2, 0]) / s, (rotation[1, 0] - rotation[0, 1]) / s
    elif rotation[0, 0] > rotation[1, 1] and rotation[0, 0] > rotation[2, 2]:
        s = math.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
        w, x, y, z = (rotation[2, 1] - rotation[1, 2]) / s, 0.25 * s, (rotation[0, 1] + rotation[1, 0]) / s, (rotation[0, 2] + rotation[2, 0]) / s
    elif rotation[1, 1] > rotation[2, 2]:
        s = math.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
        w, x, y, z = (rotation[0, 2] - rotation[2, 0]) / s, (rotation[0, 1] + rotation[1, 0]) / s, 0.25 * s, (rotation[1, 2] + rotation[2, 1]) / s
    else:
        s = math.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
        w, x, y, z = (rotation[1, 0] - rotation[0, 1]) / s, (rotation[0, 2] + rotation[2, 0]) / s, (rotation[1, 2] + rotation[2, 1]) / s, 0.25 * s
    quat = np.array([w, x, y, z], dtype=np.float64)
    return quat / np.linalg.norm(quat)


class ResetController:
    """Runs on Kit update callbacks and owns no ROS transport."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pending: Optional[ResetRequest] = None
        self._active: Optional[ResetRequest] = None
        self._completed = False
        self._settle_remaining = 0
        self._articulation: Optional[SingleArticulation] = None
        self._root_to_base: Optional[np.ndarray] = None

    def queue(self, x: float, y: float, yaw: float) -> None:
        request = ResetRequest(float(x), float(y), float(yaw))
        with self._lock:
            self._pending = request
        carb.log_info(f"[TB03ResetController] Reset request: x={x:.3f} y={y:.3f} yaw={yaw:.3f}")

    def advance(self) -> None:
        """Execute only from Isaac's Kit update thread."""
        if self._settle_remaining > 0:
            self._hold_stationary()
            self._settle_remaining -= 1
            carb.log_info(f"[TB03ResetController] Settling: {self._settle_remaining} update frames remaining")
            if self._settle_remaining == 0:
                with self._lock:
                    self._completed = True
                carb.log_info("[TB03ResetController] Reset complete")
            return

        with self._lock:
            request = self._pending
            self._pending = None
        if request is None:
            return
        if not omni.timeline.get_timeline_interface().is_playing():
            # Keep the request pending until the user resumes Play.
            with self._lock:
                self._pending = request
            carb.log_warn("[TB03ResetController] Waiting for Isaac timeline Play")
            return
        if not self._ensure_articulation():
            with self._lock:
                self._pending = request
            return
        self._apply(request)

    def consume_completion(self) -> bool:
        with self._lock:
            complete = self._completed
            self._completed = False
            return complete

    def _ensure_articulation(self) -> bool:
        if self._articulation is None:
            self._articulation = SingleArticulation(ARTICULATION_ROOT, name="tb03_reset_articulation")
        try:
            self._articulation.initialize()
            stage = omni.usd.get_context().get_stage()
            if stage is None or not stage.GetPrimAtPath(BASE_LINK).IsValid():
                carb.log_warn("[TB03ResetController] Waiting for TurtleBot base_link prim")
                return False
            root_world = _world_matrix(stage, ARTICULATION_ROOT)
            base_world = _world_matrix(stage, BASE_LINK)
            self._root_to_base = np.linalg.inv(root_world) @ base_world
            return True
        except Exception as error:
            carb.log_warn(f"[TB03ResetController] Waiting for initialized articulation: {error}")
            return False

    def _apply(self, request: ResetRequest) -> None:
        assert self._articulation is not None and self._root_to_base is not None
        stage = omni.usd.get_context().get_stage()
        assert stage is not None
        current_base = _world_matrix(stage, BASE_LINK)
        desired_base = np.eye(4, dtype=np.float64)
        c, s = math.cos(request.yaw), math.sin(request.yaw)
        desired_base[:3, :3] = np.array(((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0)))
        desired_base[0, 3], desired_base[1, 3], desired_base[2, 3] = request.x, request.y, current_base[2, 3]
        desired_root = desired_base @ np.linalg.inv(self._root_to_base)
        self._articulation.set_world_pose(desired_root[:3, 3], _quat_wxyz_from_rotation(desired_root[:3, :3]))
        self._clear_velocities()
        self._active = request
        self._settle_remaining = SETTLE_UPDATE_FRAMES
        carb.log_info("[TB03ResetController] Reset applied")

    def _clear_velocities(self) -> None:
        assert self._articulation is not None
        self._articulation.set_linear_velocity(np.zeros(3, dtype=np.float64))
        self._articulation.set_angular_velocity(np.zeros(3, dtype=np.float64))
        joint_velocities = self._articulation.get_joint_velocities()
        if joint_velocities is not None:
            self._articulation.set_joint_velocities(np.zeros_like(joint_velocities))

    def _hold_stationary(self) -> None:
        self._clear_velocities()


_controller: Optional[ResetController] = None


def install_controller(controller: ResetController) -> None:
    global _controller
    _controller = controller


def remove_controller() -> None:
    global _controller
    _controller = None


def get_controller() -> Optional[ResetController]:
    return _controller
