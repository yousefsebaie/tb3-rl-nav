#!/usr/bin/env python3
"""Headless direct-turn smoke test for the TurtleBot articulation.

This test verifies direct differential-drive yaw signs only.  It uses no ROS 2,
OmniGraph, Gymnasium, PPO, LiDAR, or navigation logic.
"""

from __future__ import annotations

import math
from pathlib import Path


SCENE_PATH = Path("/home/g8/Desktop/FInalSceneDemo.usd")
ARTICULATION_PATH = "/World/turtlebot3_burger_02"
LEFT_WHEEL_DOF = "a__namespace_wheel_left_joint"
RIGHT_WHEEL_DOF = "a__namespace_wheel_right_joint"

WHEEL_RADIUS_M = 0.033
WHEEL_BASE_M = 0.160
TURN_SPEED_RADPS = 0.80
DRIVE_FRAMES = 15
SETTLE_FRAMES = 8


# SimulationApp must be constructed before importing Kit/Isaac runtime APIs.
from isaacsim import SimulationApp


print("[TB03_TURN] Before opening USD via SimulationApp", flush=True)
simulation_app = SimulationApp(
    {
        "headless": True,
        "open_usd": str(SCENE_PATH),
        "create_new_stage": False,
        "fast_shutdown": False,
        "disable_viewport_updates": True,
    }
)
print("[TB03_TURN] SimulationApp created", flush=True)


def _format(values) -> str:
    """Format NumPy-like values without depending on a particular backend."""
    return str(values.tolist() if hasattr(values, "tolist") else values)


def _wrap_to_pi(angle: float) -> float:
    """Return an angle in [-pi, pi]."""
    return math.atan2(math.sin(angle), math.cos(angle))


def _yaw_from_wxyz(quaternion) -> float:
    """Extract planar yaw from Isaac's scalar-first world orientation."""
    w, x, y, z = (float(component) for component in quaternion)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def main() -> None:
    """Reset, turn left and right separately, then check their yaw signs."""
    import numpy as np
    import omni.usd

    from isaacsim.core.api import SimulationContext
    from isaacsim.core.experimental.prims import Articulation
    from isaacsim.robot.wheeled_robots.controllers.differential_controller import (
        DifferentialController,
    )
    from isaacsim.robot.wheeled_robots.robots.wheeled_robot import WheeledRobot

    if not SCENE_PATH.is_file():
        raise FileNotFoundError(f"Scene does not exist: {SCENE_PATH}")

    usd_context = omni.usd.get_context()
    while usd_context.get_stage_loading_status()[2] > 0:
        simulation_app.update()
    print("[TB03_TURN] USD loading complete", flush=True)

    stage = usd_context.get_stage()
    if stage is None:
        raise RuntimeError(f"No active USD stage after opening {SCENE_PATH}.")
    robot_prim = stage.GetPrimAtPath(ARTICULATION_PATH)
    if not robot_prim.IsValid():
        raise RuntimeError(f"Expected robot prim is absent: {ARTICULATION_PATH}")
    print(f"[TB03_TURN] Verified robot prim: {ARTICULATION_PATH}", flush=True)

    simulation_context = SimulationContext()
    simulation_context.initialize_physics()
    simulation_context.play()
    for _ in range(SETTLE_FRAMES):
        simulation_context.step(render=False)
    print("[TB03_TURN] Physics initialized and settled", flush=True)

    reset_robot = Articulation(ARTICULATION_PATH)
    if not reset_robot.is_physics_tensor_entity_valid():
        raise RuntimeError(f"Physics tensor state is unavailable for {ARTICULATION_PATH}.")

    drive_robot = WheeledRobot(
        prim_path=ARTICULATION_PATH,
        name="tb03_direct_turn",
        wheel_dof_names=[LEFT_WHEEL_DOF, RIGHT_WHEEL_DOF],
    )
    drive_robot.initialize()
    resolved_dof_names = list(drive_robot.dof_names)
    print(f"[TB03_TURN] Resolved articulation DOF names: {resolved_dof_names}", flush=True)
    if LEFT_WHEEL_DOF not in resolved_dof_names or RIGHT_WHEEL_DOF not in resolved_dof_names:
        raise RuntimeError(f"Expected wheel DOFs missing from: {resolved_dof_names}")
    _, wheel_indices = drive_robot.get_articulation_controller_properties()
    print(f"[TB03_TURN] Resolved wheel indices [left, right]: {wheel_indices}", flush=True)

    controller = DifferentialController(
        name="tb03_direct_turn_controller",
        wheel_radius=WHEEL_RADIUS_M,
        wheel_base=WHEEL_BASE_M,
        max_linear_speed=0.22,
        max_angular_speed=1.0,
    )
    stop_action = controller.forward(np.array([0.0, 0.0], dtype=np.float64))

    # Cache the known initialized articulation-root pose. The collection-oriented
    # experimental wrapper supplies the correct leading batch dimensions.
    default_position, default_orientation = reset_robot.get_world_poses()
    zero_root_velocity = np.zeros(tuple(default_position.shape), dtype=np.float32)
    zero_dof_velocity = np.zeros(tuple(reset_robot.get_dof_velocities().shape), dtype=np.float32)
    print(f"[TB03_TURN] Cached reset position (m): {_format(default_position)}", flush=True)
    print(f"[TB03_TURN] Cached reset orientation (wxyz): {_format(default_orientation)}", flush=True)

    def reset_to_default() -> None:
        """Apply the proven full-articulation reset and hold zero wheel motion."""
        reset_robot.set_world_poses(default_position, default_orientation)
        reset_robot.set_velocities(zero_root_velocity, zero_root_velocity)
        reset_robot.set_dof_velocities(zero_dof_velocity)
        reset_robot.set_dof_velocity_targets(zero_dof_velocity)
        for _ in range(SETTLE_FRAMES):
            drive_robot.apply_wheel_actions(stop_action)
            simulation_context.step(render=False)

    def run_turn(label: str, angular_speed_radps: float) -> tuple[float, float]:
        """Run one isolated turn and return (wrapped yaw delta, XY displacement)."""
        print(f"[TB03_TURN] {label}: resetting to known initial pose", flush=True)
        reset_to_default()
        initial_position, initial_orientation = drive_robot.get_world_pose()
        initial_yaw = _yaw_from_wxyz(initial_orientation)
        print(f"[TB03_TURN] {label}: initial position (m): {_format(initial_position)}", flush=True)
        print(f"[TB03_TURN] {label}: initial yaw (rad): {initial_yaw:.6f}", flush=True)

        command = np.array([0.0, angular_speed_radps], dtype=np.float64)
        action = controller.forward(command)
        print(f"[TB03_TURN] {label}: command [linear, angular]: {_format(command)}", flush=True)
        print(
            f"[TB03_TURN] {label}: wheel velocity [left, right] (rad/s): "
            f"{_format(action.joint_velocities)}",
            flush=True,
        )

        for _ in range(DRIVE_FRAMES):
            drive_robot.apply_wheel_actions(action)
            simulation_context.step(render=False)
        drive_robot.apply_wheel_actions(stop_action)
        print(f"[TB03_TURN] {label}: zero wheel velocity command applied", flush=True)
        for _ in range(SETTLE_FRAMES):
            simulation_context.step(render=False)

        final_position, final_orientation = drive_robot.get_world_pose()
        final_yaw = _yaw_from_wxyz(final_orientation)
        yaw_change = _wrap_to_pi(final_yaw - initial_yaw)
        xy_displacement = float(np.linalg.norm(final_position[:2] - initial_position[:2]))
        print(f"[TB03_TURN] {label}: final position (m): {_format(final_position)}", flush=True)
        print(f"[TB03_TURN] {label}: final yaw (rad): {final_yaw:.6f}", flush=True)
        print(f"[TB03_TURN] {label}: wrapped yaw change (rad): {yaw_change:.6f}", flush=True)
        print(f"[TB03_TURN] {label}: XY displacement (m): {xy_displacement:.6f}", flush=True)
        return yaw_change, xy_displacement

    left_yaw_change, _ = run_turn("TURN_LEFT", +TURN_SPEED_RADPS)
    right_yaw_change, _ = run_turn("TURN_RIGHT", -TURN_SPEED_RADPS)

    if left_yaw_change <= 0.0:
        raise RuntimeError(
            f"[TB03_TURN] FAIL: TURN_LEFT must have positive wrapped yaw change, got {left_yaw_change:.6f}"
        )
    if right_yaw_change >= 0.0:
        raise RuntimeError(
            f"[TB03_TURN] FAIL: TURN_RIGHT must have negative wrapped yaw change, got {right_yaw_change:.6f}"
        )
    print("[TB03_TURN] PASS: TURN_LEFT yaw is positive and TURN_RIGHT yaw is negative", flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        print("[TB03_TURN] Immediately before simulation_app.close()", flush=True)
        simulation_app.close()
