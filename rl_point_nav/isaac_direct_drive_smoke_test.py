#!/usr/bin/env python3
"""Headless direct-wheel-drive smoke test for the TurtleBot articulation.

This intentionally exercises only Isaac Sim's direct wheeled-robot API.  It
does not use ROS 2, OmniGraph, Gymnasium, PPO, LiDAR, or navigation code.
"""

from __future__ import annotations

from pathlib import Path


SCENE_PATH = Path("/home/g8/Desktop/FInalSceneDemo.usd")
ARTICULATION_PATH = "/World/turtlebot3_burger_02"
LEFT_WHEEL_DOF = "a__namespace_wheel_left_joint"
RIGHT_WHEEL_DOF = "a__namespace_wheel_right_joint"

# TurtleBot3 Burger physical parameters from the authored wheel collision
# geometry and the installed Burger configuration.
WHEEL_RADIUS_M = 0.033
WHEEL_BASE_M = 0.160

FORWARD_SPEED_MPS = 0.05
ANGULAR_SPEED_RADPS = 0.0
DRIVE_FRAMES = 30
SETTLE_FRAMES = 8
MIN_EXPECTED_DISPLACEMENT_M = 0.005


# SimulationApp must be constructed before importing Kit/Isaac runtime APIs.
from isaacsim import SimulationApp


print("[TB03_DRIVE] Before opening USD via SimulationApp", flush=True)
simulation_app = SimulationApp(
    {
        "headless": True,
        "open_usd": str(SCENE_PATH),
        # Without this, SimulationApp can replace open_usd with a blank stage.
        "create_new_stage": False,
        "fast_shutdown": False,
        "disable_viewport_updates": True,
    }
)
print("[TB03_DRIVE] SimulationApp created", flush=True)


def _format(values) -> str:
    """Format NumPy-like values without depending on a particular backend."""
    return str(values.tolist() if hasattr(values, "tolist") else values)


def main() -> None:
    """Drive both wheels briefly, stop them, and verify root-pose displacement."""
    import numpy as np
    import omni.usd

    from isaacsim.core.api import SimulationContext
    from isaacsim.robot.wheeled_robots.controllers.differential_controller import (
        DifferentialController,
    )
    from isaacsim.robot.wheeled_robots.robots.wheeled_robot import WheeledRobot

    if not SCENE_PATH.is_file():
        raise FileNotFoundError(f"Scene does not exist: {SCENE_PATH}")

    # Use the proven load-completion pattern from the articulation reset test.
    usd_context = omni.usd.get_context()
    while usd_context.get_stage_loading_status()[2] > 0:
        simulation_app.update()
    print("[TB03_DRIVE] USD loading complete", flush=True)

    stage = usd_context.get_stage()
    if stage is None:
        raise RuntimeError(f"No active USD stage after opening {SCENE_PATH}.")

    robot_prim = stage.GetPrimAtPath(ARTICULATION_PATH)
    if not robot_prim.IsValid():
        world_prim = stage.GetPrimAtPath("/World")
        world_children = (
            [child.GetPath().pathString for child in world_prim.GetChildren()]
            if world_prim.IsValid()
            else []
        )
        raise RuntimeError(
            f"Expected robot prim is absent from the active stage: {ARTICULATION_PATH}. "
            f"Immediate /World children: {world_children}"
        )
    print(f"[TB03_DRIVE] Verified robot prim: {ARTICULATION_PATH}", flush=True)

    print("[TB03_DRIVE] Before physics initialization", flush=True)
    simulation_context = SimulationContext()
    simulation_context.initialize_physics()
    simulation_context.play()
    print("[TB03_DRIVE] Physics initialization complete", flush=True)

    # Allow PhysX and the referenced articulation to become ready before the
    # legacy WheeledRobot wrapper resolves its named DOFs.
    for _ in range(SETTLE_FRAMES):
        simulation_context.step(render=False)
    print("[TB03_DRIVE] Initial physics settling complete", flush=True)

    print("[TB03_DRIVE] Before creating WheeledRobot", flush=True)
    robot = WheeledRobot(
        prim_path=ARTICULATION_PATH,
        name="tb03_direct_drive",
        wheel_dof_names=[LEFT_WHEEL_DOF, RIGHT_WHEEL_DOF],
    )
    robot.initialize()
    print("[TB03_DRIVE] WheeledRobot initialized", flush=True)

    resolved_dof_names = list(robot.dof_names)
    print(f"[TB03_DRIVE] Resolved articulation DOF names: {resolved_dof_names}", flush=True)
    missing_dofs = [
        dof_name
        for dof_name in (LEFT_WHEEL_DOF, RIGHT_WHEEL_DOF)
        if dof_name not in resolved_dof_names
    ]
    if missing_dofs:
        raise RuntimeError(
            f"Expected TurtleBot wheel DOFs are absent: {missing_dofs}. "
            f"Resolved DOFs: {resolved_dof_names}"
        )

    _, wheel_indices = robot.get_articulation_controller_properties()
    if wheel_indices is None or len(wheel_indices) != 2 or any(index is None or index < 0 for index in wheel_indices):
        raise RuntimeError(f"Could not resolve two valid wheel indices: {wheel_indices}")
    print(
        f"[TB03_DRIVE] Resolved wheel indices: left={wheel_indices[0]}, right={wheel_indices[1]}",
        flush=True,
    )

    controller = DifferentialController(
        name="tb03_direct_differential_controller",
        wheel_radius=WHEEL_RADIUS_M,
        wheel_base=WHEEL_BASE_M,
        max_linear_speed=0.22,
        max_angular_speed=1.0,
    )

    # Explicitly impose zero wheel targets before recording the starting pose.
    stop_action = controller.forward(np.array([0.0, 0.0], dtype=np.float64))
    robot.apply_wheel_actions(stop_action)
    for _ in range(SETTLE_FRAMES):
        simulation_context.step(render=False)

    initial_position, initial_orientation = robot.get_world_pose()
    print(f"[TB03_DRIVE] Initial position (m): {_format(initial_position)}", flush=True)
    print(f"[TB03_DRIVE] Initial orientation (wxyz): {_format(initial_orientation)}", flush=True)

    command = np.array([FORWARD_SPEED_MPS, ANGULAR_SPEED_RADPS], dtype=np.float64)
    drive_action = controller.forward(command)
    print(f"[TB03_DRIVE] Vehicle command [linear, angular]: {_format(command)}", flush=True)
    print(
        f"[TB03_DRIVE] Wheel velocity command [left, right] (rad/s): "
        f"{_format(drive_action.joint_velocities)}",
        flush=True,
    )

    print(f"[TB03_DRIVE] Applying forward command for {DRIVE_FRAMES} physics frames", flush=True)
    for _ in range(DRIVE_FRAMES):
        robot.apply_wheel_actions(drive_action)
        simulation_context.step(render=False)

    robot.apply_wheel_actions(stop_action)
    print("[TB03_DRIVE] Zero wheel velocity command applied", flush=True)
    for _ in range(SETTLE_FRAMES):
        simulation_context.step(render=False)
    print(f"[TB03_DRIVE] Settling complete after {SETTLE_FRAMES} physics frames", flush=True)

    final_position, final_orientation = robot.get_world_pose()
    xy_displacement = float(np.linalg.norm(final_position[:2] - initial_position[:2]))
    print(f"[TB03_DRIVE] Final position (m): {_format(final_position)}", flush=True)
    print(f"[TB03_DRIVE] Final orientation (wxyz): {_format(final_orientation)}", flush=True)
    print(f"[TB03_DRIVE] XY displacement (m): {xy_displacement:.6f}", flush=True)

    if xy_displacement <= MIN_EXPECTED_DISPLACEMENT_M:
        raise RuntimeError(
            "[TB03_DRIVE] FAIL: TurtleBot did not move far enough; "
            f"expected > {MIN_EXPECTED_DISPLACEMENT_M:.3f} m, observed {xy_displacement:.6f} m"
        )
    print(
        "[TB03_DRIVE] PASS: TurtleBot moved more than "
        f"{MIN_EXPECTED_DISPLACEMENT_M:.3f} m",
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    finally:
        print("[TB03_DRIVE] Immediately before simulation_app.close()", flush=True)
        simulation_app.close()
