#!/usr/bin/env python3
"""Headless direct-Isaac reset smoke test for the TurtleBot articulation.

This is intentionally not a Gymnasium environment and does not use ROS, PPO,
observations, rewards, goals, LiDAR, or robot movement actions.
"""

from __future__ import annotations

from pathlib import Path

SCENE_PATH = Path("/home/g8/Desktop/FInalSceneDemo.usd")
ARTICULATION_PATH = "/World/turtlebot3_burger_02"
SETTLE_FRAMES = 8


# SimulationApp must be constructed before importing Kit/Isaac runtime APIs.
from isaacsim import SimulationApp

# SimulationApp opens ``open_usd`` during construction, so this is necessarily
# the checkpoint immediately before the requested USD is opened.
print("[TB03_SMOKE] Before opening USD via SimulationApp", flush=True)
simulation_app = SimulationApp(
    {
        "headless": True,
        "open_usd": str(SCENE_PATH),
        # SimulationApp otherwise opens open_usd and then replaces it with a new blank stage.
        "create_new_stage": False,
        "fast_shutdown": False,
        "disable_viewport_updates": True,
    }
)
print("[TB03_SMOKE] SimulationApp created", flush=True)


def _format(values) -> str:
    """Format NumPy-like values without depending on a particular Isaac backend."""
    return str(values.tolist() if hasattr(values, "tolist") else values)


def main() -> None:
    """Initialize physics, reset the complete articulation state, and print it."""
    import numpy as np
    import omni.usd

    from isaacsim.core.experimental.prims import Articulation
    from isaacsim.core.api import SimulationContext

    if not SCENE_PATH.is_file():
        raise FileNotFoundError(f"Scene does not exist: {SCENE_PATH}")

    # Match Isaac's installed standalone examples: let Kit update while OmniUSD
    # reports outstanding stage-load work, rather than relying on a sleep duration.
    usd_context = omni.usd.get_context()
    while usd_context.get_stage_loading_status()[2] > 0:
        simulation_app.update()
    print("[TB03_SMOKE] USD loading complete", flush=True)

    stage = usd_context.get_stage()
    if stage is None:
        raise RuntimeError(f"No active USD stage after opening {SCENE_PATH}.")

    default_prim = stage.GetDefaultPrim()
    default_prim_path = default_prim.GetPath().pathString if default_prim.IsValid() else "<none>"
    print(f"[TB03_SMOKE] Active stage root layer: {stage.GetRootLayer().identifier}", flush=True)
    print(f"[TB03_SMOKE] Active stage default prim: {default_prim_path}", flush=True)

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
    print(f"[TB03_SMOKE] Verified robot prim: {ARTICULATION_PATH}", flush=True)
    print("[TB03_SMOKE] Robot prim confirmed", flush=True)

    # Only initialize physics after the expected scene and robot prim are verified.
    print("[TB03_SMOKE] Before physics initialization", flush=True)
    simulation_context = SimulationContext()
    simulation_context.initialize_physics()
    print("[TB03_SMOKE] Physics initialization complete", flush=True)
    simulation_context.play()

    # Allow PhysX and the referenced articulation to initialize before querying it.
    for _ in range(SETTLE_FRAMES):
        simulation_context.step(render=False)

    print("[TB03_SMOKE] Before creating Articulation", flush=True)
    robot = Articulation(ARTICULATION_PATH)
    print("[TB03_SMOKE] Articulation created", flush=True)
    if not robot.is_physics_tensor_entity_valid():
        raise RuntimeError(f"Physics tensor state is unavailable for {ARTICULATION_PATH}.")

    # This scene's initialized root pose is the smoke test's reset/default pose.
    initial_position, initial_orientation = robot.get_world_poses()
    print("[TB03_SMOKE] Initial root pose read", flush=True)
    print(f"[TB03_SMOKE] Initial position (m): {_format(initial_position)}", flush=True)
    print(f"[TB03_SMOKE] Initial orientation (wxyz): {_format(initial_orientation)}", flush=True)

    initial_linear_velocity, initial_angular_velocity = robot.get_velocities()
    initial_dof_velocities = robot.get_dof_velocities()
    print("[TB03_SMOKE] Initial root and DOF velocities read", flush=True)
    print(f"[TB03_SMOKE] Initial linear velocity (m/s): {_format(initial_linear_velocity)}", flush=True)
    print(f"[TB03_SMOKE] Initial angular velocity (rad/s): {_format(initial_angular_velocity)}", flush=True)
    print(f"[TB03_SMOKE] Initial DOF velocities: {_format(initial_dof_velocities)}", flush=True)

    # The experimental wrapper is collection-oriented. Preserve its actual
    # leading batch dimension instead of assuming it is always one articulation.
    zero_root_velocity = np.zeros(tuple(initial_position.shape), dtype=np.float32)
    zero_joint_velocity = np.zeros(tuple(robot.get_dof_velocities().shape), dtype=np.float32)

    # Reset the full articulation root, then clear root and all DOF state.
    print("[TB03_SMOKE] Immediately before applying reset", flush=True)
    robot.set_world_poses(initial_position, initial_orientation)
    print("[TB03_SMOKE] set_world_poses complete", flush=True)
    robot.set_velocities(zero_root_velocity, zero_root_velocity)
    print("[TB03_SMOKE] Root velocities cleared", flush=True)
    robot.set_dof_velocities(zero_joint_velocity)
    robot.set_dof_velocity_targets(zero_joint_velocity)
    print("[TB03_SMOKE] DOF velocities and targets cleared", flush=True)

    for _ in range(SETTLE_FRAMES):
        simulation_context.step(render=False)
    print("[TB03_SMOKE] Settling physics steps complete", flush=True)

    final_position, final_orientation = robot.get_world_poses()
    linear_velocity, angular_velocity = robot.get_velocities()
    joint_velocities = robot.get_dof_velocities()
    print("[TB03_SMOKE] Final state read", flush=True)
    print(f"[TB03_SMOKE] Final position (m): {_format(final_position)}", flush=True)
    print(f"[TB03_SMOKE] Final orientation (wxyz): {_format(final_orientation)}", flush=True)
    print(f"[TB03_SMOKE] Final linear velocity (m/s): {_format(linear_velocity)}", flush=True)
    print(f"[TB03_SMOKE] Final angular velocity (rad/s): {_format(angular_velocity)}", flush=True)
    print(f"[TB03_SMOKE] Final DOF velocities: {_format(joint_velocities)}", flush=True)
    # This smoke test records the final state but intentionally has no tolerance assertion yet.
    print("[TB03_SMOKE] Reset verification checkpoint reached", flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        print("[TB03_SMOKE] Immediately before simulation_app.close()", flush=True)
        simulation_app.close()
