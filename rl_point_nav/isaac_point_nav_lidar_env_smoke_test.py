#!/usr/bin/env python3
"""V2 environment smoke test: LiDAR sectors, obstacle pool and collision.

Scenarios (explicit obstacle positions via reset options, robot at the V1B start):
  EMPTY      all obstacles parked                 -> baseline clearances
  FRONT      obstacle 0.50 m ahead of the LiDAR   -> front ~0.44 m (box face)
  LEFT       obstacle 0.50 m left of the LiDAR    -> left  ~0.44 m
  RIGHT      obstacle 0.50 m right of the LiDAR   -> right ~0.44 m
  COLLISION  obstacle 0.30 m ahead of the robot, drive FORWARD until termination;
             checks collision=True, terminated=True, reward ~ -10. Front clearance is
             logged only: LiDAR and collision geometry are intentionally decoupled.

  --gui   run with the Isaac GUI (default headless)
"""

from __future__ import annotations

import argparse
import math

import numpy as np

try:
    from .config import PointNavAction
    from .isaac_point_nav_env import _yaw_from_wxyz
    from .isaac_point_nav_lidar_env import LIDAR_MOUNT_OFFSET_M, LIDAR_RANGE_SETTINGS, LIDAR_PATH, GUI_VISUAL_ROOTS, IsaacPointNavLidarEnv
except ImportError:  # pragma: no cover - direct-script convenience only
    from config import PointNavAction
    from isaac_point_nav_env import _yaw_from_wxyz
    from isaac_point_nav_lidar_env import LIDAR_MOUNT_OFFSET_M, LIDAR_RANGE_SETTINGS, LIDAR_PATH, GUI_VISUAL_ROOTS, IsaacPointNavLidarEnv


GOAL = (-2.118, 0.444)  # V1B straight goal; only used for the observation
PROBE_DISTANCE_M = 0.50
CLEARANCE_TOLERANCE_M = 0.06
COLLISION_OBSTACLE_AHEAD_M = 0.30
MAX_COLLISION_STEPS = 20
# Collision step reward = progress term (|.| <= 5.0 * 0.10) - 0.01 step penalty - 10 collision penalty.
COLLISION_REWARD_TOLERANCE = 0.51
TAG = "[TB03_V2_SMOKE]"


def log(message: str) -> None:
    print(f"{TAG} {message}", flush=True)


def _fmt(clearances: dict[str, float]) -> str:
    return " ".join(f"{name}={value:.3f}" for name, value in clearances.items())


def main() -> bool:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gui", action="store_true")
    args = parser.parse_args()

    env = IsaacPointNavLidarEnv(headless=not args.gui)
    results: dict[str, bool] = {}
    try:
        from pxr import UsdGeom

        sensor = env._stage.GetPrimAtPath(LIDAR_PATH)
        results["LIDAR_SETTINGS"] = all(
            np.isclose(sensor.GetAttribute(name).Get(), value, atol=1e-6, rtol=0)
            for name, value in LIDAR_RANGE_SETTINGS.items()
        )
        results["TOPLEVEL"] = str(sensor.GetParent().GetPath()) == "/World"
        results["HIDDEN_VISUALS"] = all(
            UsdGeom.Imageable(env._stage.GetPrimAtPath(path)).ComputeVisibility() == UsdGeom.Tokens.invisible
            for path in GUI_VISUAL_ROOTS
        )
        results["ARTICULATION_VALID"] = env._reset_robot.is_physics_tensor_entity_valid()
        start_x = float(env._default_position_host[0, 0])
        start_y = float(env._default_position_host[0, 1])
        yaw = _yaw_from_wxyz(env._default_orientation_host[0])
        lidar_x = start_x + LIDAR_MOUNT_OFFSET_M[0] * math.cos(yaw)
        lidar_y = start_y + LIDAR_MOUNT_OFFSET_M[0] * math.sin(yaw)
        half = 0.5 * env.lidar_config.obstacle_size_xy_m
        expected_face = PROBE_DISTANCE_M - half
        log(f"start=({start_x:.3f}, {start_y:.3f}) yaw={yaw:.3f} lidar_xy=({lidar_x:.3f}, {lidar_y:.3f})")

        observation, info = env.reset(options={"goal": GOAL, "obstacles": []})
        log(f"OBS_SHAPE={observation.shape} OBS={observation.round(3).tolist()}")
        log(f"EMPTY clearances_m: {_fmt(info['clearances_m'])}")
        results["OBS_SHAPE_6"] = observation.shape == (6,) and observation.dtype == np.float32 and env.observation_space.contains(observation)

        for name, offset_deg in (("FRONT", 0.0), ("LEFT", 90.0), ("RIGHT", -90.0)):
            heading = yaw + math.radians(offset_deg)
            obstacle = (
                lidar_x + PROBE_DISTANCE_M * math.cos(heading),
                lidar_y + PROBE_DISTANCE_M * math.sin(heading),
            )
            observation, info = env.reset(options={"goal": GOAL, "obstacles": [obstacle]})
            clearances = info["clearances_m"]
            measured = clearances[name.lower()]
            ok = abs(measured - expected_face) <= CLEARANCE_TOLERANCE_M
            others_clear = all(
                clearances[other] > expected_face + CLEARANCE_TOLERANCE_M
                for other in clearances
                if other != name.lower()
            )
            results[name] = ok and others_clear
            log(
                f"{name} obstacle=({obstacle[0]:.3f}, {obstacle[1]:.3f}) clearances_m: {_fmt(clearances)} "
                f"expected_{name.lower()}={expected_face:.3f} obs={observation.round(3).tolist()} "
                f"{'PASS' if results[name] else 'FAIL'}"
            )

        obstacle = (
            start_x + COLLISION_OBSTACLE_AHEAD_M * math.cos(yaw),
            start_y + COLLISION_OBSTACLE_AHEAD_M * math.sin(yaw),
        )
        observation, info = env.reset(options={"goal": GOAL, "obstacles": [obstacle]})
        fronts = [info["clearances_m"]["front"]]
        collided, collision_terminated, collision_reward, steps = False, False, None, 0
        for steps in range(1, MAX_COLLISION_STEPS + 1):
            observation, reward, terminated, truncated, info = env.step(PointNavAction.FORWARD)
            fronts.append(info["clearances_m"]["front"])
            log(
                f"COLLISION step={steps} x={info['pose']['x']:.3f} front={fronts[-1]:.3f} "
                f"reward={reward:.3f} terminated={terminated} collision={info['collision']}"
            )
            if terminated or truncated:
                collided = bool(info["collision"])
                collision_terminated = bool(terminated)
                collision_reward = float(reward)
                break
        reward_ok = collision_reward is not None and abs(
            collision_reward + env.lidar_config.collision_penalty
        ) <= COLLISION_REWARD_TOLERANCE
        results["COLLISION"] = collided and collision_terminated and reward_ok
        log(
            f"COLLISION collision={collided} terminated={collision_terminated} "
            f"reward={'n/a' if collision_reward is None else f'{collision_reward:.3f}'} reward_ok={reward_ok} "
            f"steps={steps} front_log={[round(f, 3) for f in fronts]}"
        )

        log("================ SUMMARY ================")
        for name, ok in results.items():
            log(f"{name:<12} {'PASS' if ok else 'FAIL'}")
        overall = all(results.values())
        log(f"OVERALL={'PASS' if overall else 'FAIL'}")
        if args.gui:
            input(f"{TAG} Done. Press Enter to close Isaac Sim... ")
        return overall
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
