#!/usr/bin/env python3
"""GUI visualization of the V2 LiDAR point-navigation environment.

  --model PATH   run a trained V2 PPO checkpoint (deterministic)
  (no --model)   run a simple reactive rule-based controller on the SAME 6D
                 observation, to inspect the environment before training
  --random N     number of extra randomly sampled episodes (default 3)
  --show-robot   select visible robot BEFORE physics; presentation only, not evaluation
  --no-lidar-view  do not draw the LiDAR point cloud in the viewport (drawn by default)
  --auto         run all episodes back to back without waiting for Enter
  --profile P    v2 or v3 settings; default: models/<run>/profile.txt next to --model, else v2
  --seeds S ...  replay these eval_v3.py scenarios (seeds 5000+) instead of the scripted/random ones
  --camera C     overview (default: whole scenario from above), follow (close chase) or none
  --keep-panels  keep Isaac Sim's side panels (hidden by default so the viewport fills the window)

Each step prints the action, pose, goal distance and front/left/right clearances.
The goal is shown as a green disc on the ground; pooled obstacles are orange boxes.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import time

import numpy as np

try:
    from .config import PointNavAction
    from .isaac_point_nav_lidar_env import IsaacPointNavLidarEnv, IsaacLidarNavConfig, v3_lidar_config
except ImportError:  # pragma: no cover - direct-script convenience only
    from config import PointNavAction
    from isaac_point_nav_lidar_env import IsaacPointNavLidarEnv, IsaacLidarNavConfig, v3_lidar_config


ACTION_PAUSE_S = 0.20
FAR_STRAIGHT_GOAL = (-1.718, 0.444)  # V1B straight goal, 1.6 m ahead of the start
SCRIPTED = (
    ("CLEAR_STRAIGHT", {"goal": FAR_STRAIGHT_GOAL, "obstacles": []}),
    ("BLOCKED_STRAIGHT", {"goal": FAR_STRAIGHT_GOAL, "obstacles": [(-2.518, 0.444)]}),
)
V1B_DIR = Path(__file__).resolve().parent / "models" / "isaac_point_nav_ppo_v1b"
VERIFY_GOAL = FAR_STRAIGHT_GOAL
VERIFY_SETTLE_FRAMES = 30
VERIFY_SAMPLES = 10
VERIFY_TOLERANCE = 0.03  # normalized clearance (= 3 cm)
RENDER_UPDATES_AFTER_RESTORE = 10
TAG = "[TB03_GUI_V2]"
OVERVIEW_DIRECTION = np.array([-0.45, -0.45, 1.0])   # mostly from above, slightly from the side
_camera = "overview"


def _robot_base_xyz():
    import omni.usd
    from pxr import UsdGeom

    try:
        from .isaac_point_nav_lidar_env import BASE_LINK_PATH
    except ImportError:  # pragma: no cover
        from isaac_point_nav_lidar_env import BASE_LINK_PATH
    prim = omni.usd.get_context().get_stage().GetPrimAtPath(BASE_LINK_PATH)
    if not prim.IsValid():
        return None
    t = UsdGeom.XformCache().GetLocalToWorldTransform(prim).ExtractTranslation()
    return np.array([t[0], t[1], t[2]])


def _place_camera(info, start: bool) -> None:
    """overview: frame robot, goal and obstacles once per episode. follow: chase the robot."""
    from isaacsim.core.utils.viewports import set_camera_view

    base = _robot_base_xyz()
    if base is None or _camera == "none":
        return
    if _camera == "follow":
        set_camera_view(eye=base + np.array([-0.9, -0.9, 0.8]), target=base)
        return
    if not start:
        return
    points = [base[:2], np.array([info["goal"]["x"], info["goal"]["y"]])]
    points += [np.array(o[:2]) for o in info["obstacles"]]
    points = np.array(points)
    center = (points.min(axis=0) + points.max(axis=0)) / 2
    radius = max(float(np.linalg.norm(points - center, axis=1).max()), 0.5) + 0.3
    target = np.array([center[0], center[1], base[2]])
    direction = OVERVIEW_DIRECTION / np.linalg.norm(OVERVIEW_DIRECTION)
    set_camera_view(eye=target + direction * (2.4 * radius), target=target)


def _hide_ui_panels() -> None:
    """Let the 3D viewport fill the window (same as pressing F7 in Isaac Sim)."""
    import carb.settings

    carb.settings.get_settings().set("/app/window/hideUi", True)


# Rule-based controller thresholds (normalized clearance = meters / 1.0 m).
AVOID_FRONT = 0.35
GOAL_SIDE_BLOCKED = 0.25
BEARING_ALIGNED_RAD = 0.20
COMMIT_FORWARD_STEPS = 6


class ReactiveController:
    """Turn away from a close front obstacle, commit forward, else steer to goal."""

    def __init__(self) -> None:
        self.commit = 0

    def reset(self) -> None:
        self.commit = 0

    def predict(self, observation, deterministic: bool = True):
        del deterministic
        _, sin_b, cos_b, front, left, right = (float(v) for v in observation)
        if front < AVOID_FRONT:
            self.commit = COMMIT_FORWARD_STEPS
            return (PointNavAction.TURN_LEFT if left >= right else PointNavAction.TURN_RIGHT), None
        if self.commit > 0:
            self.commit -= 1
            return PointNavAction.FORWARD, None
        bearing = math.atan2(sin_b, cos_b)
        if abs(bearing) > BEARING_ALIGNED_RAD:
            side_clear = left if bearing > 0 else right
            if side_clear >= GOAL_SIDE_BLOCKED:
                return (PointNavAction.TURN_LEFT if bearing > 0 else PointNavAction.TURN_RIGHT), None
        return PointNavAction.FORWARD, None


def _print_lidar(env, info, observation) -> None:
    """Log-only: per-sector readings from info plus totals from the scan behind the observation."""
    lines = [f"{TAG}   LiDAR:"]
    for index, (name, reading) in enumerate(info["lidar_sectors"].items()):
        valid = reading["valid_rays"]
        clearance = reading["clearance_m"]
        min_range = f"{clearance:.3f}" if valid > 0 else "n/a"
        lines += [
            f"{TAG}   {name}:",
            f"{TAG}     clearance={clearance:.3f} m (obs={float(observation[3 + index]):.3f})",
            f"{TAG}     valid_rays={valid}",
            f"{TAG}     invalid={reading['invalid']}",
            f"{TAG}     min_range={min_range}",
        ]
    scan = env._last_ranges  # the FlatScan used for this observation (read-only)
    total = 0 if scan is None else int(scan.size)
    valid_total = 0 if scan is None else int(np.isfinite(scan).sum())
    lines += [
        f"{TAG}   FLAT_valid={valid_total}/{total}",
        f"{TAG}   SELF_RETURN_MASK={env.self_mask_ray_count}",
    ]
    print("\n".join(lines), flush=True)


def _run_episode(env, policy, label: str, options, seed=None) -> str:
    if hasattr(policy, "reset"):
        policy.reset()
    observation, info = env.reset(seed=seed, options=options)
    print(
        f"{TAG} {label} goal=({info['goal']['x']:.3f}, {info['goal']['y']:.3f}) "
        f"obstacles={[(round(x, 3), round(y, 3)) for x, y in info['obstacles']]} "
        f"distance={info['distance_to_goal_m']:.3f}m",
        flush=True,
    )
    _print_lidar(env, info, observation)
    _place_camera(info, start=True)
    while True:
        action, _ = policy.predict(observation, deterministic=True)
        observation, reward, terminated, truncated, info = env.step(action)
        c = info["clearances_m"]
        print(
            f"{TAG} {label} step={info['episode_steps']:3d} action={PointNavAction(int(action)).name:<10} "
            f"pose=({info['pose']['x']:.3f}, {info['pose']['y']:.3f}, {info['yaw']:.2f}) "
            f"dist={info['distance_to_goal_m']:.3f} front={c['front']:.2f} left={c['left']:.2f} "
            f"right={c['right']:.2f} reward={reward:.2f}",
            flush=True,
        )
        _print_lidar(env, info, observation)
        _place_camera(info, start=False)
        if terminated or truncated:
            outcome = "COLLISION" if info["collision"] else ("SUCCESS" if terminated else "TIMEOUT")
            print(f"{TAG} {label} {outcome} after {info['episode_steps']} steps", flush=True)
            return outcome
        time.sleep(ACTION_PAUSE_S)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--random", type=int, default=3)
    parser.add_argument("--show-robot", action="store_true")
    parser.add_argument("--no-lidar-view", action="store_true")
    parser.add_argument("--auto", action="store_true")
    parser.add_argument("--profile", choices=("v2", "v3"), default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--camera", choices=("overview", "follow", "none"), default="overview")
    parser.add_argument("--keep-panels", action="store_true")
    args = parser.parse_args()
    global _camera
    _camera = args.camera

    if args.model is not None:
        if not args.model.is_file():
            raise FileNotFoundError(f"V2 model not found: {args.model}")
        if V1B_DIR.resolve() in args.model.resolve().parents:
            raise ValueError("V1B models use a 3D observation and cannot drive the 6D V2 environment.")

    if args.show_robot:
        print(f"{TAG} WARNING: --show-robot changes LiDAR perception. "
              "Presentation only; do not use outcomes for quantitative evaluation.", flush=True)
    profile = args.profile
    if profile is None and args.model is not None and (args.model.parent / "profile.txt").is_file():
        profile = (args.model.parent / "profile.txt").read_text().strip()
    profile = profile or "v2"
    print(f"{TAG} profile={profile}", flush=True)
    lidar_config = (v3_lidar_config(hide_robot_visuals=not args.show_robot) if profile == "v3"
                    else IsaacLidarNavConfig(hide_robot_visuals=not args.show_robot))
    env = IsaacPointNavLidarEnv(headless=False, lidar_config=lidar_config)
    if not args.keep_panels:
        _hide_ui_panels()
    try:
        if not args.no_lidar_view:
            # Draw what the policy's LiDAR sees (debug point cloud in the viewport).
            env._lidar.enable_visualization()
            print(f"{TAG} LiDAR point cloud drawn in the viewport", flush=True)
        if args.model is not None:
            from stable_baselines3 import PPO

            policy = PPO.load(str(args.model), env=env, device="cpu")
            print(f"{TAG} policy=PPO {args.model}", flush=True)
        else:
            policy = ReactiveController()
            print(f"{TAG} policy=REACTIVE_RULES (no --model given)", flush=True)

        if args.seeds:
            episodes = [(f"SCENARIO_{seed}", None, seed) for seed in args.seeds]
        else:
            episodes = [(label, options, None) for label, options in SCRIPTED]
            episodes += [(f"RANDOM_{i + 1}", None, None) for i in range(args.random)]
        outcomes = []
        for index, (label, options, seed) in enumerate(episodes):
            outcomes.append((label, _run_episode(env, policy, label, options, seed)))
            if index < len(episodes) - 1:
                if args.auto:
                    env.settle_scan(60)
                else:
                    input(f"{TAG} {label} complete. Press Enter for the next episode... ")
        for label, outcome in outcomes:
            print(f"{TAG} SUMMARY {label:<18} {outcome}", flush=True)
        if not args.auto:
            input(f"{TAG} All episodes complete. Press Enter to close Isaac Sim... ")
    finally:
        env.close()


if __name__ == "__main__":
    main()
