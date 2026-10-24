#!/usr/bin/env python3
"""GUI visualization of the V2 LiDAR point-navigation environment.

  --model PATH   run a trained V2 PPO checkpoint (deterministic)
  (no --model)   run a simple reactive rule-based controller on the SAME 6D
                 observation, to inspect the environment before training
  --random N     number of extra randomly sampled episodes (default 3)
  --show-robot   select visible robot BEFORE physics; presentation only, not evaluation

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
    from .isaac_point_nav_lidar_env import IsaacPointNavLidarEnv, IsaacLidarNavConfig
except ImportError:  # pragma: no cover - direct-script convenience only
    from config import PointNavAction
    from isaac_point_nav_lidar_env import IsaacPointNavLidarEnv, IsaacLidarNavConfig


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


def _run_episode(env, policy, label: str, options) -> str:
    if hasattr(policy, "reset"):
        policy.reset()
    observation, info = env.reset(options=options)
    print(
        f"{TAG} {label} goal=({info['goal']['x']:.3f}, {info['goal']['y']:.3f}) "
        f"obstacles={[(round(x, 3), round(y, 3)) for x, y in info['obstacles']]} "
        f"distance={info['distance_to_goal_m']:.3f}m",
        flush=True,
    )
    _print_lidar(env, info, observation)
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
    args = parser.parse_args()

    if args.model is not None:
        if not args.model.is_file():
            raise FileNotFoundError(f"V2 model not found: {args.model}")
        if V1B_DIR.resolve() in args.model.resolve().parents:
            raise ValueError("V1B models use a 3D observation and cannot drive the 6D V2 environment.")

    if args.show_robot:
        print(f"{TAG} WARNING: --show-robot changes LiDAR perception. "
              "Presentation only; do not use outcomes for quantitative evaluation.", flush=True)
    env = IsaacPointNavLidarEnv(
        headless=False, lidar_config=IsaacLidarNavConfig(hide_robot_visuals=not args.show_robot)
    )
    try:
        if args.model is not None:
            from stable_baselines3 import PPO

            policy = PPO.load(str(args.model), env=env, device="cpu")
            print(f"{TAG} policy=PPO {args.model}", flush=True)
        else:
            policy = ReactiveController()
            print(f"{TAG} policy=REACTIVE_RULES (no --model given)", flush=True)

        episodes = list(SCRIPTED) + [(f"RANDOM_{i + 1}", None) for i in range(args.random)]
        outcomes = []
        for index, (label, options) in enumerate(episodes):
            outcomes.append((label, _run_episode(env, policy, label, options)))
            if index < len(episodes) - 1:
                input(f"{TAG} {label} complete. Press Enter for the next episode... ")
        for label, outcome in outcomes:
            print(f"{TAG} SUMMARY {label:<18} {outcome}", flush=True)
        input(f"{TAG} All episodes complete. Press Enter to close Isaac Sim... ")
    finally:
        env.close()


if __name__ == "__main__":
    main()
