#!/usr/bin/env python3
"""Deterministic evaluation of a V2/V3 LiDAR policy on fixed, seeded scenarios.

Every model sees exactly the same goals and obstacles (seeds 5000..5000+N-1),
so results are comparable between checkpoints. Training success rates are lower
because PPO samples actions there; this uses the policy's best action.

  TB3_SCENE_USD=~/Desktop/FInalSceneDemo_realsize.usd \\
      ./isaac_run.sh eval_v3.py models/isaac_point_nav_lidar_ppo_v3/checkpoint_1000000_steps.zip [--episodes 100]

The profile (v2/v3) comes from profile.txt next to the model unless --profile is given.
Writes results/eval_<run>_<checkpoint>.csv and prints a summary.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

try:
    from .isaac_point_nav_lidar_env import IsaacLidarNavConfig, IsaacPointNavLidarEnv, v3_lidar_config
except ImportError:  # pragma: no cover - direct-script convenience only
    from isaac_point_nav_lidar_env import IsaacLidarNavConfig, IsaacPointNavLidarEnv, v3_lidar_config

HERE = Path(__file__).resolve().parent
SCENARIO_SEED = 5000
TAG = "[TB03_EVAL_V3]"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--profile", choices=("v2", "v3"), default=None)
    args = parser.parse_args()

    profile = args.profile
    if profile is None and (args.model.parent / "profile.txt").is_file():
        profile = (args.model.parent / "profile.txt").read_text().strip()
    profile = profile or "v2"
    config = v3_lidar_config() if profile == "v3" else IsaacLidarNavConfig()

    from stable_baselines3 import PPO

    env = IsaacPointNavLidarEnv(headless=True, lidar_config=config)
    rows = []
    try:
        model = PPO.load(str(args.model), device="cpu")
        print(f"{TAG} model={args.model} profile={profile} episodes={args.episodes}", flush=True)
        for i in range(args.episodes):
            observation, info = env.reset(seed=SCENARIO_SEED + i)
            goal, obstacles = info["goal"], len(info["obstacles"])
            widths = info.get("obstacle_sizes_m", [])
            closest = float("inf")
            while True:
                action, _ = model.predict(observation, deterministic=True)
                observation, _, terminated, truncated, info = env.step(int(action))
                closest = min(closest, info.get("obstacle_distance_m", float("inf")))
                if terminated or truncated:
                    break
            outcome = "COLLISION" if info["collision"] else ("SUCCESS" if terminated else "TIMEOUT")
            rows.append({
                "episode": i, "seed": SCENARIO_SEED + i, "outcome": outcome,
                "steps": info["episode_steps"], "final_distance_m": round(info["distance_to_goal_m"], 3),
                "goal_x": round(goal["x"], 3), "goal_y": round(goal["y"], 3), "obstacles": obstacles,
                "max_width_m": round(max(widths), 3) if widths else 0.0,
                "closest_obstacle_m": round(closest, 3) if closest != float("inf") else "",
            })
            done = len(rows)
            rate = sum(r["outcome"] == "SUCCESS" for r in rows) / done
            print(f"{TAG} {done:3d}/{args.episodes} {outcome:<9} steps={info['episode_steps']:3d} "
                  f"obstacles={obstacles} success_so_far={rate:.1%}", flush=True)
    finally:
        results = HERE / "results"
        results.mkdir(exist_ok=True)
        if rows:
            path = results / f"eval_{args.model.parent.name}_{args.model.stem}.csv"
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            n = len(rows)
            count = lambda o: sum(r["outcome"] == o for r in rows)
            with_obstacles = [r for r in rows if r["obstacles"]]
            wide = [r for r in rows if r["max_width_m"] and r["max_width_m"] >= 0.30]
            share = lambda subset: f"{sum(r['outcome'] == 'SUCCESS' for r in subset) / len(subset):.1%} of {len(subset)}" if subset else "n/a"
            print(f"{TAG} RESULT {args.model.name}: success {count('SUCCESS') / n:.1%}  "
                  f"collision {count('COLLISION') / n:.1%}  timeout {count('TIMEOUT') / n:.1%}  (n={n})", flush=True)
            print(f"{TAG} RESULT success with obstacles {share(with_obstacles)}; with a box >= 30 cm {share(wide)}", flush=True)
            print(f"{TAG} CSV {path}", flush=True)
        env.close()


if __name__ == "__main__":
    main()
