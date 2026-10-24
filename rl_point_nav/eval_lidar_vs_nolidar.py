#!/usr/bin/env python3
"""Fair comparison: navigation WITH LiDAR (PPO V2) vs WITHOUT LiDAR (PPO V1B).

All policies run in the same V2 environment on the same seeded scenarios:

  V1B (no LiDAR)   frozen V1B policy, given only obs[:3] = [d_norm, sin(b), cos(b)]
  V2  (LiDAR)      V2 checkpoint, full 6D observation
  RULES (LiDAR)    reactive rule-based controller from the GUI demo (reference)

Scenario sets:
  NO_OBSTACLES     the 7 V1B goal-bank goals, no obstacles
  OBSTACLES        N seeded scenarios sampled exactly like V2 training
                   (goal from the goal bank, 70% blocking box + 0-2 distractors)

Deterministic actions. Reports success / collision / timeout rates, mean steps
to success and mean final distance, and writes a CSV to results/.

  --v2-model PATH   default: highest checkpoint_*_steps.zip in the V2 model dir
  --episodes N      number of OBSTACLES scenarios (default 30)
  --gui             run with the Isaac GUI (default headless)
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import re
from statistics import mean

import gymnasium as gym

try:
    from .isaac_point_nav_env import LEFT_GOALS, RIGHT_GOALS, STRAIGHT_GOALS
    from .isaac_point_nav_lidar_env import IsaacPointNavLidarEnv
    from .demo_isaac_point_nav_lidar_v2_gui import ReactiveController
except ImportError:  # pragma: no cover - direct-script convenience only
    from isaac_point_nav_env import LEFT_GOALS, RIGHT_GOALS, STRAIGHT_GOALS
    from isaac_point_nav_lidar_env import IsaacPointNavLidarEnv
    from demo_isaac_point_nav_lidar_v2_gui import ReactiveController


HERE = Path(__file__).resolve().parent
V1B_MODEL = HERE / "models" / "isaac_point_nav_ppo_v1b" / "checkpoint_15000_steps.zip"
V2_DIR = HERE / "models" / "isaac_point_nav_lidar_ppo_v2"
RESULTS_DIR = HERE / "results"
SCENARIO_SEED = 1000
TAG = "[TB03_EVAL]"


def _latest_v2_checkpoint() -> Path:
    checkpoints = sorted(
        V2_DIR.glob("checkpoint_*_steps.zip"),
        key=lambda p: int(re.search(r"checkpoint_(\d+)_steps", p.name).group(1)),
    )
    if not checkpoints:
        raise FileNotFoundError(f"No V2 checkpoints in {V2_DIR}")
    return checkpoints[-1]


class _SB3Policy:
    """Deterministic SB3 policy; V1B receives only the first 3 observation entries."""

    def __init__(self, model, obs_dims: int) -> None:
        self.model = model
        self.obs_dims = obs_dims

    def reset(self) -> None:
        pass

    def act(self, observation):
        action, _ = self.model.predict(observation[: self.obs_dims], deterministic=True)
        return int(action)


class _RulePolicy:
    def __init__(self) -> None:
        self.controller = ReactiveController()

    def reset(self) -> None:
        self.controller.reset()

    def act(self, observation):
        action, _ = self.controller.predict(observation)
        return int(action)


def _scenarios(env, n_obstacle: int) -> list[tuple[str, dict]]:
    """Fixed scenario list, generated once and replayed identically for every policy."""
    scenarios = [
        (f"NO_OBSTACLES:{family}{i}", {"goal": goal, "obstacles": []})
        for family, goals in (("L", LEFT_GOALS), ("R", RIGHT_GOALS), ("S", STRAIGHT_GOALS))
        for i, goal in enumerate(goals)
    ]
    start = (float(env._default_position_host[0, 0]), float(env._default_position_host[0, 1]))
    for i in range(n_obstacle):
        gym.Env.reset(env, seed=SCENARIO_SEED + i)
        goal = env._sample_v1_goal()
        obstacles = env._sample_obstacles(start, goal)
        scenarios.append((f"OBSTACLES:{i:03d}", {"goal": goal, "obstacles": obstacles}))
    return scenarios


def _run(env, policy, options) -> dict:
    policy.reset()
    observation, info = env.reset(options=options)
    while True:
        observation, _, terminated, truncated, info = env.step(policy.act(observation))
        if terminated or truncated:
            break
    collision = bool(info["collision"])
    return {
        "outcome": "COLLISION" if collision else ("SUCCESS" if terminated else "TIMEOUT"),
        "steps": int(info["episode_steps"]),
        "final_distance_m": float(info["distance_to_goal_m"]),
        "obstacles": len(options["obstacles"]),
    }


def _summary(rows: list[dict]) -> dict:
    n = len(rows)
    successes = [r for r in rows if r["outcome"] == "SUCCESS"]
    return {
        "episodes": n,
        "success": len(successes) / n,
        "collision": sum(r["outcome"] == "COLLISION" for r in rows) / n,
        "timeout": sum(r["outcome"] == "TIMEOUT" for r in rows) / n,
        "steps_to_success": mean(r["steps"] for r in successes) if successes else float("nan"),
        "final_distance_m": mean(r["final_distance_m"] for r in rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v2-model", type=Path, default=None)
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--gui", action="store_true")
    args = parser.parse_args()

    from stable_baselines3 import PPO

    v2_path = args.v2_model or _latest_v2_checkpoint()
    for path in (V1B_MODEL, v2_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    env = IsaacPointNavLidarEnv(headless=not args.gui)
    rows: list[dict] = []
    try:
        policies = {
            "V1B (no LiDAR)": _SB3Policy(PPO.load(str(V1B_MODEL), device="cpu"), obs_dims=3),
            "V2 (LiDAR)": _SB3Policy(PPO.load(str(v2_path), device="cpu"), obs_dims=6),
            "RULES (LiDAR)": _RulePolicy(),
        }
        scenarios = _scenarios(env, args.episodes)
        print(f"{TAG} V1B={V1B_MODEL.name} V2={v2_path.name} scenarios={len(scenarios)}", flush=True)
        for name, policy in policies.items():
            for label, options in scenarios:
                result = _run(env, policy, options)
                rows.append({"policy": name, "scenario": label, **result})
                print(
                    f"{TAG} {name:<15} {label:<18} {result['outcome']:<9} steps={result['steps']:3d} "
                    f"final={result['final_distance_m']:.3f} m",
                    flush=True,
                )
    finally:
        env.close()

    RESULTS_DIR.mkdir(exist_ok=True)
    csv_path = RESULTS_DIR / f"eval_lidar_vs_nolidar_{v2_path.stem}.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"{TAG} ================ RESULTS ({v2_path.name}) ================", flush=True)
    print(
        f"{TAG} {'policy':<15} {'set':<13} {'n':>3} {'success':>8} {'collision':>10} {'timeout':>8} "
        f"{'steps→goal':>11} {'final d':>8}",
        flush=True,
    )
    for name in policies:
        for scenario_set in ("NO_OBSTACLES", "OBSTACLES"):
            subset = [r for r in rows if r["policy"] == name and r["scenario"].startswith(scenario_set)]
            s = _summary(subset)
            print(
                f"{TAG} {name:<15} {scenario_set:<13} {s['episodes']:>3} {s['success']:>8.0%} "
                f"{s['collision']:>10.0%} {s['timeout']:>8.0%} {s['steps_to_success']:>11.1f} "
                f"{s['final_distance_m']:>7.3f}m",
                flush=True,
            )
    print(f"{TAG} per-episode CSV: {csv_path}", flush=True)


if __name__ == "__main__":
    main()
