#!/usr/bin/env python3
"""Paired timing benchmark of navigation controllers on frozen, seeded scenarios.

Follows the Mission-Time & RL Performance Evaluation Framework for the single-robot
layer that exists today:
  B1 rule   - hand-coded reactive controller on the same 6-D observation
  PPO       - trained V3 policy (deterministic)
Every controller runs the SAME scenarios (seeds SCENARIO_SEED+i: identical goal,
obstacles and start), with the same simulator settings. Per episode the Mission Timing
Evaluator records sim-clock time to goal (the episode's makespan), wall time, RTF,
inference latency p50/p95/p99, path length, path efficiency and outcome.

  TB3_SCENE_USD=~/Desktop/FInalSceneDemo_realsize.usd \\
      ./isaac_run.sh eval_timing.py --episodes 30 [--controllers rule ppo]

Writes results/timing/<run>/: manifest.json (frozen scenario + settings), episodes.csv,
events/<controller>_<scenario>.jsonl, summary.json (paired comparison with 95% CIs).
MAPPO (B3) and the multi-robot conditions plug into the same harness once they exist.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import time
from pathlib import Path

try:
    from .config import PointNavAction
    from .isaac_point_nav_lidar_env import IsaacPointNavLidarEnv, v3_lidar_config
    from .mission_timing import MissionTimer, append_csv, compare_paired
except ImportError:  # pragma: no cover - direct-script convenience only
    from config import PointNavAction
    from isaac_point_nav_lidar_env import IsaacPointNavLidarEnv, v3_lidar_config
    from mission_timing import MissionTimer, append_csv, compare_paired

HERE = Path(__file__).resolve().parent
DEFAULT_MODEL = HERE / "models" / "isaac_point_nav_lidar_ppo_v3" / "checkpoint_1000000_steps.zip"
SCENARIO_SEED = 5000          # same scenario bank as eval_v3.py
TAG = "[TB03_TIMING]"


class PpoController:
    name = "ppo"

    def __init__(self, model_path: Path):
        from stable_baselines3 import PPO

        self.model_path = model_path
        self.model = PPO.load(str(model_path), device="cpu")

    def reset(self) -> None:
        pass

    def act(self, observation) -> int:
        action, _ = self.model.predict(observation, deterministic=True)
        return int(action)


class RuleController:
    name = "rule"

    def __init__(self):
        try:
            from .demo_isaac_point_nav_lidar_v2_gui import ReactiveController
        except ImportError:  # pragma: no cover
            from demo_isaac_point_nav_lidar_v2_gui import ReactiveController
        self.inner = ReactiveController()

    def reset(self) -> None:
        self.inner.reset()

    def act(self, observation) -> int:
        action, _ = self.inner.predict(observation)
        return int(action)


def run_episode(env, controller, scenario: int, run_dir: Path, manifest: dict) -> dict:
    controller.reset()
    observation, info = env.reset(seed=scenario)
    goal = (info["goal"]["x"], info["goal"]["y"])
    start = (info["pose"]["x"], info["pose"]["y"])
    straight = math.dist(start, goal)
    timer = MissionTimer(
        mission_id=f"scenario_{scenario}", controller=controller.name, sim_clock=env.sim_time,
        required_tasks=["reach_goal"],
        metadata={"scenario_id": scenario, "obstacles": len(info["obstacles"]),
                  "max_obstacle_width_m": round(max(info.get("obstacle_sizes_m") or [0.0]), 3),
                  "goal_distance_m": round(straight, 3)},
    )
    frames0 = env.physics_frames()
    timer.start()
    timer.task_start("reach_goal")
    timer.record_pose(*start)
    status, reason = "timeout", "step or time budget"
    while True:
        action = timer.time_inference(controller.act, observation)
        observation, _, terminated, truncated, info = env.step(action)
        timer.record_pose(info["pose"]["x"], info["pose"]["y"])
        if info["collision"]:
            status, reason = "collision", f"obstacle contact after {info['episode_steps']} actions"
            timer.task_failed("reach_goal", reason)
            break
        if terminated:
            status, reason = "success", ""
            timer.task_complete("reach_goal", actions=info["episode_steps"])
            timer.robot_complete("robot_0")
            break
        if truncated:
            timer.task_failed("reach_goal", reason)
            break
    frames = env.physics_frames() - frames0
    row = timer.finish(
        status, reason,
        actions=info["episode_steps"],
        final_distance_m=round(info["distance_to_goal_m"], 3),
        path_efficiency=round(straight / timer.path_length_m, 3) if status == "success" and timer.path_length_m > 0 else None,
        physics_frames=frames,
        frames_x_dt_s=round(frames * manifest["physics_dt_s"], 3),
    )
    timer.write_events(run_dir / "events" / f"{controller.name}_{scenario}.jsonl")
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--controllers", nargs="+", choices=("rule", "ppo"), default=["rule", "ppo"])
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--sensor-noise", action="store_true", help="keep V3 training noise (default: clean LiDAR)")
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()

    clean = {} if args.sensor_noise else dict(lidar_noise_std_m=0.0, lidar_ray_dropout=0.0, lidar_sector_dropout=0.0)
    config = v3_lidar_config(**clean)
    env = IsaacPointNavLidarEnv(headless=True, lidar_config=config)
    run_dir = HERE / "results" / "timing" / (args.run_name or time.strftime("%Y%m%d_%H%M%S"))
    run_dir.mkdir(parents=True, exist_ok=True)
    try:
        scenarios = [SCENARIO_SEED + i for i in range(args.episodes)]
        manifest = {
            "scenarios": scenarios, "controllers": args.controllers, "model": str(args.model),
            "scene": os.environ.get("TB3_SCENE_USD", "default"),
            "physics_dt_s": env._physics_dt_s, "isaac_config": vars(env.isaac_config),
            "lidar_config": {k: v for k, v in vars(config).items() if not k.startswith("_")},
            "max_episode_steps": env.point_nav_config.max_episode_steps,
            "action_semantics": {a.name: a.value for a in PointNavAction},
            "sensor_noise": args.sensor_noise, "python": platform.python_version(),
            "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
        controllers = [RuleController() if name == "rule" else PpoController(args.model) for name in args.controllers]
        rows = []
        for scenario in scenarios:
            for controller in controllers:
                row = run_episode(env, controller, scenario, run_dir, manifest)
                rows.append(row)
                append_csv(run_dir / "episodes.csv", row)
                print(f"{TAG} scenario={scenario} {controller.name:<4} {row['status']:<9} "
                      f"time={row['makespan_sim_s'] or row['termination_elapsed_sim_s']}s "
                      f"path={row['path_length_m']}m p95={row['inference_p95_ms']}ms rtf={row['rtf']}", flush=True)
        summary = {}
        if len(args.controllers) == 2:
            budget = env.point_nav_config.max_episode_sim_time_s
            summary = compare_paired(rows, args.controllers[0], args.controllers[1], budget_s=budget)
            summary["path_length"] = compare_paired(rows, args.controllers[0], args.controllers[1], metric="path_length_m")
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        print(f"{TAG} SUMMARY {json.dumps(summary, indent=2)}", flush=True)
        print(f"{TAG} results in {run_dir}", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
