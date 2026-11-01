#!/usr/bin/env python3
"""Paired benchmark of navigation controllers on frozen, seeded scenarios.

Implements the experimental design of the Mission-Time & RL Performance Evaluation
Framework for the single-robot layer:
  b0    existing system: the original CyberScape move_to_point controller (no RL,
        baseline_b0.py), as used under the LLM plans before RL
  rule  heuristic baseline (B1): reactive rules on the same 6-D observation
  ppo   trained V3 PPO policy (deterministic)
Every controller runs the SAME scenarios (identical goal, obstacles, start) in the
same simulator, each with its native perception on the same simulated LiDAR: PPO and
rule use the V3 sensor model PPO was trained/deployed with; b0 uses the robot node's
raw front-sector minimum. Conditions:
  nominal         seeds 5000+, V3 sensor model (the one PPO was trained with)
  generalization  seeds 6000+ (new layouts) and a random initial heading +-90 deg
  robustness      seeds 7000+, 15% of commands delayed by one decision and doubled
                  LiDAR noise / dropout
Per episode the Mission Timing Evaluator records: success / collision / timeout,
sim-clock time to goal, wall time, RTF, decision latency p50/p95/p99, path length,
path efficiency, turn-direction reversals, number of decisions and team return.

Team return is comparable across controllers with different decision rates: it is
the PPO reward written in continuous time,
    5 (d_start - d_end) - 0.02 T_sim + 10 [success] - 10 [collision],
i.e. progress reward, step penalty per ~0.5 s, success bonus, collision penalty.

  TB3_SCENE_USD=~/Desktop/FInalSceneDemo_realsize.usd \\
      ./isaac_run.sh eval_timing.py --condition nominal --episodes 30 --controllers b0 rule ppo

Writes results/timing/<run>/: manifest.json, episodes.csv, events/*.jsonl, summary.json.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import random
import time
from pathlib import Path

try:
    from .baseline_b0 import B0Driver
    from .config import PointNavAction
    from .isaac_point_nav_lidar_env import IsaacPointNavLidarEnv, v3_lidar_config
    from .mission_timing import MissionTimer, append_csv, compare_paired
except ImportError:  # pragma: no cover - direct-script convenience only
    from baseline_b0 import B0Driver
    from config import PointNavAction
    from isaac_point_nav_lidar_env import IsaacPointNavLidarEnv, v3_lidar_config
    from mission_timing import MissionTimer, append_csv, compare_paired

HERE = Path(__file__).resolve().parent
DEFAULT_MODEL = HERE / "models" / "isaac_point_nav_lidar_ppo_v3" / "checkpoint_1000000_steps.zip"
CONDITIONS = {"nominal": 5000, "generalization": 6000, "robustness": 7000}
BUDGET_S = 120.0              # PointNavConfig.max_episode_sim_time_s, same for every controller
DELAY_PROBABILITY = 0.15
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


class B0Controller:
    name = "b0"

    def reset(self) -> None:
        pass


def team_return(d_start: float, d_end: float, sim_s: float, status: str) -> float:
    return 5.0 * (d_start - d_end) - 0.02 * sim_s + (10.0 if status == "success" else 0.0) - (10.0 if status == "collision" else 0.0)


def set_initial_heading(env, offset_deg: float) -> None:
    """Generalization: turn in place to start_yaw + offset before t0 (not timed)."""
    target = env._state().yaw + math.radians(offset_deg)
    for _ in range(400):
        error = math.atan2(math.sin(target - env._state().yaw), math.cos(target - env._state().yaw))
        if abs(error) < math.radians(2.0):
            break
        env.drive_command(0.0, 0.6 * (1 if error > 0 else -1), 2)
    env.stop_robot(10)


def run_episode(env, controller, scenario: int, condition: str, run_dir: Path, manifest: dict) -> dict:
    controller.reset()
    observation, info = env.reset(seed=scenario)
    goal = (info["goal"]["x"], info["goal"]["y"])
    rng = random.Random(scenario * 7 + 1)
    heading_offset = 0.0
    if condition == "generalization":
        heading_offset = rng.uniform(-90.0, 90.0)
        set_initial_heading(env, heading_offset)
    observation, info = env.continue_to_goal(goal)       # fresh step/time budget, obstacles kept
    start = (info["pose"]["x"], info["pose"]["y"])
    d_start = info["distance_to_goal_m"]
    timer = MissionTimer(
        mission_id=f"{condition}_{scenario}", controller=controller.name, sim_clock=env.sim_time,
        required_tasks=["reach_goal"],
        metadata={"condition": condition, "scenario_id": scenario, "obstacles": len(info["obstacles"]),
                  "max_obstacle_width_m": round(max(info.get("obstacle_sizes_m") or [0.0]), 3),
                  "goal_distance_m": round(d_start, 3), "initial_heading_offset_deg": round(heading_offset, 1)},
    )
    frames0 = env.physics_frames()
    timer.start()
    timer.task_start("reach_goal")
    timer.record_pose(*start)
    turns = {"last": 0, "reversals": 0}

    def note_turn(direction: int) -> None:
        if direction and turns["last"] and direction != turns["last"]:
            turns["reversals"] += 1
        if direction:
            turns["last"] = direction

    delayed = random.Random(scenario * 13 + 5)

    if controller.name == "b0":
        def on_tick(tick_info):
            timer.record_pose(tick_info["pose"]["x"], tick_info["pose"]["y"])

        def disturb(v, w, last):
            # robustness: the previous command is still executing (one-decision delay)
            if condition == "robustness" and delayed.random() < DELAY_PROBABILITY:
                return last
            return v, w

        def decide_turn(v, w, last):
            v, w = disturb(v, w, last)
            note_turn(0 if abs(w) < 0.2 or abs(v) > 0.05 else (1 if w > 0 else -1))
            return v, w

        driver = B0Driver(env, seed=scenario, budget_s=BUDGET_S, on_tick=on_tick, disturb=decide_turn,
                          noisy_sensor=condition == "robustness")
        status, reason = driver.move_to_point(goal)   # closed-form controller: no policy inference to time
        decisions = driver.decisions
        info = env._info(env._state())
    else:
        status, reason = "timeout", "time budget"
        previous = int(PointNavAction.STAY)
        decisions = 0
        while True:
            action = timer.time_inference(controller.act, observation)
            if condition == "robustness" and delayed.random() < DELAY_PROBABILITY:
                action = previous
            previous = action
            decisions += 1
            note_turn({int(PointNavAction.TURN_LEFT): 1, int(PointNavAction.TURN_RIGHT): -1}.get(int(action), 0))
            observation, _, terminated, truncated, info = env.step(action)
            timer.record_pose(info["pose"]["x"], info["pose"]["y"])
            if info["collision"]:
                status, reason = "collision", f"obstacle contact after {info['episode_steps']} actions"
                break
            if terminated:
                status, reason = "success", ""
                break
            if truncated or env._episode_elapsed_sim_s >= BUDGET_S:
                status, reason = "timeout", "time budget"
                break
    if status == "success":
        timer.task_complete("reach_goal", decisions=decisions)
        timer.robot_complete("robot_0")
    else:
        timer.task_failed("reach_goal", reason)
    frames = env.physics_frames() - frames0
    d_end = info["distance_to_goal_m"]
    sim_s = env.sim_time() - timer.start_sim
    row = timer.finish(
        status, reason,
        decisions=decisions,
        turn_reversals=turns["reversals"],
        final_distance_m=round(d_end, 3),
        path_efficiency=round(math.dist(start, goal) / timer.path_length_m, 3) if status == "success" and timer.path_length_m > 0 else None,
        team_return=round(team_return(d_start, d_end, sim_s, status), 3),
        physics_frames=frames,
        frames_x_dt_s=round(frames * manifest["physics_dt_s"], 3),
    )
    timer.write_events(run_dir / "events" / f"{controller.name}_{scenario}.jsonl")
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=sorted(CONDITIONS), default="nominal")
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--controllers", nargs="+", choices=("b0", "rule", "ppo"), default=["b0", "rule", "ppo"])
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--clean-sensor", action="store_true", help="ideal LiDAR (default: V3 sensor model)")
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()

    overrides = {}
    if args.clean_sensor:
        overrides = dict(lidar_noise_std_m=0.0, lidar_ray_dropout=0.0, lidar_sector_dropout=0.0)
    if args.condition == "robustness":
        overrides = dict(lidar_noise_std_m=0.03, lidar_ray_dropout=0.20, lidar_sector_dropout=0.10)
    config = v3_lidar_config(**overrides)
    env = IsaacPointNavLidarEnv(headless=True, lidar_config=config)
    run_dir = HERE / "results" / "timing" / (args.run_name or f"{args.condition}_{time.strftime('%Y%m%d_%H%M%S')}")
    run_dir.mkdir(parents=True, exist_ok=True)
    try:
        scenarios = [CONDITIONS[args.condition] + i for i in range(args.episodes)]
        manifest = {
            "condition": args.condition, "scenarios": scenarios, "controllers": args.controllers,
            "model": str(args.model), "scene": os.environ.get("TB3_SCENE_USD", "default"),
            "physics_dt_s": env._physics_dt_s, "budget_s": BUDGET_S,
            "delay_probability": DELAY_PROBABILITY if args.condition == "robustness" else 0.0,
            "isaac_config": vars(env.isaac_config),
            "lidar_config": {k: v for k, v in vars(config).items() if not k.startswith("_")},
            "max_episode_steps": env.point_nav_config.max_episode_steps,
            "action_semantics": {a.name: a.value for a in PointNavAction},
            "python": platform.python_version(), "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
        controllers = []
        for name in args.controllers:
            controllers.append(B0Controller() if name == "b0" else RuleController() if name == "rule" else PpoController(args.model))
        rows = []
        for scenario in scenarios:
            for controller in controllers:
                row = run_episode(env, controller, scenario, args.condition, run_dir, manifest)
                rows.append(row)
                append_csv(run_dir / "episodes.csv", row)
                print(f"{TAG} {args.condition} scenario={scenario} {controller.name:<4} {row['status']:<9} "
                      f"time={row['makespan_sim_s'] or row['termination_elapsed_sim_s']}s path={row['path_length_m']}m "
                      f"reversals={row['turn_reversals']} return={row['team_return']} rtf={row['rtf']}", flush=True)
        summary = {}
        names = [c.name for c in controllers]
        for base in names:
            for cand in names:
                if base < cand or (base == "b0" and cand != "b0"):
                    if f"{base}_vs_{cand}" in summary or f"{cand}_vs_{base}" in summary or base == cand:
                        continue
                    summary[f"{base}_vs_{cand}"] = compare_paired(rows, base, cand, budget_s=BUDGET_S)
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        print(f"{TAG} results in {run_dir}", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
