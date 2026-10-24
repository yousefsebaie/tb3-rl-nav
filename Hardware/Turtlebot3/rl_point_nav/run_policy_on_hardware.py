#!/usr/bin/env python3
"""Drive the real TurtleBot3 Burger to a goal with a trained PPO policy.

Runs ON THE ROBOT'S PI (or any machine on the robot's ROS 2 network) with only
NumPy + ROS 2 Humble. Needs, already running on the Pi:
    1) turtlebot3 bringup   (/odom, /cmd_vel, and /scan)
    2) safety_filter_node.py (/cmd_vel_raw -> /cmd_vel)

The goal is RELATIVE TO WHERE THE ROBOT IS NOW: +x = straight ahead,
+y = to the robot's left, in metres. Trained range: 0.5-2.0 m away,
within about +/-60 deg of straight ahead.

Examples:
    # 1) No motion: print what the policy sees and would do (sanity check)
    python3 run_policy_on_hardware.py --model models/isaac_point_nav_lidar_ppo_v2/checkpoint_170000_steps.npz \
        --goal-x 1.0 --goal-y 0.0 --dry-run
    # 2) Drive: V1B (no LiDAR), 1 m straight ahead
    python3 run_policy_on_hardware.py --model models/isaac_point_nav_ppo_v1b/checkpoint_15000_steps.npz --goal-x 1.0 --goal-y 0.0
    # 3) Drive: V2 (LiDAR)
    python3 run_policy_on_hardware.py --model models/isaac_point_nav_lidar_ppo_v2/checkpoint_170000_steps.npz --goal-x 1.2 --goal-y 0.0

Ctrl-C stops the robot. Every step is logged to logs/<time>_<model>.csv.
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path

try:
    from .config import PointNavAction
    from .policy_numpy import NumpyPolicy
    from .ros_point_nav_lidar_env import RosPointNavLidarEnv
except ImportError:  # pragma: no cover - direct-script convenience only
    from config import PointNavAction
    from policy_numpy import NumpyPolicy
    from ros_point_nav_lidar_env import RosPointNavLidarEnv


HERE = Path(__file__).resolve().parent
TRAINED_MIN_M, TRAINED_MAX_M, TRAINED_MAX_BEARING_DEG = 0.5, 2.0, 65.0
DRY_RUN_READINGS = 10


def _fmt_obs(obs) -> str:
    names = ["d_norm", "sin_b", "cos_b", "front", "left", "right"]
    return " ".join(f"{n}={v:.2f}" for n, v in zip(names, obs))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a trained point-nav policy on the real TurtleBot3")
    parser.add_argument("--model", type=Path, required=True, help="exported .npz policy (see export_policy_numpy.py)")
    parser.add_argument("--goal-x", type=float, required=True, help="metres ahead of the robot")
    parser.add_argument("--goal-y", type=float, required=True, help="metres to the robot's left")
    parser.add_argument("--dry-run", action="store_true", help="read sensors and print actions; never move")
    parser.add_argument("--allow-out-of-range", action="store_true", help="allow goals outside the trained range")
    args = parser.parse_args()

    policy = NumpyPolicy(args.model)
    use_lidar = policy.obs_dim == 6
    if policy.obs_dim not in (3, 6):
        raise SystemExit(f"Unsupported policy input size {policy.obs_dim}")

    distance = math.hypot(args.goal_x, args.goal_y)
    bearing_deg = math.degrees(math.atan2(args.goal_y, args.goal_x))
    if not (TRAINED_MIN_M <= distance <= TRAINED_MAX_M and abs(bearing_deg) <= TRAINED_MAX_BEARING_DEG):
        message = (f"Goal is {distance:.2f} m at {bearing_deg:.0f} deg; the policy was trained on "
                   f"{TRAINED_MIN_M}-{TRAINED_MAX_M} m within +/-{TRAINED_MAX_BEARING_DEG:.0f} deg.")
        if not args.allow_out_of_range:
            raise SystemExit(message + " Use --allow-out-of-range to try anyway.")
        print("WARNING:", message)

    env = RosPointNavLidarEnv(0.0, 0.0, use_lidar=use_lidar)
    log_file = None
    try:
        # Convert the robot-relative goal into the odometry frame using the current pose.
        start = env._current_state()
        gx = start.x + args.goal_x * math.cos(start.yaw) - args.goal_y * math.sin(start.yaw)
        gy = start.y + args.goal_x * math.sin(start.yaw) + args.goal_y * math.cos(start.yaw)
        obs, info = env.reset(goal=(gx, gy))
        print(f"[RL] model={args.model.name} ({'V2 LiDAR' if use_lidar else 'V1B no LiDAR'}) "
              f"start=({start.x:.3f}, {start.y:.3f}, yaw {math.degrees(start.yaw):.0f} deg) "
              f"goal(odom)=({gx:.3f}, {gy:.3f}) distance={info['distance_to_goal_m']:.3f} m", flush=True)

        if args.dry_run:
            for i in range(DRY_RUN_READINGS):
                if use_lidar:
                    env.wait_for_scan_after(time.monotonic())
                obs = env._observation(env._current_state())
                action = PointNavAction(policy.act(obs)).name
                print(f"[RL][dry-run {i + 1}/{DRY_RUN_READINGS}] {_fmt_obs(obs)} -> would do {action}", flush=True)
                time.sleep(0.5)
            print("[RL] Dry run finished; the robot was never commanded to move.")
            return

        log_dir = HERE / "logs"
        log_dir.mkdir(exist_ok=True)
        log_path = log_dir / f"{time.strftime('%Y%m%d_%H%M%S')}_{args.model.stem}.csv"
        log_file = log_path.open("w", newline="")
        writer = csv.writer(log_file)
        writer.writerow(["step", "action", "x", "y", "yaw_deg", "distance_m", "front", "left", "right",
                         "reward", "collision"])

        outcome, steps = "TIMEOUT", 0
        t0 = time.monotonic()
        while True:
            action = policy.act(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            steps = info["episode_steps"]
            pose = info["pose"]
            lidar = obs[3:] if use_lidar else ["", "", ""]
            writer.writerow([steps, PointNavAction(action).name, f"{pose['x']:.4f}", f"{pose['y']:.4f}",
                             f"{math.degrees(pose['yaw']):.1f}", f"{info['distance_to_goal_m']:.4f}",
                             *lidar, f"{reward:.3f}", info["collision"]])
            print(f"[RL] step={steps:3d} {PointNavAction(action).name:<10} "
                  f"dist={info['distance_to_goal_m']:.3f} m  {_fmt_obs(obs)}", flush=True)
            if terminated or truncated:
                outcome = "COLLISION" if info["collision"] else ("SUCCESS" if terminated else "TIMEOUT")
                break

        print(f"[RL] RESULT {outcome}  steps={steps}  final_distance={info['distance_to_goal_m']:.3f} m  "
              f"time={time.monotonic() - t0:.1f} s  log={log_path}", flush=True)
    except KeyboardInterrupt:
        print("\n[RL] Stopped by user.")
    finally:
        if log_file is not None:
            log_file.close()
        env.close()   # always sends a zero velocity


if __name__ == "__main__":
    main()
