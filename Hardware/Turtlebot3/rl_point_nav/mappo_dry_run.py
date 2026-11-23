#!/usr/bin/env python3
"""MAPPO V5 on a Burger WITHOUT driving: print and log the 15 inputs and the chosen action.

Handoff section 7, steps 3 and 4. Nothing is sent to /cmd_vel: this node's only command
topic is a dummy one. Run next to bringup, robot standing still (or pushed by hand).

  # step 3: teammate fixed "far and idle"; inputs 0..10 should look like V4's
  python3 mappo_dry_run.py --robot TURTLEBOT3_1 --start 0 0 0 --goal 1.5 0

  # step 4: both robots, each told the other's IP; compare teammate_x / teammate_y
  # with a tape measure (e.g. Burger 2 1.0 m straight ahead of Burger 1 -> Burger 1 reads 0.50, 0.00)
  python3 mappo_dry_run.py --robot TURTLEBOT3_1 --start 0 0 0   --goal 1.5 0 --peer <BURGER2_IP>
  python3 mappo_dry_run.py --robot TURTLEBOT3_2 --start 1 0 180 --goal -0.5 0 --peer <BURGER1_IP>

--start is where this robot stands now in the shared mission frame: x y (m) and heading
(degrees, CCW from the frame's +x). Both robots must use the same frame.
Writes missions/<time>_mappo_dry_run_<robot>.csv (time, shared pose, 15 inputs, action).
"""

from __future__ import annotations

import argparse
import dataclasses
import math
from pathlib import Path
import time

from config import PointNavAction
from mappo_hardware import POSE_PORT, V5_INPUT_NAMES, teammate_local
from policy_numpy import NumpyPolicy
from ros_mappo_env import MappoRosEnv, TeammateLost
from ros_point_nav_lidar_env import v2_config

HERE = Path(__file__).resolve().parent
DEFAULT_MODEL = HERE / "models" / "mappo_v5_kl05" / "best_model.npz"
DUMMY_CMD_TOPIC = "/mappo_dry_run/cmd_vel_unused"


class DryRunEnv(MappoRosEnv):
    def _wait_for_command_subscriber(self, timeout_s: float) -> None:
        pass   # nothing listens to the dummy command topic, and nothing should


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--robot", required=True, choices=("TURTLEBOT3_1", "TURTLEBOT3_2"))
    parser.add_argument("--start", nargs=3, type=float, metavar=("X", "Y", "YAW_DEG"), required=True)
    parser.add_argument("--goal", nargs=2, type=float, metavar=("X", "Y"), required=True)
    parser.add_argument("--peer", default=None, help="the other Burger's IP (omit = teammate far and idle)")
    parser.add_argument("--listen-port", type=int, default=POSE_PORT)
    parser.add_argument("--peer-port", type=int, default=POSE_PORT)
    parser.add_argument("--active", action="store_true", help="tell the teammate this robot is driving")
    parser.add_argument("--rate", type=float, default=2.0, help="lines per second")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    args = parser.parse_args()

    policy = NumpyPolicy(args.model)
    if policy.obs_dim != len(V5_INPUT_NAMES):
        raise SystemExit(f"{args.model} takes {policy.obs_dim} inputs; MAPPO V5 takes {len(V5_INPUT_NAMES)}")
    config = dataclasses.replace(v2_config(), cmd_vel_topic=DUMMY_CMD_TOPIC)
    sx, sy, syaw = args.start
    env = DryRunEnv(args.robot, (sx, sy, math.radians(syaw)), config=config, peer=args.peer,
                    listen_port=args.listen_port, peer_port=args.peer_port)
    env.active = args.active
    env.set_goal_shared(*args.goal)
    env.wait_for_scan_after(time.monotonic() - 10.0, timeout_s=5.0)

    (HERE / "missions").mkdir(exist_ok=True)
    csv_path = HERE / "missions" / f"{time.strftime('%Y%m%d_%H%M%S')}_mappo_dry_run_{args.robot}.csv"
    csv = open(csv_path, "w", buffering=1)
    csv.write("t_s,x_m,y_m,yaw_deg," + ",".join(V5_INPUT_NAMES) + ",action,teammate_age_s\n")
    print(f"[MAPPO_DRY] {args.robot} role={env.role:.0f} start=({sx:.2f}, {sy:.2f}, {syaw:.0f} deg) "
          f"goal=({args.goal[0]:.2f}, {args.goal[1]:.2f}) peer={args.peer or 'none (teammate far)'}; "
          f"logging {csv_path}; NOT driving. Ctrl+C to stop.", flush=True)
    t0 = time.monotonic()
    try:
        while True:
            state = env._current_state()
            me = env.frame.to_shared((state.x, state.y, state.yaw))
            try:
                obs = env._observation(state)
            except TeammateLost as exc:
                print(f"[MAPPO_DRY] {exc}", flush=True)
                time.sleep(1.0 / args.rate)
                continue
            action = PointNavAction(policy.act(obs)).name
            mate = env.last_teammate
            age = "" if mate is None else f"{mate['age_s']:.2f}"
            csv.write(f"{time.monotonic() - t0:.2f},{me[0]:.3f},{me[1]:.3f},{math.degrees(me[2]):.1f},"
                      + ",".join(f"{v:.4f}" for v in obs) + f",{action},{age}\n")
            sectors = " ".join(f"{n}={v:.2f}" for n, v in zip(V5_INPUT_NAMES[3:11], obs[3:11]))
            line = (f"[MAPPO_DRY] me=({me[0]:+.2f},{me[1]:+.2f},{math.degrees(me[2]):+4.0f}deg) "
                    f"goal d={obs[0] * 2.0:.2f}m b={math.degrees(math.atan2(obs[1], obs[2])):+4.0f}deg | "
                    f"{sectors} | role={obs[11]:.0f} mate=({obs[12]:+.2f},{obs[13]:+.2f}) active={obs[14]:.0f}")
            if mate is not None:
                ahead, left = teammate_local(me, (mate["x"], mate["y"]))
                line += f" [{ahead:+.2f} m ahead, {left:+.2f} m left, age {mate['age_s']:.2f}s]"
            print(line + f" -> {action}", flush=True)
            time.sleep(1.0 / args.rate)
    except KeyboardInterrupt:
        pass
    finally:
        csv.close()
        env.close()
        print(f"[MAPPO_DRY] saved {csv_path}", flush=True)


if __name__ == "__main__":
    main()
