#!/usr/bin/env python3
"""Drive one Burger to a goal with MAPPO V5, next to its teammate (one phase, handoff 5.1-5.3).

The hardware counterpart of train_mappo.drive_phase() with --gap-assist, for ONE robot; run
it on both Pis at the same time, each told the other's IP:

  1. Goal Interface: turn in place until the goal is within 15 deg of ahead.
  2. MAPPO V5 (15 inputs, ros_mappo_env.py) picks each action.
  3. Follow-the-gap hand-over when the robot is stuck (< 6 cm in 18 steps), handed back
     after 1.2 m or 150 steps (mappo_hardware.GapAssist).
  4. Done at the goal (0.05 m), on contact (LiDAR within 13 cm of the centre: the phase
     fails) or after 400 steps. Then this robot reports active = 0 and keeps sending its
     pose, standing still, until the teammate is done too (or 120 s), so the teammate
     never loses it.

A robot with nothing to do this phase runs with --hold: it stays where it is, active = 0.
Commands go straight to /cmd_vel (no safety filter; the sim robots pass closer than its
0.30 m). The teammate pose must be fresh (0.5 s) or the robot holds still.

  python3 mappo_drive.py --robot TURTLEBOT3_1 --start 0 0 0      --goal 1.8 0 --peer <BURGER2_IP>
  python3 mappo_drive.py --robot TURTLEBOT3_2 --start 1.8 0 180  --goal 0 0   --peer <BURGER1_IP>

Writes missions/<time>_mappo_<robot>.csv: time, shared pose, the 15 inputs, policy action,
final action, why (policy / align / gap ...), teammate age.
"""

from __future__ import annotations

import argparse
import dataclasses
import math
from pathlib import Path
import time

from config import PointNavAction
from mappo_hardware import ALIGN_UNTIL_DEG, POSE_PORT, STAY, TURN_LEFT, TURN_RIGHT, V5_INPUT_NAMES, GapAssist, \
    free_gap_heading
from policy_numpy import NumpyPolicy
from ros_mappo_env import MappoRosEnv, TeammateLost
from ros_point_nav_lidar_env import v2_config

HERE = Path(__file__).resolve().parent
DEFAULT_MODEL = HERE / "models" / "mappo_v5_kl05" / "best_model.npz"
MISSION_STEP_LIMIT = 400        # cyberscape_mappo_mission.MISSION_STEP_LIMIT
TEAMMATE_WAIT_S = 30.0
AFTER_DONE_WAIT_S = 120.0
ROBOT_CONTACT_M = 2 * 0.105 + 0.03


def log(message: str) -> None:
    print(f"[MAPPO_DRIVE] {message}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--robot", required=True, choices=("TURTLEBOT3_1", "TURTLEBOT3_2"))
    parser.add_argument("--start", nargs=3, type=float, metavar=("X", "Y", "YAW_DEG"), required=True)
    parser.add_argument("--goal", nargs=2, type=float, metavar=("X", "Y"))
    parser.add_argument("--hold", action="store_true", help="nothing to do this phase: stay, active = 0")
    parser.add_argument("--peer", required=True, help="the other Burger's IP")
    parser.add_argument("--listen-port", type=int, default=POSE_PORT)
    parser.add_argument("--peer-port", type=int, default=POSE_PORT)
    parser.add_argument("--no-gap", action="store_true", help="MAPPO only, no follow-the-gap hand-over")
    parser.add_argument("--max-steps", type=int, default=MISSION_STEP_LIMIT)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    args = parser.parse_args()
    if args.goal is None and not args.hold:
        parser.error("give --goal, or --hold for a robot with nothing to do")

    policy = NumpyPolicy(args.model)
    if policy.obs_dim != len(V5_INPUT_NAMES):
        raise SystemExit(f"{args.model} takes {policy.obs_dim} inputs; MAPPO V5 takes {len(V5_INPUT_NAMES)}")
    config = dataclasses.replace(v2_config(), cmd_vel_topic="/cmd_vel", max_episode_steps=args.max_steps,
                                 max_episode_sim_time_s=2.0 * args.max_steps)
    sx, sy, syaw = args.start
    env = MappoRosEnv(args.robot, (sx, sy, math.radians(syaw)), config=config, peer=args.peer,
                      listen_port=args.listen_port, peer_port=args.peer_port)
    (HERE / "missions").mkdir(exist_ok=True)
    csv_path = HERE / "missions" / f"{time.strftime('%Y%m%d_%H%M%S')}_mappo_{args.robot}.csv"
    csv = open(csv_path, "w", buffering=1)
    csv.write("t_s,x_m,y_m,yaw_deg," + ",".join(V5_INPUT_NAMES) + ",policy_action,action,why,teammate_age_s\n")
    t0 = time.monotonic()
    outcome = "ABORTED"
    try:
        log(f"{args.robot} start=({sx:.2f}, {sy:.2f}, {syaw:.0f} deg) "
            + ("HOLD" if args.hold else f"goal=({args.goal[0]:.2f}, {args.goal[1]:.2f})")
            + f" peer={args.peer} gap={'off' if args.no_gap else 'on'}; log {csv_path}")
        deadline = time.monotonic() + TEAMMATE_WAIT_S
        while env.teammate() is None:
            if time.monotonic() > deadline:
                raise SystemExit(f"no pose from the teammate at {args.peer} within {TEAMMATE_WAIT_S:.0f} s")
            time.sleep(0.1)
        mate = env.teammate()
        me = env.latest_shared()
        log(f"teammate {mate['robot']} at ({mate['x']:.2f}, {mate['y']:.2f}), "
            f"{math.dist(me[:2], (mate['x'], mate['y'])):.2f} m from me")

        if args.hold:
            outcome = "HOLD"
        else:
            outcome = drive(env, policy, tuple(args.goal), args.max_steps, not args.no_gap, csv, t0)
        log(f"RESULT {outcome}")

        # Stay visible to the teammate (active = 0) until it is done too.
        env.active = False
        seen_active = not args.hold   # a holding robot first waits for the mover to start
        deadline = time.monotonic() + AFTER_DONE_WAIT_S
        while time.monotonic() < deadline:
            mate = env.teammate()
            if mate is not None and mate["active"]:
                seen_active = True
            elif seen_active:
                break
            env.stop()
            time.sleep(0.2)
        time.sleep(1.0)   # let the last poses go out
    except KeyboardInterrupt:
        log("interrupted")
    finally:
        csv.close()
        env.close()   # sends a zero velocity
        log(f"{args.robot} {outcome}; saved {csv_path}")


def drive(env: MappoRosEnv, policy: NumpyPolicy, goal: tuple[float, float], max_steps: int, use_gap: bool,
          csv, t0: float, abort=lambda: False) -> str:
    """One move: align, then MAPPO (+ gap hand-over) to ``goal`` in the shared frame.

    Returns SUCCESS, COLLISION, TIMEOUT or TEAMMATE_LOST; leaves env.active = False.
    """
    try:
        return _drive(env, policy, goal, max_steps, use_gap, csv, t0, abort)
    finally:
        env.active = False
        env.stop()


def _drive(env, policy, goal, max_steps, use_gap, csv, t0, abort) -> str:
    env.set_goal_shared(*goal)
    observation, info = env.reset()
    env.active = True
    gap = GapAssist()
    aligning = True
    lost_since = None
    for _ in range(max_steps):
        if abort():
            log("stop requested")
            return "ABORTED"
        bearing_deg = math.degrees(info["relative_goal_bearing_rad"])
        dist = info["distance_to_goal_m"]
        policy_action = policy.act(observation)
        if aligning and abs(bearing_deg) <= ALIGN_UNTIL_DEG:
            aligning = False
            gap.reset()          # fresh stuck history once MAPPO takes over
        if aligning:
            action, why = (TURN_LEFT if bearing_deg > 0 else TURN_RIGHT), "align"
        elif not use_gap:
            action, why = policy_action, "policy"
        else:
            state = env._current_state()
            action, why = gap.decide(policy_action, (state.x, state.y), dist, bearing_deg,
                                     lambda: gap_heading(env, bearing_deg))
        me = env.latest_shared()
        mate = env.last_teammate
        csv.write(f"{time.monotonic() - t0:.2f},{me[0]:.3f},{me[1]:.3f},{math.degrees(me[2]):.1f},"
                  + ",".join(f"{v:.4f}" for v in observation)
                  + f",{PointNavAction(policy_action).name},{PointNavAction(action).name},{why},"
                  + ("" if mate is None else f"{mate['age_s']:.2f}") + "\n")
        if why != "policy" and why != "align":
            log(f"step {info['episode_steps']}: {why} (goal {dist:.2f} m, {bearing_deg:+.0f} deg)")
        try:
            observation, _, terminated, truncated, info = env.step(action)
            lost_since = None
        except TeammateLost as exc:
            log(f"{exc}; holding still")
            lost_since = time.monotonic()
            while True:
                env.stop()
                time.sleep(0.3)
                state = env._current_state()
                try:
                    observation, info = env._observation(state), env._info(state)
                    break
                except TeammateLost:
                    if time.monotonic() - lost_since > 5.0:
                        log("teammate lost for 5 s; giving up")
                        return "TEAMMATE_LOST"
            continue
        if info["collision"]:
            log(f"contact: something within 13 cm after {info['episode_steps']} steps")
            return "COLLISION"
        mate, me = env.last_teammate, env.latest_shared()
        if mate is not None and math.dist(me[:2], (mate["x"], mate["y"])) < ROBOT_CONTACT_M:
            # The LDS does not see the black Burgers, so robot-robot contact is checked from the shared poses
            # (the simulator's footprint test: 2 x 0.105 m, plus a small margin).
            log(f"teammate within {ROBOT_CONTACT_M:.2f} m (centre to centre) after {info['episode_steps']} steps")
            return "COLLISION"
        if terminated:
            log(f"goal reached in {info['episode_steps']} steps, {time.monotonic() - t0:.0f} s, "
                f"{gap.handovers} gap hand-overs")
            return "SUCCESS"
        if truncated:
            break
    return "TIMEOUT"


def gap_heading(env: MappoRosEnv, bearing_deg: float):
    with env._scan_lock:
        scan = env._scan
    if scan is None:
        return None
    return free_gap_heading(env._sector_ranges(scan), scan.angle_min, scan.angle_increment, bearing_deg)


if __name__ == "__main__":
    main()
