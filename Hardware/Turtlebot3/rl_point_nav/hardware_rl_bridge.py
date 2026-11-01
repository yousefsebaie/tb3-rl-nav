#!/usr/bin/env python3
"""CyberScape -> RL on the real Burger: the same Bridge as the Isaac demo, driving real wheels.

Hardware counterpart of cyberscape_rl_bridge.py. It reuses that file's Bridge class
unchanged (goal alignment, waypoint legs, stall recovery, spin guard), so the real robot
runs the exact decision logic of the simulation demo:

  CyberScape plan -> HITL approval -> hitl-comm-server.py (Mac)
      -> this node polls /get_instruction like burger_comm_node.py
      -> Bridge -> V3 policy (NumPy) -> /cmd_vel_raw -> safety_filter_node.py -> wheels

Run on the Pi next to bringup and safety_filter_node.py:

  python3 hardware_rl_bridge.py --server http://<MAC_IP>:5001          # approved CyberScape plans
  python3 hardware_rl_bridge.py --plan rl_demo_burger_plan.json        # a plan JSON, no server

Command frame (as in the simulation): (x, y) in metres from the robot's pose when this node
starts, +x = straight ahead, +y = left. Differences from the simulation demo, all from the
hardware: no Isaac obstacles or survivors (people=()), so detect_with_camera finds nobody,
and legs default to 1.4 m because V3 stalls on clear goals beyond ~1.5 m on this robot.
"""

from __future__ import annotations

import argparse
import dataclasses
import math
from pathlib import Path
import sys
import threading
import time
import types

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))   # burger_comm_node.py lives one level up in the repo

# cyberscape_rl_bridge imports the Isaac env module (which needs gymnasium) only for its own
# main(); Bridge itself uses neither. The Pi has no gymnasium, so stand in a stub there.
try:
    import gymnasium  # noqa: F401
except ImportError:
    _stub = types.ModuleType("isaac_point_nav_lidar_env")
    _stub.IsaacPointNavLidarEnv = object
    _stub.v3_lidar_config = None
    sys.modules["isaac_point_nav_lidar_env"] = _stub

from burger_comm_node import complete_phase, get_instruction, report_error, send_heartbeat  # noqa: E402
from cyberscape_rl_bridge import ALIGN_ABOVE_DEG, ALIGN_UNTIL_DEG, Bridge, log, plan_commands  # noqa: E402
from policy_numpy import NumpyPolicy  # noqa: E402
from route_planner import RoutePlanner, lookahead, route_length  # noqa: E402
from ros_point_nav_lidar_env import RosPointNavLidarEnv, v2_config  # noqa: E402

DEFAULT_MODEL = HERE / "models" / "v3" / "checkpoint_1000000_steps.npz"
HEARTBEAT_PERIOD_S = 5.0
SETTLE_FRAME_S = 1.0 / 60.0   # Isaac settle_scan frames are 60 Hz physics frames


class NumpyController:
    """Bridge controller interface (name / reset / act) around the exported policy."""

    name = "ppo"

    def __init__(self, model_path: Path) -> None:
        self.policy = NumpyPolicy(model_path)

    def reset(self) -> None:
        pass

    def act(self, observation) -> int:
        return self.policy.act(observation)


class HardwareMissionEnv:
    """The subset of IsaacPointNavLidarEnv that Bridge uses, backed by the real robot."""

    def __init__(self, ros_env: RosPointNavLidarEnv) -> None:
        self.ros = ros_env

    def _state(self):
        return self.ros._current_state()

    def _with_lidar(self, info: dict) -> dict:
        info["clearances_m"] = dict(self.ros.last_lidar.get("clearance_m", {}))
        info.setdefault("collision", False)
        return info

    def continue_to_goal(self, goal):
        # Like Isaac's continue_to_goal: new goal from the current pose, fresh step budget.
        observation, info = self.ros.reset(goal=goal)
        return observation, self._with_lidar(info)

    def step(self, action):
        observation, reward, terminated, truncated, info = self.ros.step(action)
        return observation, reward, terminated, truncated, self._with_lidar(info)

    def _info(self, state) -> dict:
        self.ros._lidar_values()   # refresh last_lidar from the newest scan
        return self._with_lidar(self.ros._info(state))

    def settle_scan(self, frames: int) -> None:
        self.ros.stop()
        time.sleep(frames * SETTLE_FRAME_S)

    def sim_time(self) -> float:
        return time.monotonic()

    def scan_and_pose(self):
        """Newest LaserScan and the robot pose (x, y, yaw) for the route planner's map."""
        with self.ros._scan_lock:
            scan = self.ros._scan
        state = self.ros._current_state()
        return scan, (state.x, state.y, state.yaw)


class PlanningBridge(Bridge):
    """Bridge whose moves follow an A* route around the obstacles seen so far.

    Every action adds the newest scan to the map. A move re-plans the route to its
    target after each short leg (LEG_M along the route), so the route bends around an
    obstacle as soon as the LiDAR sees it. The policy drives each leg; everything else
    (rotate, scan, detect, waiting) is the unchanged simulation Bridge.
    """

    LEG_M = 0.5               # policy goal: this far along the route (V3 drives straight here)
    LEG_STEPS = 25            # policy actions per leg before re-planning anyway
    WAYPOINT_REACHED_M = 0.12
    ARRIVED_M = 0.08
    MAX_LEGS = 60
    ROUTE_CHANGED_M = 0.15    # new route this much longer/shorter than the old one's rest -> report it

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.planner = RoutePlanner(self.start[:2])
        self.route_changes = 0
        self.update_map()

    def update_map(self) -> None:
        scan, pose = self.env.scan_and_pose()
        if scan is not None:
            self.planner.add_scan(pose, scan.ranges, scan.angle_min, scan.angle_increment,
                                  scan.range_min, scan.range_max)

    def _step(self, action):
        result = super()._step(action)
        self.update_map()
        return result

    def move_to(self, x: float, y: float) -> tuple[bool, str]:
        radius = math.hypot(x, y)
        if radius > self.max_goal_radius:
            scale = self.max_goal_radius / radius
            log(f"  ({x:.2f}, {y:.2f}) is {radius:.2f} m from the start; clamped to {self.max_goal_radius:.2f} m")
            x, y = x * scale, y * scale
        target = self.to_world(x, y)
        previous = None   # (route length, robot position) of the last plan
        changes_before = self.route_changes
        best, legs_without_progress = math.inf, 0
        for _ in range(self.MAX_LEGS):
            state = self.env._state()
            remaining = math.hypot(target[0] - state.x, target[1] - state.y)
            if remaining <= self.ARRIVED_M:
                return True, f"arrived ({self.route_changes - changes_before} route changes)"
            route = self.planner.plan((state.x, state.y), target)
            if route is None:
                return False, "no route to the target: blocked on every side"
            length = route_length(route)
            if previous is None:
                log(f"  route to ({x:.2f}, {y:.2f}): {length:.2f} m")
            else:
                # Following the old route would leave (old length - distance driven) to go.
                expected = previous[0] - math.dist(previous[1], (state.x, state.y))
                if abs(length - expected) > self.ROUTE_CHANGED_M:
                    self.route_changes += 1
                    self.count("route_change")
                    log(f"  obstacle seen: route changed, now {length:.2f} m to ({x:.2f}, {y:.2f})")
            previous = (length, (state.x, state.y))
            final = length <= self.LEG_M
            ok, why = self.drive_short_leg(target if final else lookahead(route, self.LEG_M), final)
            if not ok:
                return False, why
            if remaining < best - 0.03:
                best, legs_without_progress = remaining, 0
            else:
                legs_without_progress += 1
                if legs_without_progress >= 4:
                    log("  no progress for 4 legs: recovery move, then re-plan")
                    self.count("stall_recovery")
                    self.recover(target, {})
                    legs_without_progress = 0
        return False, f"gave up after {self.MAX_LEGS} legs"

    def drive_short_leg(self, goal_world: tuple[float, float], final: bool) -> tuple[bool, str]:
        observation, info = self.env.continue_to_goal(goal_world)
        bearing = info["relative_goal_bearing_rad"]
        if abs(math.degrees(bearing)) > ALIGN_ABOVE_DEG:
            self.count("goal_alignment")
            state = self.env._state()
            self.turn_until(math.atan2(goal_world[1] - state.y, goal_world[0] - state.x), ALIGN_UNTIL_DEG)
            observation, info = self.env.continue_to_goal(goal_world)
        for _ in range(self.LEG_STEPS):
            observation, _, terminated, _, info = self.step(self.decide(observation))
            if info["collision"]:
                log("  contact guard: obstacle within 13 cm, turning away and re-planning")
                self.count("contact_recovery")
                self.recover(goal_world, info)
                return True, "contact"
            if terminated or (not final and info["distance_to_goal_m"] <= self.WAYPOINT_REACHED_M):
                break
        return True, "leg done"


def run_server(bridge: Bridge, server_url: str, robot_name: str) -> None:
    """burger_comm_node.run_mission, with each instruction executed by the Bridge."""

    def _heartbeat_loop():
        while True:
            send_heartbeat(server_url, robot_name)
            time.sleep(HEARTBEAT_PERIOD_S)

    threading.Thread(target=_heartbeat_loop, daemon=True).start()
    log(f"waiting for approved CyberScape plans for {robot_name} on {server_url}")
    phase = 1
    while True:
        data = get_instruction(server_url, robot_name, phase)
        if data is None:
            time.sleep(2.0)
            continue
        instructions = data.get("low_level_plan", [])
        if isinstance(instructions, str):
            instructions = [line.strip() for line in instructions.splitlines() if line.strip()]
        log(f"phase {phase}: {len(instructions)} instructions")
        outputs = {}
        for index, line in enumerate(instructions, start=1):
            _, cx, cy = bridge.pose()
            log(f"COMMAND {line}   (robot at ({cx:.2f}, {cy:.2f}) in the start frame)")
            try:
                ok, why = bridge.execute(line)
            except Exception as exc:  # report, keep the node alive
                ok, why = False, f"error: {exc}"
            log(("DONE    " if ok else "FAILED  ") + f"{line}: {why}")
            outputs[f"instruction_{index}"] = why
            if not ok:
                report_error(server_url, robot_name, phase, index, why)
                break
        expected = data.get("expected_outputs", {})
        complete_phase(server_url, robot_name, phase, {**{k: "" for k in expected}, **outputs})
        phase += 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--server", default="http://localhost:5001", help="CyberScape HITL server URL")
    parser.add_argument("--robot", default="TURTLEBOT3", help="robot name on the server")
    parser.add_argument("--plan", type=Path, default=None, help="run a CyberScape low-level plan JSON, no server")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--profile", default=None,
                        help="LiDAR sector profile; default: profile.txt next to --model, else v2")
    parser.add_argument("--max-goal-radius", type=float, default=3.0)
    parser.add_argument("--max-leg-m", type=float, default=1.4)
    parser.add_argument("--no-spin-guard", action="store_true",
                        help="pure RL legs (the simulation demo runs with the spin guard on)")
    parser.add_argument("--no-replan", action="store_true",
                        help="drive straight legs like the simulation demo instead of A* routes")
    parser.add_argument("--no-safety-filter", action="store_true",
                        help="publish straight to /cmd_vel; stop safety_filter_node.py first, "
                             "or its zero commands fight these")
    args = parser.parse_args()

    profile = args.profile
    profile_file = args.model.parent / "profile.txt"
    if profile is None and profile_file.is_file():
        profile = profile_file.read_text().strip()
    profile = profile or "v2"

    controller = NumpyController(args.model)
    if controller.policy.obs_dim != 6:
        raise SystemExit("The bridge needs a LiDAR policy (6 inputs), e.g. V3.")
    config = v2_config()
    if args.no_safety_filter:
        config = dataclasses.replace(config, cmd_vel_topic="/cmd_vel")
    ros_env = RosPointNavLidarEnv(0.0, 0.0, use_lidar=True, config=config, lidar_profile=profile)
    try:
        env = HardwareMissionEnv(ros_env)
        bridge_class = Bridge if args.no_replan else PlanningBridge
        bridge = bridge_class(env, controller, args.max_goal_radius, args.max_leg_m,
                              spin_guard=not args.no_spin_guard)
        log(f"model {args.model.name} ({profile} sectors); start pose = mission origin (0, 0), "
            f"+x ahead, +y left; legs <= {args.max_leg_m} m; "
            f"spin guard {'off' if args.no_spin_guard else 'ON'}; "
            f"route re-planning {'off' if args.no_replan else 'ON'}; commands to {config.cmd_vel_topic}")
        if args.plan is not None:
            commands = plan_commands(args.plan)
            log(f"running {len(commands)} commands from {args.plan}")
            for line in commands:
                ok, why = bridge.execute(line)
                log(("DONE    " if ok else "FAILED  ") + f"{line}: {why}")
                if not ok:
                    break
            return
        run_server(bridge, args.server.rstrip("/"), args.robot)
    except KeyboardInterrupt:
        log("interrupted")
    finally:
        ros_env.close()   # always sends a zero velocity


if __name__ == "__main__":
    main()
