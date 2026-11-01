#!/usr/bin/env python3
"""CyberScape -> RL bridge: Isaac Sim executes approved CyberScape commands with the PPO policy.

Replaces Hardware/Isaac/cyberscape_isaac_listener.py (which teleported the robot) with
the trained V3 policy driving the real-size TurtleBot3 through Isaac physics and LiDAR.
This is the "RL Low-Level Navigation Agent" layer of the architecture:

  CyberScape plan -> HITL approval -> cyberscape_isaac_plan_sender.py
      -> COMMAND_FILE -> this bridge -> Goal Interface -> PPO policy -> wheels

Command frame (same as the real Burger's odom): (x, y) in metres from the robot's START
pose, +x = straight ahead, +y = left. Goals farther than --max-goal-radius from the start
are clamped (the policy was trained on goals 0.5-2 m away). Long moves are cut into
waypoints of at most --max-leg-m.

  ./isaac_run.sh cyberscape_rl_bridge.py                       # wait for the plan sender
  ./isaac_run.sh cyberscape_rl_bridge.py --plan ../../../Plans/final_burger_low_level_plan.json

Status file protocol (read by the plan sender): BUSY while running, then DONE or FAILED: <why>.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import time
from typing import Optional

import numpy as np

try:
    from .config import PointNavAction
    from .isaac_point_nav_lidar_env import IsaacPointNavLidarEnv, v3_lidar_config
    from .mission_timing import MissionTimer, append_csv
except ImportError:  # pragma: no cover - direct-script convenience only
    from config import PointNavAction
    from isaac_point_nav_lidar_env import IsaacPointNavLidarEnv, v3_lidar_config
    from mission_timing import MissionTimer, append_csv

HERE = Path(__file__).resolve().parent
DEFAULT_MODEL = HERE / "models" / "isaac_point_nav_lidar_ppo_v3" / "checkpoint_1000000_steps.zip"
COMMAND_FILE = Path(os.environ.get("CYBERSCAPE_ISAAC_COMMAND_FILE", "/home/g8/cyberscape_command.txt"))
STATUS_FILE = Path(os.environ.get("CYBERSCAPE_ISAAC_STATUS_FILE", "/home/g8/cyberscape_status.txt"))
MISSIONS_DIR = HERE / "results" / "missions"      # Mission Timing Evaluator output
TAG = "[CYBERSCAPE_RL]"

# Mission area boxes in the START frame (x ahead, y left, width). They sit between the
# demo mission's checkpoints so the policy has to steer around them.
DEMO_OBSTACLES = ((0.85, 0.62, 0.25), (1.72, 0.0, 0.25), (0.85, -0.62, 0.25))
# --scenario: same proven layout, different story and look.
#   patrol:   orange boxes, checkpoints A (1.5, 0.75) and B (1.5, -0.75)
#   disaster: earthquake set dressing (disaster_scene.py): broken concrete blocks, trapped
#             survivors next to both checkpoints, collapsed ceiling, responder base, dim lights
SCENARIOS = ("patrol", "disaster")
# Survivors (chest x, chest y, body heading deg), ~0.5 m beyond each checkpoint and away
# from the routes. Bodies are below the LiDAR plane; Bridge.step's person-safety zone
# blocks any forward move into them.
SURVIVORS = ((2.45, 1.2, 30.0), (1.0, -1.3, 180.0))
CAMERA_RANGE_M = 1.5
BODY_HALF_WIDTH_M = 0.24      # survivor body: head-to-feet segment +- this
ROBOT_RADIUS_M = 0.105
PERSON_SAFETY_M = 0.15        # forward moves that would end closer than this to a body are blocked
FORWARD_STEP_M = 0.03         # a little more than one FORWARD action moves
# Training starts always faced the goal banks (mostly straight ahead), so the goal interface
# first turns the robot toward the goal, then hands over to the policy.
ALIGN_ABOVE_DEG = 15.0
ALIGN_UNTIL_DEG = 8.0
ROTATE_TOLERANCE_DEG = 6.0
STALL_STEPS = 20           # no 3 cm of progress in this many RL steps -> recovery
SPIN_GUARD_TURNS = 8       # --spin-guard: after ~65 deg of turning in a row ...
SPIN_GUARD_BEARING_DEG = 45.0   # ... drive forward if the goal is roughly ahead and
SPIN_GUARD_FRONT_M = 0.50       # ... the path in front is clear, else turn the short way to the goal
STALL_PROGRESS_M = 0.03
RECOVERY_FORWARD_STEPS = 6


def log(message: str) -> None:
    print(f"{TAG} {message}", flush=True)


def write_status(status: str) -> None:
    STATUS_FILE.write_text(status)


def parse_command(line: str):
    match = re.match(r"\s*[A-Za-z0-9_]+\.([A-Za-z0-9_]+)\((.*)\)\s*$", line)
    if match is None:
        return None, []
    numbers = [float(v) for v in re.findall(r"-?\d+(?:\.\d+)?", match.group(2))]
    return match.group(1), numbers


class Bridge:
    def __init__(self, env: IsaacPointNavLidarEnv, controller, max_goal_radius: float, max_leg: float,
                 spin_guard: bool = False, people=(), timing_meta: Optional[dict] = None):
        self.env, self.controller, self.spin_guard = env, controller, spin_guard
        self.timer: Optional[MissionTimer] = None
        self.mission_tasks: list[str] = []
        self.next_task = 0
        self.timing_meta = timing_meta or {}
        state = env._state()
        self.start = (state.x, state.y, state.yaw)
        self.people = [(x, y) for x, y, _ in people]   # chest (x, y), start frame: what the camera finds
        # Body segments head -> feet in WORLD coordinates for the person-safety zone.
        self.bodies = []
        for x, y, heading in people:
            c, s = math.cos(math.radians(heading)), math.sin(math.radians(heading))
            head = self.to_world(x - 0.42 * c, y - 0.42 * s)
            feet = self.to_world(x + 0.80 * c, y + 0.80 * s)
            self.bodies.append((head, feet))
        self.found = set()
        self.safety_active = False
        self.max_goal_radius, self.max_leg = max_goal_radius, max_leg

    # ---------------------------------------------------------------- frames
    def to_world(self, x: float, y: float) -> tuple[float, float]:
        x0, y0, yaw0 = self.start
        return x0 + math.cos(yaw0) * x - math.sin(yaw0) * y, y0 + math.sin(yaw0) * x + math.cos(yaw0) * y

    def to_mission(self, wx: float, wy: float) -> tuple[float, float]:
        x0, y0, yaw0 = self.start
        dx, dy = wx - x0, wy - y0
        return math.cos(yaw0) * dx + math.sin(yaw0) * dy, -math.sin(yaw0) * dx + math.cos(yaw0) * dy

    def person_clearance(self, wx: float, wy: float) -> float:
        """Distance from a robot centre at world (wx, wy) to the nearest survivor body surface."""
        best = math.inf
        for (ax, ay), (bx, by) in self.bodies:
            dx, dy = bx - ax, by - ay
            t = max(0.0, min(1.0, ((wx - ax) * dx + (wy - ay) * dy) / (dx * dx + dy * dy)))
            best = min(best, math.hypot(wx - ax - t * dx, wy - ay - t * dy) - BODY_HALF_WIDTH_M - ROBOT_RADIUS_M)
        return best

    def step(self, action):
        """env.step with the person-safety zone: never drive forward into a survivor."""
        action = PointNavAction(int(action))
        if action == PointNavAction.FORWARD and self.bodies:
            state = self.env._state()
            nx, ny = state.x + FORWARD_STEP_M * math.cos(state.yaw), state.y + FORWARD_STEP_M * math.sin(state.yaw)
            now, after = self.person_clearance(state.x, state.y), self.person_clearance(nx, ny)
            if after < PERSON_SAFETY_M and after < now:
                # Turn away from the nearest body instead (toward the side that increases clearance).
                left = self.person_clearance(state.x + FORWARD_STEP_M * math.cos(state.yaw + 0.5),
                                             state.y + FORWARD_STEP_M * math.sin(state.yaw + 0.5))
                right = self.person_clearance(state.x + FORWARD_STEP_M * math.cos(state.yaw - 0.5),
                                              state.y + FORWARD_STEP_M * math.sin(state.yaw - 0.5))
                action = PointNavAction.TURN_LEFT if left >= right else PointNavAction.TURN_RIGHT
                if not self.safety_active:
                    log(f"  person safety: {max(now, 0.0):.2f} m from a survivor, turning away instead of driving forward")
                self.count("person_safety_override")
                self.safety_active = True
                return self._step(action)
        self.safety_active = False
        return self._step(action)

    def _step(self, action):
        result = self.env.step(int(action))
        if self.timer is not None:
            self.timer.record_pose(result[4]["pose"]["x"], result[4]["pose"]["y"])
        return result

    def count(self, name: str) -> None:
        if self.timer is not None:
            self.timer.count(name)

    def decide(self, observation) -> PointNavAction:
        """Controller decision, timed on the decision clock when a mission is being timed."""
        if self.timer is not None:
            return PointNavAction(self.timer.time_inference(self.controller.act, observation))
        return PointNavAction(self.controller.act(observation))

    def pose(self):
        state = self.env._state()
        mx, my = self.to_mission(state.x, state.y)
        return state, mx, my

    # ---------------------------------------------------------------- skills
    def turn_until(self, target_yaw: float, tolerance_deg: float, limit: int = 60) -> None:
        """Goal Interface alignment: discrete TURN actions until the heading is close."""
        for _ in range(limit):
            error = math.atan2(math.sin(target_yaw - self.env._state().yaw), math.cos(target_yaw - self.env._state().yaw))
            if abs(math.degrees(error)) <= tolerance_deg:
                return
            self.step(PointNavAction.TURN_LEFT if error > 0 else PointNavAction.TURN_RIGHT)

    def drive_leg(self, goal_world: tuple[float, float]) -> tuple[bool, str]:
        observation, info = self.env.continue_to_goal(goal_world)
        bearing = info["relative_goal_bearing_rad"]
        if abs(math.degrees(bearing)) > ALIGN_ABOVE_DEG:
            state = self.env._state()
            log(f"  goal is {math.degrees(bearing):+.0f} deg off the heading: turning toward it first")
            self.count("goal_alignment")
            self.turn_until(math.atan2(goal_world[1] - state.y, goal_world[0] - state.x), ALIGN_UNTIL_DEG)
            observation, info = self.env.continue_to_goal(goal_world)
        history = []
        turns = 0
        while True:
            history.append(info["distance_to_goal_m"])
            if len(history) > STALL_STEPS and history[-STALL_STEPS - 1] - history[-1] < STALL_PROGRESS_M:
                log(f"  no progress for {STALL_STEPS} RL steps: recovery move, then RL again")
                self.count("stall_recovery")
                observation, info = self.recover(goal_world, info)
                if info.get("collision"):
                    return False, "collision during recovery"
                history = []
                continue
            action = self.decide(observation)
            turning = action in (PointNavAction.TURN_LEFT, PointNavAction.TURN_RIGHT)
            turns = turns + 1 if turning else 0
            if self.spin_guard and turning and turns > SPIN_GUARD_TURNS:
                bearing = info["relative_goal_bearing_rad"]
                if abs(math.degrees(bearing)) < SPIN_GUARD_BEARING_DEG and info["clearances_m"]["front"] > SPIN_GUARD_FRONT_M:
                    log("  spin guard: goal ahead and path clear, driving forward instead of turning")
                    self.count("spin_guard_override")
                    action, turns = PointNavAction.FORWARD, 0
                else:
                    toward = PointNavAction.TURN_LEFT if bearing > 0 else PointNavAction.TURN_RIGHT
                    if action != toward:
                        log("  spin guard: turning the short way back toward the goal")
                        self.count("spin_guard_override")
                    action = toward
            observation, _, terminated, truncated, info = self.step(action)
            c = info["clearances_m"]
            if info["episode_steps"] % 10 == 0:
                log(f"  RL step {info['episode_steps']:3d} {PointNavAction(int(action)).name:<10} "
                    f"dist={info['distance_to_goal_m']:.2f} m  front={c['front']:.2f} left={c['left']:.2f} right={c['right']:.2f}")
            if info["collision"]:
                return False, f"collision after {info['episode_steps']} RL steps"
            if terminated:
                return True, f"reached in {info['episode_steps']} RL steps"
            if truncated:
                return False, f"timeout, {info['distance_to_goal_m']:.2f} m from the goal"

    def recover(self, goal_world, info):
        """Turn toward the goal, or toward the clearer side if the goal side is blocked, and drive a bit."""
        state = self.env._state()
        goal_yaw = math.atan2(goal_world[1] - state.y, goal_world[0] - state.x)
        self.turn_until(goal_yaw, 15.0)
        observation, info = self.env.continue_to_goal(goal_world)
        c = info["clearances_m"]
        if c["front"] < 0.45:
            side = 1.0 if c["left"] >= c["right"] else -1.0
            self.turn_until(self.env._state().yaw + side * math.radians(60), 15.0)
        for _ in range(RECOVERY_FORWARD_STEPS):
            observation, info = self.env.continue_to_goal(goal_world)
            if info["clearances_m"]["front"] < 0.30:
                break
            observation, _, terminated, _, info = self.step(PointNavAction.FORWARD)
            if info["collision"]:
                return observation, info
            if terminated:
                break
        return self.env.continue_to_goal(goal_world)

    def move_to(self, x: float, y: float) -> tuple[bool, str]:
        radius = math.hypot(x, y)
        if radius > self.max_goal_radius:
            scale = self.max_goal_radius / radius
            log(f"  ({x:.2f}, {y:.2f}) is {radius:.2f} m from the start; clamped to {self.max_goal_radius:.2f} m")
            x, y = x * scale, y * scale
        _, cx, cy = self.pose()
        distance = math.hypot(x - cx, y - cy)
        legs = max(1, math.ceil(distance / self.max_leg))
        for leg in range(1, legs + 1):
            wx, wy = cx + (x - cx) * leg / legs, cy + (y - cy) * leg / legs
            log(f"  RL waypoint {leg}/{legs}: ({wx:.2f}, {wy:.2f})")
            ok, why = self.drive_leg(self.to_world(wx, wy))
            log(f"  waypoint {leg}/{legs}: {why}")
            if not ok:
                return False, why
        return True, "arrived"

    def rotate(self, degrees: float) -> tuple[bool, str]:
        state = self.env._state()
        target = state.yaw + math.radians(degrees)
        # A far goal along the new heading keeps the episode active while turning in place.
        self.env.continue_to_goal((state.x + 10 * math.cos(target), state.y + 10 * math.sin(target)))
        self.turn_until(target, ROTATE_TOLERANCE_DEG, limit=int(abs(degrees) / 5) + 20)
        return True, f"rotated {degrees:.0f} deg"

    def detect(self) -> tuple[bool, str]:
        """Camera stand-in: nearest person within CAMERA_RANGE_M (simulation ground truth).

        The robot turns to face them, as a real camera check would need.
        """
        _, cx, cy = self.pose()
        candidates = [(math.hypot(px - cx, py - cy), i, px, py) for i, (px, py) in enumerate(self.people)]
        candidates = [c for c in candidates if c[0] <= CAMERA_RANGE_M]
        if not candidates:
            return True, "no survivor in camera range"
        distance, index, px, py = min(candidates)
        wx, wy = self.to_world(px, py)
        state = self.env._state()
        # Far goal in the person's direction keeps the episode active while turning in place.
        self.env.continue_to_goal((state.x + 10 * (wx - state.x), state.y + 10 * (wy - state.y)))
        self.turn_until(math.atan2(wy - state.y, wx - state.x), 10.0)
        self.found.add(index)
        return True, (f"SURVIVOR DETECTED at ({px:.2f}, {py:.2f}), {distance:.2f} m away "
                      f"({len(self.found)}/{len(self.people)} found)")

    def execute(self, line: str) -> tuple[bool, str]:
        action, args = parse_command(line)
        _, cx, cy = self.pose()
        if action == "move_to_point" and len(args) >= 2:
            return self.move_to(args[0], args[1])
        if action == "move_forward" and args:
            yaw = self.env._state().yaw - self.start[2]
            return self.move_to(cx + args[0] * math.cos(yaw), cy + args[0] * math.sin(yaw))
        if action == "return_to_base":
            return self.move_to(0.0, 0.0)
        if action == "rotate" and args:
            return self.rotate(args[0])
        if action == "scan_with_lidar":
            c = self.env._info(self.env._state())["clearances_m"]
            return True, f"scan: front={c['front']:.2f} m left={c['left']:.2f} m right={c['right']:.2f} m"
        if action == "detect_with_camera":
            return self.detect()
        if action in ("wait_for_signal", "communicate_with_apm", "retrieve_object"):
            self.env.settle_scan(30)
            return True, "simulated"
        return True, "not supported in simulation, skipped"

    # ------------------------------------------------------- mission timing
    def begin_mission(self, commands: list[str], mission_id: str) -> None:
        """Mission Timing Evaluator: one task per plan command, t0 = now (first command)."""
        self.mission_tasks = [f"t{i:02d} {c}" for i, c in enumerate(commands)]
        self.next_task = 0
        self.timer = MissionTimer(mission_id, self.controller.name, self.env.sim_time, self.mission_tasks,
                                  metadata=dict(self.timing_meta))
        self.timer.start()
        _, cx, cy = self.pose()
        self.timer.record_pose(*self.to_world(cx, cy))
        log(f"timing mission {mission_id}: {len(commands)} tasks, controller {self.controller.name}")

    def end_mission(self, status: str, reason: str = "") -> None:
        if self.timer is None:
            return
        if status == "success":
            self.timer.robot_complete("turtlebot3")
        row = self.timer.finish(status, reason)
        folder = MISSIONS_DIR / self.timer.mission_id
        self.timer.write_events(folder / "events.jsonl")
        append_csv(MISSIONS_DIR / "missions.csv", row)
        log(f"MISSION {row['status']}: makespan {row['makespan_sim_s']} s sim "
            f"(terminated at {row['termination_elapsed_sim_s']} s), wall {row['wall_elapsed_s']} s, RTF {row['rtf']}, "
            f"path {row['path_length_m']} m, inference p95 {row['inference_p95_ms']} ms -> {folder}")
        self.timer = None

    def run(self, line: str) -> bool:
        _, cx, cy = self.pose()
        log(f"COMMAND {line.strip()}   (robot at ({cx:.2f}, {cy:.2f}) in the start frame)")
        write_status("BUSY")
        task = None
        if self.timer is not None and self.next_task < len(self.mission_tasks):
            task = self.mission_tasks[self.next_task]
            if task.split(" ", 1)[1] != line.strip():
                log(f"  timing: command differs from plan task {task!r}; not timed")
                task = None
            else:
                self.next_task += 1
                self.timer.task_start(task, "turtlebot3")
        try:
            ok, why = self.execute(line)
        except Exception as exc:  # report, keep the bridge alive
            ok, why = False, f"error: {exc}"
        log(("DONE    " if ok else "FAILED  ") + f"{line.strip()}: {why}")
        write_status("DONE" if ok else f"FAILED: {why}")
        if task is not None:
            if ok:
                self.timer.task_complete(task, result=why)
                if self.next_task == len(self.mission_tasks):
                    self.end_mission("success")
            else:
                self.timer.task_failed(task, why)
                self.end_mission("collision" if "collision" in why else "timeout" if "timeout" in why else "failed", why)
        return ok


def plan_commands(path: Path) -> list[str]:
    data = json.loads(path.read_text())
    commands = []
    for phase in data.get("phases", []):
        plan = phase.get("low_level_plan", "")
        lines = plan if isinstance(plan, list) else str(plan).splitlines()
        commands += [line.strip() for line in lines if line.strip()]
    return commands


def dress_scenario(bridge: Bridge, name: str) -> None:
    """Visual-only props for the scenario (the RL obstacles and goals do not change)."""
    if name == "disaster":
        try:
            from .disaster_scene import build
        except ImportError:  # pragma: no cover - direct-script convenience only
            from disaster_scene import build
        build(bridge.env, bridge.to_world, bridge.start[2], DEMO_OBSTACLES, SURVIVORS)


def frame_camera(bridge: Bridge, scenario: str = "patrol") -> None:
    from isaacsim.core.utils.viewports import set_camera_view

    z = bridge.env._ground_z
    if scenario == "disaster":
        # Lower, from behind the responder base: debris, survivors and collapse in one shot.
        tx, ty = bridge.to_world(1.5, 0.0)
        ex, ey = bridge.to_world(-1.0, 1.6)
        set_camera_view(eye=np.array([ex, ey, z + 1.35]), target=np.array([tx, ty, z]))
        return
    cx, cy = bridge.to_world(0.8, 0.0)
    target = np.array([cx, cy, z])
    yaw = bridge.start[2]
    # From behind-left of the start, high up: whole mission area in view.
    eye = target + np.array([-1.6 * math.cos(yaw) + 1.0 * math.sin(yaw), -1.6 * math.sin(yaw) - 1.0 * math.cos(yaw), 2.6])
    set_camera_view(eye=eye, target=target)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--plan", type=Path, default=None, help="run a CyberScape low-level plan JSON directly")
    parser.add_argument("--max-goal-radius", type=float, default=1.9)
    parser.add_argument("--max-leg-m", type=float, default=1.9)
    parser.add_argument("--no-obstacles", action="store_true")
    parser.add_argument("--hide-robot", action="store_true")
    parser.add_argument("--scenario", choices=SCENARIOS, default="patrol")
    parser.add_argument("--sensor-noise", action="store_true",
                        help="keep V3's training-time LiDAR noise/dropout (default: clean simulated LiDAR)")
    parser.add_argument("--snapshot", type=Path, default=None,
                        help="save a PNG of the viewport after setting up the scene, then quit")
    parser.add_argument("--headless", action="store_true",
                        help="no window (batch evaluation with --plan; exits when the plan is done)")
    parser.add_argument("--controller", choices=("ppo", "rule"), default="ppo",
                        help="ppo (trained policy) or rule (B1 hand-coded reactive baseline); same safety layer")
    parser.add_argument("--mission-plan", type=Path, default=None,
                        help="plan JSON the incoming commands come from (to time the mission in watch mode)")
    parser.add_argument("--spin-guard", action="store_true",
                        help="override long in-place turns with FORWARD when the goal is ahead and the path is clear")
    args = parser.parse_args()

    try:
        from .eval_timing import PpoController, RuleController
    except ImportError:  # pragma: no cover
        from eval_timing import PpoController, RuleController

    # V3 trains with real-sensor noise and dropouts (sim-to-real augmentation); the simulated
    # LiDAR is ideal, so the demo feeds clean readings unless --sensor-noise.
    clean = {} if args.sensor_noise else dict(lidar_noise_std_m=0.0, lidar_ray_dropout=0.0, lidar_sector_dropout=0.0)
    env = IsaacPointNavLidarEnv(headless=args.headless,
                                lidar_config=v3_lidar_config(hide_robot_visuals=args.hide_robot, **clean))
    try:
        if not args.headless:
            import carb.settings

            carb.settings.get_settings().set("/app/window/hideUi", True)
            env._lidar.enable_visualization()
        controller = PpoController(args.model) if args.controller == "ppo" else RuleController()
        _, info = env.reset(seed=0, options={"goal": (0.0, 0.0), "obstacles": []})
        people = SURVIVORS if args.scenario == "disaster" else ()
        timing_meta = {"scenario": args.scenario, "spin_guard": args.spin_guard, "sensor_noise": args.sensor_noise,
                       "model": args.model.name if args.controller == "ppo" else "", "plan": ""}
        bridge = Bridge(env, controller, args.max_goal_radius, args.max_leg_m, spin_guard=args.spin_guard,
                        people=people, timing_meta=timing_meta)
        obstacles = [] if args.no_obstacles else [(*bridge.to_world(x, y), w) for x, y, w in DEMO_OBSTACLES]
        env._place_obstacles(obstacles)
        x0, y0, _ = bridge.start
        if not args.headless:
            dress_scenario(bridge, args.scenario)
        env.continue_to_goal((x0, y0))
        if not args.headless:
            frame_camera(bridge, args.scenario)
        if args.snapshot is not None:
            from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport

            env.settle_scan(120)
            capture_viewport_to_file(get_active_viewport(), str(args.snapshot))
            env.settle_scan(60)
            log(f"snapshot saved to {args.snapshot}")
            return
        log(f"scenario {args.scenario}; controller {args.controller}; start pose = mission origin (0, 0), +x ahead, +y left; "
            f"{len(obstacles)} obstacles placed; spin guard {'ON' if args.spin_guard else 'off'}")

        if args.plan is not None:
            commands = plan_commands(args.plan)
            log(f"running {len(commands)} commands from {args.plan}")
            bridge.timing_meta["plan"] = args.plan.name
            bridge.begin_mission(commands, f"{time.strftime('%Y%m%d_%H%M%S')}_{args.scenario}_{args.controller}")
            for line in commands:
                if not bridge.run(line):
                    break
            bridge.end_mission("failed", "plan aborted")   # no-op when the mission already ended
            log("plan finished" + ("" if args.headless else "; close the window to quit"))
            while not args.headless and env._simulation_app.is_running():
                env.settle_scan(1)
            return

        write_status("DONE")
        last = COMMAND_FILE.stat().st_mtime if COMMAND_FILE.exists() else 0.0
        log(f"waiting for CyberScape commands in {COMMAND_FILE} (approve a plan in the HITL dashboard)")
        while env._simulation_app.is_running():
            if COMMAND_FILE.exists() and COMMAND_FILE.stat().st_mtime != last:
                last = COMMAND_FILE.stat().st_mtime
                line = COMMAND_FILE.read_text().strip()
                if line:
                    if bridge.timer is None and args.mission_plan is not None and args.mission_plan.exists():
                        commands = plan_commands(args.mission_plan)
                        if commands and commands[0] == line:
                            bridge.timing_meta["plan"] = args.mission_plan.name
                            bridge.begin_mission(commands, f"{time.strftime('%Y%m%d_%H%M%S')}_{args.scenario}_{args.controller}")
                    bridge.run(line)
            else:
                env.settle_scan(2)
                time.sleep(0.05)
    finally:
        env.close()


if __name__ == "__main__":
    main()
