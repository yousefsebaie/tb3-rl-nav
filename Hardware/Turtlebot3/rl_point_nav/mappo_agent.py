#!/usr/bin/env python3
"""One Burger in a CyberScape two-robot mission: runs each phase it is sent, with MAPPO V5.

The hardware half of train_mappo.py --mission-listen. In simulation one process drove both
robots; here each Pi runs this agent and the Mac's cyberscape_mappo_hw_sender.py is the
phase barrier: it sends phase n to both agents and waits until both report done.

Per phase, like train_mappo.run_phase():
* this robot's move_to_point((x, y)) / return_to_base(): align, MAPPO V5 + follow-the-gap
  (mappo_drive.drive) in the shared mission frame; a robot with no move holds still and
  reports active = 0 to its teammate;
* then its other commands in plan order: scan_with_lidar (sector distances),
  detect_with_camera (survivor within 1.5 m from the layout's known positions, the sim's
  camera stand-in), communicate_with_apm and the rest are logged as done.
* the phase fails if the move did not reach its goal.

The pose exchange with the teammate runs for the agent's whole life, so the teammate is
always visible, also while this robot holds still between phases.

HTTP on --port (default 8765):
  POST /phase  {"phase": n, "lines": [...this robot's low-level lines...]}  -> 202, or 409 if busy
  GET  /status -> {"phase", "state": IDLE|BUSY|DONE|FAILED, "summary", "pose"}
  POST /stop   -> stops the current move (the phase fails)

  python3 mappo_agent.py --robot TURTLEBOT3_1 --start 0 0 0 --peer <BURGER2_IP> \
      --survivors 1.6,-1.1 0.4,-1.1
"""

from __future__ import annotations

import argparse
import dataclasses
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import threading
import time

from cyberscape_mappo_mission import CAMERA_RANGE_M, MISSION_STEP_LIMIT, parse_line
from mappo_drive import DEFAULT_MODEL, drive
from mappo_hardware import POSE_PORT, V5_INPUT_NAMES
from policy_numpy import NumpyPolicy
from ros_mappo_env import MappoRosEnv
from ros_point_nav_lidar_env import v2_config

HERE = Path(__file__).resolve().parent
AGENT_PORT = 8765


def log(message: str) -> None:
    print(f"[CYBERSCAPE_MAPPO] {message}", flush=True)


class Agent:
    def __init__(self, args) -> None:
        self.args = args
        self.policy = NumpyPolicy(args.model)
        if self.policy.obs_dim != len(V5_INPUT_NAMES):
            raise SystemExit(f"{args.model} takes {self.policy.obs_dim} inputs; MAPPO V5 takes {len(V5_INPUT_NAMES)}")
        config = dataclasses.replace(v2_config(), cmd_vel_topic="/cmd_vel", max_episode_steps=args.max_steps,
                                     max_episode_sim_time_s=2.0 * args.max_steps)
        sx, sy, syaw = args.start
        self.base = (sx, sy)
        self.env = MappoRosEnv(args.robot, (sx, sy, math.radians(syaw)), config=config, peer=args.peer,
                               listen_port=args.listen_port, peer_port=args.peer_port)
        self.survivors = [tuple(float(v) for v in s.split(",")) for s in args.survivors]
        self.found: set[int] = set()
        self.lock = threading.Lock()
        self.status = {"phase": 0, "state": "IDLE", "summary": ""}
        self.stop_requested = threading.Event()
        (HERE / "missions").mkdir(exist_ok=True)
        self.csv = open(HERE / "missions" / f"{time.strftime('%Y%m%d_%H%M%S')}_mappo_mission_{args.robot}.csv",
                        "w", buffering=1)
        self.csv.write("t_s,x_m,y_m,yaw_deg," + ",".join(V5_INPUT_NAMES)
                       + ",policy_action,action,why,teammate_age_s\n")
        self.t0 = time.monotonic()
        log(f"{args.robot} ready: start=({sx:.2f}, {sy:.2f}, {syaw:.0f} deg) peer={args.peer} "
            f"survivors={self.survivors} gap={'off' if args.no_gap else 'on'}; log {self.csv.name}")

    # ------------------------------------------------------------ phases

    def start_phase(self, number: int, lines: list[str]) -> bool:
        with self.lock:
            if self.status["state"] == "BUSY":
                return False
            self.status = {"phase": number, "state": "BUSY", "summary": ""}
        self.stop_requested.clear()
        threading.Thread(target=self._run_phase, args=(number, lines), daemon=True).start()
        return True

    def _run_phase(self, number: int, lines: list[str]) -> None:
        try:
            ok, summary = self.run_phase(number, lines)
        except Exception as exc:  # report, keep the agent alive
            ok, summary = False, f"error: {exc}"
            self.env.stop()
        log(f"phase {number} {'DONE' if ok else 'FAILED'}: {summary}")
        with self.lock:
            self.status = {"phase": number, "state": "DONE" if ok else "FAILED", "summary": summary}

    def run_phase(self, number: int, lines: list[str]) -> tuple[bool, str]:
        moves, after = [], []
        for line in lines:
            _, action, nums = parse_line(line)
            if action == "move_to_point" and len(nums) >= 2:
                moves.append((nums[0], nums[1]))
            elif action == "return_to_base":
                moves.append(self.base)
            else:
                after.append((action, line))
        results = []
        if not moves:
            log(f"phase {number}: {self.args.robot} holds its position")
        for goal in moves:
            log(f"phase {number}: {self.args.robot} -> ({goal[0]:.2f}, {goal[1]:.2f})")
            outcome = drive(self.env, self.policy, goal, self.args.max_steps, not self.args.no_gap,
                            self.csv, self.t0, abort=self.stop_requested.is_set)
            results.append(outcome)
            if outcome != "SUCCESS":
                return False, f"{self.args.robot}={outcome} on the way to ({goal[0]:.2f}, {goal[1]:.2f})"
        for action, line in after:   # scans, camera checks and reports run after the move
            result = "done"
            if action == "scan_with_lidar":
                result = self.scan()
            elif action == "detect_with_camera":
                result = self.detect_survivor()
            log(f"phase {number}: {line} -> {result}")
        return True, f"{self.args.robot}=" + (",".join(results) if results else "HOLD")

    def scan(self) -> str:
        self.env._lidar_values()   # refresh last_lidar from the newest scan
        clear = self.env.last_lidar.get("clearance_m", {})
        return " ".join(f"{name}={min(v, 9.99):.2f}m" for name, v in clear.items())

    def detect_survivor(self) -> str:
        """Camera stand-in (as in simulation): nearest survivor within CAMERA_RANGE_M of the layout's positions."""
        x, y, _ = self.env.latest_shared()
        near = sorted((math.hypot(sx - x, sy - y), k, sx, sy) for k, (sx, sy) in enumerate(self.survivors))
        near = [n for n in near if n[0] <= CAMERA_RANGE_M]
        if not near:
            return "no survivor in camera range"
        distance, k, sx, sy = near[0]
        self.found.add(k)
        return (f"SURVIVOR DETECTED at ({sx:.2f}, {sy:.2f}), {distance:.2f} m away "
                f"({len(self.found)}/{len(self.survivors)} found by {self.args.robot})")

    def snapshot(self) -> dict:
        with self.lock:
            status = dict(self.status)
        pose = self.env.latest_shared()
        status["pose"] = None if pose is None else [round(pose[0], 3), round(pose[1], 3), round(math.degrees(pose[2]), 1)]
        mate = self.env.teammate()
        status["teammate"] = None if mate is None else {"x": round(mate["x"], 3), "y": round(mate["y"], 3),
                                                        "active": mate["active"], "age_s": round(mate["age_s"], 2)}
        return status


def make_handler(agent: Agent):
    class Handler(BaseHTTPRequestHandler):
        def _reply(self, code: int, body: dict) -> None:
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/status":
                self._reply(200, agent.snapshot())
            else:
                self._reply(404, {"error": "unknown path"})

        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length) or b"{}")
            if self.path == "/phase":
                if agent.start_phase(int(body["phase"]), list(body.get("lines", []))):
                    self._reply(202, {"accepted": True})
                else:
                    self._reply(409, {"error": "busy with phase " + str(agent.status["phase"])})
            elif self.path == "/stop":
                agent.stop_requested.set()
                agent.env.stop()
                self._reply(200, {"stopping": True})
            else:
                self._reply(404, {"error": "unknown path"})

        def log_message(self, *args):   # keep the console for the mission log
            pass

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--robot", required=True, choices=("TURTLEBOT3_1", "TURTLEBOT3_2"))
    parser.add_argument("--start", nargs=3, type=float, metavar=("X", "Y", "YAW_DEG"), required=True,
                        help="where this robot stands now in the shared mission frame (its base)")
    parser.add_argument("--peer", required=True, help="the other Burger's IP")
    parser.add_argument("--survivors", nargs="*", default=[], metavar="X,Y",
                        help="survivor positions in the mission frame (the camera stand-in)")
    parser.add_argument("--port", type=int, default=AGENT_PORT)
    parser.add_argument("--listen-port", type=int, default=POSE_PORT)
    parser.add_argument("--peer-port", type=int, default=POSE_PORT)
    parser.add_argument("--no-gap", action="store_true")
    parser.add_argument("--max-steps", type=int, default=MISSION_STEP_LIMIT)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    args = parser.parse_args()

    agent = Agent(args)
    server = ThreadingHTTPServer(("0.0.0.0", args.port), make_handler(agent))
    log(f"{args.robot} waiting for phases on port {args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log("interrupted")
    finally:
        agent.stop_requested.set()
        server.server_close()
        agent.csv.close()
        agent.env.close()   # sends a zero velocity


if __name__ == "__main__":
    main()
