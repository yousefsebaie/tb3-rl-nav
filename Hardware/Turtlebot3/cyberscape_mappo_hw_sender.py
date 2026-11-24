#!/usr/bin/env python3
"""Send approved CyberScape two-robot plans to the two real Burgers, one phase at a time.

Hardware counterpart of Hardware/Isaac/cyberscape_mappo_plan_sender.py. The plans are merged
by phase_number the same way (cyberscape_mappo_mission.merge_plans); each phase goes to both
robots' mappo_agent.py over HTTP (a robot with no lines in that phase holds still), and the
next phase starts only when BOTH report done: the phase barrier the simulator had for free.

The HITL server launches this after approval when CYBERSCAPE_ISAAC_SENDER points at it
(Hardware/Turtlebot3/rl_point_nav/run_cyberscape_mappo_hardware.sh).

  MAPPO_AGENTS="http://<BURGER1_IP>:8765,http://<BURGER2_IP>:8765"   (TurtleBot3_1 first)
"""

import json
import os
import re
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "rl_point_nav"))
from cyberscape_mappo_mission import ROBOT_NAMES, merge_plans, parse_line  # noqa: E402

SERVER_URL = os.environ.get("CYBERSCAPE_SERVER_URL", "http://localhost:5001")
AGENTS = [a.strip().rstrip("/") for a in os.environ.get("MAPPO_AGENTS", "").split(",") if a.strip()]
POLL_DELAY = 0.5
TIMEOUT_SECONDS = float(os.environ.get("CYBERSCAPE_ISAAC_COMMAND_TIMEOUT", "900"))


def http(method, url, body=None, timeout=5.0):
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_approved_plans():
    return http("GET", SERVER_URL.rstrip("/") + "/hitl/get_approved_plans", timeout=10)["plans"]


def stop_all():
    for agent in AGENTS:
        try:
            http("POST", agent + "/stop", {})
        except (urllib.error.URLError, OSError):
            pass


FOUND = {}   # survivor name -> [x, y], reported by a robot's camera in an earlier phase
_NAMED = re.compile(r"""(deliver_kit|move_to_point|look_at)\(\s*['"]([A-Za-z0-9_]+)['"]\s*\)""")


def fill_survivors(line):
    """deliver_kit('BOB2') -> deliver_kit((x, y)) with the location the camera robot reported."""
    def sub(match):
        name = match.group(2).upper()
        if name not in FOUND:
            raise LookupError(f"{name} has not been confirmed by a camera yet")
        x, y = FOUND[name]
        return f"{match.group(1)}(({x:.3f}, {y:.3f}))"
    return _NAMED.sub(sub, line)


def run_phase(number, lines):
    try:
        lines = [fill_survivors(line) for line in lines]
    except LookupError as exc:
        print(f"MAPPO hw sender: phase {number} cannot start: {exc}; stopping", flush=True)
        return False
    per_robot = [[] for _ in ROBOT_NAMES]
    for line in lines:
        index = parse_line(line)[0]
        if index is not None:
            per_robot[index].append(line)
        else:
            print(f"MAPPO hw sender: phase {number}: no robot for '{line}', skipped", flush=True)
    for name, robot_lines in zip(ROBOT_NAMES, per_robot):
        print(f"MAPPO hw sender: phase {number}: {name}: " + ("; ".join(robot_lines) or "hold"), flush=True)
    for agent, robot_lines in zip(AGENTS, per_robot):
        http("POST", agent + "/phase", {"phase": number, "lines": robot_lines})
    start = time.time()
    while time.time() - start < TIMEOUT_SECONDS:
        states = []
        for agent in AGENTS:
            try:
                status = http("GET", agent + "/status")
            except (urllib.error.URLError, OSError) as exc:
                print(f"MAPPO hw sender: {agent} not answering ({exc}); stopping both", flush=True)
                stop_all()
                return False
            states.append(status if status.get("phase") == number else {"state": "BUSY"})
        if any(s["state"] == "FAILED" for s in states):
            print(f"MAPPO hw sender: phase {number} FAILED: "
                  + " | ".join(s.get("summary", "") for s in states), flush=True)
            stop_all()
            return False
        if all(s["state"] == "DONE" for s in states):
            print(f"MAPPO robots completed phase {number}: "
                  + " | ".join(s.get("summary", "") for s in states), flush=True)
            for name, status in zip(ROBOT_NAMES, states):
                for survivor, where in (status.get("outputs") or {}).items():
                    FOUND[survivor.upper()] = where
                    print(f"MAPPO hw sender: {name} reports survivor {survivor.upper()} at "
                          f"({where[0]:.2f}, {where[1]:.2f}); shared with the team", flush=True)
            return True
        time.sleep(POLL_DELAY)
    print(f"MAPPO hw sender: timeout waiting for phase {number}; stopping both", flush=True)
    stop_all()
    return False


def main():
    if len(AGENTS) != len(ROBOT_NAMES):
        raise SystemExit(f"MAPPO_AGENTS must list {len(ROBOT_NAMES)} agent URLs (TurtleBot3_1 first), got {AGENTS}")
    try:
        plans = fetch_approved_plans()
    except urllib.error.URLError as exc:
        raise SystemExit(f"Could not fetch approved plans from {SERVER_URL}: {exc}")
    phases = merge_plans(plans)
    if not phases:
        print("No MULTI_TURTLEBOT / TURTLEBOT3_1 / TURTLEBOT3_2 plan approved.", flush=True)
        return
    print(f"Loaded approved plans {sorted(plans)}: {len(phases)} phases; agents {AGENTS}", flush=True)
    for number, lines in phases:
        if not run_phase(number, lines):
            print("MAPPO mission stopped", flush=True)
            return
    print("MAPPO mission complete", flush=True)


if __name__ == "__main__":
    main()
