#!/usr/bin/env python3
"""Send approved CyberScape two-robot plans (MULTI_TURTLEBOT, or TURTLEBOT3_1 + TURTLEBOT3_2) to the MAPPO bridge in Isaac Sim, one phase at a time.

Same file protocol as cyberscape_isaac_plan_sender.py, but each command is one PHASE: JSON
{"phase": n, "lines": [...]} with every robot's low-level lines for that phase_number, so the
robots drive together (train_mappo.py --mission-listen). The HITL server launches this sender
after approval when CYBERSCAPE_ISAAC_SENDER points at it (run_cyberscape_mappo_demo.sh).
"""

import json
import os
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Turtlebot3", "rl_point_nav"))
from cyberscape_mappo_mission import merge_plans, phase_command  # noqa: E402

SERVER_URL = os.environ.get("CYBERSCAPE_SERVER_URL", "http://localhost:5001")
COMMAND_FILE = os.environ.get("CYBERSCAPE_ISAAC_COMMAND_FILE", os.path.expanduser("~/cyberscape_command.txt"))
STATUS_FILE = os.environ.get("CYBERSCAPE_ISAAC_STATUS_FILE", os.path.expanduser("~/cyberscape_status.txt"))
POLL_DELAY = 0.2
TIMEOUT_SECONDS = float(os.environ.get("CYBERSCAPE_ISAAC_COMMAND_TIMEOUT", "900"))


def fetch_approved_plans():
    with urllib.request.urlopen(SERVER_URL.rstrip("/") + "/hitl/get_approved_plans", timeout=10) as response:
        return json.loads(response.read().decode("utf-8"))["plans"]


def read_status():
    if not os.path.exists(STATUS_FILE):
        return ""
    with open(STATUS_FILE) as file:
        return file.read().strip()


def run_phase(number, lines):
    with open(STATUS_FILE, "w") as file:
        file.write("WAITING")
    with open(COMMAND_FILE, "w") as file:
        file.write(phase_command(number, lines))
    print(f"MAPPO sender: phase {number}:", "; ".join(lines), flush=True)
    start = time.time()
    while time.time() - start < TIMEOUT_SECONDS:
        status = read_status()
        if status == "DONE":
            print(f"MAPPO bridge completed phase {number}", flush=True)
            return True
        if status.startswith("FAILED"):
            print(f"MAPPO bridge failed phase {number}: {status}", flush=True)
            return False
        time.sleep(POLL_DELAY)
    print(f"MAPPO sender: timeout waiting for phase {number}", flush=True)
    return False


def main():
    try:
        plans = fetch_approved_plans()
    except urllib.error.URLError as exc:
        raise SystemExit(f"Could not fetch approved plans from {SERVER_URL}: {exc}")
    phases = merge_plans(plans)
    if not phases:
        print("No MULTI_TURTLEBOT / TURTLEBOT3_1 / TURTLEBOT3_2 plan approved.", flush=True)
        return
    print(f"Loaded approved plans {sorted(plans)}: {len(phases)} phases", flush=True)
    for number, lines in phases:
        if not run_phase(number, lines):
            print("MAPPO mission stopped", flush=True)
            return
    print("MAPPO mission complete", flush=True)


if __name__ == "__main__":
    main()
