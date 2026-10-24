#!/usr/bin/env python3
"""Mirror approved CyberScape TurtleBot3 plans to Isaac Sim.

The real Burger receives instructions by polling /get_instruction. Isaac Sim
uses a lightweight file bridge: this sender writes one instruction at a time to
/home/g8/cyberscape_command.txt, and the Isaac Script Editor listener marks the
command DONE in /home/g8/cyberscape_status.txt.
"""

import json
import os
import time
import urllib.error
import urllib.request


SERVER_URL = os.environ.get("CYBERSCAPE_SERVER_URL", "http://localhost:5001")
COMMAND_FILE = os.environ.get("CYBERSCAPE_ISAAC_COMMAND_FILE", "/home/g8/cyberscape_command.txt")
STATUS_FILE = os.environ.get("CYBERSCAPE_ISAAC_STATUS_FILE", "/home/g8/cyberscape_status.txt")
POLL_DELAY = float(os.environ.get("CYBERSCAPE_ISAAC_POLL_DELAY", "0.2"))
TIMEOUT_SECONDS = float(os.environ.get("CYBERSCAPE_ISAAC_COMMAND_TIMEOUT", "180"))


def fetch_approved_plans():
    url = SERVER_URL.rstrip("/") + "/hitl/get_approved_plans"
    with urllib.request.urlopen(url, timeout=10) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return payload["plans"]


def flatten_commands(plans):
    commands = []
    for robot_name, plan in plans.items():
        if robot_name.upper() != "TURTLEBOT3":
            continue

        for phase in plan.get("phases", []):
            low_level_plan = phase.get("low_level_plan", "")
            if isinstance(low_level_plan, list):
                lines = low_level_plan
            else:
                lines = str(low_level_plan).splitlines()

            for line in lines:
                command = line.strip()
                if command:
                    commands.append(command)

    return commands


def write_status(status):
    with open(STATUS_FILE, "w") as file:
        file.write(status)


def read_status():
    if not os.path.exists(STATUS_FILE):
        return ""
    with open(STATUS_FILE, "r") as file:
        return file.read().strip()


def send_command(command):
    write_status("WAITING")
    with open(COMMAND_FILE, "w") as file:
        file.write(command)
    print("Isaac sender sent:", command, flush=True)


def wait_until_done(command):
    start_time = time.time()
    while True:
        if read_status() == "DONE":
            print("Isaac completed:", command, flush=True)
            return True

        if time.time() - start_time > TIMEOUT_SECONDS:
            print("Isaac timeout waiting for:", command, flush=True)
            return False

        time.sleep(POLL_DELAY)


def main():
    try:
        plans = fetch_approved_plans()
    except urllib.error.URLError as exc:
        raise SystemExit(f"Could not fetch approved plans from {SERVER_URL}: {exc}")

    commands = flatten_commands(plans)
    print("Loaded approved TurtleBot3 commands:", len(commands), flush=True)

    if not commands:
        print("No TurtleBot3 commands found for Isaac.", flush=True)
        return

    write_status("DONE")
    for command in commands:
        send_command(command)
        if not wait_until_done(command):
            break

    print("Isaac mirrored CyberScape plan finished", flush=True)


if __name__ == "__main__":
    main()
