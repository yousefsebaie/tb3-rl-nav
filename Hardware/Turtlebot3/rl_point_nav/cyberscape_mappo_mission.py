"""CyberScape -> MAPPO: two TurtleBot3s execute an approved MULTI_TURTLEBOT plan in Isaac Sim.

  CyberScape plan (Plans/rl_mappo_*_plan.json) -> HITL approval
      -> cyberscape_mappo_plan_sender.py (one PHASE per command, both robots' lines)
      -> COMMAND_FILE -> train_mappo.py --mission-listen -> MAPPO actor + gap controller -> wheels

Phases with the same phase_number run together: every robot with a move_to_point drives to it
in one MAPPO episode, robots without one hold their position. detect_with_camera('survivor')
marks the nearest survivor within CAMERA_RANGE_M of that robot (simulation ground truth, a
camera stand-in) with a green beacon. Other commands (scan_with_lidar, communicate_with_apm,
...) are logged as done. Coordinates are metres in the room (Isaac world) frame.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

ROBOT_NAMES = ("TurtleBot3_1", "TurtleBot3_2")
CAMERA_RANGE_M = 1.5
MISSION_STEP_LIMIT = 400          # a doorway leg with a gap hand-over can take ~250 steps
# Same files and environment variables as Hardware/Isaac/cyberscape_mappo_plan_sender.py.
COMMAND_FILE = Path(os.environ.get("CYBERSCAPE_ISAAC_COMMAND_FILE", Path.home() / "cyberscape_command.txt"))
STATUS_FILE = Path(os.environ.get("CYBERSCAPE_ISAAC_STATUS_FILE", Path.home() / "cyberscape_status.txt"))
_CALL = re.compile(r"^\s*([A-Za-z0-9_]+)\.([A-Za-z_]+)\((.*)\)\s*$")


def _doorway(x: float, y: float, gap: float, box: float = 0.4) -> tuple:
    """Four wall boxes along y = const with a gap centred on x (the obstacle pool holds 6)."""
    return tuple((x + s * (0.5 * gap + k * box), y, box) for s in (-1, 1) for k in (0.5, 1.5))


# Each scenario: bases (x, y, yaw) per robot, obstacles (x, y, width), survivors (chest x, chest y,
# body heading deg), the camera box (x range, y range) and, for disaster, the set dressing inputs.
SCENARIOS = {
    # Plans/rl_mappo_demo_plan.json: swap stations (crossing), doorway into a storage room.
    "warehouse": dict(
        bases=((-3.32, 0.444, 0.0), (-1.32, 0.444, 3.1416)),
        obstacles=_doorway(-2.3, -0.2, 0.80) + ((-0.75, 1.85, 0.3), (-3.6, 1.95, 0.3)),
        survivors=(),
        view=((-3.75, -0.6), (-1.1, 2.1)),
    ),
    # Plans/rl_mappo_disaster_plan.json: earthquake search and rescue in a collapsed building.
    #   top:    responder base; a 0.75 m x 0.8 m corridor through the rubble leads down to the hub
    #   middle: hub between the corridor and the fallen wall (0.8 m strip); open west and east sectors
    #   bottom: fallen wall (y = -0.1) with two 0.8 m doorways into two rooms split by a partition
    # Four survivors: west and east sectors, west room, east room.
    "disaster": dict(
        bases=((-2.3, 1.95, -1.5708), (-1.2, 1.95, -1.5708)),
        obstacles=(
            (-3.6, -0.1, 0.4), (-2.4, -0.1, 0.4), (-2.0, -0.1, 0.4), (-0.8, -0.1, 0.4), (-0.45, -0.1, 0.4),
            (-2.2, -0.5, 0.4), (-2.2, -0.9, 0.4), (-2.2, -1.15, 0.4),           # partition between the rooms
            (-2.875, 1.1, 0.4), (-2.875, 1.5, 0.4), (-1.725, 1.1, 0.4), (-1.725, 1.5, 0.4),   # corridor
            (-3.6, 1.9, 0.3), (-0.6, 1.95, 0.3)),
        survivors=((-3.7, 0.9, 90.0), (-0.5, 1.1, -90.0), (-3.45, -0.95, 0.0), (-0.95, -0.95, 180.0)),
        routes=(((-2.3, 1.95), (-2.3, 0.5)), ((-1.2, 1.95), (-1.0, 0.9)),
                ((-2.3, 0.5), (-3.2, 0.6)), ((-1.0, 0.9), (-1.4, 0.5)),
                ((-3.2, 0.6), (-3.0, -0.55)), ((-1.4, 0.5), (-1.4, -0.55)),
                ((-1.4, 0.5), (-2.3, 0.5))),
        base_xy=(-1.75, 1.95),
        wing=(-3.85, -0.35, -1.15, -0.25),
        view=((-3.9, -0.3), (-1.35, 2.25)),
    ),
    # Plans/rl_mappo_disaster_simple_plan.json: the first disaster layout (one doorway, three
    # survivors); a shorter fallback demo.
    "disaster_simple": dict(
        bases=((-2.75, 1.85, -1.5708), (-1.85, 1.85, -1.5708)),
        obstacles=_doorway(-2.3, -0.2, 0.80) + ((-3.5, 1.8, 0.3), (-0.75, 1.8, 0.3)),
        survivors=((-3.7, 0.55, 90.0), (-0.6, 0.55, 90.0), (-1.2, -0.9, 0.0)),
        routes=(((-2.75, 1.85), (-1.1, 0.55)), ((-1.85, 1.85), (-3.2, 0.55)),
                ((-1.1, 0.55), (-2.3, 0.55)), ((-3.2, 0.55), (-1.85, 1.85)),
                ((-2.3, 0.55), (-2.3, -0.85)), ((-1.85, 1.85), (-2.3, 0.75))),
        base_xy=(-2.3, 1.9),
        wing=(-3.85, -0.4, -1.15, -0.45),
        view=((-3.85, -0.4), (-1.45, 2.2)),
    ),
}


def _robot_index(name: str):
    """TurtleBot3_1 / TURTLEBOT3_1 / Turtlebot31 (the low-level planner's prefix) -> 0."""
    key = name.replace("_", "").lower()
    return next((i for i, n in enumerate(ROBOT_NAMES) if n.replace("_", "").lower() == key), None)


def parse_line(line: str):
    """'TurtleBot3_2.move_to_point((-3.0, 1.5))' -> (robot index or None, action, [numbers])."""
    m = _CALL.match(line)
    if not m:
        return None, line.strip(), []
    name, action, args = m.groups()
    return _robot_index(name), action, [float(v) for v in re.findall(r"-?\d+(?:\.\d+)?", args)]


def phase_groups(plan: dict) -> list[tuple[int, list[str]]]:
    """[(phase_number, [low-level lines of every robot in that phase])] in phase order."""
    groups: dict[int, list[str]] = {}
    for phase in plan.get("phases", []):
        lines = phase.get("low_level_plan", "")
        lines = lines if isinstance(lines, list) else str(lines).splitlines()
        groups.setdefault(int(phase.get("phase_number", len(groups) + 1)), []).extend(
            l.strip() for l in lines if l.strip())
    return sorted(groups.items())


def merge_plans(plans: dict) -> list[tuple[int, list[str]]]:
    """Approved CyberScape plans -> phases. A MULTI_TURTLEBOT plan holds both robots; per-robot plans
    (TURTLEBOT3_1, TURTLEBOT3_2, as the LLM pipeline writes them) are merged by phase_number, and
    each line is addressed to that plan's robot whatever prefix the LLM used."""
    groups: dict[int, list[str]] = {}
    for name, plan in sorted(plans.items()):
        index = _robot_index(name)
        if index is None and name.upper() != "MULTI_TURTLEBOT":
            continue
        for number, lines in phase_groups(plan):
            if index is not None:
                lines = [f"{ROBOT_NAMES[index]}.{l.split('.', 1)[1]}" if _CALL.match(l) else l for l in lines]
            groups.setdefault(number, []).extend(lines)
    return sorted(groups.items())


def phase_command(number: int, lines: list[str]) -> str:
    return json.dumps({"phase": number, "lines": lines})


def load_plan(path: Path) -> list[tuple[int, list[str]]]:
    return phase_groups(json.loads(Path(path).read_text()))
