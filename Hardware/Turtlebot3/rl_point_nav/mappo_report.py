#!/usr/bin/env python3
"""Per-episode metrics from a MAPPO / IPPO / baseline log ([TB03_MAPPO] episode lines).

An episode counts ONCE:
  collision     any robot collided at any point in the episode (robot-robot share shown too)
  both_reached  every robot reached its goal
  timeout       neither of the above (the 250-step / 120 s limit ended it)

  python3 mappo_report.py results/mappo_baseline_v4_encounters.log
  python3 mappo_report.py results/train_mappo_v4.log --last 500     # the last 500 training episodes
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import re


def episodes(path: str) -> list[tuple[str, list[str], list[str]]]:
    rows = []
    for line in open(path, errors="ignore"):
        if not line.startswith("[TB03_MAPPO] episode="):
            continue
        encounter = re.search(r"encounter=(\S+)", line)
        rows.append((
            encounter[1] if encounter else "random",
            re.search(r"outcomes=(\S+)", line)[1].split(","),
            re.search(r"collisions=(\S+)", line)[1].split(","),
        ))
    return rows


def summary(rows) -> str:
    n = len(rows)
    collision = sum("COLLISION" in outcomes for _, outcomes, _ in rows)
    robot_robot = sum("robot" in kinds for _, _, kinds in rows)
    reached = sum(all(o == "SUCCESS" for o in outcomes) for _, outcomes, _ in rows)
    timeout = n - collision - reached
    return (f"episodes={n:4d}  collision={collision / n:6.1%} (robot-robot {robot_robot / n:5.1%})  "
            f"both_reached={reached / n:6.1%}  timeout={timeout / n:6.1%}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("log")
    parser.add_argument("--last", type=int, default=0, help="only the last N episodes")
    args = parser.parse_args()
    rows = episodes(args.log)
    if args.last:
        rows = rows[-args.last:]
    if not rows:
        raise SystemExit(f"No [TB03_MAPPO] episode lines in {args.log}")
    print(f"{'ALL':18s} {summary(rows)}")
    groups = defaultdict(list)
    for row in rows:
        groups[row[0]].append(row)
    for name in sorted(groups):
        print(f"{name:18s} {summary(groups[name])}")


if __name__ == "__main__":
    main()
