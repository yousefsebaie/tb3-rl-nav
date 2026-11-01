#!/usr/bin/env python3
"""Live terminal view of a running training log ([TB03_PROGRESS] lines).

  python3 training_progress.py results/train_v3_realsize_long.log

Refreshes every few seconds. Ctrl-C only closes this view; training keeps running.
"""

from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path
import re
import time

FIELD = re.compile(r"(\w+)=(\S+)")
BAR_WIDTH = 40
HISTORY = 12   # success-rate samples in the trend line


def duration(seconds: float) -> str:
    if seconds < 0:
        return "--"
    seconds = int(seconds)
    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes = seconds // 60
    return f"{days}d {hours:02d}h {minutes:02d}m" if days else f"{hours:02d}h {minutes:02d}m"


def bar(fraction: float) -> str:
    filled = int(round(BAR_WIDTH * min(max(fraction, 0.0), 1.0)))
    return "█" * filled + "░" * (BAR_WIDTH - filled)


def read(log: Path):
    progress, events = None, []
    trend = deque(maxlen=HISTORY)
    with log.open(errors="replace") as handle:
        for line in handle:
            if "[TB03_PROGRESS]" in line:
                progress = dict(FIELD.findall(line))
                rate = next((v for k, v in progress.items() if k.endswith("_success")), None)
                if rate is not None and int(progress.get("episode", 0)) % 25 == 0:
                    trend.append(float(rate))
            elif "[TB03_PPO_V2]" in line and ("resumed" in line or "complete" in line or "training" in line):
                events.append(line.split("[TB03_PPO_V2]", 1)[1].strip())
            elif "Traceback" in line or "Error" in line and "[Error]" not in line and "Warning" not in line:
                events.append(line.strip()[:120])
    return progress, events[-3:], list(trend)


def render(log: Path, models: Path | None) -> str:
    progress, events, trend = read(log)
    lines = [f"TurtleBot3 PPO training   log: {log}", ""]
    if progress is None:
        lines.append("Waiting for the first finished episode (Isaac start-up takes ~1-2 min)...")
    else:
        steps, total = int(progress["steps"]), int(progress["total"])
        rate = float(progress["steps_per_s"])
        window = next((k for k in progress if k.endswith("_success")), "last100_success").split("_")[0]
        get = lambda name: float(progress.get(f"{window}_{name}", 0.0))
        lines += [
            f"Progress  {bar(steps / total)}  {100 * steps / total:5.1f}%",
            f"Steps     {steps:,} / {total:,}      speed {rate:.2f} steps/s ({rate * 3600:,.0f}/h)",
            f"Time      running {duration(float(progress['elapsed_s']))}   ETA {duration(float(progress['eta_s']))}",
            f"Episodes  {int(progress['episode']):,}   last: {progress.get('outcome', '?')} in {progress.get('length', '?')} steps",
            "",
            f"Over the {window.replace('last', 'last ')} episodes:",
            f"  success   {bar(get('success'))[:30]} {100 * get('success'):5.1f}%",
            f"  collision {bar(get('collision'))[:30]} {100 * get('collision'):5.1f}%",
            f"  timeout   {bar(get('timeout'))[:30]} {100 * get('timeout'):5.1f}%",
        ]
        if trend:
            lines.append("  success trend (every 25 episodes): " + " ".join(f"{100 * v:.0f}%" for v in trend))
    if models is not None and models.is_dir():
        checkpoints = sorted(models.glob("checkpoint_*_steps.zip"),
                             key=lambda p: int(re.search(r"(\d+)_steps", p.name).group(1)))
        if checkpoints:
            lines += ["", f"Latest checkpoint: {checkpoints[-1]}"]
    if events:
        lines += ["", "Events:"] + [f"  {event}" for event in events]
    age = time.time() - log.stat().st_mtime
    lines += ["", f"Log updated {age:.0f} s ago" + ("   ⚠ no output for 10+ min: check the training process" if age > 600 else ""),
              "Ctrl-C closes this view only; training keeps running."]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    parser.add_argument("--models", type=Path, default=None, help="model folder, to show the latest checkpoint")
    parser.add_argument("--every", type=float, default=5.0)
    args = parser.parse_args()
    try:
        while True:
            if args.log.exists():
                text = render(args.log, args.models)
            else:
                text = f"Waiting for {args.log} ..."
            print("\033[2J\033[H" + text, flush=True)
            time.sleep(args.every)
    except KeyboardInterrupt:
        print()


if __name__ == "__main__":
    main()
