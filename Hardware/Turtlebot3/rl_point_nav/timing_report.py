#!/usr/bin/env python3
"""Print the timing study tables from the Mission Timing Evaluator outputs.

  python3 timing_report.py [--run results/timing/<run>] [--missions results/missions/missions.csv]

Episode table: success / collision / timeout over ALL attempts, then time statistics on
scenarios where BOTH controllers succeeded (paired), with bootstrap 95% CIs and the
failure-penalized mean. Mission table: one row per timed CyberScape mission.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from mission_timing import compare_paired

HERE = Path(__file__).resolve().parent


def fmt(value, digits=2, suffix=""):
    if value is None or value == "":
        return "-"
    if isinstance(value, (tuple, list)):
        return "[" + ", ".join(fmt(v, digits) for v in value) + "]" + suffix
    try:
        return f"{float(value):.{digits}f}{suffix}"
    except (TypeError, ValueError):
        return str(value)


def episode_report(run: Path) -> None:
    rows = list(csv.DictReader((run / "episodes.csv").open()))
    controllers = list(dict.fromkeys(r["controller"] for r in rows))
    manifest = json.loads((run / "manifest.json").read_text())
    print(f"\nEPISODES  {run.name}: {len(manifest['scenarios'])} scenarios x {controllers}, "
          f"sensor noise {'on' if manifest['sensor_noise'] else 'off'}, model {Path(manifest['model']).name}")
    print(f"{'controller':<10}{'success':>9}{'collision':>11}{'timeout':>9}{'p50 ms':>9}{'p95 ms':>9}{'p99 ms':>9}{'RTF':>7}")
    for c in controllers:
        mine = [r for r in rows if r["controller"] == c]
        share = lambda s: sum(r["status"] == s for r in mine) / len(mine)
        med = lambda k: sorted(float(r[k]) for r in mine if r[k])[len(mine) // 2] if any(r[k] for r in mine) else None
        print(f"{c:<10}{share('success'):>9.0%}{share('collision'):>11.0%}{share('timeout'):>9.0%}"
              f"{fmt(med('inference_p50_ms'), 3):>9}{fmt(med('inference_p95_ms'), 3):>9}{fmt(med('inference_p99_ms'), 3):>9}"
              f"{fmt(med('rtf')):>7}")
    if len(controllers) == 2:
        base, cand = controllers
        t = compare_paired(rows, base, cand, budget_s=120.0)   # PointNavConfig.max_episode_sim_time_s
        p = compare_paired(rows, base, cand, metric="path_length_m")
        print(f"\nPaired time to goal (sim s), {t['scenarios_both_successful']} scenarios where both succeeded "
              f"(baseline {base}, candidate {cand}; positive = {cand} faster):")
        print(f"  mean   {base} {fmt(t['mean_s'][base])} s   {cand} {fmt(t['mean_s'][cand])} s")
        print(f"  median {base} {fmt(t['median_s'][base])} s   {cand} {fmt(t['median_s'][cand])} s")
        print(f"  mean paired delta {fmt(t['mean_delta_s'])} s, 95% CI {fmt(t['delta_ci95_s'])} s")
        print(f"  mean paired improvement {fmt(t['mean_paired_improvement_pct'], 1)} %, 95% CI {fmt(t['paired_improvement_ci95_pct'], 1)} %; "
              f"change of aggregate means {fmt(t['aggregate_mean_change_pct'], 1)} %")
        print(f"  failure-penalized mean (failures count as 120 s): {base} {fmt(t['penalized_mean_s'][base])} s, "
              f"{cand} {fmt(t['penalized_mean_s'][cand])} s")
        print(f"  path length mean {base} {fmt(p['mean_s'][base])} m, {cand} {fmt(p['mean_s'][cand])} m")


def mission_report(path: Path) -> None:
    if not path.exists():
        return
    rows = list(csv.DictReader(path.open()))
    print(f"\nMISSIONS  {path}")
    print(f"{'mission':<34}{'ctrl':<6}{'status':<10}{'tasks':>6}{'makespan':>10}{'mean task':>10}{'wall':>8}{'RTF':>6}"
          f"{'path':>7}{'p95 ms':>8}  overrides")
    for r in rows:
        overrides = ", ".join(f"{k[6:]}={v}" for k, v in r.items() if k.startswith("count_") and v)
        print(f"{r['mission_id'][:33]:<34}{r['controller']:<6}{r['status']:<10}"
              f"{r['tasks_completed'] + '/' + r['tasks_required']:>6}{fmt(r['makespan_sim_s'], 1, 's'):>10}"
              f"{fmt(r['mean_task_completion_sim_s'], 1, 's'):>10}{fmt(r['wall_elapsed_s'], 0, 's'):>8}{fmt(r['rtf']):>6}"
              f"{fmt(r['path_length_m'], 1, 'm'):>7}{fmt(r['inference_p95_ms'], 3):>8}  {overrides}")
    planning = path.parent / "planning.csv"
    if planning.exists():
        times = [float(r["planning_wall_s"]) for r in csv.DictReader(planning.open()) if r["ok"] == "True"]
        if times:
            print(f"\nLLM planning overhead (wall): {len(times)} runs, mean {sum(times) / len(times):.1f} s, "
                  f"min {min(times):.1f} s, max {max(times):.1f} s")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=None, help="results/timing/<run> (default: latest)")
    parser.add_argument("--missions", type=Path, default=HERE / "results" / "missions" / "missions.csv")
    args = parser.parse_args()
    run = args.run
    if run is None:
        runs = sorted((HERE / "results" / "timing").glob("*/episodes.csv"))
        run = runs[-1].parent if runs else None
    if run is not None:
        episode_report(run)
    mission_report(args.missions)


if __name__ == "__main__":
    main()
