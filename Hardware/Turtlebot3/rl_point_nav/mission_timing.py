"""Mission Timing Evaluator (MTE): event-based timing for missions and episodes.

Implements the timing contract of the "Mission-Time & RL Performance Evaluation
Framework" (sections 2-4, 7, 8). It is instrumentation, not a policy: the same code
times every controller (rule-based B1, PPO, later MAPPO).

Three clocks (section 3):
  * simulation clock  - Isaac's authoritative time, read AFTER physics steps
                        (never "number of actions x assumed duration");
  * monotonic wall    - time.perf_counter_ns(), for real elapsed time and RTF;
  * decision clock    - wall time of each policy call (inference latency).

Timestamps are stored relative to a shared mission start t0. A task's completion time
is written once (first transition to complete) and never overwritten. Makespan and
completion means are reported only for successful missions; failures keep their
termination time and reason instead.

Outputs (section 7, EXPORT): one JSONL event log per mission/episode plus one summary
row (dict) that callers append to a CSV.

Stdlib only; unit tests in test_mission_timing.py.
"""

from __future__ import annotations

import csv
import json
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, median
from typing import Callable, Iterable, Optional


def wall_now() -> float:
    """Monotonic wall clock in seconds."""
    return time.perf_counter_ns() / 1e9


def percentile(values: list[float], q: float) -> Optional[float]:
    """Linear-interpolated percentile (q in 0..100); None for no data."""
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q / 100.0
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


@dataclass
class MissionTimer:
    """Event recorder for one mission (or one evaluation episode).

    sim_clock: callable returning the simulator's current time in seconds.
    required_tasks: every mandatory task id; success needs all of them complete.
    """

    mission_id: str
    controller: str
    sim_clock: Callable[[], float]
    required_tasks: list[str]
    metadata: dict = field(default_factory=dict)
    wall_clock: Callable[[], float] = wall_now

    start_sim: Optional[float] = None
    start_wall: Optional[float] = None
    task_start_sim: dict = field(default_factory=dict)
    task_finish_sim: dict = field(default_factory=dict)
    task_owner: dict = field(default_factory=dict)
    robot_finish_sim: dict = field(default_factory=dict)
    inference_ms: list = field(default_factory=list)
    counters: dict = field(default_factory=dict)
    path_length_m: float = 0.0
    events: list = field(default_factory=list)
    _last_sim: Optional[float] = None
    _last_xy: Optional[tuple] = None
    _result: Optional[dict] = None

    # ------------------------------------------------------------- events
    def _event(self, kind: str, **data) -> None:
        sim = self.sim_clock()
        if self._last_sim is not None and sim < self._last_sim:
            raise ValueError(f"non-monotonic simulator time: {sim} < {self._last_sim}")
        self._last_sim = sim
        record = {"event": kind, "sim_s": None if self.start_sim is None else sim - self.start_sim,
                  "sim_abs_s": sim, "wall_s": None if self.start_wall is None else self.wall_clock() - self.start_wall}
        record.update(data)
        self.events.append(record)

    def start(self) -> None:
        """Mission t0: after reset/warm-up, at the first controller-enabled instant."""
        if self.start_sim is not None:
            raise RuntimeError("mission already started")
        self.start_sim, self.start_wall = self.sim_clock(), self.wall_clock()
        self._event("mission_start", tasks=list(self.required_tasks), **self.metadata)

    def task_start(self, task_id: str, robot_id: str = "robot_0", **data) -> None:
        self._check_task(task_id)
        if task_id not in self.task_start_sim:
            self.task_start_sim[task_id] = self.sim_clock()
            self.task_owner[task_id] = robot_id
            self._event("task_start", task=task_id, robot=robot_id, **data)

    def task_complete(self, task_id: str, **data) -> None:
        """First transition to complete only; never overwrites the completion time."""
        self._check_task(task_id)
        if task_id in self.task_finish_sim:
            return
        sim = self.sim_clock()
        if sim < self.start_sim:
            raise ValueError("task completed before mission start")
        self.task_finish_sim[task_id] = sim
        self._event("task_complete", task=task_id, robot=self.task_owner.get(task_id), **data)

    def task_failed(self, task_id: str, reason: str) -> None:
        self._check_task(task_id)
        self._event("task_failed", task=task_id, reason=reason)

    def robot_complete(self, robot_id: str) -> None:
        if robot_id not in self.robot_finish_sim:
            self.robot_finish_sim[robot_id] = self.sim_clock()
            self._event("robot_complete", robot=robot_id)

    def count(self, name: str, **data) -> None:
        """Safety overrides, recoveries, disturbances... (counted and logged)."""
        self.counters[name] = self.counters.get(name, 0) + 1
        self._event(name, **data)

    def record_pose(self, x: float, y: float) -> None:
        """Accumulate traversed path length (metres) from successive positions."""
        if self._last_xy is not None:
            self.path_length_m += math.hypot(x - self._last_xy[0], y - self._last_xy[1])
        self._last_xy = (x, y)

    def time_inference(self, fn: Callable, *args, **kwargs):
        """Run a policy call and record its wall latency in ms (decision clock)."""
        begin = time.perf_counter_ns()
        result = fn(*args, **kwargs)
        self.inference_ms.append((time.perf_counter_ns() - begin) / 1e6)
        return result

    def _check_task(self, task_id: str) -> None:
        if self.start_sim is None:
            raise RuntimeError("call start() first")
        if task_id not in self.required_tasks:
            raise ValueError(f"unexpected task {task_id!r}")

    # ---------------------------------------------------------- aggregate
    def finish(self, status: str, reason: str = "", warmup_decisions: int = 1, **extra) -> dict:
        """status: 'success', 'collision', 'timeout', 'failed', 'error'. Returns the summary row."""
        if self._result is not None:
            return self._result
        end_sim, end_wall = self.sim_clock(), self.wall_clock()
        self._event("mission_end", status=status, reason=reason)
        success = status == "success" and set(self.required_tasks) <= set(self.task_finish_sim)
        task_elapsed = [self.task_finish_sim[t] - self.start_sim for t in self.required_tasks if t in self.task_finish_sim]
        robot_elapsed = [t - self.start_sim for t in self.robot_finish_sim.values()]
        sim_elapsed, wall_elapsed = end_sim - self.start_sim, end_wall - self.start_wall
        latency = self.inference_ms[warmup_decisions:] if len(self.inference_ms) > warmup_decisions else self.inference_ms
        row = {
            "mission_id": self.mission_id,
            "controller": self.controller,
            "status": status if not (status == "success" and not success) else "incomplete",
            "success": success,
            "reason": reason,
            "tasks_required": len(self.required_tasks),
            "tasks_completed": len(self.task_finish_sim),
            "makespan_sim_s": round(max(task_elapsed), 3) if success else None,
            "mean_task_completion_sim_s": round(mean(task_elapsed), 3) if success else None,
            "mean_robot_finish_sim_s": round(mean(robot_elapsed), 3) if success and robot_elapsed else None,
            "termination_elapsed_sim_s": round(sim_elapsed, 3),
            "wall_elapsed_s": round(wall_elapsed, 3),
            "rtf": round(sim_elapsed / wall_elapsed, 4) if wall_elapsed > 0 else None,
            "path_length_m": round(self.path_length_m, 3),
            "decisions": len(self.inference_ms),
            "inference_p50_ms": _round(percentile(latency, 50)),
            "inference_p95_ms": _round(percentile(latency, 95)),
            "inference_p99_ms": _round(percentile(latency, 99)),
            "inference_mean_ms": _round(mean(latency) if latency else None),
        }
        for name, value in sorted(self.counters.items()):
            row[f"count_{name}"] = value
        row.update(self.metadata)
        row.update(extra)
        self._result = row
        return row

    def write_events(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as handle:
            for record in self.events:
                handle.write(json.dumps(record) + "\n")


def _round(value: Optional[float], digits: int = 3) -> Optional[float]:
    return None if value is None else round(value, digits)


def append_csv(path: Path, row: dict) -> None:
    """Append one summary row; adds new columns by rewriting the file when needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    if path.exists():
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
    rows.append({k: ("" if v is None else v) for k, v in row.items()})
    fields: list[str] = []
    for r in rows:
        fields += [k for k in r if k not in fields]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


# ------------------------------------------------------------ comparisons (section 4)
def paired_improvement_pct(baseline_s: float, candidate_s: float) -> float:
    """Positive = candidate faster: 100 * (baseline - candidate) / baseline."""
    if baseline_s <= 0:
        raise ValueError("baseline completion time must be positive")
    return 100.0 * (baseline_s - candidate_s) / baseline_s


def bootstrap_ci(values: list[float], statistic=mean, resamples: int = 5000,
                 level: float = 0.95, seed: int = 0) -> tuple[Optional[float], Optional[float]]:
    """Percentile bootstrap CI over scenarios (resampling the paired differences)."""
    if len(values) < 2:
        return None, None
    rng = random.Random(seed)
    stats = sorted(statistic([rng.choice(values) for _ in values]) for _ in range(resamples))
    tail = (1.0 - level) / 2.0
    return stats[int(tail * (resamples - 1))], stats[int((1.0 - tail) * (resamples - 1))]


def compare_paired(rows: Iterable[dict], baseline: str, candidate: str,
                   metric: str = "makespan_sim_s", key: str = "scenario_id",
                   budget_s: Optional[float] = None) -> dict:
    """Paired comparison of two controllers on matched scenarios (sections 4, 9).

    Time statistics use only scenarios where BOTH succeeded. Success/collision/timeout
    rates use all attempts. With budget_s, also a failure-aware penalized mean where a
    failed run counts as budget_s.
    """
    by_controller: dict[str, dict] = {baseline: {}, candidate: {}}
    for row in rows:
        if row["controller"] in by_controller:
            by_controller[row["controller"]][str(row[key])] = row

    def rate(controller, status):
        runs = list(by_controller[controller].values())
        return sum(r["status"] == status for r in runs) / len(runs) if runs else None

    common = sorted(set(by_controller[baseline]) & set(by_controller[candidate]))
    both = [s for s in common
            if _truthy(by_controller[baseline][s]["success"]) and _truthy(by_controller[candidate][s]["success"])]
    base_t = [float(by_controller[baseline][s][metric]) for s in both]
    cand_t = [float(by_controller[candidate][s][metric]) for s in both]
    deltas = [b - c for b, c in zip(base_t, cand_t)]
    pcts = [paired_improvement_pct(b, c) for b, c in zip(base_t, cand_t)]
    result = {
        "baseline": baseline, "candidate": candidate, "metric": metric,
        "scenarios_matched": len(common), "scenarios_both_successful": len(both),
        "success_rate": {c: rate(c, "success") for c in by_controller},
        "collision_rate": {c: rate(c, "collision") for c in by_controller},
        "timeout_rate": {c: rate(c, "timeout") for c in by_controller},
        "mean_s": {baseline: _mean(base_t), candidate: _mean(cand_t)},
        "median_s": {baseline: _median(base_t), candidate: _median(cand_t)},
        "mean_delta_s": _mean(deltas),
        "delta_ci95_s": bootstrap_ci(deltas),
        "mean_paired_improvement_pct": _mean(pcts),
        "paired_improvement_ci95_pct": bootstrap_ci(pcts),
        "aggregate_mean_change_pct": (paired_improvement_pct(_mean(base_t), _mean(cand_t)) if base_t else None),
    }
    if budget_s is not None:
        def penalized(controller):
            values = [float(r[metric]) if _truthy(r["success"]) else budget_s
                      for s, r in by_controller[controller].items() if s in common]
            return _mean(values)
        result["penalized_mean_s"] = {c: penalized(c) for c in by_controller}
    return result


def _truthy(value) -> bool:
    return value is True or str(value).lower() == "true"


def _mean(values):
    return mean(values) if values else None


def _median(values):
    return median(values) if values else None
