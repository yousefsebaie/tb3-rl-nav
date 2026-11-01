"""Unit tests for mission_timing.py (framework backlog P0: parallel tasks, unequal action
durations, failed runs, asynchronous starts).

  python3 -m unittest test_mission_timing -v
"""

import unittest

from mission_timing import MissionTimer, bootstrap_ci, compare_paired, paired_improvement_pct, percentile


class FakeClock:
    def __init__(self, t: float = 100.0):
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


def timer(tasks, clock, controller="ppo", **meta):
    return MissionTimer("m1", controller, clock, list(tasks), metadata=meta, wall_clock=lambda: 0.0)


class MissionTimerTests(unittest.TestCase):
    def test_worked_example_three_parallel_robots(self):
        # Framework section 4: robots finish at 12, 15, 18 s from a common t0.
        clock = FakeClock()
        t = timer(["a", "b", "c"], clock)
        t.start()
        for task, robot in (("a", "r1"), ("b", "r2"), ("c", "r3")):
            t.task_start(task, robot)
        for task, robot, at in (("a", "r1", 12), ("b", "r2", 15), ("c", "r3", 18)):
            clock.t = 100.0 + at
            t.task_complete(task)
            t.robot_complete(robot)
        row = t.finish("success")
        self.assertEqual(row["makespan_sim_s"], 18.0)
        self.assertEqual(row["mean_task_completion_sim_s"], 15.0)
        self.assertEqual(row["mean_robot_finish_sim_s"], 15.0)

    def test_unequal_action_durations_use_simulator_clock(self):
        # FORWARD (30 frames) and TURN (15 frames) + 8 settle frames at 60 Hz: the timer
        # must report what the simulator clock says, not actions x a constant.
        clock = FakeClock(0.0)
        t = timer(["go"], clock)
        t.start()
        t.task_start("go")
        for frames in (38, 23, 23, 38, 38):
            clock.advance(frames / 60.0)
        t.task_complete("go")
        row = t.finish("success")
        self.assertAlmostEqual(row["makespan_sim_s"], 160 / 60.0, places=3)
        self.assertNotAlmostEqual(row["makespan_sim_s"], 5 * 0.5, places=2)

    def test_failed_run_has_no_makespan_but_keeps_termination_time(self):
        clock = FakeClock()
        t = timer(["a", "b"], clock)
        t.start()
        clock.advance(4.0)
        t.task_complete("a")
        clock.advance(6.0)
        row = t.finish("collision", reason="hit debris")
        self.assertFalse(row["success"])
        self.assertIsNone(row["makespan_sim_s"])
        self.assertIsNone(row["mean_task_completion_sim_s"])
        self.assertEqual(row["termination_elapsed_sim_s"], 10.0)
        self.assertEqual(row["tasks_completed"], 1)

    def test_success_claim_without_all_tasks_is_incomplete(self):
        clock = FakeClock()
        t = timer(["a", "b"], clock)
        t.start()
        t.task_complete("a")
        row = t.finish("success")
        self.assertFalse(row["success"])
        self.assertEqual(row["status"], "incomplete")
        self.assertIsNone(row["makespan_sim_s"])

    def test_async_start_uses_shared_t0(self):
        # Robot 2 activates 5 s late and works 3 s: its task ends at 8 s from t0, not 3 s.
        clock = FakeClock()
        t = timer(["early", "late"], clock)
        t.start()
        t.task_start("early", "r1")
        clock.advance(5.0)
        t.task_start("late", "r2")
        clock.advance(1.0)
        t.task_complete("early")
        clock.advance(2.0)
        t.task_complete("late")
        row = t.finish("success")
        self.assertEqual(row["makespan_sim_s"], 8.0)

    def test_completion_never_overwritten(self):
        clock = FakeClock()
        t = timer(["a"], clock)
        t.start()
        clock.advance(2.0)
        t.task_complete("a")
        clock.advance(5.0)
        t.task_complete("a")
        self.assertEqual(t.finish("success")["makespan_sim_s"], 2.0)

    def test_nonmonotonic_clock_rejected(self):
        clock = FakeClock()
        t = timer(["a"], clock)
        t.start()
        clock.t -= 1.0
        with self.assertRaises(ValueError):
            t.task_complete("a")

    def test_unknown_task_rejected(self):
        clock = FakeClock()
        t = timer(["a"], clock)
        t.start()
        with self.assertRaises(ValueError):
            t.task_complete("zzz")

    def test_path_counters_latency(self):
        clock = FakeClock()
        t = timer(["a"], clock)
        t.start()
        for xy in ((0, 0), (3, 4), (3, 5)):
            t.record_pose(*xy)
        t.count("spin_guard")
        t.count("spin_guard")
        for _ in range(5):
            t.time_inference(lambda: None)
        clock.advance(1.0)
        t.task_complete("a")
        row = t.finish("success")
        self.assertEqual(row["path_length_m"], 6.0)
        self.assertEqual(row["count_spin_guard"], 2)
        self.assertEqual(row["decisions"], 5)
        self.assertIsNotNone(row["inference_p95_ms"])


class ComparisonTests(unittest.TestCase):
    def test_paired_improvement_sign(self):
        self.assertEqual(paired_improvement_pct(24.0, 18.0), 25.0)
        with self.assertRaises(ValueError):
            paired_improvement_pct(0.0, 1.0)

    def test_percentile(self):
        self.assertEqual(percentile([1, 2, 3, 4, 5], 50), 3)
        self.assertIsNone(percentile([], 50))

    def test_compare_uses_only_both_successful_and_reports_failures(self):
        rows = [
            {"controller": "rule", "scenario_id": 1, "success": True, "status": "success", "makespan_sim_s": 24.0},
            {"controller": "ppo", "scenario_id": 1, "success": True, "status": "success", "makespan_sim_s": 18.0},
            {"controller": "rule", "scenario_id": 2, "success": True, "status": "success", "makespan_sim_s": 20.0},
            {"controller": "ppo", "scenario_id": 2, "success": False, "status": "timeout", "makespan_sim_s": None},
        ]
        result = compare_paired(rows, "rule", "ppo", budget_s=60.0)
        self.assertEqual(result["scenarios_both_successful"], 1)
        self.assertEqual(result["mean_delta_s"], 6.0)
        self.assertEqual(result["mean_paired_improvement_pct"], 25.0)
        self.assertEqual(result["timeout_rate"]["ppo"], 0.5)
        self.assertEqual(result["penalized_mean_s"]["ppo"], (18.0 + 60.0) / 2)

    def test_bootstrap_ci_brackets_mean(self):
        low, high = bootstrap_ci([1.0, 2.0, 3.0, 4.0, 5.0])
        self.assertLessEqual(low, 3.0)
        self.assertGreaterEqual(high, 3.0)


if __name__ == "__main__":
    unittest.main()
