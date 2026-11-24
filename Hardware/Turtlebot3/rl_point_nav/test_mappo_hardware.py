"""Unit tests for the MAPPO V5 hardware inputs (mappo_hardware.py, lidar_sectors.py "v5").

Each test checks the hardware code against the simulator's formulas
(isaac_multi_point_nav_env.py, train_mappo.py) or the handoff's worked examples.

  python3 -m unittest test_mappo_hardware -v
"""

import math
import random
import time
import unittest
from pathlib import Path

import numpy as np

import lidar_sectors
from lidar_sectors import STOCK_MOUNT_M, sector_names, sim_matched_sectors
from mappo_hardware import (BODY_RADIUS_AT_SCAN_M, FAR_TEAMMATE, FORWARD, GAP_STUCK_STEPS, STAY, TURN_LEFT,
                            V5_INPUT_NAMES, GapAssist, PoseLink, SharedFrame, free_gap_heading, head_on_starts,
                            lidar_pose, merge_teammate_circle, teammate_inputs)

HERE = Path(__file__).resolve().parent
V5_MODEL = HERE / "models" / "mappo_v5_kl05" / "best_model.npz"
V4_MODEL = HERE / "models" / "isaac_point_nav_lidar_ppo_v3_random" / "checkpoint_1570000_steps.npz"


def sim_teammate_ranges(sensor, azimuth_deg, teammate_xy, radius=0.105, near=0.10):
    """isaac_multi_point_nav_env._teammate_ranges(), one teammate."""
    px, py, yaw = sensor
    angles = yaw + np.radians(azimuth_deg)
    ux, uy = np.cos(angles), np.sin(angles)
    dx, dy = teammate_xy[0] - px, teammate_xy[1] - py
    along = ux * dx + uy * dy
    perp_sq = dx * dx + dy * dy - along * along
    inside = perp_sq <= radius * radius
    with np.errstate(invalid="ignore"):
        distance = along - np.sqrt(np.maximum(radius * radius - perp_sq, 0.0))
    hits = np.where(inside & (distance > 0.0), distance, np.nan)
    with np.errstate(invalid="ignore"):
        hits[hits < near] = np.nan
    return hits


class V5InputTests(unittest.TestCase):
    def test_input_names_match_the_exported_policy(self):
        self.assertEqual(V5_INPUT_NAMES, tuple(str(n) for n in np.load(V5_MODEL)["inputs"]))

    def test_handoff_example_teammate_one_metre_ahead(self):
        # Handoff step 4: Burger 2 1.0 m directly ahead of Burger 1 -> Burger 1 reads (0.5, 0.0).
        me, other = (0.3, -0.2, math.radians(40)), None
        other = (0.3 + math.cos(math.radians(40)), -0.2 + math.sin(math.radians(40)))
        role, tx, ty, active = teammate_inputs(me, other, True, 1.0)
        self.assertEqual((role, active), (1.0, 1.0))
        self.assertAlmostEqual(tx, 0.5)
        self.assertAlmostEqual(ty, 0.0)

    def test_teammate_inputs_match_sim_global_state(self):
        rng = random.Random(3)
        for _ in range(200):
            me = (rng.uniform(-3, 3), rng.uniform(-3, 3), rng.uniform(-math.pi, math.pi))
            other = (rng.uniform(-3, 3), rng.uniform(-3, 3))
            dx, dy = other[0] - me[0], other[1] - me[1]
            c, s = math.cos(me[2]), math.sin(me[2])
            expect = (np.clip((c * dx + s * dy) / 2.0, -1, 1), np.clip((-s * dx + c * dy) / 2.0, -1, 1))
            got = teammate_inputs(me, other, False, 0.0)
            self.assertAlmostEqual(got[1], expect[0])
            self.assertAlmostEqual(got[2], expect[1])
            self.assertEqual(got[3], 0.0)

    def test_far_teammate(self):
        self.assertEqual(teammate_inputs((0, 0, 0), None, True, 1.0), [1.0, *FAR_TEAMMATE])

    def test_teammate_circle_matches_sim(self):
        rng = random.Random(5)
        n = 230
        inc = 2 * math.pi / n
        azimuth = np.degrees(np.arange(n) * inc)
        for _ in range(100):
            sensor = (rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(-math.pi, math.pi))
            mate = (sensor[0] + rng.uniform(-1.5, 1.5), sensor[1] + rng.uniform(-1.5, 1.5))
            real = [rng.choice([math.inf, 0.0, rng.uniform(0.15, 3.0)]) for _ in range(n)]
            merged = merge_teammate_circle(real, 0.0, inc, 0.12, 3.5, sensor, mate)
            hits = sim_teammate_ranges(sensor, azimuth, mate)
            for i in range(n):
                valid = math.isfinite(real[i]) and 0.12 <= real[i] <= 3.5
                if np.isnan(hits[i]):
                    self.assertEqual(merged[i], real[i])
                elif valid:
                    self.assertAlmostEqual(merged[i], min(real[i], hits[i]))
                else:
                    self.assertAlmostEqual(merged[i], hits[i])

    def test_v5_sectors_match_sim_masks_at_the_training_mount(self):
        # With the LDS at the training mount there is no shift, so every sector must equal
        # the simulator's mask minimum over its own rays (overlapping sectors included).
        saved = lidar_sectors.REAL_MOUNT_M
        lidar_sectors.REAL_MOUNT_M = STOCK_MOUNT_M
        try:
            rng = random.Random(7)
            n = 360
            inc = 2 * math.pi / n
            start = math.radians(0.37)   # keep rays off the exact sector bounds (rounding decides there)
            azimuth = (np.degrees(start + np.arange(n) * inc) + 180) % 360 - 180
            bounds = lidar_sectors.PROFILES["v5"]["sectors_deg"]
            for _ in range(100):
                ranges = [rng.uniform(0.15, 3.0) for _ in range(n)]
                values, _ = sim_matched_sectors(ranges, start, inc, 0.12, 3.5, {}, profile="v5")
                for name, value in zip(sector_names("v5"), values):
                    lo, hi = bounds[name]
                    mask = (azimuth >= lo) & (azimuth <= hi)
                    expect = min(float(np.min(np.array(ranges)[mask])), 1.0)
                    self.assertAlmostEqual(value, expect, msg=name)
        finally:
            lidar_sectors.REAL_MOUNT_M = saved

    def test_v5_with_far_teammate_mostly_drives_like_v4(self):
        # Handoff step 3: with the teammate far and idle, V5 should act like a single robot.
        from policy_numpy import NumpyPolicy
        v5, v4 = NumpyPolicy(V5_MODEL), NumpyPolicy(V4_MODEL)
        rng = np.random.default_rng(11)
        agree = 0
        for _ in range(2000):
            bearing = rng.uniform(-math.pi, math.pi)
            front, left, right = rng.uniform(0.1, 1.0, 3)
            six = [rng.uniform(0.05, 1.0), math.sin(bearing), math.cos(bearing), front, left, right]
            extra = [front, min(front, left), min(front, right), left, right]   # consistent finer sectors
            agree += v5.act(six + extra + [1.0, *FAR_TEAMMATE]) == v4.act(six)
        print(f"\n  V5 (far teammate) chose V4's action in {agree / 20:.1f}% of 2000 random states")
        self.assertGreater(agree, 1000)


class GapAndPlacementTests(unittest.TestCase):
    def test_free_gap_is_found_and_closest_to_the_goal_wins(self):
        n = 232
        inc = 2 * math.pi / n
        ranges = []
        for i in range(n):
            a = (math.degrees(i * inc) + 180) % 360 - 180
            ranges.append(3.0 if 20 <= a <= 35 or -60 <= a <= -45 else 0.5)
        self.assertAlmostEqual(free_gap_heading(ranges, 0.0, inc, 0.0), 27.0, delta=2.0)
        self.assertAlmostEqual(free_gap_heading(ranges, 0.0, inc, -50.0), -52.0, delta=2.0)
        self.assertIsNone(free_gap_heading([0.5] * n, 0.0, inc, 0.0))

    def test_a_clear_goal_direction_is_taken_straight(self):
        # Burger 2's case: one wide free arc (-10..+95 deg), obstacles left and right further out,
        # goal 3 deg off: steer at the goal, not at the arc's centre (~+42 deg).
        n = 232
        inc = 2 * math.pi / n
        ranges = []
        for i in range(n):
            a = (math.degrees(i * inc) + 180) % 360 - 180
            ranges.append(2.0 if -10 <= a <= 95 else 0.5)
        self.assertAlmostEqual(free_gap_heading(ranges, 0.0, inc, 3.0), 3.0, delta=1.0)
        self.assertAlmostEqual(free_gap_heading(ranges, 0.0, inc, -30.0), 0.0, delta=2.5)

    def test_gap_assist_hands_over_when_stuck_and_back_after_1_2_m(self):
        gap = GapAssist()
        for _ in range(GAP_STUCK_STEPS - 1):
            self.assertEqual(gap.decide(STAY, (0.0, 0.0), 1.0, 0.0, lambda: 30.0), (STAY, "policy"))
        action, why = gap.decide(STAY, (0.0, 0.0), 1.0, 0.0, lambda: 30.0)
        self.assertEqual((action, why), (TURN_LEFT, "gap turn +30"))
        self.assertEqual(gap.decide(STAY, (0.5, 0.0), 1.0, 0.0, lambda: 2.0), (FORWARD, "gap forward +2"))
        self.assertEqual(gap.decide(STAY, (0.0, 0.0), 1.0, 90.0, lambda: 2.0), (TURN_LEFT, "gap face goal"))
        self.assertEqual(gap.decide(STAY, (1.3, 0.0), 1.0, 0.0, lambda: 2.0), (STAY, "gap handback"))
        self.assertEqual(gap.handovers, 1)

    def test_not_stuck_near_the_goal(self):
        gap = GapAssist()
        for _ in range(GAP_STUCK_STEPS + 5):
            self.assertEqual(gap.decide(STAY, (0.0, 0.0), 0.2, 0.0, lambda: 30.0)[1], "policy")

    def test_head_on_placement_from_the_two_lidars(self):
        m = lidar_sectors.REAL_MOUNT_M[0]
        for x2, y2, yaw2_deg in ((1.8, 0.0, 180.0), (1.7, 0.15, 170.0), (2.0, -0.1, -165.0)):
            yaw2 = math.radians(yaw2_deg)
            l1 = (m, 0.0)
            l2 = (x2 + m * math.cos(yaw2), y2 + m * math.sin(yaw2))
            reach = math.dist(l1, l2)
            seen_by_1 = (reach - BODY_RADIUS_AT_SCAN_M, math.degrees(math.atan2(l2[1] - l1[1], l2[0] - l1[0])))
            back = math.atan2(l1[1] - l2[1], l1[0] - l2[0]) - yaw2
            seen_by_2 = (reach - BODY_RADIUS_AT_SCAN_M, math.degrees(math.atan2(math.sin(back), math.cos(back))))
            start1, start2 = head_on_starts(seen_by_1, seen_by_2)
            self.assertEqual(start1, (0.0, 0.0, 0.0))
            self.assertAlmostEqual(start2[0], x2)
            self.assertAlmostEqual(start2[1], y2)
            self.assertAlmostEqual(math.cos(start2[2]), math.cos(yaw2))
            self.assertAlmostEqual(math.sin(start2[2]), math.sin(yaw2))


class FrameAndLinkTests(unittest.TestCase):
    def test_shared_frame_round_trip_and_placement(self):
        # Robot switched on somewhere (odometry (0.4, -0.1, 30 deg) at mission start),
        # standing on the tape mark (1.0, 2.0) facing -90 deg in the mission frame.
        frame = SharedFrame((1.0, 2.0, -math.pi / 2), (0.4, -0.1, math.radians(30)))
        x, y, yaw = frame.to_shared((0.4, -0.1, math.radians(30)))
        self.assertAlmostEqual(x, 1.0)
        self.assertAlmostEqual(y, 2.0)
        self.assertAlmostEqual(yaw, -math.pi / 2)
        # Driving 0.5 m forward in odometry = 0.5 m along -y in the mission frame.
        ox, oy = 0.4 + 0.5 * math.cos(math.radians(30)), -0.1 + 0.5 * math.sin(math.radians(30))
        x, y, _ = frame.to_shared((ox, oy, math.radians(30)))
        self.assertAlmostEqual(x, 1.0)
        self.assertAlmostEqual(y, 1.5)
        back = frame.to_odom(x, y)
        self.assertAlmostEqual(back[0], ox)
        self.assertAlmostEqual(back[1], oy)

    def test_lidar_sits_ahead_of_the_axle(self):
        x, y, _ = lidar_pose((1.0, 1.0, math.pi / 2))
        self.assertAlmostEqual(x, 1.0)
        self.assertAlmostEqual(y, 1.0 + lidar_sectors.REAL_MOUNT_M[0])

    def test_pose_link_over_localhost(self):
        a = PoseLink("TURTLEBOT3_1", "127.0.0.1", lambda: (1.0, 2.0, 0.5, True), 47210, 47211, 50.0)
        b = PoseLink("TURTLEBOT3_2", "127.0.0.1", lambda: (3.0, 4.0, -0.5, False), 47211, 47210, 50.0)
        try:
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline and (a.teammate() is None or b.teammate() is None):
                time.sleep(0.02)
            ta, tb = a.teammate(), b.teammate()
            self.assertEqual((ta["robot"], ta["x"], ta["active"]), ("TURTLEBOT3_2", 3.0, False))
            self.assertEqual((tb["robot"], tb["y"], tb["active"]), ("TURTLEBOT3_1", 2.0, True))
            self.assertIsNone(a.teammate(max_age_s=-1.0))   # stale poses are refused
        finally:
            a.close()
            b.close()


if __name__ == "__main__":
    unittest.main()
