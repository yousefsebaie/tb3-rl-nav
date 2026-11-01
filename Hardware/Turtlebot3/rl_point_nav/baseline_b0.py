"""B0: the original CyberScape execution controller (no RL), ported to Isaac Sim.

Faithful port of BurgerDriver.move_to_point / rotate / _avoid_obstacle from
Hardware/Turtlebot3/burger_comm_node.py (the controller the LLM plans ran on before
RL): rotate in place until the heading error is under 0.1 rad, then drive with
v = min(0.15, 0.5 d), w = 1.5 * heading error, 10 Hz control. If the front LiDAR
sector (+-18 deg) is closer than 0.35 m: stop, turn 90 deg to a random side (50/50, as
on the robot), drive 0.25 m if the front is clear to 0.50 m, else try the other side;
3 attempts per block and at most 12 avoidance manoeuvres per goal.

Differences from the robot node, all to run it in the same simulator and clocks as
B1/PPO: commands go through IsaacPointNavLidarEnv.drive_command (same wheels, physics
and simulated LiDAR); front clearance = min valid range within +-18 deg, like the
node's _sector_min_distance (env.raw_sector_clearance); the random side choice uses a
seeded RNG; the time budget is the same 120 s simulated as the RL episodes; collision =
the env's contact check.
"""

from __future__ import annotations

import math
import random

LINEAR_SPEED = 0.15
ANGULAR_SPEED = 0.5
GOAL_TOLERANCE = 0.05
OBSTACLE_STOP_DISTANCE = 0.35
OBSTACLE_CLEAR_DISTANCE = 0.50
AVOIDANCE_TURN_DEG = 90.0
AVOIDANCE_STEP_DISTANCE = 0.25
MAX_AVOIDANCE_ATTEMPTS = 12
CONTROL_PERIOD_S = 0.1
HEADING_DEADBAND_RAD = 0.1


def _wrap(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


class _Abort(Exception):
    def __init__(self, status: str, reason: str):
        super().__init__(reason)
        self.status, self.reason = status, reason


class B0Driver:
    """Runs one move_to_point on the env; on_tick(info) is called after every control period."""

    name = "b0"

    def __init__(self, env, seed: int, budget_s: float = 120.0, on_tick=None, disturb=None,
                 noisy_sensor: bool = False):
        self.noisy_sensor = noisy_sensor
        self.env, self.rng, self.budget_s = env, random.Random(seed), budget_s
        self.on_tick = on_tick or (lambda info: None)
        self.disturb = disturb          # optional callable(v, w) -> (v, w), robustness tests
        self.frames = max(1, round(CONTROL_PERIOD_S / env._physics_dt_s))
        self.decisions = 0
        self.trace = []                 # avoidance decisions (front clearance seen), for diagnosis
        self._last = (0.0, 0.0)

    # ------------------------------------------------------------ helpers
    def _front(self) -> float:
        # The robot node's own perception: min valid range within +-18 deg (no RL sector model).
        return self.env.raw_sector_clearance("front", perturb=self.noisy_sensor)

    def _command(self, v: float, w: float) -> dict:
        if self.disturb is not None:
            v, w = self.disturb(v, w, self._last)
        self._last = (v, w)
        info = self.env.drive_command(v, w, self.frames)
        self.decisions += 1
        self.on_tick(info)
        if info["collision"]:
            raise _Abort("collision", "obstacle contact")
        if self.env._episode_elapsed_sim_s >= self.budget_s:
            raise _Abort("timeout", "time budget")
        return info

    def _rotate(self, degrees: float) -> None:
        target = math.radians(degrees)
        accumulated, last = 0.0, self.env._state().yaw
        w = ANGULAR_SPEED * (1 if target >= 0 else -1)
        while abs(accumulated) < abs(target):
            self._command(0.0, w)
            yaw = self.env._state().yaw
            accumulated += _wrap(yaw - last)
            last = yaw
        self.env.stop_robot()

    def _drive_forward_for(self, distance: float) -> bool:
        state = self.env._state()
        start = (state.x, state.y)
        while True:
            state = self.env._state()
            travelled = math.hypot(state.x - start[0], state.y - start[1])
            if travelled >= distance:
                self.env.stop_robot()
                return True
            if self._front() < OBSTACLE_STOP_DISTANCE:
                self.env.stop_robot()
                return False
            self._command(min(LINEAR_SPEED * 0.75, distance - travelled), 0.0)

    def _avoid(self) -> bool:
        self.env.stop_robot()
        self.trace.append(("avoid", round(self._front(), 3)))
        for _ in range(3):
            turn = (1 if self.rng.random() < 0.5 else -1) * AVOIDANCE_TURN_DEG
            self._rotate(turn)
            self.trace.append(("after_turn", turn, round(self._front(), 3)))
            if self._front() >= OBSTACLE_CLEAR_DISTANCE and self._drive_forward_for(AVOIDANCE_STEP_DISTANCE):
                return True
            self._rotate(-turn)
            self._rotate(-turn)
            if self._front() >= OBSTACLE_CLEAR_DISTANCE and self._drive_forward_for(AVOIDANCE_STEP_DISTANCE):
                return True
        self.env.stop_robot()
        return False

    # ---------------------------------------------------------- move_to_point
    def move_to_point(self, goal) -> tuple[str, str]:
        """Returns (status, reason): success / collision / timeout / failed."""
        attempts = 0
        try:
            while True:
                state = self.env._state()
                dx, dy = goal[0] - state.x, goal[1] - state.y
                dist = math.hypot(dx, dy)
                if dist < GOAL_TOLERANCE:
                    self.env.stop_robot()
                    return "success", ""
                error = _wrap(math.atan2(dy, dx) - state.yaw)
                if abs(error) > HEADING_DEADBAND_RAD:
                    self._command(0.0, ANGULAR_SPEED * (1 if error > 0 else -1))
                    continue
                if self._front() < OBSTACLE_STOP_DISTANCE:
                    attempts += 1
                    if attempts > MAX_AVOIDANCE_ATTEMPTS:
                        self.env.stop_robot()
                        return "failed", "too many obstacle avoidance attempts"
                    if not self._avoid():
                        return "failed", "no clear local bypass"
                    continue
                self._command(min(LINEAR_SPEED, dist * 0.5), 1.5 * error)
        except _Abort as stop:
            self.env.stop_robot()
            return stop.status, stop.reason
