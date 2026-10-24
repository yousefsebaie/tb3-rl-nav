#!/usr/bin/env python3
"""Odometry-only, discrete point-navigation environment for the Isaac TurtleBot.

This module deliberately contains no PPO implementation, obstacle logic, LiDAR,
reset service, mission polling, or LLM integration.  It is a small transport and
reward layer that a future trainer can call through ``reset`` and ``step``.
"""

from __future__ import annotations

import argparse
import math
import threading
import time
from dataclasses import dataclass
from typing import Optional, Sequence

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.context import Context
from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile

try:  # Supports both python -m ... and direct execution of this file.
    from .config import PointNavAction, PointNavConfig
except ImportError:  # pragma: no cover - direct-script convenience only
    from config import PointNavAction, PointNavConfig


def wrap_to_pi(angle: float) -> float:
    """Return an angle in [-pi, pi]."""
    return math.atan2(math.sin(angle), math.cos(angle))


@dataclass(frozen=True)
class OdomState:
    """A measured planar robot state from one Odometry message."""

    x: float
    y: float
    yaw: float
    stamp_s: float
    sequence: int


class StaleOdometryError(RuntimeError):
    """Raised when Isaac does not supply the odometry needed to bound an action."""


class RosPointNavEnv(Node):
    """A small Gymnasium-like environment driven by measured ROS 2 odometry.

    ``reset`` starts bookkeeping for an episode but intentionally does not move
    the robot.  For V1, reset the TurtleBot manually in Isaac Sim before calling
    it.  Each ``step`` sends exactly one bounded low-level action and always sends
    a zero Twist before returning or raising an error.
    """

    def __init__(
        self,
        goal_x: float,
        goal_y: float,
        config: Optional[PointNavConfig] = None,
        context: Optional[Context] = None,
    ) -> None:
        self._context = context or rclpy.get_default_context()
        self._owns_context = not rclpy.ok(context=self._context)
        if self._owns_context:
            rclpy.init(context=self._context)

        super().__init__("tb03_point_nav_env", context=self._context)
        self.config = config or PointNavConfig()
        self._goal_x = float(goal_x)
        self._goal_y = float(goal_y)

        self._state_lock = threading.Condition()
        self._state: Optional[OdomState] = None
        self._episode_start_stamp_s: Optional[float] = None
        self._episode_steps = 0
        self._closed = False
        self._close_lock = threading.Lock()
        self._spin_stop = threading.Event()

        self._cmd_pub = self.create_publisher(Twist, self.config.cmd_vel_topic, 10)
        self._odom_sub = self.create_subscription(
            Odometry,
            self.config.odom_topic,
            self._odom_callback,
            QoSProfile(depth=10),
        )
        self._executor = SingleThreadedExecutor(context=self._context)
        self._executor.add_node(self)
        self._spin_thread = threading.Thread(
            target=self._spin,
            daemon=True,
            name="tb03-point-nav-ros-spin",
        )
        self._spin_thread.start()

    def _spin(self) -> None:
        """Run callbacks until close() asks this environment's executor to stop."""
        try:
            while not self._spin_stop.is_set() and rclpy.ok(context=self._context):
                self._executor.spin_once(timeout_sec=0.1)
        except ExternalShutdownException:
            # Expected if another owner shuts down a shared context.
            pass

    @property
    def goal(self) -> tuple[float, float]:
        return self._goal_x, self._goal_y

    def set_goal(self, goal_x: float, goal_y: float) -> None:
        """Set the target used for the next observation and episode."""
        self._goal_x = float(goal_x)
        self._goal_y = float(goal_y)

    def _odom_callback(self, msg: Odometry) -> None:
        q = msg.pose.pose.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        stamp_s = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9

        with self._state_lock:
            sequence = 1 if self._state is None else self._state.sequence + 1
            self._state = OdomState(
                x=float(msg.pose.pose.position.x),
                y=float(msg.pose.pose.position.y),
                yaw=math.atan2(siny, cosy),
                stamp_s=stamp_s,
                sequence=sequence,
            )
            self._state_lock.notify_all()

    def _wait_for_state_after(
        self,
        sequence: int,
        timeout_s: Optional[float] = None,
    ) -> OdomState:
        """Wait for a new measured odometry state, using a wall-clock watchdog."""
        timeout_s = self.config.stale_odom_timeout_s if timeout_s is None else timeout_s
        deadline = time.monotonic() + timeout_s
        with self._state_lock:
            while self._state is None or self._state.sequence <= sequence:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    raise StaleOdometryError(
                        f"No fresh odometry on {self.config.odom_topic} within {timeout_s:.1f} s."
                    )
                self._state_lock.wait(timeout=remaining)
            return self._state

    def _current_state(self) -> OdomState:
        with self._state_lock:
            sequence = 0 if self._state is None else self._state.sequence
        return self._wait_for_state_after(sequence, self.config.stale_odom_timeout_s)

    def _publish_twist(self, linear_x: float = 0.0, angular_z: float = 0.0) -> None:
        msg = Twist()
        msg.linear.x = linear_x
        msg.angular.z = angular_z
        self._cmd_pub.publish(msg)

    def stop(self) -> None:
        """Explicitly stop the robot. Called automatically at the end of every step."""
        self._publish_twist()

    def _action_command(self, action: PointNavAction) -> tuple[float, float, float]:
        if action == PointNavAction.FORWARD:
            return self.config.forward_speed_mps, 0.0, self.config.forward_duration_s
        if action == PointNavAction.TURN_LEFT:
            return 0.0, self.config.turn_speed_radps, self.config.turn_duration_s
        if action == PointNavAction.TURN_RIGHT:
            return 0.0, -self.config.turn_speed_radps, self.config.turn_duration_s
        if action == PointNavAction.STAY:
            return 0.0, 0.0, self.config.stay_duration_s
        raise ValueError(f"Unknown point-navigation action: {action!r}")

    def _distance_and_bearing(self, state: OdomState) -> tuple[float, float]:
        dx = self._goal_x - state.x
        dy = self._goal_y - state.y
        distance = math.hypot(dx, dy)
        bearing = wrap_to_pi(math.atan2(dy, dx) - state.yaw)
        return distance, bearing

    def _observation(self, state: OdomState) -> list[float]:
        distance, bearing = self._distance_and_bearing(state)
        normalized_distance = min(distance / self.config.distance_normalization_m, 1.0)
        return [normalized_distance, math.sin(bearing), math.cos(bearing)]

    def _info(
        self,
        state: OdomState,
        action: Optional[PointNavAction] = None,
        progress_m: Optional[float] = None,
    ) -> dict:
        distance, bearing = self._distance_and_bearing(state)
        return {
            "pose": {"x": state.x, "y": state.y, "yaw": state.yaw},
            "goal": {"x": self._goal_x, "y": self._goal_y},
            "distance_to_goal_m": distance,
            "relative_goal_bearing_rad": bearing,
            "action": None if action is None else action.name,
            "progress_m": progress_m,
            "odom_timestamp_s": state.stamp_s,
            "odom_sequence": state.sequence,
            "episode_steps": self._episode_steps,
        }

    def reset(
        self,
        goal: Optional[Sequence[float]] = None,
    ) -> tuple[list[float], dict]:
        """Begin an episode without changing the physical robot pose.

        The caller must manually reset/reposition the TurtleBot before invoking
        this method.  ``goal`` may be a two-value ``(x, y)`` sequence.
        """
        if goal is not None:
            if len(goal) != 2:
                raise ValueError("goal must contain exactly (x, y)")
            self.set_goal(goal[0], goal[1])
        self.stop()
        state = self._current_state()
        self._episode_start_stamp_s = state.stamp_s
        self._episode_steps = 0
        return self._observation(state), self._info(state)

    def step(self, action: PointNavAction | int) -> tuple[list[float], float, bool, bool, dict]:
        """Execute one bounded action and return measured transition data.

        Action duration is determined from odometry simulation timestamps, not
        wall time.  A two-second wall-clock watchdog prevents a paused simulator
        or stopped odometry stream from leaving a velocity command active.
        """
        # Start from a known zero command before validating the requested
        # action or waiting for odometry, so every exception path is stopped.
        self.stop()
        try:
            try:
                action = PointNavAction(action)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Action must be one of {[a.name for a in PointNavAction]}") from exc
            if self._episode_start_stamp_s is None:
                raise RuntimeError("Call reset() before step().")

            before = self._current_state()
            distance_before, _ = self._distance_and_bearing(before)
            linear_x, angular_z, duration_s = self._action_command(action)
            target_stamp_s = before.stamp_s + duration_s
            latest_sequence = before.sequence
            deadline = time.monotonic() + self.config.stale_odom_timeout_s

            while True:
                self._publish_twist(linear_x, angular_z)
                with self._state_lock:
                    latest = self._state
                    if latest is not None:
                        latest_sequence = latest.sequence
                        if latest.stamp_s >= target_stamp_s:
                            break
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    raise StaleOdometryError(
                        "Odometry time did not advance through the action interval."
                    )
                time.sleep(min(self.config.command_publish_period_s, remaining))

            after = self._wait_for_state_after(latest_sequence)
            self._episode_steps += 1
            distance_after, _ = self._distance_and_bearing(after)
            progress_m = distance_before - distance_after
            reward = (
                self.config.progress_reward_scale
                * max(-self.config.progress_clip_m, min(progress_m, self.config.progress_clip_m))
                - self.config.step_penalty
            )
            terminated = distance_after <= self.config.goal_tolerance_m
            if terminated:
                reward += self.config.success_bonus

            elapsed_sim_s = after.stamp_s - self._episode_start_stamp_s
            truncated = (
                not terminated
                and (
                    self._episode_steps >= self.config.max_episode_steps
                    or elapsed_sim_s >= self.config.max_episode_sim_time_s
                )
            )
            info = self._info(after, action=action, progress_m=progress_m)
            info["episode_elapsed_sim_s"] = elapsed_sim_s
            return self._observation(after), reward, terminated, truncated, info
        finally:
            # Always stop after an action, failed odometry wait, or any other
            # exception raised while executing a physical-robot transition.
            self.stop()

    def close(self) -> None:
        """Stop the robot and deterministically release owned ROS resources."""
        with self._close_lock:
            if self._closed:
                return
            self._closed = True

        try:
            if rclpy.ok(context=self._context):
                self.stop()
        finally:
            # Stop callbacks before destroying objects they can access.
            self._spin_stop.set()
            self._executor.shutdown(timeout_sec=1.0)
            if rclpy.ok(context=self._context):
                self.destroy_node()
            if self._owns_context and rclpy.ok(context=self._context):
                rclpy.shutdown(context=self._context)
            if self._spin_thread.is_alive():
                self._spin_thread.join(timeout=2.0)


def _parse_action(value: str) -> PointNavAction:
    try:
        return PointNavAction[value.upper()]
    except KeyError as exc:
        choices = ", ".join(action.name for action in PointNavAction)
        raise argparse.ArgumentTypeError(f"action must be one of: {choices}") from exc


def main() -> None:
    """Manual one-action smoke test; no command is sent unless --action is supplied."""
    parser = argparse.ArgumentParser(description="Manual TB03 odometry point-navigation smoke test")
    parser.add_argument("--goal-x", type=float, required=True)
    parser.add_argument("--goal-y", type=float, required=True)
    parser.add_argument(
        "--action",
        type=_parse_action,
        help="Send exactly one action: FORWARD, TURN_LEFT, TURN_RIGHT, or STAY.",
    )
    args = parser.parse_args()

    env = RosPointNavEnv(args.goal_x, args.goal_y)
    try:
        observation, before = env.reset()
        print("odom before:", before)
        print("observation before:", observation)
        if args.action is None:
            print("No --action supplied; no movement command was published.")
            return
        observation, reward, terminated, truncated, after = env.step(args.action)
        print("odom after:", after)
        print("observation after:", observation)
        print(
            f"action={args.action.name} progress={after['progress_m']:.4f} m "
            f"reward={reward:.4f} terminated={terminated} truncated={truncated}"
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
