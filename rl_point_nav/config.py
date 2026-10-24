"""Configuration for the odometry-only TurtleBot point-navigation prototype."""

from dataclasses import dataclass
from enum import IntEnum


class PointNavAction(IntEnum):
    """Discrete robot-relative actions used by the point-navigation environment."""

    FORWARD = 0
    TURN_LEFT = 1
    TURN_RIGHT = 2
    STAY = 3


@dataclass(frozen=True)
class PointNavConfig:
    """Fixed V1 parameters; no obstacle sensing or reset support is included."""

    # Real Burger deployment path: the hardware safety filter receives raw
    # policy commands here before forwarding permitted motion to /cmd_vel.
    cmd_vel_topic: str = "/cmd_vel_raw"
    odom_topic: str = "/odom"

    forward_speed_mps: float = 0.05
    turn_speed_radps: float = 0.80
    forward_duration_s: float = 0.50
    turn_duration_s: float = 0.25
    stay_duration_s: float = 0.25
    command_publish_period_s: float = 0.05

    distance_normalization_m: float = 2.0
    goal_tolerance_m: float = 0.05
    max_episode_steps: int = 150
    max_episode_sim_time_s: float = 60.0
    stale_odom_timeout_s: float = 2.0

    progress_clip_m: float = 0.10
    progress_reward_scale: float = 5.0
    step_penalty: float = 0.01
    success_bonus: float = 10.0
