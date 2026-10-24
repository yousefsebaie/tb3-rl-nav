"""Direct-Isaac, odometry-free TurtleBot point-navigation environment.

Isaac Sim is started once by :class:`IsaacPointNavEnv` and stays alive across
episodes.  This V1 environment deliberately has no ROS 2, Action Graph,
LiDAR, obstacle logic, PPO, or random start-pose sampling.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Optional, Sequence

import gymnasium as gym
from gymnasium import spaces
import numpy as np

try:  # Supports package and direct-script imports.
    from .config import PointNavAction, PointNavConfig
except ImportError:  # pragma: no cover - direct-script convenience only
    from config import PointNavAction, PointNavConfig


SCENE_PATH = Path("/home/g8/Desktop/FInalSceneDemo.usd")
ARTICULATION_PATH = "/World/turtlebot3_burger_02"
LEFT_WHEEL_DOF = "a__namespace_wheel_left_joint"
RIGHT_WHEEL_DOF = "a__namespace_wheel_right_joint"


# Approved V1B fixed-start goal banks. The eastbound straight bank is backed
# by the historical trace; the left/right banks are deliberate visual probes
# and are not presented as collision-clearance assertions.
LEFT_GOALS = (
    (-2.818, 1.310),
    (-2.718, 1.483),
)
RIGHT_GOALS = (
    (-2.818, -0.422),
    (-2.718, -0.595),
)
STRAIGHT_GOALS = (
    (-2.718, 0.444),
    (-2.118, 0.444),
    (-1.718, 0.444),
)


@dataclass(frozen=True)
class IsaacPointNavConfig:
    """Direct-Isaac action parameters validated by the smoke tests."""

    wheel_radius_m: float = 0.033
    wheel_base_m: float = 0.160
    max_linear_speed_mps: float = 0.22
    max_angular_speed_radps: float = 1.0

    forward_speed_mps: float = 0.05
    turn_speed_radps: float = 0.80
    forward_frames: int = 30
    turn_frames: int = 15
    stay_frames: int = 15
    settle_frames: int = 8

    reset_position_tolerance_m: float = 0.01
    reset_yaw_tolerance_rad: float = 0.02
    # V1 direct-Isaac point-navigation success radius: 5 cm.
    goal_tolerance_m: float = 0.05

    # All approved V1B goals are 0.5--2.0 m from the fixed reset pose.
    min_goal_distance_m: float = 0.50
    max_goal_distance_m: float = 2.00


@dataclass(frozen=True)
class IsaacPoseState:
    """Measured articulation-root pose used for V1 point navigation."""

    x: float
    y: float
    yaw: float


def wrap_to_pi(angle: float) -> float:
    """Return an angle in [-pi, pi]."""
    return math.atan2(math.sin(angle), math.cos(angle))


def _yaw_from_wxyz(quaternion: Sequence[float]) -> float:
    """Extract planar yaw from Isaac's scalar-first world quaternion."""
    w, x, y, z = (float(component) for component in quaternion)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _host_array(values) -> np.ndarray:
    """Copy an Isaac/Warp getter result to an indexable host NumPy array."""
    return values.numpy() if hasattr(values, "numpy") else np.asarray(values)


class IsaacPointNavEnv(gym.Env):
    """A persistent direct-Isaac Gymnasium environment for point navigation.

    The state and goal are expressed in the world XY frame of the TurtleBot
    articulation root.  ``reset`` uses only the cached initial root pose in V1;
    random start-pose sampling is intentionally outside this environment.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        goal_x: float = 1.0,
        goal_y: float = 0.0,
        *,
        point_nav_config: Optional[PointNavConfig] = None,
        isaac_config: Optional[IsaacPointNavConfig] = None,
        headless: bool = True,
    ) -> None:
        super().__init__()

        # SimulationApp must be constructed before all Kit/Isaac runtime imports.
        from isaacsim import SimulationApp

        self._closed = False
        self._headless = bool(headless)
        self._simulation_app = None
        self._simulation_context = None
        self._reset_robot = None
        self._drive_robot = None
        self._stop_action = None
        self._physics_dt_s = 0.0
        self._default_position = None
        self._default_orientation = None
        self._default_position_host = None
        self._default_orientation_host = None
        self._zero_root_velocity = None
        self._zero_dof_velocity = None
        self._goal_x = float(goal_x)
        self._goal_y = float(goal_y)
        self._episode_steps = 0
        self._episode_elapsed_sim_s = 0.0
        self._episode_active = False

        self.point_nav_config = point_nav_config or PointNavConfig()
        self.isaac_config = isaac_config or IsaacPointNavConfig()
        self.action_space = spaces.Discrete(len(PointNavAction))
        self.observation_space = spaces.Box(
            low=np.array([0.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        try:
            mode = "headless" if self._headless else "GUI"
            print(f"[TB03_ENV] Starting {mode} Isaac Sim", flush=True)
            self._simulation_app = SimulationApp(
                {
                    "headless": self._headless,
                    "open_usd": str(SCENE_PATH),
                    "create_new_stage": False,
                    "fast_shutdown": False,
                    "disable_viewport_updates": self._headless,
                }
            )
            self._initialize_isaac()
            print(
                f"[TB03_ENV] Initialized; physics_dt={self._physics_dt_s:.6f}s, "
                f"goal=({self._goal_x:.3f}, {self._goal_y:.3f})",
                flush=True,
            )
        except Exception:
            self.close()
            raise

    def _initialize_isaac(self) -> None:
        """Load, validate, and initialize the one persistent Isaac simulation."""
        import omni.usd

        from isaacsim.core.api import SimulationContext
        from isaacsim.core.experimental.prims import Articulation
        from isaacsim.robot.wheeled_robots.controllers.differential_controller import (
            DifferentialController,
        )
        from isaacsim.robot.wheeled_robots.robots.wheeled_robot import WheeledRobot

        if not SCENE_PATH.is_file():
            raise FileNotFoundError(f"Scene does not exist: {SCENE_PATH}")

        usd_context = omni.usd.get_context()
        while usd_context.get_stage_loading_status()[2] > 0:
            self._simulation_app.update()
        stage = usd_context.get_stage()
        if stage is None:
            raise RuntimeError(f"No active USD stage after opening {SCENE_PATH}.")
        if not stage.GetPrimAtPath(ARTICULATION_PATH).IsValid():
            world_prim = stage.GetPrimAtPath("/World")
            children = [child.GetPath().pathString for child in world_prim.GetChildren()] if world_prim.IsValid() else []
            raise RuntimeError(
                f"Expected robot prim is absent: {ARTICULATION_PATH}. "
                f"Immediate /World children: {children}"
            )
        print(f"[TB03_ENV] Verified robot prim: {ARTICULATION_PATH}", flush=True)

        self._simulation_context = SimulationContext()
        self._simulation_context.initialize_physics()
        self._simulation_context.play()
        self._physics_dt_s = float(self._simulation_context.get_physics_dt())
        if self._physics_dt_s <= 0.0:
            raise RuntimeError(f"Invalid Isaac physics timestep: {self._physics_dt_s}")
        self._step_frames(self.isaac_config.settle_frames, count_episode_time=False)

        self._reset_robot = Articulation(ARTICULATION_PATH)
        if not self._reset_robot.is_physics_tensor_entity_valid():
            raise RuntimeError(f"Physics tensor state is unavailable for {ARTICULATION_PATH}.")

        self._drive_robot = WheeledRobot(
            prim_path=ARTICULATION_PATH,
            name="tb03_point_nav_direct_drive",
            wheel_dof_names=[LEFT_WHEEL_DOF, RIGHT_WHEEL_DOF],
        )
        self._drive_robot.initialize()
        dof_names = list(self._drive_robot.dof_names)
        missing_dofs = [name for name in (LEFT_WHEEL_DOF, RIGHT_WHEEL_DOF) if name not in dof_names]
        if missing_dofs:
            raise RuntimeError(f"Expected wheel DOFs are absent: {missing_dofs}; available: {dof_names}")
        _, wheel_indices = self._drive_robot.get_articulation_controller_properties()
        if wheel_indices is None or len(wheel_indices) != 2 or any(index is None or index < 0 for index in wheel_indices):
            raise RuntimeError(f"Could not resolve valid left/right wheel indices: {wheel_indices}")
        print(
            f"[TB03_ENV] Wheel DOFs resolved: left={wheel_indices[0]}, right={wheel_indices[1]}",
            flush=True,
        )

        controller = DifferentialController(
            name="tb03_point_nav_differential_controller",
            wheel_radius=self.isaac_config.wheel_radius_m,
            wheel_base=self.isaac_config.wheel_base_m,
            max_linear_speed=self.isaac_config.max_linear_speed_mps,
            max_angular_speed=self.isaac_config.max_angular_speed_radps,
        )
        self._controller = controller
        self._stop_action = controller.forward(np.array([0.0, 0.0], dtype=np.float64))

        self._default_position, self._default_orientation = self._reset_robot.get_world_poses()
        # Experimental prim getters return Warp arrays. Keep their original
        # values for the proven setters and host copies for scalar inspection.
        self._default_position_host = _host_array(self._default_position)
        self._default_orientation_host = _host_array(self._default_orientation)
        self._zero_root_velocity = np.zeros(tuple(self._default_position.shape), dtype=np.float32)
        self._zero_dof_velocity = np.zeros(
            tuple(self._reset_robot.get_dof_velocities().shape), dtype=np.float32
        )
        self._apply_stop()
        self._step_frames(self.isaac_config.settle_frames, count_episode_time=False, hold_stop=True)

    def _step_frames(self, frames: int, *, count_episode_time: bool, hold_stop: bool = False) -> None:
        """Advance exactly ``frames`` physics frames and optionally hold zero wheels."""
        for _ in range(frames):
            if hold_stop:
                self._apply_stop()
            self._simulation_context.step(render=not self._headless)
        if count_episode_time:
            self._episode_elapsed_sim_s += frames * self._physics_dt_s

    def _apply_stop(self) -> None:
        """Apply the validated zero wheel-velocity action."""
        if self._drive_robot is not None and self._stop_action is not None:
            self._drive_robot.apply_wheel_actions(self._stop_action)

    def _state(self) -> IsaacPoseState:
        """Read measured root state from the direct Isaac articulation wrapper."""
        position, orientation = self._reset_robot.get_world_poses()
        position = _host_array(position)
        orientation = _host_array(orientation)
        return IsaacPoseState(
            x=float(position[0, 0]),
            y=float(position[0, 1]),
            yaw=_yaw_from_wxyz(orientation[0]),
        )

    @property
    def goal(self) -> tuple[float, float]:
        return self._goal_x, self._goal_y

    def _set_goal(self, goal: Sequence[float]) -> None:
        if len(goal) != 2:
            raise ValueError("goal must contain exactly (x, y)")
        self._goal_x = float(goal[0])
        self._goal_y = float(goal[1])

    def _sample_v1_goal(self) -> tuple[float, float]:
        """Sample LEFT, RIGHT, or STRAIGHT uniformly from the approved V1B banks."""
        goal_banks = (LEFT_GOALS, RIGHT_GOALS, STRAIGHT_GOALS)
        bank = goal_banks[int(self.np_random.integers(len(goal_banks)))]
        goal_x, goal_y = bank[int(self.np_random.integers(len(bank)))]

        start_x = float(self._default_position_host[0, 0])
        start_y = float(self._default_position_host[0, 1])
        distance = math.hypot(goal_x - start_x, goal_y - start_y)
        if not self.isaac_config.min_goal_distance_m <= distance <= self.isaac_config.max_goal_distance_m:
            raise RuntimeError(
                f"Approved V1B goal {(goal_x, goal_y)} violates the configured "
                f"distance range: {distance:.6f} m"
            )
        return float(goal_x), float(goal_y)

    def _distance_and_bearing(self, state: IsaacPoseState) -> tuple[float, float]:
        dx = self._goal_x - state.x
        dy = self._goal_y - state.y
        distance = math.hypot(dx, dy)
        bearing = wrap_to_pi(math.atan2(dy, dx) - state.yaw)
        return distance, bearing

    def _observation(self, state: IsaacPoseState) -> np.ndarray:
        distance, bearing = self._distance_and_bearing(state)
        return np.array(
            [
                np.clip(distance / self.point_nav_config.distance_normalization_m, 0.0, 1.0),
                math.sin(bearing),
                math.cos(bearing),
            ],
            dtype=np.float32,
        )

    def _info(
        self,
        state: IsaacPoseState,
        *,
        action: Optional[PointNavAction] = None,
        progress_m: Optional[float] = None,
    ) -> dict[str, Any]:
        distance, bearing = self._distance_and_bearing(state)
        return {
            "pose": {"x": state.x, "y": state.y},
            "yaw": state.yaw,
            "goal": {"x": self._goal_x, "y": self._goal_y},
            "distance_to_goal_m": distance,
            "relative_goal_bearing_rad": bearing,
            "action": None if action is None else action.name,
            "progress_m": progress_m,
            "episode_steps": self._episode_steps,
            "episode_elapsed_sim_s": self._episode_elapsed_sim_s,
        }

    def _reset_articulation(self) -> IsaacPoseState:
        """Restore the full articulation to the cached initial root pose."""
        self._apply_stop()
        self._reset_robot.set_world_poses(self._default_position, self._default_orientation)
        self._reset_robot.set_velocities(self._zero_root_velocity, self._zero_root_velocity)
        self._reset_robot.set_dof_velocities(self._zero_dof_velocity)
        self._reset_robot.set_dof_velocity_targets(self._zero_dof_velocity)
        self._step_frames(self.isaac_config.settle_frames, count_episode_time=False, hold_stop=True)

        state = self._state()
        expected_yaw = _yaw_from_wxyz(self._default_orientation_host[0])
        position_error = math.hypot(
            state.x - float(self._default_position_host[0, 0]),
            state.y - float(self._default_position_host[0, 1]),
        )
        yaw_error = abs(wrap_to_pi(state.yaw - expected_yaw))
        if position_error > self.isaac_config.reset_position_tolerance_m or yaw_error > self.isaac_config.reset_yaw_tolerance_rad:
            raise RuntimeError(
                "Articulation reset did not settle at the cached root pose: "
                f"position_error={position_error:.6f} m, yaw_error={yaw_error:.6f} rad"
            )
        return state

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset to the cached V1 start pose and return the initial transition."""
        if self._closed:
            raise RuntimeError("Environment is closed.")
        super().reset(seed=seed)
        options = {} if options is None else options
        unknown_options = set(options) - {"goal"}
        if unknown_options:
            raise ValueError(f"Unsupported V1 reset option(s): {sorted(unknown_options)}")
        if "goal" in options:
            self._set_goal(options["goal"])
        else:
            self._goal_x, self._goal_y = self._sample_v1_goal()

        state = self._reset_articulation()
        self._episode_steps = 0
        self._episode_elapsed_sim_s = 0.0
        self._episode_active = True
        print(
            f"[TB03_ENV] Reset: pose=({state.x:.3f}, {state.y:.3f}, {state.yaw:.3f}), "
            f"goal=({self._goal_x:.3f}, {self._goal_y:.3f})",
            flush=True,
        )
        return self._observation(state), self._info(state)

    def _action_spec(self, action: PointNavAction) -> tuple[np.ndarray, int]:
        """Return direct differential-drive command and calibrated frame count."""
        if action == PointNavAction.FORWARD:
            return np.array([self.isaac_config.forward_speed_mps, 0.0], dtype=np.float64), self.isaac_config.forward_frames
        if action == PointNavAction.TURN_LEFT:
            return np.array([0.0, self.isaac_config.turn_speed_radps], dtype=np.float64), self.isaac_config.turn_frames
        if action == PointNavAction.TURN_RIGHT:
            return np.array([0.0, -self.isaac_config.turn_speed_radps], dtype=np.float64), self.isaac_config.turn_frames
        if action == PointNavAction.STAY:
            return np.array([0.0, 0.0], dtype=np.float64), self.isaac_config.stay_frames
        raise ValueError(f"Unknown point-navigation action: {action!r}")

    def step(self, action: PointNavAction | int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Execute one bounded direct wheel action and return a Gym transition."""
        if self._closed:
            raise RuntimeError("Environment is closed.")
        if not self._episode_active:
            raise RuntimeError("Call reset() before step().")
        try:
            point_nav_action = PointNavAction(action)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Action must be one of {[item.name for item in PointNavAction]}") from exc

        before = self._state()
        distance_before, _ = self._distance_and_bearing(before)
        command, drive_frames = self._action_spec(point_nav_action)
        wheel_action = self._controller.forward(command)
        for _ in range(drive_frames):
            self._drive_robot.apply_wheel_actions(wheel_action)
            self._simulation_context.step(render=not self._headless)
        self._episode_elapsed_sim_s += drive_frames * self._physics_dt_s

        self._apply_stop()
        self._step_frames(
            self.isaac_config.settle_frames,
            count_episode_time=True,
            hold_stop=True,
        )

        after = self._state()
        self._episode_steps += 1
        distance_after, _ = self._distance_and_bearing(after)
        progress_m = distance_before - distance_after
        reward = (
            self.point_nav_config.progress_reward_scale
            * float(np.clip(progress_m, -self.point_nav_config.progress_clip_m, self.point_nav_config.progress_clip_m))
            - self.point_nav_config.step_penalty
        )
        terminated = distance_after <= self.isaac_config.goal_tolerance_m
        if terminated:
            reward += self.point_nav_config.success_bonus

        truncated = not terminated and (
            self._episode_steps >= self.point_nav_config.max_episode_steps
            or self._episode_elapsed_sim_s >= self.point_nav_config.max_episode_sim_time_s
        )
        if terminated or truncated:
            self._episode_active = False
        return self._observation(after), float(reward), terminated, truncated, self._info(
            after,
            action=point_nav_action,
            progress_m=progress_m,
        )

    def close(self) -> None:
        """Stop direct wheel motion and close this environment's Isaac process."""
        if self._closed:
            return
        self._closed = True
        try:
            self._apply_stop()
        finally:
            if self._simulation_app is not None:
                print("[TB03_ENV] Closing Isaac Sim", flush=True)
                self._simulation_app.close()
                self._simulation_app = None

