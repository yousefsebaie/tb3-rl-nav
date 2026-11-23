"""MAPPO environment: several TurtleBot3 Burgers in one Isaac scene, each a V4 robot.

Every robot keeps the single-robot V4 setup unchanged (isaac_point_nav_lidar_env.py with
v3_lidar_config() + the --random-obstacles overrides of train_isaac_point_nav_lidar_ppo_v2.py):

* its own top-level RTX LiDAR (Example_Rotary_2D, nearRangeM = minDistBetweenEchosM = 0.10 m,
  farRangeM = 200 m) following its base_link at the real-size mount (-0.032, 0, 0.172) m,
  synchronized before and after every physics step;
* the same 6D observation [d_norm, sin(bearing), cos(bearing), front, left, right], sectors
  front +/-18 deg, left 18..120 deg, right -120..-18 deg, the hardware empty-sector rule,
  range noise 0.015 m, ray dropout 0.10, sector dropout 0.05, minimum valid range 0.15 m;
* the same discrete actions and frame counts (forward 0.05 m/s x 30, turn 0.8 rad/s x 15,
  stay 15, settle 8, physics dt 1/60 s), goal tolerance 0.05 m, 250 steps / 120 s per episode;
* the same reward: 5 x progress (clipped +/-0.10 m) - 0.01 per step, +10 at the goal,
  proximity penalty up to 0.30 inside 0.30 m, -10 and termination on collision
  (footprint radius 0.105 m);
* the same obstacle pool (6 boxes, 10-45 cm, created once, teleported, never deleted).

What is new for several robots:

* Robot 0 is the scene's Burger. Each further robot is a copy of that prim (made once,
  before physics, so it carries the real-size drive fixes), parked 2.0 m along +x facing -x.
* Each episode draws an ENCOUNTER (two robots) that forces the robots to interact:
    head_on   30%  start 1.5-2.0 m apart on one line, each driving to the other's start, any
                   heading; half of them inside a 0.55-0.75 m corridor of boxes (axis-aligned)
    crossing  30%  routes of 1.4-2.0 m crossing at 60-120 deg, both starts the same distance
                   from the crossing, so the robots arrive there together
    doorway   25%  head-on through the only 0.40-0.60 m gap of a wall of 4 boxes (axis-aligned),
                   wide enough for one robot at a time
    random    15%  the fixed starts (scene start, and 2.0 m along +x facing -x) with robot 0's
                   V1B goal banks and the mirrored banks for robot 1
  Starts face along the route +/-0.5 rad. Every route stays within 2.0 m (V4's goal range)
  and inside the area V4 already drove in. The remaining pool boxes are V4 distractors,
  kept 0.5 m from the encounter's centre when it has a corridor or wall.
* Robot bodies are hidden from the RTX LiDARs (as in V4 training, so a robot does not see
  itself). The teammate is added to each scan by an exact ray-circle intersection with its
  footprint at the LiDAR plane, BEFORE the V4 sensor model (near range, noise, dropout,
  minimum range), so it reads like any other obstacle.
* A teammate counts in the proximity penalty and in collisions (centre distance < 2 x 0.105 m).
* cooperative_multi_config() (MAPPO v2) adds team shaping on top of the V4 reward: hitting a
  teammate costs 20 instead of 10, the proximity penalty for a teammate starts at 0.45 m
  instead of 0.30 m, each active robot also receives 0.5 x the sum of its teammates' rewards
  that step (so waiting while the other gets through pays), and doorway gaps are 0.50-0.70 m.
  The defaults (MultiNavConfig()) are the first MAPPO run's task, unchanged.
* Robots act at the same time: each runs its own action's frames, a robot with a shorter
  action holds still for the rest, then all settle. A robot that reaches its goal or
  collides stops and stays in place (an obstacle for the others) until every robot is
  done or the episode is truncated.
* global_state(i) is the centralized critic input: robot i's observation, then for each
  other robot its observation, its position relative to robot i (robot i frame, / 2.0 m),
  sin/cos of the relative yaw, and whether it is still active.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Optional, Sequence

import numpy as np

try:  # Supports package and direct-script imports.
    from .config import PointNavAction, PointNavConfig
    from .isaac_point_nav_env import (
        ARTICULATION_PATH,
        LEFT_GOALS,
        LEFT_WHEEL_DOF,
        RIGHT_GOALS,
        RIGHT_WHEEL_DOF,
        SCENE_PATH,
        STRAIGHT_GOALS,
        IsaacPointNavConfig,
        IsaacPoseState,
        _host_array,
        _yaw_from_wxyz,
        wrap_to_pi,
    )
    from .isaac_point_nav_lidar_env import (
        FLAT_SCAN_ANNOTATOR,
        INVALID_RANGE_EPSILON_M,
        LIDAR_CONFIG,
        LIDAR_MOUNT_OFFSET_M,
        LIDAR_NEAR_RANGE_M,
        LIDAR_RANGE_SETTINGS,
        REAL_SIZE_LIDAR_MOUNT_OFFSET_M,
        UNIT_SCALE_TOLERANCE,
        IsaacLidarNavConfig,
        _v2_point_nav_config,
        _wxyz,
        v3_lidar_config,
    )
except ImportError:  # pragma: no cover - direct-script convenience only
    from config import PointNavAction, PointNavConfig
    from isaac_point_nav_env import (
        ARTICULATION_PATH,
        LEFT_GOALS,
        LEFT_WHEEL_DOF,
        RIGHT_GOALS,
        RIGHT_WHEEL_DOF,
        SCENE_PATH,
        STRAIGHT_GOALS,
        IsaacPointNavConfig,
        IsaacPoseState,
        _host_array,
        _yaw_from_wxyz,
        wrap_to_pi,
    )
    from isaac_point_nav_lidar_env import (
        FLAT_SCAN_ANNOTATOR,
        INVALID_RANGE_EPSILON_M,
        LIDAR_CONFIG,
        LIDAR_MOUNT_OFFSET_M,
        LIDAR_NEAR_RANGE_M,
        LIDAR_RANGE_SETTINGS,
        REAL_SIZE_LIDAR_MOUNT_OFFSET_M,
        UNIT_SCALE_TOLERANCE,
        IsaacLidarNavConfig,
        _v2_point_nav_config,
        _wxyz,
        v3_lidar_config,
    )


OBSTACLE_ROOT = "/World/obstacles"
LIDAR_PATH_PREFIX = "/World/tb03_mappo_lidar"
GOAL_MARKER_PREFIX = "/World/tb03_mappo_goal"
CLONE_PATH_PREFIX = "/World/turtlebot3_burger_mappo"
VISUAL_ROOT_NAMES = (
    "a__namespace_base_link/visuals",
    "a__namespace_base_scan/visuals",
    "a__namespace_wheel_left_link/visuals",
    "a__namespace_wheel_right_link/visuals",
)
GOAL_COLORS = ((0.10, 0.80, 0.20), (0.15, 0.45, 0.95), (0.95, 0.70, 0.10), (0.80, 0.20, 0.80))

# train_isaac_point_nav_lidar_ppo_v2.py --random-obstacles (the V4 training run).
V4_OBSTACLE_OVERRIDES = dict(
    obstacle_count=6,
    max_distractors=5,
    distractor_margin_m=0.80,
    obstacle_size_range_m=(0.10, 0.45),
)


def v4_lidar_config(**overrides) -> IsaacLidarNavConfig:
    """The exact single-robot V4 training configuration (V3 profile + random obstacles)."""
    values = dict(V4_OBSTACLE_OVERRIDES, hide_robot_visuals=True)
    values.update(overrides)
    return v3_lidar_config(**values)


@dataclass(frozen=True)
class MultiNavConfig:
    """Settings that only exist with several robots."""

    robot_count: int = 2
    partner_start_offset_m: float = 2.0      # robot k starts this far along +x from robot 0, facing -x
    min_goal_separation_m: float = 0.50      # between any two robots' goals
    min_goal_start_separation_m: float = 0.40  # a goal vs another robot's start
    goal_attempts: int = 200
    teammate_radius_m: float = 0.105         # teammate footprint seen by the LiDAR and used for collisions
    teammate_obs_scale_m: float = 2.0        # relative position normalization in the critic state
    # Team shaping (defaults = no shaping, the first MAPPO run; see cooperative_multi_config()).
    robot_collision_penalty: float = 10.0    # replaces the V4 collision penalty for a teammate hit
    teammate_proximity_threshold_m: float = 0.30
    team_reward_weight: float = 0.0          # x the sum of the teammates' rewards, added per step
    obstacle_collision_penalty: float = 0.0  # > 0 replaces the V4 collision penalty for an obstacle hit
    proximity_budget: float = 0.0            # > 0 caps a robot's summed proximity penalty per episode
    park_teammate: bool = False              # robot 1 waits at its goal in the far corner: a solo test
    # V5: finer LiDAR sectors appended AFTER V4's six inputs, as (name, lo_deg, hi_deg).
    extra_sectors: tuple = ()
    invalid_as_clear: bool = False           # a sector with no returns reads max range, not 0 m (V4 rule)
    # Encounters (see module docstring); weights need not sum to 1.
    encounter_weights: tuple = (("head_on", 0.30), ("crossing", 0.30), ("doorway", 0.25), ("random", 0.15))
    area_x_m: tuple[float, float] = (-3.9, -0.3)   # every start and goal inside this box
    area_y_m: tuple[float, float] = (-1.2, 2.2)
    centre_x_m: tuple[float, float] = (-3.0, -1.2)
    centre_y_m: tuple[float, float] = (-0.3, 1.3)
    route_length_m: tuple[float, float] = (1.5, 2.0)
    crossing_length_m: tuple[float, float] = (1.4, 2.0)
    crossing_angle_deg: tuple[float, float] = (60.0, 120.0)
    lane_offset_m: float = 0.10                    # head-on lanes are offset by up to this
    corridor_probability: float = 0.5
    corridor_width_m: tuple[float, float] = (0.55, 0.75)
    doorway_gap_m: tuple[float, float] = (0.40, 0.60)
    wall_box_m: tuple[float, float] = (0.35, 0.45)
    start_yaw_noise_rad: float = 0.5
    structure_clearance_m: float = 0.5             # distractors vs the centre of a corridor / wall


def cooperative_multi_config(**overrides) -> MultiNavConfig:
    """MAPPO v2 task: team-shaped reward and doorways a robot can actually pass."""
    values = dict(robot_collision_penalty=20.0, teammate_proximity_threshold_m=0.45,
                  team_reward_weight=0.5, doorway_gap_m=(0.50, 0.70))
    values.update(overrides)
    return MultiNavConfig(**values)


# V4's three sectors cannot see a gap's edges (0/40 solo doorways and corridors); these split them.
V5_SECTORS = (("narrow_front", -6.0, 6.0), ("front_left", 18.0, 45.0), ("front_right", -45.0, -18.0),
              ("left_side", 45.0, 100.0), ("right_side", -100.0, -45.0))


def wide_passage_config(**overrides) -> MultiNavConfig:
    """V5 training task: the v2 cooperative task with passages from the test range up to easy widths."""
    values = dict(doorway_gap_m=(0.50, 0.90), corridor_width_m=(0.55, 0.95))
    values.update(overrides)
    return cooperative_multi_config(**values)


def benchmark_config(**overrides) -> MultiNavConfig:
    """Two-robot benchmark matching the PPO evaluation conditions: crossing, head-on and random-goal
    encounters with scattered obstacles (no doorway or corridor structures)."""
    values = dict(encounter_weights=(("head_on", 0.30), ("crossing", 0.30), ("random", 0.15)),
                  corridor_probability=0.0)
    values.update(overrides)
    return cooperative_multi_config(**values)


def solo_passage_config(**overrides) -> MultiNavConfig:
    """Diagnostic: robot 0 alone through a v2 doorway or corridor; robot 1 is parked in a far corner."""
    values = dict(park_teammate=True, encounter_weights=(("doorway", 0.5), ("head_on", 0.5)),
                  corridor_probability=1.0)
    values.update(overrides)
    return cooperative_multi_config(**values)


def passage_training_config(**overrides) -> MultiNavConfig:
    """Passage curriculum: the solo task with wider gaps, so a lone robot can learn to drive through
    a doorway or corridor at all (0/40 for V4 and MAPPO v2 on solo_passage_config)."""
    values = dict(doorway_gap_m=(0.55, 0.90), corridor_width_m=(0.65, 0.95))
    values.update(overrides)
    return solo_passage_config(**values)


def cooperative_v3_config(**overrides) -> MultiNavConfig:
    """MAPPO v3 task: v2 plus a crash that always costs more than waiting.

    In v2 a 250-step doorway wait could cost -25 to -75 in proximity penalty while a wall hit
    cost -10 and ended the episode, so the policy learned to crash out of deadlocks. Here an
    obstacle hit costs 30, a teammate hit 40, and the summed proximity penalty is capped at 8.
    """
    values = dict(robot_collision_penalty=40.0, obstacle_collision_penalty=30.0, proximity_budget=8.0)
    values.update(overrides)
    return cooperative_multi_config(**values)


def _mirror_goal(goal: Sequence[float], mirror_x: float) -> tuple[float, float]:
    return 2.0 * mirror_x - float(goal[0]), float(goal[1])


class _Robot:
    """Per-robot Isaac handles and episode state."""

    def __init__(self, index: int, path: str) -> None:
        self.index = index
        self.path = path
        self.base_link_path = f"{path}/a__namespace_base_link"
        self.lidar_path = f"{LIDAR_PATH_PREFIX}_{index}"
        self.articulation = None
        self.drive = None
        self.lidar = None
        self.annotator = None
        self.mount_offset_m = LIDAR_MOUNT_OFFSET_M
        self.default_position = None
        self.default_orientation = None
        self.default_position_host = None
        self.default_orientation_host = None
        self.zero_root_velocity = None
        self.zero_dof_velocity = None
        self.sector_masks: dict[str, np.ndarray] = {}
        self.baseline_sector_valid: dict[str, int] = {}
        self.last_ranges: Optional[np.ndarray] = None
        self.stale_scan_reads = 0
        self.readings_cache: Optional[tuple[int, dict]] = None
        self.previous_clearance: dict[str, float] = {}
        self.goal = (0.0, 0.0)
        self.goal_marker = None
        self.active = False
        self.outcome: Optional[str] = None
        self.collision_kind: Optional[str] = None
        self.consecutive_turns = 0
        self.proximity_penalty = 0.0
        self.steps = 0
        self.episode_start: Optional[tuple[float, float, float]] = None   # (x, y, yaw) this episode

    @property
    def home(self) -> tuple[float, float]:
        """The robot's fixed start (scene pose for robot 0, the copy's pose for the others)."""
        return float(self.default_position_host[0, 0]), float(self.default_position_host[0, 1])

    @property
    def start(self) -> tuple[float, float]:
        if self.episode_start is None:
            return self.home
        return self.episode_start[0], self.episode_start[1]


class IsaacMultiPointNavEnv:
    """Several V4 TurtleBots sharing one persistent Isaac simulation (MAPPO / IPPO)."""

    SECTOR_NAMES = ("front", "left", "right")
    OBS_DIM = 6

    def __init__(
        self,
        *,
        multi_config: Optional[MultiNavConfig] = None,
        lidar_config: Optional[IsaacLidarNavConfig] = None,
        point_nav_config: Optional[PointNavConfig] = None,
        isaac_config: Optional[IsaacPointNavConfig] = None,
        headless: bool = True,
        seed: Optional[int] = None,
    ) -> None:
        from isaacsim import SimulationApp

        self.multi_config = multi_config or MultiNavConfig()
        self.lidar_config = lidar_config or v4_lidar_config()
        self.point_nav_config = point_nav_config or _v2_point_nav_config()
        self.isaac_config = isaac_config or IsaacPointNavConfig()
        if self.multi_config.robot_count < 2:
            raise ValueError("Use IsaacPointNavLidarEnv for a single robot")
        self.np_random = np.random.default_rng(seed)
        self._headless = bool(headless)
        self.invalid_sector_reads = 0            # sector reads hit by the V4 empty-sector rule
        self._closed = False
        self._simulation_app = None
        self._simulation_context = None
        self._stage = None
        self._controller = None
        self._stop_action = None
        self._physics_dt_s = 0.0
        self._meters_per_unit = 1.0
        self._ground_z = 0.0
        self._frames = 0
        self._sensor_azimuth_deg: Optional[np.ndarray] = None
        self._obstacle_paths: list[str] = []
        self._obstacle_xy: list[Optional[tuple[float, float]]] = []
        self._obstacle_size: list[float] = []
        self._episode_steps = 0
        self._episode_elapsed_sim_s = 0.0
        self._episode_active = False
        self._mirror_x = 0.0
        self.encounter = "random"
        self.robots = [_Robot(0, ARTICULATION_PATH)] + [
            _Robot(k, f"{CLONE_PATH_PREFIX}_{k}") for k in range(1, self.multi_config.robot_count)
        ]
        self.action_count = len(PointNavAction)

        try:
            mode = "headless" if self._headless else "GUI"
            print(f"[TB03_MULTI] Starting {mode} Isaac Sim with {len(self.robots)} robots", flush=True)
            self._simulation_app = SimulationApp(
                {
                    "headless": self._headless,
                    "open_usd": str(SCENE_PATH),
                    "create_new_stage": False,
                    "fast_shutdown": False,
                    "disable_viewport_updates": self._headless,
                }
            )
            self._initialize()
        except Exception:
            import sys
            import traceback

            print("[TB03_MULTI] INITIALIZATION FAILED (before Isaac shutdown):", flush=True)
            traceback.print_exc(file=sys.stdout)
            sys.stdout.flush()
            self.close()
            raise

    @property
    def obs_dim(self) -> int:
        return self.OBS_DIM + len(self.multi_config.extra_sectors)

    @property
    def state_dim(self) -> int:
        return self.obs_dim + (len(self.robots) - 1) * (self.obs_dim + 5)

    # ------------------------------------------------------------ initialization

    def _initialize(self) -> None:
        import omni.timeline
        import omni.usd
        from pxr import UsdGeom

        context = omni.usd.get_context()
        while context.get_stage_loading_status()[2] > 0:
            self._simulation_app.update()
        self._stage = context.get_stage()
        if self._stage is None:
            raise RuntimeError(f"No active USD stage after opening {SCENE_PATH}")
        if not self._stage.GetPrimAtPath(ARTICULATION_PATH).IsValid():
            raise RuntimeError(f"Expected robot prim is absent: {ARTICULATION_PATH}")
        self._meters_per_unit = UsdGeom.GetStageMetersPerUnit(self._stage)
        if not math.isfinite(self._meters_per_unit) or self._meters_per_unit <= 0:
            raise RuntimeError(f"Invalid stage units: {self._meters_per_unit}")
        omni.timeline.get_timeline_interface().stop()

        # Before physics: copy the robot, hide every robot's visuals (as V4 training does).
        for robot in self.robots[1:]:
            self._clone_robot(robot)
        for robot in self.robots:
            self._hide_robot_visuals(robot)

        self._initialize_physics()
        self._initialize_scene()

    def _clone_robot(self, robot: _Robot) -> None:
        """Copy robot 0's prim (reference + real-size drive overrides) and place it facing -x."""
        from pxr import Gf, Sdf, UsdGeom

        if self._stage.GetPrimAtPath(robot.path).IsValid():
            raise RuntimeError(f"Robot copy path already exists: {robot.path}")
        source = self._stage.GetPrimAtPath(ARTICULATION_PATH)
        layer = self._stage.GetRootLayer()
        if layer.GetPrimAtPath(ARTICULATION_PATH) is None:
            raise RuntimeError(f"{ARTICULATION_PATH} has no spec in the root layer; cannot copy it")
        if not Sdf.CopySpec(layer, ARTICULATION_PATH, layer, robot.path):
            raise RuntimeError(f"Sdf.CopySpec failed for {robot.path}")
        clone = self._stage.GetPrimAtPath(robot.path)
        if not clone.IsValid():
            raise RuntimeError(f"Robot copy did not compose: {robot.path}")

        offset = self.multi_config.partner_start_offset_m * robot.index / self._meters_per_unit
        ops = {op.GetOpName(): op for op in UsdGeom.Xformable(clone).GetOrderedXformOps()}
        source_ops = {op.GetOpName(): op for op in UsdGeom.Xformable(source).GetOrderedXformOps()}
        if "xformOp:translate" not in ops or "xformOp:orient" not in ops:
            raise RuntimeError(f"Robot root needs translate + orient ops; has {list(ops)}")
        translate = source_ops["xformOp:translate"].Get()
        ops["xformOp:translate"].Set(type(translate)(translate[0] + offset, translate[1], translate[2]))
        orient = source_ops["xformOp:orient"].Get()
        half_turn = type(orient)(0.0, type(orient.GetImaginary())(0.0, 0.0, 1.0))  # 180 deg about +z
        ops["xformOp:orient"].Set(half_turn * orient)
        print(f"[TB03_MULTI] robot {robot.index}: copied {ARTICULATION_PATH} -> {robot.path}, "
              f"+{self.multi_config.partner_start_offset_m * robot.index:.2f} m along x, facing -x", flush=True)

    def _hide_robot_visuals(self, robot: _Robot) -> None:
        """Hide only the four validated visual roots (no physics/collision prims)."""
        from pxr import Usd, UsdGeom, UsdPhysics

        if not self.lidar_config.hide_robot_visuals:
            print("[TB03_MULTI] WARNING: visible robots change LiDAR perception; PRESENTATION ONLY", flush=True)
        for name in VISUAL_ROOT_NAMES:
            prim = self._stage.GetPrimAtPath(f"{robot.path}/{name}")
            if not prim.IsValid() or prim.IsInstanceProxy():
                raise RuntimeError(f"Visual root missing or not editable: {robot.path}/{name}")
            if any(p.HasAPI(UsdPhysics.CollisionAPI) or p.HasAPI(UsdPhysics.RigidBodyAPI)
                   or p.HasAPI(UsdPhysics.ArticulationRootAPI) or p.IsA(UsdPhysics.Joint)
                   or "collision" in str(p.GetPath()).lower()
                   for p in Usd.PrimRange(prim, Usd.TraverseInstanceProxies())):
                raise RuntimeError(f"Refusing to change a visual root containing physics: {prim.GetPath()}")
            visible = not self.lidar_config.hide_robot_visuals
            UsdGeom.Imageable(prim).GetVisibilityAttr().Set(
                UsdGeom.Tokens.inherited if visible else UsdGeom.Tokens.invisible
            )

    def _initialize_physics(self) -> None:
        from isaacsim.core.api import SimulationContext
        from isaacsim.core.experimental.prims import Articulation
        from isaacsim.robot.wheeled_robots.controllers.differential_controller import DifferentialController
        from isaacsim.robot.wheeled_robots.robots.wheeled_robot import WheeledRobot

        self._simulation_context = SimulationContext()
        self._simulation_context.initialize_physics()
        self._simulation_context.play()
        self._physics_dt_s = float(self._simulation_context.get_physics_dt())
        if self._physics_dt_s <= 0.0:
            raise RuntimeError(f"Invalid Isaac physics timestep: {self._physics_dt_s}")
        self._step_frames(self.isaac_config.settle_frames, count_episode_time=False)

        cfg = self.isaac_config
        self._controller = DifferentialController(
            name="tb03_mappo_differential_controller",
            wheel_radius=cfg.wheel_radius_m,
            wheel_base=cfg.wheel_base_m,
            max_linear_speed=cfg.max_linear_speed_mps,
            max_angular_speed=cfg.max_angular_speed_radps,
        )
        self._stop_action = self._controller.forward(np.array([0.0, 0.0], dtype=np.float64))
        for robot in self.robots:
            robot.articulation = Articulation(robot.path)
            if not robot.articulation.is_physics_tensor_entity_valid():
                raise RuntimeError(f"Physics tensor state is unavailable for {robot.path}")
            robot.drive = WheeledRobot(
                prim_path=robot.path,
                name=f"tb03_mappo_drive_{robot.index}",
                wheel_dof_names=[LEFT_WHEEL_DOF, RIGHT_WHEEL_DOF],
            )
            robot.drive.initialize()
            _, wheel_indices = robot.drive.get_articulation_controller_properties()
            if wheel_indices is None or len(wheel_indices) != 2 or any(i is None or i < 0 for i in wheel_indices):
                raise RuntimeError(f"Could not resolve wheel indices for {robot.path}: {wheel_indices}")
        self._apply_stop()
        self._step_frames(cfg.settle_frames, count_episode_time=False, hold_stop=True)
        for robot in self.robots:
            robot.default_position, robot.default_orientation = robot.articulation.get_world_poses()
            robot.default_position_host = _host_array(robot.default_position)
            robot.default_orientation_host = _host_array(robot.default_orientation)
            robot.zero_root_velocity = np.zeros(tuple(robot.default_position.shape), dtype=np.float32)
            robot.zero_dof_velocity = np.zeros(tuple(robot.articulation.get_dof_velocities().shape), dtype=np.float32)
            state = self._state(robot)
            print(f"[TB03_MULTI] robot {robot.index} start=({state.x:.3f}, {state.y:.3f}, yaw {state.yaw:.3f})",
                  flush=True)
        self._mirror_x = 0.5 * (self.robots[0].home[0] + self.robots[1].home[0])

    def _initialize_scene(self) -> None:
        from pxr import Gf, UsdGeom
        from isaacsim.sensors.rtx import LidarRtx

        stage = self._stage
        for path in [OBSTACLE_ROOT] + [r.lidar_path for r in self.robots]:
            if stage.GetPrimAtPath(path).IsValid():
                raise RuntimeError(f"MAPPO prim path already exists in the scene: {path}")
        self._ground_z = self._measure_floor_z()

        # Obstacle pool: created once, parked, never deleted.
        UsdGeom.Xform.Define(stage, OBSTACLE_ROOT)
        cfg = self.lidar_config
        for index in range(cfg.obstacle_count):
            path = f"{OBSTACLE_ROOT}/obstacle_{index}"
            cube = UsdGeom.Cube.Define(stage, path)
            cube.CreateSizeAttr(1.0)
            UsdGeom.XformCommonAPI(cube.GetPrim()).SetScale(Gf.Vec3f(
                cfg.obstacle_size_xy_m / self._meters_per_unit,
                cfg.obstacle_size_xy_m / self._meters_per_unit,
                cfg.obstacle_height_m / self._meters_per_unit,
            ))
            cube.CreateDisplayColorAttr([Gf.Vec3f(0.85, 0.35, 0.10)])
            self._obstacle_paths.append(path)
            self._obstacle_xy.append(None)
            self._obstacle_size.append(cfg.obstacle_size_xy_m)
            self._park_obstacle(index)
        print(f"[TB03_MULTI] obstacle pool: {cfg.obstacle_count} prims under {OBSTACLE_ROOT}", flush=True)

        if not self._headless:
            for robot in self.robots:
                marker = UsdGeom.Cylinder.Define(stage, f"{GOAL_MARKER_PREFIX}_{robot.index}")
                marker.CreateRadiusAttr(0.05 / self._meters_per_unit)
                marker.CreateHeightAttr(0.005 / self._meters_per_unit)
                marker.CreateAxisAttr(UsdGeom.Tokens.z)
                marker.CreateDisplayColorAttr([Gf.Vec3f(*GOAL_COLORS[robot.index % len(GOAL_COLORS)])])
                robot.goal_marker = marker

        for robot in self.robots:
            _, _, scale, _, _, _ = UsdGeom.XformCache().GetLocalToWorldTransform(
                stage.GetPrimAtPath(robot.path)
            ).Factor()
            if all(abs(float(s) - 1.0) <= UNIT_SCALE_TOLERANCE for s in scale):
                robot.mount_offset_m = REAL_SIZE_LIDAR_MOUNT_OFFSET_M
            robot.lidar = LidarRtx(
                prim_path=robot.lidar_path,
                name=f"tb03_mappo_lidar_{robot.index}",
                config_file_name=LIDAR_CONFIG,
                **LIDAR_RANGE_SETTINGS,
            )
            product = robot.lidar._render_product
            if product is None or not robot.lidar.get_render_product_path():
                raise RuntimeError(f"LiDAR render product was not created for robot {robot.index}")
            product.hydra_texture.set_updates_enabled(True)
            robot.lidar.initialize()
            robot.lidar.set_local_scale(np.ones(3))
            prim = stage.GetPrimAtPath(robot.lidar.prim_path)
            for name, expected in LIDAR_RANGE_SETTINGS.items():
                value = prim.GetAttribute(name).Get()
                if value is None or not np.isclose(value, expected, atol=1e-6, rtol=0):
                    raise RuntimeError(f"LiDAR {robot.index} setting not applied: {name}={value}")
            self._sync_lidar(robot)
            robot.lidar.attach_annotator(FLAT_SCAN_ANNOTATOR)
            robot.annotator = robot.lidar.get_annotators()[FLAT_SCAN_ANNOTATOR]
            print(f"[TB03_MULTI] robot {robot.index} scale={tuple(round(float(s), 3) for s in scale)} "
                  f"lidar={robot.lidar_path} mount={robot.mount_offset_m} m", flush=True)

        self._wait_for_scans()
        self._build_baselines()

    # ------------------------------------------------------------ physics stepping

    def _sync_lidar(self, robot: _Robot) -> None:
        """Top-level LiDAR from the rigid base pose (no inherited scale), as in V4."""
        from pxr import Gf, UsdGeom

        base_link = self._stage.GetPrimAtPath(robot.base_link_path)
        if not base_link.IsValid():
            raise RuntimeError(f"Missing TurtleBot base link: {robot.base_link_path}")
        _, _, _, rotation_matrix, translation, _ = UsdGeom.XformCache().GetLocalToWorldTransform(base_link).Factor()
        rotation = rotation_matrix.ExtractRotationQuat().GetNormalized()
        offset = Gf.Vec3d(*(c / self._meters_per_unit for c in robot.mount_offset_m))
        world_offset = Gf.Rotation(rotation).TransformDir(offset)
        robot.lidar.set_world_pose(
            position=np.asarray(translation + world_offset, dtype=np.float64),
            orientation=np.asarray(_wxyz(rotation), dtype=np.float64),
        )

    def _physics_step(self) -> None:
        # RTX LiDAR needs a rendered frame every step; keep every LiDAR mounted.
        self._frames += 1
        for robot in self.robots:
            self._sync_lidar(robot)
        self._simulation_context.step(render=True)
        for robot in self.robots:
            self._sync_lidar(robot)

    def _step_frames(self, frames: int, *, count_episode_time: bool, hold_stop: bool = False) -> None:
        for _ in range(frames):
            if hold_stop:
                self._apply_stop()
            if self.robots[0].lidar is None:
                self._simulation_context.step(render=not self._headless)
            else:
                self._physics_step()
        if count_episode_time:
            self._episode_elapsed_sim_s += frames * self._physics_dt_s

    def _apply_stop(self, robots: Optional[Sequence[_Robot]] = None) -> None:
        for robot in robots if robots is not None else self.robots:
            if robot.drive is not None and self._stop_action is not None:
                robot.drive.apply_wheel_actions(self._stop_action)

    def settle_scan(self, frames: int) -> None:
        self._step_frames(frames, count_episode_time=False, hold_stop=True)

    def _state(self, robot: _Robot) -> IsaacPoseState:
        position, orientation = robot.articulation.get_world_poses()
        position = _host_array(position)
        orientation = _host_array(orientation)
        return IsaacPoseState(x=float(position[0, 0]), y=float(position[0, 1]), yaw=_yaw_from_wxyz(orientation[0]))

    # ------------------------------------------------------------ LiDAR

    def _flat_ranges(self, robot: _Robot) -> np.ndarray:
        data = robot.annotator.get_data()
        if not isinstance(data, dict):
            return np.zeros(0)
        ranges = np.asarray(data.get("linearDepthData", []), dtype=np.float64).reshape(-1)
        return np.where(np.isfinite(ranges) & (ranges > INVALID_RANGE_EPSILON_M) & (ranges >= 0.10) & (ranges <= 200.0),
                        ranges, np.nan)

    def _wait_for_scans(self) -> None:
        cfg = self.lidar_config
        accepted = {r.index: 0 for r in self.robots}
        last_size: dict[int, Optional[int]] = {r.index: None for r in self.robots}
        ready: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for frame in range(1, cfg.scan_wait_max_frames + 1):
            self._step_frames(1, count_episode_time=False, hold_stop=True)
            for robot in self.robots:
                if robot.index in ready:
                    continue
                data = robot.annotator.get_data()
                if not isinstance(data, dict):
                    continue
                ranges = np.asarray(data.get("linearDepthData", []), dtype=np.float64).reshape(-1).copy()
                cols = int(data.get("numCols", 0))
                bounds = np.asarray(data.get("azimuthRange", []), dtype=float).reshape(-1)
                depth = np.asarray(data.get("depthRange", []), dtype=float).reshape(-1)
                if not (cols > 0 and cols == ranges.size and bounds.size == 2
                        and np.isfinite(bounds).all() and depth.size == 2):
                    continue
                if not np.allclose(depth, [0.10, 200.0], atol=1e-5, rtol=0):
                    raise RuntimeError(f"FlatScan depthRange does not match configuration: {depth}")
                valid = np.isfinite(ranges) & (ranges > INVALID_RANGE_EPSILON_M) & (ranges >= 0.10) & (ranges <= 200.0)
                if not valid.any():
                    continue
                accepted[robot.index] = accepted[robot.index] + 1 if ranges.size == last_size[robot.index] else 1
                last_size[robot.index] = ranges.size
                if accepted[robot.index] >= cfg.scan_stable_frames:
                    azimuth = (np.linspace(bounds[0], bounds[1], ranges.size) + 180) % 360 - 180
                    ready[robot.index] = (np.where(valid, ranges, np.nan), azimuth)
            if len(ready) == len(self.robots):
                break
        else:
            raise RuntimeError(f"RTX LiDAR initialization timed out; accepted={accepted}")
        for robot in self.robots:
            robot.last_ranges, azimuth = ready[robot.index]
            if self._sensor_azimuth_deg is None:
                self._sensor_azimuth_deg = azimuth
            elif azimuth.shape != self._sensor_azimuth_deg.shape or not np.allclose(azimuth, self._sensor_azimuth_deg):
                raise RuntimeError("The robots' LiDARs report different azimuth layouts")
            robot.sector_masks = {
                name: self._angle_mask(getattr(cfg, f"{name}_sector_deg")) for name in self.SECTOR_NAMES
            }
            for name, lo, hi in self.multi_config.extra_sectors:
                robot.sector_masks[name] = self._angle_mask((lo, hi))
            print(f"[TB03_MULTI] robot {robot.index} LiDAR ready: rays={azimuth.size} rendered_updates={frame} "
                  + " ".join(f"{n}_rays={int(m.sum())}" for n, m in robot.sector_masks.items()), flush=True)

    def _angle_mask(self, sector: tuple[float, float]) -> np.ndarray:
        lo, hi = sector
        return (self._sensor_azimuth_deg >= lo) & (self._sensor_azimuth_deg <= hi)

    def _build_baselines(self) -> None:
        """Valid rays each sector normally sees with obstacles parked (V4 'too close' rule)."""
        scans: dict[int, list[np.ndarray]] = {r.index: [] for r in self.robots}
        for _ in range(self.lidar_config.self_mask_frames):
            self._step_frames(1, count_episode_time=False, hold_stop=True)
            for robot in self.robots:
                ranges = self._flat_ranges(robot)
                if ranges.size == self._sensor_azimuth_deg.size:
                    scans[robot.index].append(ranges)
        for robot in self.robots:
            if not scans[robot.index]:
                raise RuntimeError(f"No LiDAR scans for robot {robot.index} baseline")
            robot.baseline_sector_valid = {
                name: max(int((np.isfinite(scan) & mask).sum()) for scan in scans[robot.index])
                for name, mask in robot.sector_masks.items()
            }
            print(f"[TB03_MULTI] robot {robot.index} baseline sector valid rays: "
                  + " ".join(f"{n}={c}" for n, c in robot.baseline_sector_valid.items()), flush=True)

    def _current_ranges(self, robot: _Robot) -> np.ndarray:
        ranges = self._flat_ranges(robot)
        if ranges.size == self._sensor_azimuth_deg.size:
            robot.last_ranges = ranges
        else:
            robot.stale_scan_reads += 1
        return robot.last_ranges

    def _lidar_pose(self, robot: _Robot) -> tuple[float, float, float]:
        """World (x, y, yaw) of a robot's LiDAR, in meters."""
        from pxr import UsdGeom

        world = UsdGeom.XformCache().GetLocalToWorldTransform(self._stage.GetPrimAtPath(robot.lidar_path))
        _, _, _, rotation_matrix, translation, _ = world.Factor()
        quat = rotation_matrix.ExtractRotationQuat().GetNormalized()
        yaw = _yaw_from_wxyz(_wxyz(quat))
        return float(translation[0]) * self._meters_per_unit, float(translation[1]) * self._meters_per_unit, yaw

    def _teammate_ranges(self, robot: _Robot) -> np.ndarray:
        """Exact ray-circle hits on the other robots' footprints (nan = no hit)."""
        px, py, yaw = self._lidar_pose(robot)
        angles = yaw + np.radians(self._sensor_azimuth_deg)
        ux, uy = np.cos(angles), np.sin(angles)
        radius = self.multi_config.teammate_radius_m
        hits = np.full(angles.size, np.nan)
        for other in self.robots:
            if other is robot:
                continue
            state = self._state(other)
            dx, dy = state.x - px, state.y - py
            along = ux * dx + uy * dy
            perp_sq = dx * dx + dy * dy - along * along
            inside = perp_sq <= radius * radius
            with np.errstate(invalid="ignore"):
                distance = along - np.sqrt(np.maximum(radius * radius - perp_sq, 0.0))
            distance = np.where(inside & (distance > 0.0), distance, np.nan)
            hits = np.fmin(hits, distance)
        # Same validity as the RTX FlatScan: nothing closer than the near range.
        with np.errstate(invalid="ignore"):
            hits[hits < LIDAR_NEAR_RANGE_M] = np.nan
        return hits

    def _perturb_ranges(self, ranges: np.ndarray) -> np.ndarray:
        """V4 sensor model: noise, per-ray dropout, minimum valid range."""
        cfg = self.lidar_config
        if not (cfg.lidar_noise_std_m > 0.0 or cfg.lidar_ray_dropout > 0.0 or cfg.lidar_min_valid_m):
            return ranges
        noisy = ranges.copy()
        finite = np.isfinite(noisy)
        if cfg.lidar_noise_std_m > 0.0:
            noisy[finite] += self.np_random.normal(0.0, cfg.lidar_noise_std_m, int(finite.sum()))
        if cfg.lidar_ray_dropout > 0.0:
            noisy[self.np_random.random(noisy.size) < cfg.lidar_ray_dropout] = np.nan
        if cfg.lidar_min_valid_m:
            with np.errstate(invalid="ignore"):
                noisy[noisy < cfg.lidar_min_valid_m] = np.nan
        return noisy

    def sector_readings(self, robot: _Robot) -> dict[str, dict[str, Any]]:
        """Per sector: valid rays, invalid flag and clearance (V4 rules, teammates included)."""
        if robot.readings_cache is not None and robot.readings_cache[0] == self._frames:
            return robot.readings_cache[1]
        ranges = np.fmin(self._current_ranges(robot), self._teammate_ranges(robot))
        ranges = self._perturb_ranges(ranges)
        readings = {}
        for name, mask in robot.sector_masks.items():
            values = ranges[mask]
            values = values[np.isfinite(values)]
            invalid = values.size == 0 and robot.baseline_sector_valid.get(name, 0) > 0
            self.invalid_sector_reads += invalid
            if self.multi_config.invalid_as_clear:
                invalid = False
            clearance = float(values.min()) if values.size else (0.0 if invalid else math.inf)
            readings[name] = {"valid_rays": int(values.size), "invalid": invalid, "clearance_m": clearance}
        cfg = self.lidar_config
        if cfg.lidar_sector_dropout > 0.0:
            for name, reading in readings.items():
                if self.np_random.random() < cfg.lidar_sector_dropout:
                    previous = robot.previous_clearance.get(name, math.inf)
                    reading.update(valid_rays=0, invalid=False, dropped=True,
                                   clearance_m=0.0 if previous < cfg.ambiguous_close_m else math.inf)
        robot.previous_clearance = {name: r["clearance_m"] for name, r in readings.items()}
        robot.readings_cache = (self._frames, readings)
        return readings

    # ------------------------------------------------------------ observations

    def _distance_and_bearing(self, robot: _Robot, state: IsaacPoseState) -> tuple[float, float]:
        dx, dy = robot.goal[0] - state.x, robot.goal[1] - state.y
        return math.hypot(dx, dy), wrap_to_pi(math.atan2(dy, dx) - state.yaw)

    def _observation(self, robot: _Robot, state: IsaacPoseState) -> np.ndarray:
        distance, bearing = self._distance_and_bearing(robot, state)
        norm = self.lidar_config.clearance_normalization_m
        readings = self.sector_readings(robot)
        names = list(self.SECTOR_NAMES) + [n for n, _, _ in self.multi_config.extra_sectors]
        lidar = [float(np.clip(readings[n]["clearance_m"] / norm, 0.0, 1.0)) for n in names]
        return np.array(
            [np.clip(distance / self.point_nav_config.distance_normalization_m, 0.0, 1.0),
             math.sin(bearing), math.cos(bearing), *lidar],
            dtype=np.float32,
        )

    def observations(self) -> list[np.ndarray]:
        return [self._observation(robot, self._state(robot)) for robot in self.robots]

    def global_state(self, index: int, observations: Sequence[np.ndarray]) -> np.ndarray:
        """Centralized critic input for robot `index` (see module docstring)."""
        robot = self.robots[index]
        me = self._state(robot)
        scale = self.multi_config.teammate_obs_scale_m
        parts = [np.asarray(observations[index], dtype=np.float32)]
        for other in self.robots[index + 1:] + self.robots[:index]:
            them = self._state(other)
            dx, dy = them.x - me.x, them.y - me.y
            cos_y, sin_y = math.cos(me.yaw), math.sin(me.yaw)
            local_x, local_y = cos_y * dx + sin_y * dy, -sin_y * dx + cos_y * dy
            relative_yaw = wrap_to_pi(them.yaw - me.yaw)
            parts.append(np.asarray(observations[other.index], dtype=np.float32))
            parts.append(np.array([
                np.clip(local_x / scale, -1.0, 1.0), np.clip(local_y / scale, -1.0, 1.0),
                math.sin(relative_yaw), math.cos(relative_yaw), 1.0 if other.active else 0.0,
            ], dtype=np.float32))
        return np.concatenate(parts).astype(np.float32)

    # ------------------------------------------------------------ obstacles and geometry

    def _measure_floor_z(self) -> float:
        from omni.physx import get_physx_scene_query_interface
        from pxr import Gf, UsdGeom

        base = UsdGeom.XformCache().GetLocalToWorldTransform(
            self._stage.GetPrimAtPath(self.robots[0].base_link_path)
        ).ExtractTranslation()
        units = self._meters_per_unit
        hits = []
        robot_paths = [r.path for r in self.robots]

        def report(hit):
            paths = (str(hit.collision), str(hit.rigid_body))
            if any(p == rp or p.startswith(rp + "/") for p in paths for rp in robot_paths):
                return True
            z = float(hit.position.z)
            if math.isfinite(z) and z <= float(base[2]) + 1e-4 / units and float(hit.normal.z) > 0.5:
                hits.append((z, str(hit.collision)))
            return True

        origin = Gf.Vec3f(float(base[0]), float(base[1]), float(base[2]) + 0.5 / units)
        get_physx_scene_query_interface().raycast_all(origin, Gf.Vec3f(0, 0, -1), 5.0 / units, report)
        if not hits:
            raise RuntimeError("No supporting floor found below robot 0")
        z, path = max(hits, key=lambda hit: hit[0])
        print(f"[TB03_MULTI] floor collider={path} z={z * units:.6f} m", flush=True)
        return z * units

    def _placement_xyz(self, x: float, y: float, z: float):
        from pxr import Gf

        coords = (float(x), float(y), float(z))
        if not all(math.isfinite(v) and abs(v) < 1e12 for v in coords):
            raise ValueError(f"Invalid placement coordinates (meters): {coords}")
        return Gf.Vec3d(*(v / self._meters_per_unit for v in coords))

    def _set_obstacle_xy(self, index: int, xy: tuple[float, float], size: Optional[float] = None) -> None:
        from pxr import Gf, UsdGeom

        cfg = self.lidar_config
        api = UsdGeom.XformCommonAPI(self._stage.GetPrimAtPath(self._obstacle_paths[index]))
        api.SetTranslate(self._placement_xyz(xy[0], xy[1], self._ground_z + 0.5 * cfg.obstacle_height_m))
        size = cfg.obstacle_size_xy_m if size is None else float(size)
        if size != self._obstacle_size[index]:
            # Resize the pooled prim (never delete / re-create while the LiDARs run).
            units = self._meters_per_unit
            api.SetScale(Gf.Vec3f(size / units, size / units, cfg.obstacle_height_m / units))
            self._obstacle_size[index] = size

    def _park_obstacle(self, index: int) -> None:
        cfg = self.lidar_config
        self._set_obstacle_xy(index, (cfg.park_origin_m[0] + index * cfg.park_spacing_m, cfg.park_origin_m[1]))
        self._obstacle_xy[index] = None

    def _place_obstacles(self, positions: Sequence[Sequence[float]]) -> None:
        if len(positions) > len(self._obstacle_paths):
            raise ValueError(f"At most {len(self._obstacle_paths)} obstacles are pooled; got {len(positions)}")
        for index in range(len(self._obstacle_paths)):
            if index < len(positions):
                xy = (float(positions[index][0]), float(positions[index][1]))
                size = float(positions[index][2]) if len(positions[index]) > 2 else None
                self._set_obstacle_xy(index, xy, size)
                self._obstacle_xy[index] = xy
            else:
                self._park_obstacle(index)

    def _sample_obstacles(self, starts: Sequence[tuple[float, float]], goals: Sequence[tuple[float, float]],
                          placed: Optional[list[tuple]] = None, keep_clear: Optional[tuple[float, float]] = None,
                          blocking: bool = True) -> list[tuple]:
        """V4 sampling per route: one blocking box per start-goal segment (p = 0.7), then
        0..max_distractors distractors around all routes, clear of every start and goal."""
        cfg = self.lidar_config
        rng = self.np_random
        sized = cfg.obstacle_size_range_m is not None
        placed = list(placed or [])

        def draw_size() -> float:
            return float(rng.uniform(*cfg.obstacle_size_range_m)) if sized else cfg.obstacle_size_xy_m

        def valid(p, size) -> bool:
            extra = 0.5 * (size - cfg.obstacle_size_xy_m)
            if keep_clear is not None and math.dist(p, keep_clear) < self.multi_config.structure_clearance_m + extra:
                return False
            return (
                all(math.dist(p, s) >= cfg.min_obstacle_start_clearance_m + extra for s in starts)
                and all(math.dist(p, g) >= cfg.min_obstacle_goal_clearance_m + extra for g in goals)
                and all(math.dist(p, q[:2]) >= cfg.min_obstacle_spacing_m + extra
                        + 0.5 * ((q[2] if len(q) > 2 else cfg.obstacle_size_xy_m) - cfg.obstacle_size_xy_m)
                        for q in placed)
            )

        def item(p, size):
            return (p[0], p[1], size) if sized else p

        for start, goal in zip(starts, goals) if blocking else ():
            if len(placed) >= cfg.obstacle_count:
                break
            dx, dy = goal[0] - start[0], goal[1] - start[1]
            length = math.hypot(dx, dy)
            size = draw_size()
            extra = 0.5 * (size - cfg.obstacle_size_xy_m)
            lo = cfg.min_obstacle_start_clearance_m + extra
            hi = length - cfg.min_obstacle_goal_clearance_m - extra
            if length > 0.0 and hi > lo and rng.random() < cfg.blocking_probability:
                ux, uy = dx / length, dy / length
                for _ in range(cfg.placement_attempts):
                    s = rng.uniform(lo, hi)
                    lateral = rng.uniform(-cfg.blocking_lateral_offset_m, cfg.blocking_lateral_offset_m)
                    p = (start[0] + s * ux - lateral * uy, start[1] + s * uy + lateral * ux)
                    if valid(p, size):
                        placed.append(item(p, size))
                        break

        n_distractors = min(int(rng.integers(0, cfg.max_distractors + 1)), cfg.obstacle_count - len(placed))
        points = list(starts) + list(goals)
        x_lo = min(p[0] for p in points) - cfg.distractor_margin_m
        x_hi = max(p[0] for p in points) + cfg.distractor_margin_m
        y_lo = min(p[1] for p in points) - cfg.distractor_margin_m
        y_hi = max(p[1] for p in points) + cfg.distractor_margin_m
        for _ in range(n_distractors):
            size = draw_size()
            for _ in range(cfg.placement_attempts):
                p = (rng.uniform(x_lo, x_hi), rng.uniform(y_lo, y_hi))
                if valid(p, size):
                    placed.append(item(p, size))
                    break
        return placed

    def _sample_goals(self) -> list[tuple[float, float]]:
        """Robot 0: the V1B goal banks. Other robots: the same banks mirrored about the midpoint."""
        mc = self.multi_config
        banks = (LEFT_GOALS, RIGHT_GOALS, STRAIGHT_GOALS)
        starts = [r.home for r in self.robots]
        for _ in range(mc.goal_attempts):
            goals = []
            for robot in self.robots:
                bank = banks[int(self.np_random.integers(len(banks)))]
                goal = bank[int(self.np_random.integers(len(bank)))]
                goals.append((float(goal[0]), float(goal[1])) if robot.index == 0
                             else _mirror_goal(goal, self._mirror_x))
            ok = all(math.dist(goals[a], goals[b]) >= mc.min_goal_separation_m
                     for a in range(len(goals)) for b in range(a + 1, len(goals)))
            ok = ok and all(math.dist(goals[a], starts[b]) >= mc.min_goal_start_separation_m
                            for a in range(len(goals)) for b in range(len(starts)) if a != b)
            if ok:
                return goals
        raise RuntimeError("Could not sample separated goals for all robots")

    def _in_area(self, points) -> bool:
        mc = self.multi_config
        return all(mc.area_x_m[0] <= p[0] <= mc.area_x_m[1] and mc.area_y_m[0] <= p[1] <= mc.area_y_m[1]
                   for p in points)

    def _sample_encounter(self, name: Optional[str] = None):
        """(name, starts [(x, y, yaw)], goals [(x, y)], structure boxes [(x, y, w)], centre or None)."""
        mc = self.multi_config
        if len(self.robots) != 2:
            name = "random"
        if name is None:
            names = [n for n, _ in mc.encounter_weights]
            weights = np.array([w for _, w in mc.encounter_weights], dtype=float)
            name = names[int(self.np_random.choice(len(names), p=weights / weights.sum()))]
        rng = self.np_random
        if name == "random":
            goals = self._sample_goals()
            starts = [(*r.home, _yaw_from_wxyz(r.default_orientation_host[0])) for r in self.robots]
            return name, starts, goals, [], None

        def yaw_along(dx, dy):
            return wrap_to_pi(math.atan2(dy, dx) + rng.uniform(-mc.start_yaw_noise_rad, mc.start_yaw_noise_rad))

        for _ in range(mc.goal_attempts):
            cx, cy = rng.uniform(*mc.centre_x_m), rng.uniform(*mc.centre_y_m)
            structure: list[tuple[float, float, float]] = []
            centre = None
            label = name
            if name == "crossing":
                heading = rng.uniform(-math.pi, math.pi)
                turn = math.radians(rng.uniform(*mc.crossing_angle_deg)) * (1 if rng.random() < 0.5 else -1)
                legs = []
                for angle in (heading, heading + turn):
                    half = 0.5 * rng.uniform(*mc.crossing_length_m)
                    ux, uy = math.cos(angle), math.sin(angle)
                    legs.append(((cx - half * ux, cy - half * uy), (cx + half * ux, cy + half * uy)))
                starts = [(s[0], s[1], yaw_along(g[0] - s[0], g[1] - s[1])) for s, g in legs]
                goals = [g for _, g in legs]
            else:
                axis_aligned = name == "doorway" or rng.random() < mc.corridor_probability
                heading = int(rng.integers(4)) * 0.5 * math.pi if axis_aligned else rng.uniform(-math.pi, math.pi)
                ux, uy = math.cos(heading), math.sin(heading)
                vx, vy = -uy, ux
                half = 0.5 * rng.uniform(*mc.route_length_m)
                lane = rng.uniform(-mc.lane_offset_m, mc.lane_offset_m)
                a = (cx - half * ux + 0.5 * lane * vx, cy - half * uy + 0.5 * lane * vy)
                b = (cx + half * ux - 0.5 * lane * vx, cy + half * uy - 0.5 * lane * vy)
                starts = [(a[0], a[1], yaw_along(b[0] - a[0], b[1] - a[1])),
                          (b[0], b[1], yaw_along(a[0] - b[0], a[1] - b[1]))]
                goals = [b, a]
                if name == "doorway":
                    gap = rng.uniform(*mc.doorway_gap_m)
                    side = rng.uniform(*mc.wall_box_m)
                    for sign in (-1, 1):
                        for k in (0.5, 1.5):
                            d = sign * (0.5 * gap + k * side)
                            structure.append((cx + d * vx, cy + d * vy, side))
                    centre = (cx, cy)
                elif axis_aligned:
                    width = rng.uniform(*mc.corridor_width_m)
                    side = rng.uniform(*mc.wall_box_m)
                    for sign in (-1, 1):
                        for k in (-0.5, 0.5):
                            d = sign * (0.5 * width + 0.5 * side)
                            structure.append((cx + d * vx + k * side * ux, cy + d * vy + k * side * uy, side))
                    centre = (cx, cy)
                    label = "head_on_corridor"
            if self._in_area([s[:2] for s in starts] + goals):
                if mc.park_teammate:   # robot 1 starts on its goal in the corner farthest from the encounter
                    corners = [(x, y) for x in mc.area_x_m for y in mc.area_y_m]
                    px, py = max(corners, key=lambda c: math.hypot(c[0] - cx, c[1] - cy))
                    px += 0.25 if px == mc.area_x_m[0] else -0.25
                    py += 0.25 if py == mc.area_y_m[0] else -0.25
                    starts[1], goals[1] = (px, py, 0.0), (px, py)
                return label, starts, goals, structure, centre
        raise RuntimeError(f"Could not place a {name} encounter inside the area")

    def _obstacle_surface_distance(self, state: IsaacPoseState) -> float:
        best = math.inf
        for xy, size in zip(self._obstacle_xy, self._obstacle_size):
            if xy is None:
                continue
            half = 0.5 * size
            best = min(best, math.hypot(max(abs(state.x - xy[0]) - half, 0.0), max(abs(state.y - xy[1]) - half, 0.0)))
        return best

    def _teammate_surface_distance(self, robot: _Robot, states: dict[int, IsaacPoseState]) -> float:
        me = states[robot.index]
        return min((math.hypot(states[o.index].x - me.x, states[o.index].y - me.y) - self.multi_config.teammate_radius_m
                    for o in self.robots if o is not robot), default=math.inf)

    def _proximity_penalty(self, distance: float, threshold: Optional[float] = None) -> float:
        cfg = self.lidar_config
        threshold = cfg.proximity_threshold_m if threshold is None else threshold
        if threshold <= 0.0 or cfg.proximity_penalty_max <= 0.0 or distance >= threshold:
            return 0.0
        span = max(threshold - cfg.robot_radius_m, 1e-6)
        return cfg.proximity_penalty_max * min((threshold - distance) / span, 1.0)

    # ------------------------------------------------------------ episode API

    def _move_goal_marker(self, robot: _Robot) -> None:
        if robot.goal_marker is None:
            return
        from pxr import UsdGeom

        UsdGeom.XformCommonAPI(robot.goal_marker.GetPrim()).SetTranslate(
            self._placement_xyz(robot.goal[0], robot.goal[1], self._ground_z + 0.005)
        )

    def _reset_articulations(self) -> None:
        self._apply_stop()
        for robot in self.robots:
            x, y, yaw = robot.episode_start
            z = float(robot.default_position_host[0, 2])
            robot.articulation.set_world_poses(
                np.array([[x, y, z]], dtype=np.float32),
                np.array([[math.cos(0.5 * yaw), 0.0, 0.0, math.sin(0.5 * yaw)]], dtype=np.float32),
            )
            robot.articulation.set_velocities(robot.zero_root_velocity, robot.zero_root_velocity)
            robot.articulation.set_dof_velocities(robot.zero_dof_velocity)
            robot.articulation.set_dof_velocity_targets(robot.zero_dof_velocity)
        self._step_frames(self.isaac_config.settle_frames, count_episode_time=False, hold_stop=True)
        for robot in self.robots:
            state = self._state(robot)
            expected_yaw = robot.episode_start[2]
            position_error = math.hypot(state.x - robot.start[0], state.y - robot.start[1])
            yaw_error = abs(wrap_to_pi(state.yaw - expected_yaw))
            if (position_error > self.isaac_config.reset_position_tolerance_m
                    or yaw_error > self.isaac_config.reset_yaw_tolerance_rad):
                raise RuntimeError(f"Robot {robot.index} reset did not settle: "
                                   f"position_error={position_error:.6f} m, yaw_error={yaw_error:.6f} rad")
        # Let the RTX scans catch up with the teleported robots and obstacles.
        self._step_frames(self.lidar_config.reset_scan_frames, count_episode_time=False, hold_stop=True)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None
              ) -> tuple[list[np.ndarray], dict[str, Any]]:
        """Reset every robot to its start, sample goals and obstacles.

        options: encounter (name), starts (one (x, y, yaw) per robot), goals (one (x, y) per robot),
        obstacles (list of (x, y[, width]); [] parks all). starts/goals override the encounter's.
        """
        if self._closed:
            raise RuntimeError("Environment is closed.")
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        options = dict(options or {})
        unknown = set(options) - {"encounter", "starts", "goals", "obstacles"}
        if unknown:
            raise ValueError(f"Unsupported reset option(s): {sorted(unknown)}")
        custom = "starts" in options or "goals" in options
        name, starts, goals, structure, centre = self._sample_encounter("random" if custom else options.get("encounter"))
        if custom:
            name, structure, centre = options.get("encounter", "custom"), [], None
        starts = [tuple(s) for s in options.get("starts", starts)]
        goals = [tuple(g) for g in options.get("goals", goals)]
        if len(goals) != len(self.robots) or len(starts) != len(self.robots):
            raise ValueError(f"Need one start and one goal per robot ({len(self.robots)})")
        self.encounter = name
        for robot, start in zip(self.robots, starts):
            robot.episode_start = (float(start[0]), float(start[1]), float(start[2]))
        if "obstacles" in options:
            obstacles = options["obstacles"]
        else:
            obstacles = self._sample_obstacles([s[:2] for s in starts], goals, placed=structure,
                                               keep_clear=centre, blocking=name == "random")
        self._place_obstacles(obstacles)
        for robot, goal in zip(self.robots, goals):
            robot.goal = (float(goal[0]), float(goal[1]))
            robot.active = True
            robot.outcome = None
            robot.collision_kind = None
            robot.consecutive_turns = 0
            robot.proximity_penalty = 0.0
            robot.proximity_total = 0.0
            robot.previous_clearance = {}
            robot.readings_cache = None
            robot.steps = 0
            self._move_goal_marker(robot)
        self._reset_articulations()
        self._episode_steps = 0
        self._episode_elapsed_sim_s = 0.0
        self._episode_active = True
        observations = self.observations()
        info = self._info()
        print(f"[TB03_MULTI] reset encounter={self.encounter} "
              f"starts={[(round(r.episode_start[0], 3), round(r.episode_start[1], 3), round(r.episode_start[2], 2)) for r in self.robots]} "
              f"goals={[(round(g[0], 3), round(g[1], 3)) for g in goals]} "
              f"obstacles={[(round(o[0], 3), round(o[1], 3)) for o in info['obstacles']]}", flush=True)
        return observations, info

    def _action_spec(self, action: PointNavAction) -> tuple[np.ndarray, int]:
        cfg = self.isaac_config
        if action == PointNavAction.FORWARD:
            return np.array([cfg.forward_speed_mps, 0.0]), cfg.forward_frames
        if action == PointNavAction.TURN_LEFT:
            return np.array([0.0, cfg.turn_speed_radps]), cfg.turn_frames
        if action == PointNavAction.TURN_RIGHT:
            return np.array([0.0, -cfg.turn_speed_radps]), cfg.turn_frames
        if action == PointNavAction.STAY:
            return np.array([0.0, 0.0]), cfg.stay_frames
        raise ValueError(f"Unknown point-navigation action: {action!r}")

    def step(self, actions: Sequence[int]) -> tuple[list[np.ndarray], list[float], list[bool], list[bool], dict[str, Any]]:
        """One simultaneous action per robot (ignored for robots that are done).

        Returns per-robot observations, rewards, terminated (goal or collision this step),
        truncated (episode limit this step, active robots only) and a team info dict.
        """
        if self._closed:
            raise RuntimeError("Environment is closed.")
        if not self._episode_active:
            raise RuntimeError("Call reset() before step().")
        if len(actions) != len(self.robots):
            raise ValueError(f"Need one action per robot ({len(self.robots)})")
        active = [r for r in self.robots if r.active]
        parsed = {r.index: PointNavAction(int(actions[r.index])) for r in active}
        if self.multi_config.park_teammate and 1 in parsed:
            parsed[1] = PointNavAction.STAY   # the parked robot never moves
        before = {r.index: self._distance_and_bearing(r, self._state(r))[0] for r in active}
        plans = {}
        for robot in active:
            command, frames = self._action_spec(parsed[robot.index])
            plans[robot.index] = (self._controller.forward(command), frames)
        total = max(frames for _, frames in plans.values())
        for frame in range(total):
            for robot in self.robots:
                wheel, frames = plans.get(robot.index, (self._stop_action, 0))
                robot.drive.apply_wheel_actions(wheel if frame < frames else self._stop_action)
            self._physics_step()
        self._episode_elapsed_sim_s += total * self._physics_dt_s
        self._apply_stop()
        self._step_frames(self.isaac_config.settle_frames, count_episode_time=True, hold_stop=True)
        self._episode_steps += 1

        states = {r.index: self._state(r) for r in self.robots}
        pnc, cfg = self.point_nav_config, self.lidar_config
        rewards = [0.0] * len(self.robots)
        terminated = [False] * len(self.robots)
        for robot in active:
            state = states[robot.index]
            robot.steps += 1
            after = self._distance_and_bearing(robot, state)[0]
            progress = before[robot.index] - after
            reward = pnc.progress_reward_scale * float(np.clip(progress, -pnc.progress_clip_m, pnc.progress_clip_m)) - pnc.step_penalty
            if after <= self.isaac_config.goal_tolerance_m:
                reward += pnc.success_bonus
                terminated[robot.index] = True
                robot.outcome = "SUCCESS"
            obstacle_distance = self._obstacle_surface_distance(state)
            teammate_distance = self._teammate_surface_distance(robot, states)
            robot.proximity_penalty = max(
                self._proximity_penalty(obstacle_distance),
                self._proximity_penalty(teammate_distance, self.multi_config.teammate_proximity_threshold_m))
            budget = self.multi_config.proximity_budget
            if budget > 0.0:
                robot.proximity_penalty = min(robot.proximity_penalty, max(budget - robot.proximity_total, 0.0))
                robot.proximity_total += robot.proximity_penalty
            reward -= robot.proximity_penalty
            action = parsed[robot.index]
            turning = action in (PointNavAction.TURN_LEFT, PointNavAction.TURN_RIGHT)
            robot.consecutive_turns = robot.consecutive_turns + 1 if turning else 0
            spin = cfg.spin_penalty if robot.consecutive_turns > cfg.spin_free_turns else 0.0
            if action == PointNavAction.STAY:
                spin += cfg.stay_penalty
            reward -= spin
            if not terminated[robot.index]:
                if obstacle_distance < cfg.robot_radius_m:
                    robot.collision_kind = "obstacle"
                elif teammate_distance < cfg.robot_radius_m:
                    robot.collision_kind = "robot"
                if robot.collision_kind is not None:
                    reward -= (self.multi_config.robot_collision_penalty if robot.collision_kind == "robot"
                               else self.multi_config.obstacle_collision_penalty or cfg.collision_penalty)
                    terminated[robot.index] = True
                    robot.outcome = "COLLISION"
            rewards[robot.index] = float(reward)
        weight = self.multi_config.team_reward_weight
        if weight:
            own = list(rewards)
            for robot in active:
                rewards[robot.index] = own[robot.index] + weight * sum(
                    own[o.index] for o in active if o is not robot)

        limit = (self._episode_steps >= pnc.max_episode_steps
                 or self._episode_elapsed_sim_s >= pnc.max_episode_sim_time_s)
        truncated = [False] * len(self.robots)
        for robot in active:
            if terminated[robot.index]:
                robot.active = False
            elif limit:
                truncated[robot.index] = True
                robot.active = False
                robot.outcome = "TIMEOUT"
        if not any(r.active for r in self.robots):
            self._episode_active = False
        observations = [self._observation(r, states[r.index]) for r in self.robots]
        return observations, rewards, terminated, truncated, self._info(states)

    def _info(self, states: Optional[dict[int, IsaacPoseState]] = None) -> dict[str, Any]:
        states = states or {r.index: self._state(r) for r in self.robots}
        robots = []
        for robot in self.robots:
            state = states[robot.index]
            distance, bearing = self._distance_and_bearing(robot, state)
            robots.append({
                "pose": {"x": state.x, "y": state.y}, "yaw": state.yaw,
                "goal": {"x": robot.goal[0], "y": robot.goal[1]},
                "distance_to_goal_m": distance, "relative_goal_bearing_rad": bearing,
                "clearances_m": {n: r["clearance_m"] for n, r in self.sector_readings(robot).items()},
                "active": robot.active, "outcome": robot.outcome, "collision_kind": robot.collision_kind,
                "proximity_penalty": robot.proximity_penalty, "steps": robot.steps,
                "stale_scan_reads": robot.stale_scan_reads,
            })
        return {
            "encounter": self.encounter,
            "robots": robots,
            "episode_steps": self._episode_steps,
            "episode_elapsed_sim_s": self._episode_elapsed_sim_s,
            "episode_done": not self._episode_active,
            "obstacles": [xy for xy in self._obstacle_xy if xy is not None],
            "obstacle_sizes_m": [s for xy, s in zip(self._obstacle_xy, self._obstacle_size) if xy is not None],
        }

    def sim_time(self) -> float:
        return float(self._simulation_context.current_time)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._apply_stop()
            for robot in self.robots:
                if robot.lidar is None:
                    continue
                try:
                    robot.lidar.pause()
                    robot.lidar.detach_all_annotators()
                    robot.lidar.detach_all_writers()
                except Exception:
                    pass
                robot.lidar = None
        finally:
            if self._simulation_app is not None:
                print("[TB03_MULTI] Closing Isaac Sim", flush=True)
                self._simulation_app.close()
                self._simulation_app = None
