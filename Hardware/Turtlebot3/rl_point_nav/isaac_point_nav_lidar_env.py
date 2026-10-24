"""V2 direct-Isaac TurtleBot point navigation with RTX LiDAR and pooled obstacles.

Builds on the frozen V1B :class:`IsaacPointNavEnv` (same Isaac startup, robot,
discrete actions, reset and progress reward) and adds:

* the validated RTX LiDAR: top-level Example_Rotary_2D, nearRangeM = minDistBetweenEchosM = 0.10 m, farRangeM = 200 m,
  following base_link by translation + normalized rotation only (offset
  (0.10, 0.0, 0.22), unit scale), synchronized before and after every physics
  step, using the returned FlatScan azimuth endpoints;
* a fixed obstacle pool created ONCE at startup under /World/obstacles and
  never deleted: reset only moves (teleports) existing prims;
* sector clearances (front / left / right) in the observation;
* geometric collision termination against the pooled obstacles;
* robot visual geometry hidden at runtime (physics/collision untouched) so the
  LiDAR does not see the ~4-7x scaled TurtleBot body. The GUI demo can choose
  visible visuals BEFORE physics initialization (--show-robot, presentation only).

Observation (6D):
    [d_norm, sin(bearing), cos(bearing), front, left, right]
    clearance = clip(min sector range / clearance_normalization_m, 0, 1)

V1B's 3D model cannot be used with this environment; V2 must be trained.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Optional, Sequence

from gymnasium import spaces
import gymnasium as gym
import numpy as np

try:  # Supports package and direct-script imports.
    from .config import PointNavAction, PointNavConfig
    from .isaac_point_nav_env import ARTICULATION_PATH, IsaacPointNavEnv, IsaacPoseState
except ImportError:  # pragma: no cover - direct-script convenience only
    from config import PointNavAction, PointNavConfig
    from isaac_point_nav_env import ARTICULATION_PATH, IsaacPointNavEnv, IsaacPoseState


BASE_LINK_PATH = f"{ARTICULATION_PATH}/a__namespace_base_link"
LIDAR_PATH = "/World/tb03_v2_lidar"
OBSTACLE_ROOT = "/World/obstacles"
GOAL_MARKER_PATH = "/World/tb03_v2_goal_marker"
ROBOT_PROXY_PATH = "/World/tb03_v2_robot_footprint"
# Original TurtleBot visual roots, restored by restore_robot_visuals_for_gui() (GUI demo only).
GUI_VISUAL_ROOTS = (
    f"{ARTICULATION_PATH}/a__namespace_base_link/visuals",
    f"{ARTICULATION_PATH}/a__namespace_base_scan/visuals",
    f"{ARTICULATION_PATH}/a__namespace_wheel_left_link/visuals",
    f"{ARTICULATION_PATH}/a__namespace_wheel_right_link/visuals",
)
LIDAR_CONFIG = "Example_Rotary_2D"
FLAT_SCAN_ANNOTATOR = "IsaacComputeRTXLidarFlatScan"
NEAR_RANGE_ATTRIBUTE = "omni:sensor:Core:nearRangeM"
LIDAR_RANGE_SETTINGS = {
    NEAR_RANGE_ATTRIBUTE: 0.10,
    "omni:sensor:Core:minDistBetweenEchosM": 0.10,
    "omni:sensor:Core:farRangeM": 200.0,
}
INVALID_RANGE_EPSILON_M = 1e-3

# Validated LiDAR constants (isaac_lidar_*_test.py). Do not change here.
LIDAR_MOUNT_OFFSET_M = (0.10, 0.0, 0.22)
LIDAR_NEAR_RANGE_M = 0.10



@dataclass(frozen=True)
class IsaacLidarNavConfig:
    """V2 LiDAR / obstacle / episode parameters."""

    # Sensor-frame sectors in degrees (CCW positive, 0 = robot forward).
    front_sector_deg: tuple[float, float] = (-30.0, 30.0)
    left_sector_deg: tuple[float, float] = (30.0, 120.0)
    right_sector_deg: tuple[float, float] = (-120.0, -30.0)
    clearance_normalization_m: float = 1.0

    # Kept for configuration compatibility; distance alone no longer masks rays.
    self_mask_max_range_m: float = 0.60
    self_mask_frames: int = 30
    scan_wait_max_frames: int = 600
    scan_stable_frames: int = 10
    # Extra rendered frames after a reset teleport so the scan reflects the new scene.
    reset_scan_frames: int = 6

    # Obstacle pool: axis-aligned boxes standing on the ground.
    obstacle_count: int = 4
    obstacle_size_xy_m: float = 0.12
    obstacle_height_m: float = 0.50
    park_origin_m: tuple[float, float] = (1000.0, 1000.0)  # far beyond farRangeM
    park_spacing_m: float = 5.0

    # Per-episode sampling.
    blocking_probability: float = 0.7    # one obstacle on the start-goal segment
    max_distractors: int = 2
    blocking_lateral_offset_m: float = 0.10
    min_obstacle_start_clearance_m: float = 0.40
    min_obstacle_goal_clearance_m: float = 0.30
    min_obstacle_spacing_m: float = 0.30
    distractor_margin_m: float = 0.40
    placement_attempts: int = 200

    # Collision: robot footprint radius vs obstacle box (TurtleBot3 Burger ~0.105 m).
    robot_radius_m: float = 0.105
    collision_penalty: float = 10.0

    # Hide the scaled robot's non-collision visual geometry (runtime only, never saved).
    hide_robot_visuals: bool = True
    robot_proxy_height_m: float = 0.02
    show_footprint_disc: bool = False  # optional GUI footprint disc; off by default


def _v2_point_nav_config() -> PointNavConfig:
    """V1B reward/timing, with a longer horizon for detours around obstacles."""
    return PointNavConfig(max_episode_steps=250, max_episode_sim_time_s=120.0)


# ---------------------------------------------------------------- validated LiDAR mount
# Copied from isaac_lidar_dynamic_follow_test.py (verified architecture).

def _wxyz(quaternion) -> tuple[float, float, float, float]:
    normalized = quaternion.GetNormalized()
    imaginary = normalized.GetImaginary()
    return (
        float(normalized.GetReal()),
        float(imaginary[0]),
        float(imaginary[1]),
        float(imaginary[2]),
    )


def _base_rigid_pose(stage, UsdGeom, Gf):
    """Return scale-free base pose and stage-unit mount offset."""
    base_link = stage.GetPrimAtPath(BASE_LINK_PATH)
    if not base_link.IsValid():
        raise RuntimeError(f"Missing TurtleBot base link: {BASE_LINK_PATH}")
    base_world = UsdGeom.XformCache().GetLocalToWorldTransform(base_link)
    _, _, inherited_scale, rotation_matrix, translation, _ = base_world.Factor()
    rotation = rotation_matrix.ExtractRotationQuat().GetNormalized()
    meters_per_unit = UsdGeom.GetStageMetersPerUnit(stage)
    if meters_per_unit <= 0.0:
        raise RuntimeError(f"Invalid stage meters-per-unit: {meters_per_unit}")
    offset_stage = Gf.Vec3d(*(component / meters_per_unit for component in LIDAR_MOUNT_OFFSET_M))
    return translation, rotation, inherited_scale, offset_stage, meters_per_unit


def _synchronize_lidar(lidar, stage, UsdGeom, Gf) -> None:
    """Synchronize top-level LiDAR from rigid base pose (no inherited scale)."""
    base_translation, base_rotation, _, offset_stage, _ = _base_rigid_pose(stage, UsdGeom, Gf)
    world_offset = Gf.Rotation(base_rotation).TransformDir(offset_stage)
    lidar.set_world_pose(
        position=np.asarray(base_translation + world_offset, dtype=np.float64),
        orientation=np.asarray(_wxyz(base_rotation), dtype=np.float64),
    )

    prim = stage.GetPrimAtPath(LIDAR_PATH)
    world = UsdGeom.XformCache().GetLocalToWorldTransform(prim)
    ok, _, scale, _, _, _ = world.Factor()
    if str(prim.GetParent().GetPath()) != "/World" or not ok or not np.allclose(scale, 1, atol=1e-6, rtol=0):
        raise RuntimeError("V2 LiDAR must remain top-level with unit world scale")


def _release_lidar(lidar, stage) -> None:
    """Copied from isaac_lidar_dynamic_follow_test.py."""
    if lidar is not None:
        try:
            lidar.pause()
        except Exception:
            pass
        for callback_name, method_name in (
            ("_acquisition_callback", "reset"),
            ("_stage_open_callback", "reset"),
            ("_timer_reset_callback", "unsubscribe"),
        ):
            callback = getattr(lidar, callback_name, None)
            if callback is not None:
                try:
                    getattr(callback, method_name)()
                except Exception:
                    pass
                setattr(lidar, callback_name, None)
        try:
            lidar.detach_all_annotators()
            lidar.detach_all_writers()
        except Exception:
            pass
        render_product = getattr(lidar, "_render_product", None)
        if render_product is not None:
            try:
                render_product.destroy()
            except Exception:
                pass
            lidar._render_product = None
    if stage is not None and stage.GetPrimAtPath(LIDAR_PATH).IsValid():
        stage.RemovePrim(LIDAR_PATH)


class _LidarSyncedSimulation:
    """Wraps SimulationContext so every V1B physics step keeps the LiDAR mounted.

    V1B steps with render=not headless; RTX LiDAR needs a rendered frame every
    step, so render is forced on. All other attributes delegate unchanged.
    """

    def __init__(self, simulation_context, sync) -> None:
        self._simulation_context = simulation_context
        self._sync = sync

    def step(self, render: bool = True) -> None:
        del render
        self._sync()
        self._simulation_context.step(render=True)
        self._sync()

    def __getattr__(self, name):
        return getattr(self._simulation_context, name)


# ---------------------------------------------------------------- environment

class IsaacPointNavLidarEnv(IsaacPointNavEnv):
    """V1B point navigation + validated RTX LiDAR sectors + pooled obstacles."""

    SECTOR_NAMES = ("front", "left", "right")

    def __init__(
        self,
        goal_x: float = 1.0,
        goal_y: float = 0.0,
        *,
        point_nav_config: Optional[PointNavConfig] = None,
        isaac_config=None,
        lidar_config: Optional[IsaacLidarNavConfig] = None,
        headless: bool = True,
    ) -> None:
        # Must exist before IsaacPointNavEnv.__init__ calls _initialize_isaac().
        self.lidar_config = lidar_config or IsaacLidarNavConfig()
        self._stage = None
        self._lidar = None
        self._flat_annotator = None
        self._sensor_azimuth_deg = None
        self._self_mask = None
        self._baseline_sector_valid: Optional[dict[str, int]] = None
        self._sector_masks = None
        self._last_ranges = None
        self._stale_scan_reads = 0
        self._ground_z = 0.0
        self._meters_per_unit = 1.0
        self._obstacle_paths: list[str] = []
        self._obstacle_xy: list[Optional[tuple[float, float]]] = []
        self._goal_marker = None
        self._robot_proxy = None
        # (path, had_authored_visibility, original_visibility) for every prim hidden at startup.
        self._hidden_visual_records: list[tuple[str, bool, Any]] = []
        self._last_collision = False

        super().__init__(
            goal_x,
            goal_y,
            point_nav_config=point_nav_config or _v2_point_nav_config(),
            isaac_config=isaac_config,
            headless=headless,
        )
        low = np.array([0.0, -1.0, -1.0] + [0.0] * len(self.SECTOR_NAMES), dtype=np.float32)
        high = np.ones(3 + len(self.SECTOR_NAMES), dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    # ------------------------------------------------------------ initialization

    def _initialize_isaac(self) -> None:
        # V1B closes SimulationApp on initialization errors. Emit the traceback
        # here, BEFORE shutdown can tear down Kit's stdout/stderr handlers.
        import sys
        import traceback

        try:
            self._initialize_v2_runtime()
        except Exception:
            print("[TB03_ENV_V2] INITIALIZATION FAILED (before Isaac shutdown):", flush=True)
            traceback.print_exc(file=sys.stdout)
            sys.stdout.flush()
            raise

    def _initialize_v2_runtime(self) -> None:
        # Called by V1B after SimulationApp opens the stage, but BEFORE it
        # constructs SimulationContext or any articulation/tensor views.
        import omni.usd
        import omni.timeline

        context = omni.usd.get_context()
        while context.get_stage_loading_status()[2] > 0:
            self._simulation_app.update()
        self._stage = context.get_stage()
        if self._stage is None:
            raise RuntimeError("No loaded stage for pre-physics visual configuration")
        omni.timeline.get_timeline_interface().stop()
        if self.lidar_config.hide_robot_visuals:
            self._hide_robot_visuals()
        else:
            self.restore_robot_visuals_for_gui()
        super()._initialize_isaac()
        self._initialize_v2_scene()

    def _initialize_v2_scene(self) -> None:
        import omni.usd
        from pxr import Gf, Usd, UsdGeom
        from isaacsim.sensors.rtx import LidarRtx

        stage = omni.usd.get_context().get_stage()
        self._stage = stage
        self._meters_per_unit = UsdGeom.GetStageMetersPerUnit(stage)
        for path in (LIDAR_PATH, OBSTACLE_ROOT, GOAL_MARKER_PATH, ROBOT_PROXY_PATH):
            if stage.GetPrimAtPath(path).IsValid():
                raise RuntimeError(f"V2 prim path already exists in the scene: {path}")

        if not math.isfinite(self._meters_per_unit) or self._meters_per_unit <= 0:
            raise RuntimeError(f"Invalid stage units: {self._meters_per_unit}")
        # Hidden robot visuals can yield an EMPTY visibility-filtered BBox with
        # a finite ~3.4e38 minimum. Never use a rendered robot bound as a floor.
        self._ground_z = self._measure_floor_z()
        print(f"[TB03_ENV_V2] floor_z={self._ground_z:.6f} m", flush=True)

        # 1) Obstacle pool: created once, parked, never deleted (validated MOVE_EXISTING pattern).
        UsdGeom.Xform.Define(stage, OBSTACLE_ROOT)
        cfg = self.lidar_config
        for index in range(cfg.obstacle_count):
            path = f"{OBSTACLE_ROOT}/obstacle_{index}"
            cube = UsdGeom.Cube.Define(stage, path)
            cube.CreateSizeAttr(1.0)
            xform = UsdGeom.XformCommonAPI(cube.GetPrim())
            xform.SetScale(
                Gf.Vec3f(
                    cfg.obstacle_size_xy_m / self._meters_per_unit,
                    cfg.obstacle_size_xy_m / self._meters_per_unit,
                    cfg.obstacle_height_m / self._meters_per_unit,
                )
            )
            cube.CreateDisplayColorAttr([Gf.Vec3f(0.85, 0.35, 0.10)])
            self._obstacle_paths.append(path)
            self._obstacle_xy.append(None)
            self._park_obstacle(index)
        print(f"[TB03_ENV_V2] obstacle pool: {cfg.obstacle_count} prims under {OBSTACLE_ROOT}", flush=True)

        # Goal marker (GUI only): flat disc on the ground, below the LiDAR plane.
        if not self._headless:
            marker = UsdGeom.Cylinder.Define(stage, GOAL_MARKER_PATH)
            marker.CreateRadiusAttr(0.05 / self._meters_per_unit)
            marker.CreateHeightAttr(0.005 / self._meters_per_unit)
            marker.CreateAxisAttr(UsdGeom.Tokens.z)
            marker.CreateDisplayColorAttr([Gf.Vec3f(0.1, 0.8, 0.2)])
            self._goal_marker = marker
        # Optional robot footprint disc (true Burger radius), on the ground below the LiDAR plane.
        if not self._headless and self.lidar_config.show_footprint_disc:
            proxy = UsdGeom.Cylinder.Define(stage, ROBOT_PROXY_PATH)
            proxy.CreateRadiusAttr(self.lidar_config.robot_radius_m / self._meters_per_unit)
            proxy.CreateHeightAttr(self.lidar_config.robot_proxy_height_m / self._meters_per_unit)
            proxy.CreateAxisAttr(UsdGeom.Tokens.z)
            proxy.CreateDisplayColorAttr([Gf.Vec3f(0.15, 0.35, 0.90)])
            self._robot_proxy = proxy
            self._move_robot_proxy(self._state())

        # 2) Validated LiDAR.
        self._lidar = LidarRtx(
            prim_path=LIDAR_PATH,
            name="tb03_v2_lidar",
            config_file_name=LIDAR_CONFIG,
            **LIDAR_RANGE_SETTINGS,
        )
        # LidarRtx owns an independent Hydra render product. Headless V1B
        # disables only the display viewport; explicitly keep the sensor active.
        product = self._lidar._render_product
        if product is None or not self._lidar.get_render_product_path():
            raise RuntimeError("LiDAR render product was not created")
        product.hydra_texture.set_updates_enabled(True)
        print(f"[TB03_ENV_V2] sensor render product enabled: {self._lidar.get_render_product_path()} "
              f"headless={self._headless}", flush=True)
        self._lidar.initialize()
        self._lidar.set_local_scale(np.ones(3))
        prim = stage.GetPrimAtPath(self._lidar.prim_path)
        for name, expected in LIDAR_RANGE_SETTINGS.items():
            value = prim.GetAttribute(name).Get()
            if value is None or not np.isclose(value, expected, atol=1e-6, rtol=0):
                raise RuntimeError(f"LiDAR setting not applied: {name}={value}")
            print(f"[TB03_ENV_V2] {name}={value}", flush=True)
        self._sync_lidar()
        self._lidar.attach_annotator(FLAT_SCAN_ANNOTATOR)
        self._flat_annotator = self._lidar.get_annotators()[FLAT_SCAN_ANNOTATOR]

        # From here on every V1B physics step keeps the LiDAR mounted and renders.
        self._simulation_context = _LidarSyncedSimulation(self._simulation_context, self._sync_lidar)

        self._wait_for_scan()
        self._build_self_mask()

    def _sync_lidar(self) -> None:
        from pxr import Gf, UsdGeom

        _synchronize_lidar(self._lidar, self._stage, UsdGeom, Gf)

    def _wait_for_scan(self) -> None:
        cfg = self.lidar_config
        accepted, last_size = 0, None
        last_status = None
        for frame in range(1, cfg.scan_wait_max_frames + 1):
            # The synced context calls step(render=True) -> Kit app.update().
            # This drives the sensor render product without a display viewport.
            # Do not also call app.update(): that could advance physics twice.
            self._step_frames(1, count_episode_time=False, hold_stop=True)
            data = self._flat_annotator.get_data()
            if not isinstance(data, dict):
                continue
            # Use ONE annotator snapshot for both payload and metadata. Empty
            # inter-scan updates are normal for a rotating RTX sensor.
            ranges = np.asarray(data.get("linearDepthData", []), dtype=np.float64).reshape(-1).copy()
            size = ranges.size
            cols = int(data.get("numCols", 0))
            bounds = np.asarray(data.get("azimuthRange", []), dtype=float).reshape(-1)
            depth = np.asarray(data.get("depthRange", []), dtype=float).reshape(-1)
            last_status = {"numCols": cols, "samples": size, "depthRange": depth.tolist()}
            complete = (cols > 0 and cols == size and bounds.size == 2
                        and np.isfinite(bounds).all() and depth.size == 2)
            if complete:
                if not np.allclose(depth, [0.10, 200.0], atol=1e-5, rtol=0):
                    raise RuntimeError(f"FlatScan depthRange does not match configuration: {depth}")
                valid = np.isfinite(ranges) & (ranges > INVALID_RANGE_EPSILON_M) & (ranges >= 0.10) & (ranges <= 200.0)
                if valid.any():
                    accepted = accepted + 1 if size == last_size else 1
                    last_size = size
                    if accepted >= cfg.scan_stable_frames:
                        self._last_ranges = np.where(valid, ranges, np.nan)
                        self._sensor_azimuth_deg = (np.linspace(bounds[0], bounds[1], size) + 180) % 360 - 180
                        self._sector_masks = {
                            name: self._angle_mask(getattr(cfg, f"{name}_sector_deg")) for name in self.SECTOR_NAMES
                        }
                        print(f"[TB03_ENV_V2] LiDAR ready: rays={size} rendered_updates={frame} "
                              f"accepted_scans={accepted} "
                              + " ".join(f"{n}_rays={int(m.sum())}" for n, m in self._sector_masks.items()), flush=True)
                        return
            if frame % 60 == 0:
                print(f"[TB03_ENV_V2] LiDAR warm-up update={frame} "
                      f"accepted={accepted} last={last_status}", flush=True)
        raise RuntimeError(f"RTX LiDAR initialization timed out after {cfg.scan_wait_max_frames} "
                           f"rendered updates; accepted={accepted}, last={last_status}")

    def _angle_mask(self, sector: tuple[float, float]) -> np.ndarray:
        lo, hi = sector
        return (self._sensor_azimuth_deg >= lo) & (self._sensor_azimuth_deg <= hi)

    def _build_self_mask(self) -> None:
        """Measure valid-ray baselines with all obstacles parked; no range mask."""
        cfg = self.lidar_config
        hits = np.zeros(self._sensor_azimuth_deg.size, dtype=int)
        scans = []
        for _ in range(cfg.self_mask_frames):
            self._step_frames(1, count_episode_time=False, hold_stop=True)
            ranges = self._flat_ranges()
            if ranges.size != hits.size:
                continue
            scans.append(ranges)
            with np.errstate(invalid="ignore"):
                hits += np.isfinite(ranges).astype(int)
        if not scans:
            raise RuntimeError("No LiDAR scans available to build the self-return mask.")
        # Visual hiding handles self-returns. Never suppress genuine near
        # obstacles just because a baseline ray was shorter than 0.60 m.
        self._self_mask = np.zeros(hits.size, dtype=bool)
        # Baseline: how many self-filtered valid rays each sector normally sees
        # (all obstacles parked). Used to tell "too close to measure" from "open space".
        self._baseline_sector_valid = {
            name: max(int((np.isfinite(scan) & mask & ~self._self_mask).sum()) for scan in scans)
            for name, mask in self._sector_masks.items()
        }
        print(
            "[TB03_ENV_V2] baseline sector valid rays: "
            + " ".join(f"{name}={count}" for name, count in self._baseline_sector_valid.items()),
            flush=True,
        )
        angles = self._sensor_azimuth_deg[self._self_mask]
        print(
            f"[TB03_ENV_V2] self-return mask: {int(self._self_mask.sum())} rays < "
            f"{cfg.self_mask_max_range_m} m"
            + ("" if angles.size == 0 else f" (sensor angles {angles.min():.1f}..{angles.max():.1f} deg)"),
            flush=True,
        )

    # ------------------------------------------------------------ LiDAR observation

    def _flat_ranges(self) -> np.ndarray:
        data = self._flat_annotator.get_data()
        if not isinstance(data, dict):
            return np.zeros(0)
        ranges = np.asarray(data.get("linearDepthData", []), dtype=np.float64).reshape(-1)
        return np.where(np.isfinite(ranges) & (ranges > INVALID_RANGE_EPSILON_M) & (ranges >= 0.10) & (ranges <= 200.0), ranges, np.nan)

    def _current_ranges(self) -> np.ndarray:
        ranges = self._flat_ranges()
        if self._sensor_azimuth_deg is not None and ranges.size == self._sensor_azimuth_deg.size:
            self._last_ranges = ranges
        else:
            self._stale_scan_reads += 1
        if self._last_ranges is None:
            raise RuntimeError("No valid LiDAR scan has been received yet.")
        return self._last_ranges

    def sector_readings(self) -> dict[str, dict[str, Any]]:
        """Per sector: valid ray count, invalid flag and clearance in meters.

        valid rays > 0                    -> clearance = minimum valid range
        valid rays == 0, baseline had rays -> INVALID: obstacle closer than the
                                             LiDAR near range; clearance = 0.0 (blocked)
        valid rays == 0, baseline had none -> open space; clearance = inf
        """
        ranges = self._current_ranges()
        readings = {}
        for name, mask in self._sector_masks.items():
            values = ranges[mask & ~self._self_mask]
            values = values[np.isfinite(values)]
            invalid = values.size == 0 and self._baseline_sector_valid.get(name, 0) > 0
            if values.size:
                clearance = float(values.min())
            else:
                clearance = 0.0 if invalid else math.inf
            readings[name] = {"valid_rays": int(values.size), "invalid": invalid, "clearance_m": clearance}
        return readings

    def sector_clearances_m(self) -> dict[str, float]:
        """Clearance per sector in meters (0.0 = blocked/too close, inf = open space)."""
        return {name: reading["clearance_m"] for name, reading in self.sector_readings().items()}

    def _observation(self, state: IsaacPoseState) -> np.ndarray:
        base = super()._observation(state)
        norm = self.lidar_config.clearance_normalization_m
        readings = self.sector_readings()
        lidar = [float(np.clip(readings[name]["clearance_m"] / norm, 0.0, 1.0)) for name in self.SECTOR_NAMES]
        if any(reading["invalid"] for reading in readings.values()):
            print(
                "[TB03_ENV_V2] sector diagnostics: "
                + " ".join(
                    f"{name}(valid={r['valid_rays']} invalid={r['invalid']} clearance={r['clearance_m']:.3f})"
                    for name, r in readings.items()
                ),
                flush=True,
            )
        return np.concatenate([base, np.asarray(lidar, dtype=np.float32)]).astype(np.float32)

    def _info(self, state: IsaacPoseState, **kwargs) -> dict[str, Any]:
        info = super()._info(state, **kwargs)
        readings = self.sector_readings()
        info["clearances_m"] = {name: r["clearance_m"] for name, r in readings.items()}
        info["lidar_sectors"] = readings
        info["collision"] = self._last_collision
        info["obstacles"] = [xy for xy in self._obstacle_xy if xy is not None]
        info["stale_scan_reads"] = self._stale_scan_reads
        return info

    # ------------------------------------------------------------ obstacles

    def _measure_floor_z(self) -> float:
        """Read the supporting collision surface below the start base (no edits)."""
        from omni.physx import get_physx_scene_query_interface
        from pxr import Gf, UsdGeom

        base = UsdGeom.XformCache().GetLocalToWorldTransform(
            self._stage.GetPrimAtPath(BASE_LINK_PATH)
        ).ExtractTranslation()
        units = self._meters_per_unit
        if not all(math.isfinite(float(v)) and abs(float(v) * units) < 1e12 for v in base):
            raise RuntimeError(f"Invalid base pose for floor query: {base}")
        hits = []

        def report(hit):
            paths = (str(hit.collision), str(hit.rigid_body))
            if any(path == ARTICULATION_PATH or path.startswith(ARTICULATION_PATH + "/") for path in paths):
                return True
            z = float(hit.position.z)
            # Accept only upward-facing support at/below the base, never a
            # ceiling or geometry above it. Distances are in stage units.
            if math.isfinite(z) and z <= float(base[2]) + 1e-4 / units and float(hit.normal.z) > 0.5:
                hits.append((z, str(hit.collision)))
            return True

        origin = Gf.Vec3f(float(base[0]), float(base[1]), float(base[2]) + 0.5 / units)
        get_physx_scene_query_interface().raycast_all(
            origin, Gf.Vec3f(0, 0, -1), 5.0 / units, report
        )
        if not hits:
            raise RuntimeError("No supporting floor found below the robot; refusing invalid Z placement")
        z, path = max(hits, key=lambda hit: hit[0])
        floor_z = z * units
        if not math.isfinite(floor_z) or abs(floor_z) >= 1e12:
            raise RuntimeError(f"Invalid measured floor Z: {floor_z}")
        print(f"[TB03_ENV_V2] floor collider={path} z={floor_z:.6f} m", flush=True)
        return floor_z

    def _placement_xyz(self, x: float, y: float, z: float):
        """Validate meter coordinates before converting to a USD translation."""
        from pxr import Gf

        coords = (float(x), float(y), float(z))
        # Reject finite empty-range sentinels too, not just NaN/infinity.
        if not all(math.isfinite(v) and abs(v) < 1e12 for v in coords):
            raise ValueError(f"Invalid V2 placement coordinates (meters): {coords}")
        units = self._meters_per_unit
        if not math.isfinite(units) or units <= 0:
            raise ValueError(f"Invalid stage units: {units}")
        local = tuple(v / units for v in coords)
        if not all(math.isfinite(v) and abs(v) < 1e12 for v in local):
            raise ValueError(f"Invalid V2 placement coordinates (stage units): {local}")
        return Gf.Vec3d(*local)

    def _set_obstacle_xy(self, index: int, xy: tuple[float, float]) -> None:
        from pxr import Gf, UsdGeom

        z = self._ground_z + 0.5 * self.lidar_config.obstacle_height_m
        prim = self._stage.GetPrimAtPath(self._obstacle_paths[index])
        UsdGeom.XformCommonAPI(prim).SetTranslate(
            self._placement_xyz(xy[0], xy[1], z)
        )

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
                self._set_obstacle_xy(index, xy)
                self._obstacle_xy[index] = xy
            else:
                self._park_obstacle(index)

    def _sample_obstacles(self, start: tuple[float, float], goal: tuple[float, float]) -> list[tuple[float, float]]:
        cfg = self.lidar_config
        rng = self.np_random
        placed: list[tuple[float, float]] = []

        def valid(p) -> bool:
            return (
                math.dist(p, start) >= cfg.min_obstacle_start_clearance_m
                and math.dist(p, goal) >= cfg.min_obstacle_goal_clearance_m
                and all(math.dist(p, q) >= cfg.min_obstacle_spacing_m for q in placed)
            )

        dx, dy = goal[0] - start[0], goal[1] - start[1]
        length = math.hypot(dx, dy)
        lo = cfg.min_obstacle_start_clearance_m
        hi = length - cfg.min_obstacle_goal_clearance_m
        if length > 0.0 and hi > lo and rng.random() < cfg.blocking_probability:
            ux, uy = dx / length, dy / length
            for _ in range(cfg.placement_attempts):
                s = rng.uniform(lo, hi)
                lateral = rng.uniform(-cfg.blocking_lateral_offset_m, cfg.blocking_lateral_offset_m)
                p = (start[0] + s * ux - lateral * uy, start[1] + s * uy + lateral * ux)
                if valid(p):
                    placed.append(p)
                    break

        n_distractors = int(rng.integers(0, cfg.max_distractors + 1))
        n_distractors = min(n_distractors, cfg.obstacle_count - len(placed))
        x_lo, x_hi = min(start[0], goal[0]) - cfg.distractor_margin_m, max(start[0], goal[0]) + cfg.distractor_margin_m
        y_lo, y_hi = min(start[1], goal[1]) - cfg.distractor_margin_m, max(start[1], goal[1]) + cfg.distractor_margin_m
        for _ in range(n_distractors):
            for _ in range(cfg.placement_attempts):
                p = (rng.uniform(x_lo, x_hi), rng.uniform(y_lo, y_hi))
                if valid(p):
                    placed.append(p)
                    break
        return placed

    def _collides(self, state: IsaacPoseState) -> bool:
        """Robot footprint circle vs axis-aligned obstacle boxes (ground truth)."""
        half = 0.5 * self.lidar_config.obstacle_size_xy_m
        for xy in self._obstacle_xy:
            if xy is None:
                continue
            dx = max(abs(state.x - xy[0]) - half, 0.0)
            dy = max(abs(state.y - xy[1]) - half, 0.0)
            if math.hypot(dx, dy) < self.lidar_config.robot_radius_m:
                return True
        return False

    def _configure_robot_visuals(self, visible: bool) -> int:
        """Only the four validated visual roots, before any physics views exist."""
        from pxr import Usd, UsdGeom, UsdPhysics

        if self._simulation_context is not None:
            raise RuntimeError("Visual visibility is fixed at startup; start a new process to change it")
        roots = []
        for path in GUI_VISUAL_ROOTS:
            prim = self._stage.GetPrimAtPath(path)
            if not prim.IsValid() or prim.IsInstanceProxy():
                raise RuntimeError(f"Visual root missing or not editable: {path}")
            subtree = list(Usd.PrimRange(prim, Usd.TraverseInstanceProxies()))
            if any(p.HasAPI(UsdPhysics.CollisionAPI)
                   or p.HasAPI(UsdPhysics.RigidBodyAPI)
                   or p.HasAPI(UsdPhysics.ArticulationRootAPI)
                   or p.IsA(UsdPhysics.Joint)
                   or "collision" in str(p.GetPath()).lower() for p in subtree):
                raise RuntimeError(f"Refusing to change a visual root containing physics/collision: {path}")
            roots.append(prim)
        # Validate every root first; no physics, transform or asset edits.
        for prim in roots:
            attr = UsdGeom.Imageable(prim).GetVisibilityAttr()
            self._hidden_visual_records.append((str(prim.GetPath()), attr.HasAuthoredValue(), attr.Get()))
            attr.Set(UsdGeom.Tokens.inherited if visible else UsdGeom.Tokens.invisible)
            computed = UsdGeom.Imageable(prim).ComputeVisibility()
            if (computed != UsdGeom.Tokens.invisible) != visible:
                raise RuntimeError(f"Visibility composition mismatch: {prim.GetPath()}")
            print(f"[TB03_ENV_V2] pre-physics visual {prim.GetPath()}={computed}", flush=True)
        return len(roots)

    def _hide_robot_visuals(self) -> None:
        self._configure_robot_visuals(visible=False)

    @property
    def self_mask_ray_count(self) -> int:
        return int(self._self_mask.sum()) if self._self_mask is not None else 0

    def restore_robot_visuals_for_gui(self) -> int:
        """Startup-only presentation setting; never recompose live physics views."""
        print("[TB03_ENV_V2] WARNING: visible robot changes LiDAR perception; "
              "PRESENTATION ONLY, not quantitative evaluation", flush=True)
        return self._configure_robot_visuals(visible=True)

    def force_render_updates(self, updates: int) -> None:
        """GUI: pump the Kit app loop, then re-sync the LiDAR with rendered stop-steps."""
        for _ in range(updates):
            self._simulation_app.update()
        self.settle_scan(2)

    def rehide_robot_visuals(self) -> int:
        """Startup only; training never toggles visuals."""
        return self._configure_robot_visuals(visible=False)

    def settle_scan(self, frames: int) -> None:
        """Advance rendered frames with zero wheel velocity (no episode time)."""
        self._step_frames(frames, count_episode_time=False, hold_stop=True)

    def lidar_observation(self) -> np.ndarray:
        """Current 3 LiDAR entries of the observation (front, left, right)."""
        return self._observation(self._state())[3:]

    def _move_robot_proxy(self, state: IsaacPoseState) -> None:
        if self._robot_proxy is None:
            return
        from pxr import Gf, UsdGeom

        UsdGeom.XformCommonAPI(self._robot_proxy.GetPrim()).SetTranslate(
            self._placement_xyz(
                state.x, state.y, self._ground_z + 0.5 * self.lidar_config.robot_proxy_height_m
            )
        )

    def _move_goal_marker(self) -> None:
        if self._goal_marker is None:
            return
        from pxr import Gf, UsdGeom

        UsdGeom.XformCommonAPI(self._goal_marker.GetPrim()).SetTranslate(
            self._placement_xyz(self._goal_x, self._goal_y, self._ground_z + 0.005)
        )

    # ------------------------------------------------------------ gym API

    def _reset_articulation(self) -> IsaacPoseState:
        super()._reset_articulation()
        # Let the RTX scan catch up with the teleported robot and obstacles.
        self._step_frames(self.lidar_config.reset_scan_frames, count_episode_time=False, hold_stop=True)
        return self._state()

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset robot (V1B start pose), goal and obstacle pool.

        options:
            goal:       (x, y) explicit goal; default: V1B goal bank
            obstacles:  list of (x, y) world positions; default: random sampling;
                        [] parks every obstacle
        """
        if self._closed:
            raise RuntimeError("Environment is closed.")
        gym.Env.reset(self, seed=seed)
        options = {} if options is None else dict(options)
        unknown = set(options) - {"goal", "obstacles"}
        if unknown:
            raise ValueError(f"Unsupported V2 reset option(s): {sorted(unknown)}")

        goal = tuple(options["goal"]) if "goal" in options else self._sample_v1_goal()
        start = (float(self._default_position_host[0, 0]), float(self._default_position_host[0, 1]))
        obstacles = options["obstacles"] if "obstacles" in options else self._sample_obstacles(start, goal)
        self._place_obstacles(obstacles)
        self._last_collision = False
        self._set_goal(goal)
        self._move_goal_marker()

        # V1B reset: robot pose, settle, goal already fixed (seed consumed above).
        observation, info = super().reset(seed=None, options={"goal": goal})
        self._move_robot_proxy(self._state())
        print(
            f"[TB03_ENV_V2] obstacles={[(round(x, 3), round(y, 3)) for x, y in info['obstacles']]} "
            f"clearances={ {k: round(v, 3) for k, v in info['clearances_m'].items()} } "
            f"valid_rays={ {k: r['valid_rays'] for k, r in info['lidar_sectors'].items()} } "
            f"invalid={ {k: r['invalid'] for k, r in info['lidar_sectors'].items()} }",
            flush=True,
        )
        return observation, info

    def step(self, action: PointNavAction | int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        observation, reward, terminated, truncated, info = super().step(action)
        state = self._state()
        self._move_robot_proxy(state)
        if not terminated and self._collides(state):
            self._last_collision = True
            reward -= self.lidar_config.collision_penalty
            terminated, truncated = True, False
            self._episode_active = False
            info["collision"] = True
        return observation, float(reward), terminated, truncated, info

    def close(self) -> None:
        if not self._closed and self._lidar is not None:
            try:
                _release_lidar(self._lidar, self._stage)
            finally:
                self._lidar = None
        super().close()
