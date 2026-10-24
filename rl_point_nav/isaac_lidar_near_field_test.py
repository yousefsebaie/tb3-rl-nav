#!/usr/bin/env python3
"""Runtime-only RTX LiDAR near-field / orientation diagnostic.

One temporary top-level Example_Rotary_2D LiDAR with the verified runtime
override omni:sensor:Core:nearRangeM = 0.10 m is measured in three
environments with identical temporary walls:

  EMPTY          sensor pinned far from the scene (50, 50, 1), identity rotation
  ROBOT_VISIBLE  frozen follow architecture: base-link offset (0.10, 0.0, 0.22),
                 rigid translation + normalized rotation, unit scale
  ROBOT_HIDDEN   same follow pose, TurtleBot made invisible (restored afterwards)

Walls are placed in the SENSOR frame at +X, -X, +Y, -Y and 0.20 / 0.30 / 0.50 m
(1.00 m as a positive control). Comparisons answer:

  1. azimuth convention ("looking backwards?")  -> EMPTY +X / +Y wall azimuths
  2. robot blocking the near field?              -> ROBOT_VISIBLE vs ROBOT_HIDDEN
  3. sensor rotation wrong on the robot?         -> ROBOT_HIDDEN vs EMPTY azimuths

Nothing is saved; the NVIDIA asset and the scene USD are never modified.
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path


SCENE_PATH = Path("/home/g8/Desktop/FInalSceneDemo.usd")
ARTICULATION_PATH = "/World/turtlebot3_burger_02"
BASE_LINK_PATH = f"{ARTICULATION_PATH}/a__namespace_base_link"
TEMP_SENSOR_PATH = "/World/tb03_near_field_lidar"
TEMP_WALL_PATH = "/World/tb03_near_field_wall"
LEFT_WHEEL_DOF = "a__namespace_wheel_left_joint"
RIGHT_WHEEL_DOF = "a__namespace_wheel_right_joint"
LIDAR_CONFIG = "Example_Rotary_2D"
POINT_CLOUD_ANNOTATOR = "IsaacExtractRTXSensorPointCloudNoAccumulator"
FLAT_SCAN_ANNOTATOR = "IsaacComputeRTXLidarFlatScan"
# base_scan frame expressed in the scale-free base_link frame (isaac_robot_geometry_inspect.py).
# Override without editing:  --offset X Y Z   (e.g. --offset 0.10 0.0 0.22 for the previous mount)
DEFAULT_MOUNT_OFFSET_M = (-0.143, 0.0, 0.685)
_arg_parser = argparse.ArgumentParser(add_help=False)
_arg_parser.add_argument("--offset", type=float, nargs=3, metavar=("X", "Y", "Z"), default=DEFAULT_MOUNT_OFFSET_M)
_args = _arg_parser.parse_known_args()[0]
MOUNT_OFFSET_M = tuple(_args.offset)
NEAR_RANGE_ATTRIBUTE = "omni:sensor:Core:nearRangeM"  # identified by isaac_lidar_range_test.py
FAR_RANGE_ATTRIBUTE = "omni:sensor:Core:farRangeM"
NEAR_RANGE_M = 0.10
WHEEL_RADIUS_M = 0.033
WHEEL_BASE_M = 0.160
SETTLE_FRAMES = 8
INVALID_RANGE_EPSILON_M = 1e-3

EMPTY_POSITION_M = (50.0, 50.0, 1.0)  # proven clean area (isaac_lidar_scene_control_test.py)
ENVIRONMENTS = ("EMPTY", "ROBOT_VISIBLE", "ROBOT_HIDDEN")
DIRECTIONS = {"+X": 0.0, "-X": 180.0, "+Y": 90.0, "-Y": -90.0}  # yaw in sensor frame, deg
WALL_DISTANCES_M = (0.20, 0.30, 0.50, 1.00)
WALL_THICKNESS_M = 0.02
WALL_WIDTH_M = 0.30
WALL_HEIGHT_M = 0.60
WALL_BAND_M = 0.05
WALL_MIN_EXTRA_RAYS = 3
BASELINE_FRAMES = 60
BASELINE_DISCARD_FRAMES = 15
WALL_FRAMES = 45
WALL_FLUSH_FRAMES = 15
ROBOT_GEOMETRY_REPORT_RADIUS_M = 0.60
TAG = "[TB03_LIDAR_NEAR]"


from isaacsim import SimulationApp


simulation_app = SimulationApp(
    {
        "headless": False,
        "open_usd": str(SCENE_PATH),
        "create_new_stage": False,
        "fast_shutdown": False,
        "disable_viewport_updates": False,
    }
)


def log(message: str) -> None:
    print(f"{TAG} {message}", flush=True)


def _fmt_m(value) -> str:
    return "n/a" if value is None else f"{value:.4f} m"


def _fmt_deg(value) -> str:
    return "n/a" if value is None else f"{value:.2f} deg"


@dataclass(frozen=True)
class WallResult:
    environment: str
    direction: str
    distance_m: float
    wall_position: tuple[float, float, float]
    point_count: int
    finite_valid: int
    flat_min: float | None
    closest_angle_deg: float | None
    band_rays: int
    baseline_band_rays: int
    band_azimuth_deg: float | None

    @property
    def detected(self) -> bool:
        return self.band_rays >= self.baseline_band_rays + WALL_MIN_EXTRA_RAYS


# ---------------------------------------------------------------- helpers
# Pose / follow / release helpers are copied from isaac_lidar_dynamic_follow_test.py
# (verified architecture), parameterized only by sensor path.

def _wxyz(quaternion) -> tuple[float, float, float, float]:
    normalized = quaternion.GetNormalized()
    imaginary = normalized.GetImaginary()
    return (
        float(normalized.GetReal()),
        float(imaginary[0]),
        float(imaginary[1]),
        float(imaginary[2]),
    )


def _yaw_from_quat(quaternion) -> float:
    w, x, y, z = _wxyz(quaternion)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _point_count(data) -> int:
    if not isinstance(data, dict):
        return 0
    points = data.get("data")
    if points is None:
        return 0
    shape = getattr(points, "shape", None)
    if shape is not None and len(shape) >= 1:
        return int(shape[0])
    try:
        return len(points)
    except TypeError:
        return 0


def _flat_ranges(data):
    import numpy as np

    if not isinstance(data, dict):
        return np.zeros(0)
    return np.asarray(data.get("linearDepthData", []), dtype=np.float64).reshape(-1)


def _valid_mask(ranges):
    import numpy as np

    return np.isfinite(ranges) & (ranges > INVALID_RANGE_EPSILON_M)


def _azimuths_deg(data, size: int):
    """Per-ray azimuth as azimuthRange[0] + index * horizontalResolution."""
    import numpy as np

    if not isinstance(data, dict) or data.get("azimuthRange") is None or data.get("horizontalResolution") is None:
        return None
    return float(data["azimuthRange"][0]) + np.arange(size) * float(data["horizontalResolution"])


def _closest_return(data):
    """(distance, azimuth_deg, index) of the nearest valid FlatScan ray."""
    import numpy as np

    ranges = _flat_ranges(data)
    valid = _valid_mask(ranges)
    if not valid.any():
        return None, None, None
    index = int(np.argmin(np.where(valid, ranges, np.inf)))
    azimuths = _azimuths_deg(data, ranges.size)
    return float(ranges[index]), None if azimuths is None else float(azimuths[index]), index


def _band_indices(data, distance_m: float):
    import numpy as np

    ranges = _flat_ranges(data)
    return np.nonzero(_valid_mask(ranges) & (np.abs(ranges - distance_m) <= WALL_BAND_M))[0]


def _band_azimuth_deg(data, distance_m: float) -> float | None:
    """Circular mean azimuth of in-band rays."""
    import numpy as np

    indices = _band_indices(data, distance_m)
    azimuths = _azimuths_deg(data, _flat_ranges(data).size)
    if indices.size == 0 or azimuths is None:
        return None
    radians = np.deg2rad(azimuths[indices])
    return float(np.rad2deg(math.atan2(np.sin(radians).mean(), np.cos(radians).mean())))


def _finite_valid(data) -> int:
    return int(_valid_mask(_flat_ranges(data)).sum())


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
    offset_stage = Gf.Vec3d(*(component / meters_per_unit for component in MOUNT_OFFSET_M))
    return translation, rotation, inherited_scale, offset_stage, meters_per_unit


def _release_lidar(lidar, sensor_path, stage) -> None:
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
    if stage.GetPrimAtPath(sensor_path).IsValid():
        stage.RemovePrim(sensor_path)


# ---------------------------------------------------------------- sensor rig

class SensorRig:
    """Temporary top-level LiDAR that either follows the robot or is pinned."""

    def __init__(self, ctx):
        from isaacsim.sensors.rtx import LidarRtx

        self.ctx = ctx
        self.mode = "EMPTY"
        self.max_offset_error_m = 0.0
        self.scale_is_unit = True
        self.lidar = LidarRtx(
            prim_path=TEMP_SENSOR_PATH,
            name="tb03_near_field_lidar",
            config_file_name=LIDAR_CONFIG,
            **{NEAR_RANGE_ATTRIBUTE: NEAR_RANGE_M},
        )
        self.lidar.initialize()
        self.prim = ctx["stage"].GetPrimAtPath(self.lidar.prim_path)
        self._sync()
        self.lidar.attach_annotator(POINT_CLOUD_ANNOTATOR)
        self.lidar.attach_annotator(FLAT_SCAN_ANNOTATOR)
        annotators = self.lidar.get_annotators()
        self.point_annotator = annotators[POINT_CLOUD_ANNOTATOR]
        self.flat_annotator = annotators[FLAT_SCAN_ANNOTATOR]

    def _sync(self) -> None:
        import numpy as np

        c = self.ctx
        stage, UsdGeom, Gf = c["stage"], c["UsdGeom"], c["Gf"]
        mpu = UsdGeom.GetStageMetersPerUnit(stage)
        if self.mode == "EMPTY":
            self.lidar.set_world_pose(
                position=np.asarray([v / mpu for v in EMPTY_POSITION_M], dtype=np.float64),
                orientation=np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
            )
            base_translation = base_rotation = offset_stage = None
        else:
            base_translation, base_rotation, _, offset_stage, _ = _base_rigid_pose(stage, UsdGeom, Gf)
            world_offset = Gf.Rotation(base_rotation).TransformDir(offset_stage)
            self.lidar.set_world_pose(
                position=np.asarray(base_translation + world_offset, dtype=np.float64),
                orientation=np.asarray(_wxyz(base_rotation), dtype=np.float64),
            )
        world = UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath(TEMP_SENSOR_PATH))
        _, _, lidar_scale, _, measured_position, _ = world.Factor()
        self.scale_is_unit = self.scale_is_unit and all(abs(float(v) - 1.0) <= 1e-6 for v in lidar_scale)
        if base_translation is not None:
            relative = Gf.Rotation(base_rotation).GetInverse().TransformDir(measured_position - base_translation)
            error_m = math.sqrt(sum(float(v) ** 2 for v in (relative - offset_stage))) * mpu
            self.max_offset_error_m = max(self.max_offset_error_m, error_m)

    def step(self, frames: int) -> list:
        """Step the stationary robot; return per-frame (point_count, flat_data)."""
        c = self.ctx
        samples = []
        for _ in range(frames):
            self._sync()
            c["robot"].apply_wheel_actions(c["stop_action"])
            c["simulation_context"].step(render=True)
            self._sync()
            samples.append((_point_count(self.point_annotator.get_data()), self.flat_annotator.get_data()))
        return samples

    def world_pose(self):
        c = self.ctx
        world = c["UsdGeom"].XformCache().GetLocalToWorldTransform(c["stage"].GetPrimAtPath(TEMP_SENSOR_PATH))
        _, _, _, rotation_matrix, translation, _ = world.Factor()
        return translation, rotation_matrix.ExtractRotationQuat().GetNormalized()

    def release(self) -> None:
        _release_lidar(self.lidar, TEMP_SENSOR_PATH, self.ctx["stage"])
        self.lidar = None


def _best_flat(samples, key):
    best, best_key = None, None
    for _, flat_data in samples:
        k = key(flat_data)
        if best_key is None or k > best_key:
            best, best_key = flat_data, k
    return best


# ---------------------------------------------------------------- robot geometry report

def _report_robot_geometry(rig: SensorRig) -> None:
    """Which TurtleBot geometry lies near the sensor origin (world axis-aligned bounds)?"""
    from pxr import Usd

    c = rig.ctx
    stage, UsdGeom, Gf = c["stage"], c["UsdGeom"], c["Gf"]
    mpu = UsdGeom.GetStageMetersPerUnit(stage)
    sensor_position, _ = rig.world_pose()
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])

    def distance_m(box) -> float:
        lo, hi = box.GetMin(), box.GetMax()
        delta = [max(lo[i] - sensor_position[i], 0.0, sensor_position[i] - hi[i]) for i in range(3)]
        return math.sqrt(sum(d * d for d in delta)) * mpu

    robot_box = cache.ComputeWorldBound(stage.GetPrimAtPath(ARTICULATION_PATH)).ComputeAlignedRange()
    size = robot_box.GetSize()
    log(f"ROBOT_WORLD_BBOX_MIN={tuple(round(float(v), 4) for v in robot_box.GetMin())}")
    log(f"ROBOT_WORLD_BBOX_MAX={tuple(round(float(v), 4) for v in robot_box.GetMax())}")
    log(f"ROBOT_WORLD_BBOX_SIZE_M={tuple(round(float(v) * mpu, 4) for v in size)}")
    log(f"SENSOR_WORLD_POSITION={tuple(round(float(v), 4) for v in sensor_position)}")
    log(f"SENSOR_INSIDE_ROBOT_BBOX={distance_m(robot_box) == 0.0}")

    nearby = []
    column_top = None
    for prim in Usd.PrimRange(stage.GetPrimAtPath(ARTICULATION_PATH)):
        if not prim.IsA(UsdGeom.Gprim):
            continue
        box = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        if box.IsEmpty():
            continue
        d = distance_m(box)
        if d <= ROBOT_GEOMETRY_REPORT_RADIUS_M:
            nearby.append((d, str(prim.GetPath()), box))
        lo, hi = box.GetMin(), box.GetMax()
        if lo[0] <= sensor_position[0] <= hi[0] and lo[1] <= sensor_position[1] <= hi[1]:
            column_top = hi[2] if column_top is None else max(column_top, hi[2])
    nearby.sort()
    log(f"MOUNT_OFFSET_M={MOUNT_OFFSET_M}")
    log(f"ROBOT_GPRIMS_WITHIN_{ROBOT_GEOMETRY_REPORT_RADIUS_M:.2f}M_OF_SENSOR={len(nearby)}")
    for d, path, box in nearby:
        log(
            f"    NEAR_GPRIM dist={d:.4f} m path={path} "
            f"size_m={tuple(round(float(v) * mpu, 4) for v in box.GetSize())}"
        )
    log(f"GPRIMS_CONTAINING_SENSOR={[path for d, path, _ in nearby if d == 0.0]}")
    # Highest geometry directly above/below the sensor x/y (world AABBs, robot upright).
    if column_top is None:
        log("COLUMN_TOP_ABOVE_SENSOR_XY=none")
    else:
        rise_m = (column_top - sensor_position[2]) * mpu
        log(f"COLUMN_TOP_Z_MINUS_SENSOR_Z={rise_m:.4f} m")
        log(
            f"SUGGESTED_OFFSET_ABOVE_COLUMN=({MOUNT_OFFSET_M[0]:.3f}, {MOUNT_OFFSET_M[1]:.3f}, "
            f"{MOUNT_OFFSET_M[2] + max(rise_m, 0.0) + 0.02:.3f})"
        )


# ---------------------------------------------------------------- measurement

def _make_wall(rig: SensorRig, direction_yaw_deg: float, distance_m: float):
    """Opaque vertical wall facing the sensor, front face distance_m along a sensor-frame yaw."""
    c = rig.ctx
    stage, UsdGeom, Gf = c["stage"], c["UsdGeom"], c["Gf"]
    mpu = UsdGeom.GetStageMetersPerUnit(stage)
    sensor_position, sensor_rotation = rig.world_pose()
    world_yaw_deg = math.degrees(_yaw_from_quat(sensor_rotation)) + direction_yaw_deg
    direction = Gf.Vec3d(math.cos(math.radians(world_yaw_deg)), math.sin(math.radians(world_yaw_deg)), 0.0)
    center = sensor_position + direction * ((distance_m + 0.5 * WALL_THICKNESS_M) / mpu)
    cube = UsdGeom.Cube.Define(stage, TEMP_WALL_PATH)
    cube.CreateSizeAttr(1.0)
    xform = UsdGeom.XformCommonAPI(cube.GetPrim())
    xform.SetTranslate(Gf.Vec3d(center[0], center[1], center[2]))
    xform.SetRotate(Gf.Vec3f(0.0, 0.0, world_yaw_deg))
    xform.SetScale(Gf.Vec3f(WALL_THICKNESS_M / mpu, WALL_WIDTH_M / mpu, WALL_HEIGHT_M / mpu))
    cube.CreateDisplayColorAttr([Gf.Vec3f(0.8, 0.2, 0.2)])
    return tuple(round(float(v) * mpu, 4) for v in center)


def _run_environment(rig: SensorRig, environment: str) -> list[WallResult]:
    stage = rig.ctx["stage"]
    log(f"================ ENVIRONMENT={environment} ================")
    baseline = rig.step(BASELINE_FRAMES)[BASELINE_DISCARD_FRAMES:]
    base_flat = _best_flat(baseline, lambda d: (_finite_valid(d), _flat_ranges(d).size))
    sensor_position, sensor_rotation = rig.world_pose()
    closest_d, closest_az, closest_i = _closest_return(base_flat)
    log(f"SENSOR_WORLD_POSITION={tuple(round(float(v), 4) for v in sensor_position)}")
    log(f"SENSOR_WORLD_YAW_DEG={math.degrees(_yaw_from_quat(sensor_rotation)):.3f}")
    if isinstance(base_flat, dict):
        log(
            f"FLAT_depthRange={[float(v) for v in base_flat.get('depthRange', [])]} "
            f"FLAT_azimuthRange={[float(v) for v in base_flat.get('azimuthRange', [])]} "
            f"FLAT_horizontalResolution={base_flat.get('horizontalResolution')} "
            f"FLAT_numCols={base_flat.get('numCols')}"
        )
    log(f"NO_WALL POINT_COUNT={max(count for count, _ in baseline)}")
    log(f"NO_WALL FLAT_finite_valid={_finite_valid(base_flat)}")
    log(f"NO_WALL FLAT_min={_fmt_m(closest_d)}")
    log(f"NO_WALL closest_return_angle={_fmt_deg(closest_az)} (index={closest_i})")
    if environment != "EMPTY":
        _report_robot_geometry(rig)

    results = []
    for direction, direction_yaw in DIRECTIONS.items():
        for distance_m in WALL_DISTANCES_M:
            baseline_band = max(len(_band_indices(flat, distance_m)) for _, flat in baseline)
            try:
                wall_position = _make_wall(rig, direction_yaw, distance_m)
                samples = rig.step(WALL_FRAMES)
            finally:
                if stage.GetPrimAtPath(TEMP_WALL_PATH).IsValid():
                    stage.RemovePrim(TEMP_WALL_PATH)
            rig.step(WALL_FLUSH_FRAMES)
            flat = _best_flat(samples, lambda d: (len(_band_indices(d, distance_m)), _finite_valid(d)))
            flat_min, closest_angle, _ = _closest_return(flat)
            result = WallResult(
                environment,
                direction,
                distance_m,
                wall_position,
                max(count for count, _ in samples),
                _finite_valid(flat),
                flat_min,
                closest_angle,
                len(_band_indices(flat, distance_m)),
                baseline_band,
                _band_azimuth_deg(flat, distance_m),
            )
            results.append(result)
            log(f"--- ENV={environment} DIR={direction} WALL_DISTANCE={distance_m:.2f} m")
            log(f"    wall_position={result.wall_position}")
            log(f"    closest_return_distance={_fmt_m(result.flat_min)}")
            log(f"    closest_return_angle={_fmt_deg(result.closest_angle_deg)}")
            log(f"    POINT_COUNT={result.point_count}")
            log(f"    FLAT_finite_valid={result.finite_valid}")
            log(f"    FLAT_min={_fmt_m(result.flat_min)}")
            log(f"    wall_band_rays={result.band_rays} (no-wall baseline={result.baseline_band_rays})")
            log(f"    wall_return_angle={_fmt_deg(result.band_azimuth_deg)}")
            log(f"    DETECTED={'yes' if result.detected else 'no'}")
    return results


def _summarize(results: list[WallResult]) -> None:
    log("================ SUMMARY (DETECTED / wall_return_angle) ================")
    header = "ENV            DIR " + " ".join(f"{d:>16.2f}m" for d in WALL_DISTANCES_M)
    log(header)
    for environment in ENVIRONMENTS:
        for direction in DIRECTIONS:
            cells = []
            for distance_m in WALL_DISTANCES_M:
                r = next(
                    x for x in results
                    if x.environment == environment and x.direction == direction and x.distance_m == distance_m
                )
                angle = "n/a" if r.band_azimuth_deg is None else f"{r.band_azimuth_deg:7.1f}"
                cells.append(f"{'yes' if r.detected else 'no ':>3} @{angle:>8}deg")
            log(f"{environment:<14} {direction:<3} " + " ".join(f"{c:>17}" for c in cells))

    def azimuth(environment, direction, distance_m=0.50):
        return next(
            x.band_azimuth_deg for x in results
            if x.environment == environment and x.direction == direction and x.distance_m == distance_m
        )

    # Q1: FlatScan azimuth of sensor +X / +Y in the clean EMPTY area.
    plus_x, plus_y = azimuth("EMPTY", "+X"), azimuth("EMPTY", "+Y")
    log(f"Q1 EMPTY sensor +X appears at FlatScan azimuth {_fmt_deg(plus_x)}")
    log(f"Q1 EMPTY sensor +Y appears at FlatScan azimuth {_fmt_deg(plus_y)}")
    if plus_x is not None and plus_y is not None:
        turn = (plus_y - plus_x + 180.0) % 360.0 - 180.0
        log(f"Q1 AZIMUTH_HANDEDNESS={'CCW (+Y = +90)' if turn > 0 else 'CW (+Y = -90)'} (delta={turn:.1f} deg)")

    # Q2: near-field walls blocked only when the robot is visible.
    blocked = [
        (v.direction, v.distance_m) for v in results
        if v.environment == "ROBOT_VISIBLE" and not v.detected
        and any(
            h.detected for h in results
            if h.environment == "ROBOT_HIDDEN" and h.direction == v.direction and h.distance_m == v.distance_m
        )
    ]
    log(f"Q2 ROBOT_BLOCKED_CASES={blocked if blocked else 'none'}")

    # Q3: on-robot azimuths vs clean-room azimuths for the same sensor-frame direction.
    for direction in DIRECTIONS:
        empty_az, robot_az = azimuth("EMPTY", direction), azimuth("ROBOT_HIDDEN", direction)
        if empty_az is None or robot_az is None:
            log(f"Q3 DIR={direction} EMPTY={_fmt_deg(empty_az)} ROBOT_HIDDEN={_fmt_deg(robot_az)} delta=n/a")
        else:
            delta = (robot_az - empty_az + 180.0) % 360.0 - 180.0
            log(f"Q3 DIR={direction} EMPTY={empty_az:.2f} ROBOT_HIDDEN={robot_az:.2f} delta={delta:.2f} deg")


# ---------------------------------------------------------------- main

def main() -> bool:
    import gc
    import numpy as np
    import omni.usd
    from pxr import Gf, UsdGeom
    from isaacsim.core.api import SimulationContext
    from isaacsim.robot.wheeled_robots.controllers.differential_controller import DifferentialController
    from isaacsim.robot.wheeled_robots.robots.wheeled_robot import WheeledRobot

    stage = None
    robot = None
    stop_action = None
    rig = None
    visibility_attr = None
    had_authored_visibility = False
    original_visibility = None
    try:
        if not SCENE_PATH.is_file():
            raise FileNotFoundError(f"Scene does not exist: {SCENE_PATH}")
        usd_context = omni.usd.get_context()
        while usd_context.get_stage_loading_status()[2] > 0:
            simulation_app.update()
        stage = usd_context.get_stage()
        if stage is None:
            raise RuntimeError("No active USD stage after opening the scene")
        if not stage.GetPrimAtPath(ARTICULATION_PATH).IsValid() or not stage.GetPrimAtPath(BASE_LINK_PATH).IsValid():
            raise RuntimeError("TurtleBot articulation or base link is absent")
        for path in (TEMP_SENSOR_PATH, TEMP_WALL_PATH):
            if stage.GetPrimAtPath(path).IsValid():
                raise RuntimeError(f"Temporary path already exists: {path}")

        simulation_context = SimulationContext()
        simulation_context.initialize_physics()
        simulation_context.play()
        for _ in range(SETTLE_FRAMES):
            simulation_context.step(render=True)

        robot = WheeledRobot(
            prim_path=ARTICULATION_PATH,
            name="tb03_near_field_robot",
            wheel_dof_names=[LEFT_WHEEL_DOF, RIGHT_WHEEL_DOF],
        )
        robot.initialize()
        controller = DifferentialController(
            name="tb03_near_field_controller",
            wheel_radius=WHEEL_RADIUS_M,
            wheel_base=WHEEL_BASE_M,
            max_linear_speed=0.22,
            max_angular_speed=1.0,
        )
        stop_action = controller.forward(np.array([0.0, 0.0], dtype=np.float64))
        robot.apply_wheel_actions(stop_action)
        ctx = {
            "stage": stage,
            "UsdGeom": UsdGeom,
            "Gf": Gf,
            "robot": robot,
            "stop_action": stop_action,
            "simulation_context": simulation_context,
        }

        rig = SensorRig(ctx)
        configured_min = rig.prim.GetAttribute(NEAR_RANGE_ATTRIBUTE).Get()
        configured_max = rig.prim.GetAttribute(FAR_RANGE_ATTRIBUTE).Get()
        log(f"CONFIGURED_MIN_RANGE={configured_min}")
        log(f"CONFIGURED_MAX_RANGE={configured_max}")
        if configured_min is None or abs(float(configured_min) - NEAR_RANGE_M) > 1e-3:
            raise RuntimeError(f"nearRangeM override not applied: {configured_min}")

        results = []
        for environment in ENVIRONMENTS:
            rig.mode = environment
            if environment == "ROBOT_HIDDEN":
                visibility_attr = UsdGeom.Imageable(stage.GetPrimAtPath(ARTICULATION_PATH)).GetVisibilityAttr()
                had_authored_visibility = visibility_attr.HasAuthoredValueOpinion()
                original_visibility = visibility_attr.Get()
                visibility_attr.Set(UsdGeom.Tokens.invisible)
            results.extend(_run_environment(rig, environment))

        _summarize(results)
        log(f"FOLLOW_OFFSET_ERROR_MAX={rig.max_offset_error_m:.8f} m")
        log(f"LIDAR_SCALE_UNIT={'PASS' if rig.scale_is_unit else 'FAIL'}")
        return True
    except Exception as error:
        log(f"FAIL: {error}")
        return False
    finally:
        if visibility_attr is not None:
            if had_authored_visibility:
                visibility_attr.Set(original_visibility)
            else:
                visibility_attr.Clear()
        if robot is not None and stop_action is not None:
            try:
                robot.apply_wheel_actions(stop_action)
            except Exception:
                pass
        if stage is not None and stage.GetPrimAtPath(TEMP_WALL_PATH).IsValid():
            stage.RemovePrim(TEMP_WALL_PATH)
        if rig is not None:
            try:
                rig.release()
            except Exception:
                pass
        gc.collect()


if __name__ == "__main__":
    try:
        main()
    finally:
        log("closing SimulationApp without saving stage")
        simulation_app.close()
