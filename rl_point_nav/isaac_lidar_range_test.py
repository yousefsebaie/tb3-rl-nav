#!/usr/bin/env python3
"""Runtime-only default/0.10 m RTX LiDAR range comparison.

All sensors are fresh direct /World children, following the measured base-link
rigid pose at (0.10, 0.0, 0.22) m without inheriting its scale. First verify the
live near/far attributes and FlatScan metadata. Then test six independent
wall-before-sensor cases while commanding only zero wheel velocities.

The installed 5.1 OmniSensor schema defines omni:sensor:Core:nearRangeM and
farRangeM as float distances in meters. LidarRtx/IsaacSensorCreateRtxLidar
accept exact attribute-name kwargs before render-product creation. Only the
nearRangeM attribute is overridden; no profile asset or scene is saved.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path


SCENE_PATH = Path("/home/g8/Desktop/FInalSceneDemo.usd")
ARTICULATION_PATH = "/World/turtlebot3_burger_02"
BASE_LINK_PATH = f"{ARTICULATION_PATH}/a__namespace_base_link"
LEFT_WHEEL_DOF = "a__namespace_wheel_left_joint"
RIGHT_WHEEL_DOF = "a__namespace_wheel_right_joint"
LIDAR_CONFIG = "Example_Rotary_2D"
POINT_CLOUD_ANNOTATOR = "IsaacExtractRTXSensorPointCloudNoAccumulator"
FLAT_SCAN_ANNOTATOR = "IsaacComputeRTXLidarFlatScan"
MOUNT_OFFSET_M = (0.10, 0.0, 0.22)
NEAR_RANGE_ATTRIBUTE = "omni:sensor:Core:nearRangeM"
FAR_RANGE_ATTRIBUTE = "omni:sensor:Core:farRangeM"
TARGET_MIN_RANGE_M = 0.10
EXPECTED_DEFAULT_MIN_M = 1.0
EXPECTED_MAX_M = 200.0
RANGE_MATCH_TOLERANCE_M = 1e-4
ATTRIBUTE_KEYWORDS = ("range", "depth", "min", "max")
WHEEL_RADIUS_M = 0.033
WHEEL_BASE_M = 0.160
SETTLE_FRAMES = 8
COLLECT_FRAMES = 180
INVALID_RANGE_EPSILON_M = 1e-3
WALL_DISTANCES_M = (0.30, 0.60, 1.20)
WALL_THICKNESS_M = 0.02
WALL_WIDTH_M = 0.80
WALL_HEIGHT_M = 0.60
WALL_RANGE_TOLERANCE_M = 0.04
FRONT_HALF_ANGLE_DEG = 10.0
MIN_WALL_RAYS = 5
MAX_STATIONARY_DRIFT_M = 0.005
MAX_STATIONARY_ROTATION_DEG = 0.5
TAG = "[TB03_LIDAR_RANGE]"


def log(message: str) -> None:
    print(f"{TAG} {message}", flush=True)


@dataclass
class Collection:
    point_count: int
    flat: dict
    wall_rays: dict[float, int]

    @property
    def scan_alive(self) -> bool:
        import numpy as np

        cols, samples, finite, _, _ = _flat_stats(self.flat)
        bounds = np.asarray(self.flat.get("azimuthRange", []), dtype=float).reshape(-1)
        angles_valid = (
            bounds.size == 2 and np.isfinite(bounds).all() and bounds[1] > bounds[0]
        )
        return (
            self.point_count > 0 and cols > 1 and samples == cols
            and finite > 0 and angles_valid
        )


@dataclass
class WallResult:
    mode: str
    distance_m: float
    detected: bool
    expected: bool
    passed: bool


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


def _flat_stats(data):
    ranges = _flat_ranges(data)
    valid = ranges[_valid_mask(ranges)]
    num_cols = int(data.get("numCols", 0)) if isinstance(data, dict) else 0
    return (
        num_cols,
        int(ranges.size),
        int(valid.size),
        float(valid.min()) if valid.size else None,
        float(valid.max()) if valid.size else None,
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
    offset_stage = Gf.Vec3d(*(component / meters_per_unit for component in MOUNT_OFFSET_M))
    return translation, rotation, inherited_scale, offset_stage, meters_per_unit


def _synchronize_lidar(lidar, sensor_path, stage, UsdGeom, Gf) -> tuple[float, bool]:
    """Synchronize top-level LiDAR from rigid base pose; return offset error in m."""
    import numpy as np

    base_translation, base_rotation, _, offset_stage, meters_per_unit = _base_rigid_pose(stage, UsdGeom, Gf)
    world_offset = Gf.Rotation(base_rotation).TransformDir(offset_stage)
    lidar_position = base_translation + world_offset
    lidar.set_world_pose(
        position=np.asarray(lidar_position, dtype=np.float64),
        orientation=np.asarray(_wxyz(base_rotation), dtype=np.float64),
    )

    lidar_world = UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath(sensor_path))
    _, _, lidar_scale, _, measured_position, _ = lidar_world.Factor()
    relative_stage = Gf.Rotation(base_rotation).GetInverse().TransformDir(measured_position - base_translation)
    error_stage = relative_stage - offset_stage
    error_m = math.sqrt(sum(float(component) ** 2 for component in error_stage)) * meters_per_unit
    scale_is_unit = all(abs(float(component) - 1.0) <= 1e-6 for component in lidar_scale)
    return error_m, scale_is_unit


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


def _depth_range(flat):
    import numpy as np

    values = np.asarray(flat.get("depthRange", []), dtype=float).reshape(-1)
    return tuple(float(v) for v in values) if values.size == 2 else None


def _wall_rays(flat, distance_m: float) -> int:
    """Match the known front plane using actual FlatScan azimuth metadata.

    Isaac 5.1 test_annotators.py reconstructs angles with linspace(min, max,
    numCols), with +X at zero azimuth. Inspect only +/-10 degrees and require
    several ranges consistent with the wall front face at d/cos(azimuth).
    The 4 cm tolerance covers the profile's 2 cm range accuracy, its small
    emitter elevation, and the bounded stationary drift.
    """
    import numpy as np

    ranges = _flat_ranges(flat)
    cols = int(flat.get("numCols", 0))
    bounds = np.asarray(flat.get("azimuthRange", []), dtype=float).reshape(-1)
    if cols <= 1 or ranges.size != cols or bounds.size != 2:
        return 0
    if not np.isfinite(bounds).all() or bounds[1] <= bounds[0]:
        return 0
    angles = np.deg2rad(np.linspace(bounds[0], bounds[1], cols))
    angles = (angles + np.pi) % (2 * np.pi) - np.pi
    front = np.abs(angles) <= math.radians(FRONT_HALF_ANGLE_DEG)
    angles, measured = angles[front], ranges[front]
    expected = distance_m / np.cos(angles)
    inside_wall = np.abs(distance_m * np.tan(angles)) < WALL_WIDTH_M / 2
    return int(np.count_nonzero(
        _valid_mask(measured) & inside_wall
        & (np.abs(measured - expected) <= WALL_RANGE_TOLERANCE_M)
    ))


def _mount_pose(ctx):
    translation, rotation, _, offset, mpu = _base_rigid_pose(
        ctx["stage"], ctx["UsdGeom"], ctx["Gf"]
    )
    return translation + ctx["Gf"].Rotation(rotation).TransformDir(offset), rotation, mpu


def _live_ranges(prim):
    values = []
    if prim.GetTypeName() != "OmniLidar":
        raise RuntimeError(f"Expected OmniLidar, got {prim.GetTypeName()}")
    for name in (NEAR_RANGE_ATTRIBUTE, FAR_RANGE_ATTRIBUTE):
        attribute = prim.GetAttribute(name)
        if not attribute.IsValid() or str(attribute.GetTypeName()) != "float":
            raise RuntimeError(f"Required installed range attribute absent/invalid: {name}")
        value = attribute.Get()
        if value is None or not math.isfinite(float(value)):
            raise RuntimeError(f"Invalid live range value: {name}={value}")
        values.append(float(value))
    return tuple(values)


def _inspect_attributes(prim) -> None:
    log(f"APPLIED_SCHEMAS={list(prim.GetAppliedSchemas())}")
    for attribute in prim.GetAttributes():
        if any(word in attribute.GetName().lower() for word in ATTRIBUTE_KEYWORDS):
            log(f"ATTRIBUTE_NAME={attribute.GetName()} "
                f"ATTRIBUTE_VALUE={attribute.Get()} "
                f"ATTRIBUTE_TYPE={attribute.GetTypeName()}")


class SensorRig:
    """One fresh top-level sensor; every physics/render step holds zero wheels."""

    def __init__(self, label, sensor_path, minimum_override, ctx):
        from isaacsim.sensors.rtx import LidarRtx

        self.label, self.sensor_path, self.ctx = label, sensor_path, ctx
        self.lidar = None
        self.max_offset_error_m = 0.0
        self.initial_pose = None
        stage = ctx["stage"]
        if stage.GetPrimAtPath(sensor_path).IsValid():
            raise RuntimeError(f"Temporary sensor path already exists: {sensor_path}")
        overrides = {} if minimum_override is None else {NEAR_RANGE_ATTRIBUTE: minimum_override}
        try:
            # Installed creation command authors these kwargs only if the exact
            # attribute exists, before LidarRtx creates its render product.
            self.lidar = LidarRtx(
                prim_path=sensor_path,
                name=sensor_path.rsplit("/", 1)[1],
                config_file_name=LIDAR_CONFIG,
                **overrides,
            )
            self.prim = stage.GetPrimAtPath(self.lidar.prim_path)
            if str(self.prim.GetPath()) != sensor_path or str(self.prim.GetParent().GetPath()) != "/World":
                raise RuntimeError("LiDAR must remain at its exact top-level /World path")
            _live_ranges(self.prim)  # Fail if the real schema attributes are absent.
            self.lidar.initialize()
            self.configured_range = _live_ranges(self.prim)
            self._sync()
            self.render_product = self.lidar.get_render_product_path()
            if not self.render_product:
                raise RuntimeError("LidarRtx render product is missing")
            self.lidar.attach_annotator(POINT_CLOUD_ANNOTATOR)
            self.lidar.attach_annotator(FLAT_SCAN_ANNOTATOR)
            self.point_annotator = self.lidar.get_annotators()[POINT_CLOUD_ANNOTATOR]
            self.flat_annotator = self.lidar.get_annotators()[FLAT_SCAN_ANNOTATOR]
            log(f"SENSOR={label} SENSOR_PATH={sensor_path}")
            log(f"RENDER_PRODUCT_PATH={self.render_product}")
            log(f"{NEAR_RANGE_ATTRIBUTE}={self.configured_range[0]}")
            log(f"{FAR_RANGE_ATTRIBUTE}={self.configured_range[1]}")
        except Exception:
            self.release()
            raise

    def _sync(self) -> None:
        c = self.ctx
        error, unit_scale = _synchronize_lidar(
            self.lidar, self.sensor_path, c["stage"], c["UsdGeom"], c["Gf"]
        )
        self.max_offset_error_m = max(self.max_offset_error_m, error)
        if not unit_scale or error > 1e-5:
            raise RuntimeError(f"LiDAR rigid mount check failed: offset_error={error}, unit_scale={unit_scale}")
        position, rotation, mpu = _mount_pose(c)
        if self.initial_pose is None:
            # Keep the same reference across baseline and all wall cases.
            self.initial_pose = c.setdefault("stationary_reference", (position, rotation))
        initial_position, initial_rotation = self.initial_pose
        drift = (position - initial_position).GetLength() * mpu
        relative = initial_rotation.GetInverse() * rotation
        angle = abs(c["Gf"].Rotation(relative.GetNormalized()).GetAngle())
        angle = min(angle, abs(360.0 - angle))
        if drift > MAX_STATIONARY_DRIFT_M or angle > MAX_STATIONARY_ROTATION_DEG:
            raise RuntimeError(f"Stationary test moved: drift={drift:.6f} m, rotation={angle:.4f} deg")

    def collect(self, wall_distance=None) -> Collection:
        from copy import deepcopy

        best_flat, best_score, point_count = {}, (-1, -1, -1), 0
        wall_rays = {distance: 0 for distance in WALL_DISTANCES_M}
        c = self.ctx
        for update in range(1, COLLECT_FRAMES + 1):
            if not c["app"].is_running():
                raise RuntimeError("Isaac window closed before collection finished")
            self._sync()
            c["robot"].apply_wheel_actions(c["stop_action"])
            c["simulation_context"].step(render=True)
            self._sync()
            point_count = max(point_count, _point_count(self.point_annotator.get_data()))
            data = self.flat_annotator.get_data()
            if not isinstance(data, dict):
                continue
            cols, samples, finite, _, _ = _flat_stats(data)
            for distance in WALL_DISTANCES_M:
                wall_rays[distance] = max(wall_rays[distance], _wall_rays(data, distance))
            matches = _wall_rays(data, wall_distance) if wall_distance is not None else 0
            score = (matches, finite, samples)
            if score > best_score:
                # Replicator may reuse output buffers on the next update.
                best_flat, best_score = deepcopy(data), score
            if update % 60 == 0:
                log(f"SENSOR={self.label} UPDATE={update}/{COLLECT_FRAMES} "
                    f"POINT_COUNT={point_count} FLAT_numCols={cols}")
        return Collection(point_count, best_flat, wall_rays)

    def release(self) -> None:
        _release_lidar(self.lidar, self.sensor_path, self.ctx["stage"])
        self.lidar = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.release()


def _range_verified(rig, collection, expected_min, expected_max) -> bool:
    import numpy as np

    live = _live_ranges(rig.prim)
    depth = _depth_range(collection.flat)
    expected = (expected_min, expected_max)
    log(f"CONFIGURED_MIN_RANGE={live[0]}")
    log(f"CONFIGURED_MAX_RANGE={live[1]}")
    log(f"FlatScan_depthRange={depth}")
    return (
        depth is not None
        and np.allclose(live, expected, rtol=0, atol=RANGE_MATCH_TOLERANCE_M)
        and np.allclose(depth, expected, rtol=0, atol=RANGE_MATCH_TOLERANCE_M)
    )


def _print_scan(collection) -> None:
    cols, samples, finite, minimum, maximum = _flat_stats(collection.flat)
    log(f"POINT_COUNT={collection.point_count}")
    log(f"FLAT_numCols={cols}")
    log(f"FLAT_samples={samples}")
    log(f"FLAT_finite_valid={finite}")
    log(f"FLAT_min={'n/a' if minimum is None else f'{minimum:.6f}'}")
    log(f"FLAT_max={'n/a' if maximum is None else f'{maximum:.6f}'}")


def _make_wall(ctx, path, distance_m):
    """Create opaque geometry BEFORE the sensor; distance means nearest face.

    The robot stays still. The wall is vertical, perpendicular to the rigid
    sensor's horizontal forward direction. It has no collision/rigid-body API.
    """
    c = ctx
    stage, UsdGeom, Gf = c["stage"], c["UsdGeom"], c["Gf"]
    if stage.GetPrimAtPath(path).IsValid():
        raise RuntimeError(f"Temporary wall path already exists: {path}")
    position, rotation, mpu = _mount_pose(c)
    up = Gf.Rotation(rotation).TransformDir(Gf.Vec3d(0, 0, 1))
    tilt_deg = math.degrees(math.acos(max(-1.0, min(1.0, float(up[2])))))
    if tilt_deg > 5.0:
        raise RuntimeError(f"Robot tilt {tilt_deg:.3f} deg invalidates the vertical-wall range approximation")
    yaw = _yaw_from_quat(rotation)
    forward = Gf.Vec3d(math.cos(yaw), math.sin(yaw), 0.0)
    center = position + forward * ((distance_m + WALL_THICKNESS_M / 2) / mpu)
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    xform = UsdGeom.XformCommonAPI(cube.GetPrim())
    xform.SetTranslate(center)
    xform.SetRotate(Gf.Vec3f(0, 0, math.degrees(yaw)))
    xform.SetScale(Gf.Vec3f(WALL_THICKNESS_M / mpu, WALL_WIDTH_M / mpu, WALL_HEIGHT_M / mpu))
    cube.CreateDisplayColorAttr([Gf.Vec3f(0.8, 0.2, 0.2)])
    cube.CreateDisplayOpacityAttr([1.0])
    # /World must be a rigid identity parent for these world-space wall ops.
    world = UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath("/World"))
    if not Gf.IsClose(world, Gf.Matrix4d(1.0), 1e-6):
        raise RuntimeError("Non-identity /World would invalidate known wall distances")
    log(f"WALL_CREATED_BEFORE_SENSOR={path} FRONT_FACE_DISTANCE_M={distance_m:.2f}")


def _wall_case(ctx, mode, distance, baseline, expected_max) -> WallResult:
    suffix = f"{mode.lower()}_{round(distance * 100):03d}"
    path = f"/World/tb03_range_wall_{suffix}"
    sensor_path = f"/World/tb03_range_lidar_{suffix}"
    minimum_override = TARGET_MIN_RANGE_M if mode == "OVERRIDE" else None
    expected_min = TARGET_MIN_RANGE_M if mode == "OVERRIDE" else EXPECTED_DEFAULT_MIN_M
    stage = ctx["stage"]
    if stage.GetPrimAtPath(path).IsValid() or stage.GetPrimAtPath(sensor_path).IsValid():
        raise RuntimeError(f"Temporary path already exists for case {suffix}")
    try:
        _make_wall(ctx, path, distance)
        with SensorRig(mode, sensor_path, minimum_override, ctx) as rig:
            data = rig.collect(wall_distance=distance)
            log(f"MODE={mode}")
            log(f"WALL_DISTANCE={distance:.2f}")
            _print_scan(data)
            range_ok = _range_verified(rig, data, expected_min, expected_max)
            matches = data.wall_rays[distance]
            baseline_matches = baseline.wall_rays[distance]
            # Existing geometry at this same front distance makes attribution
            # ambiguous. Report FAIL instead of declaring a wall detection.
            unambiguous = baseline_matches < MIN_WALL_RAYS
            detected = matches >= MIN_WALL_RAYS and unambiguous
            expected = mode == "OVERRIDE" or distance >= EXPECTED_DEFAULT_MIN_M
            passed = range_ok and data.scan_alive and unambiguous and detected == expected
            log(f"WALL_MATCHED_RAYS={matches} NO_WALL_MATCHED_RAYS={baseline_matches}")
            log(f"DETECTED={'yes' if detected else 'no'}")
            log(f"EXPECTED_DETECTED={'yes' if expected else 'no'}")
            if not unambiguous:
                log("FAIL: existing front geometry matches the wall distance; result is ambiguous")
            if not data.scan_alive:
                log("FAIL: sensor data/angles missing; this cannot prove minimum-range rejection")
            log(f"CASE_PASS={'PASS' if passed else 'FAIL'}")
            log(f"FOLLOW_OFFSET_ERROR_MAX={rig.max_offset_error_m:.8f}")
            return WallResult(mode, distance, detected, expected, passed)
    finally:
        if stage.GetPrimAtPath(path).IsValid():
            stage.RemovePrim(path)


def main(simulation_app) -> bool:
    import gc
    import numpy as np
    import omni.usd
    from pxr import Gf, UsdGeom
    from isaacsim.core.api import SimulationContext
    from isaacsim.robot.wheeled_robots.controllers.differential_controller import DifferentialController
    from isaacsim.robot.wheeled_robots.robots.wheeled_robot import WheeledRobot

    robot = stop_action = simulation_context = None
    stage = previous_edit_target = None
    default_depth = override_depth = None
    override_applied = overall = False
    results = []
    try:
        usd_context = omni.usd.get_context()
        while usd_context.get_stage_loading_status()[2] > 0:
            simulation_app.update()
        stage = usd_context.get_stage()
        if stage is None or not stage.GetPrimAtPath(BASE_LINK_PATH).IsValid():
            raise RuntimeError("Scene loading failed or TurtleBot base link is absent")
        # All diagnostic opinions go to the unsaved session layer.
        previous_edit_target = stage.GetEditTarget()
        stage.SetEditTarget(stage.GetSessionLayer())
        simulation_context = SimulationContext()
        simulation_context.initialize_physics()
        simulation_context.play()
        for _ in range(SETTLE_FRAMES):
            simulation_context.step(render=True)
        robot = WheeledRobot(
            prim_path=ARTICULATION_PATH, name="tb03_range_robot",
            wheel_dof_names=[LEFT_WHEEL_DOF, RIGHT_WHEEL_DOF],
        )
        robot.initialize()
        controller = DifferentialController(
            name="tb03_range_controller", wheel_radius=WHEEL_RADIUS_M,
            wheel_base=WHEEL_BASE_M, max_linear_speed=0.22, max_angular_speed=1.0,
        )
        stop_action = controller.forward(np.array([0.0, 0.0], dtype=np.float64))
        for _ in range(SETTLE_FRAMES):
            robot.apply_wheel_actions(stop_action)
            simulation_context.step(render=True)
        ctx = dict(stage=stage, UsdGeom=UsdGeom, Gf=Gf, app=simulation_app,
                   robot=robot, stop_action=stop_action, simulation_context=simulation_context)
        log(f"MOUNT_OFFSET_M={MOUNT_OFFSET_M} UNIT_SCALE_REQUIRED=True")

        # No wall exists during these two profile checks; each uses a fresh sensor.
        with SensorRig("DEFAULT", "/World/tb03_range_control", None, ctx) as rig:
            _inspect_attributes(rig.prim)
            default = rig.collect()
            _print_scan(default)
            default_depth = _depth_range(default.flat)
            if not _range_verified(rig, default, EXPECTED_DEFAULT_MIN_M, EXPECTED_MAX_M) or not default.scan_alive:
                raise RuntimeError("Default sensor/range control failed; wall tests skipped")
            default_max = _live_ranges(rig.prim)[1]

        with SensorRig("OVERRIDE", "/World/tb03_range_override", TARGET_MIN_RANGE_M, ctx) as rig:
            overridden = rig.collect()
            _print_scan(overridden)
            override_depth = _depth_range(overridden.flat)
            override_applied = _range_verified(rig, overridden, TARGET_MIN_RANGE_M, default_max)
            log(f"OVERRIDE_APPLIED={'PASS' if override_applied else 'FAIL'}")
            if not override_applied:
                raise RuntimeError("Live/FlatScan range is not [0.10, unchanged maximum]; wall tests skipped")
            if not overridden.scan_alive:
                raise RuntimeError("Override sensor has no valid scan; wall tests skipped")

        for mode, baseline in (("DEFAULT", default), ("OVERRIDE", overridden)):
            for distance in WALL_DISTANCES_M:
                results.append(_wall_case(ctx, mode, distance, baseline, default_max))
                gc.collect()
        overall = override_applied and len(results) == 6 and all(result.passed for result in results)
    except Exception as error:
        log(f"FAIL: {error}")
    finally:
        cleanup = []
        if robot is not None and stop_action is not None:
            cleanup.append(lambda: robot.apply_wheel_actions(stop_action))
        if simulation_context is not None:
            cleanup.append(simulation_context.stop)
        if stage is not None and previous_edit_target is not None:
            cleanup.append(lambda: stage.SetEditTarget(previous_edit_target))
        for operation in cleanup:
            try:
                operation()
            except Exception as error:
                log(f"CLEANUP_ERROR={error}")
                overall = False
        log(f"DEFAULT_DEPTH_RANGE={default_depth}")
        log(f"OVERRIDE_DEPTH_RANGE={override_depth}")
        for mode in ("DEFAULT", "OVERRIDE"):
            log(f"{mode}:")
            for distance in WALL_DISTANCES_M:
                result = next((r for r in results if r.mode == mode and r.distance_m == distance), None)
                if result is None:
                    log(f"{distance:.2f} SKIPPED")
                else:
                    log(f"{distance:.2f} DETECTED={'yes' if result.detected else 'no'} "
                        f"EXPECTED={'yes' if result.expected else 'no'} "
                        f"{'PASS' if result.passed else 'FAIL'}")
        log(f"OVERRIDE_APPLIED={'PASS' if override_applied else 'FAIL'}")
        log(f"OVERALL_PASS={'PASS' if overall else 'FAIL'}")
    return overall


if __name__ == "__main__":
    from isaacsim import SimulationApp

    if not SCENE_PATH.is_file():
        raise FileNotFoundError(f"Scene does not exist: {SCENE_PATH}")
    simulation_app = SimulationApp({
        "headless": False, "disable_viewport_updates": False,
        "open_usd": str(SCENE_PATH), "create_new_stage": False, "fast_shutdown": False,
    })
    try:
        success = main(simulation_app)
    finally:
        log("closing SimulationApp without saving stage")
        simulation_app.close()
    raise SystemExit(0 if success else 1)
