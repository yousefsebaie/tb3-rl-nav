#!/usr/bin/env python3
"""Runtime-only RTX LiDAR azimuth calibration in a clean area.

The same temporary top-level Example_Rotary_2D LiDAR (unit scale,
nearRangeM = 0.10 m) is placed in an empty region far from the robot scene
(CLEAN_CENTER_M, the area proven clean by isaac_lidar_scene_control_test.py).
Its ORIENTATION follows the TurtleBot base_link rigid rotation (normalized, no
scale), so the fitted mapping is expressed in the same sensor frame the robot
mount uses; its translation is pinned to the clean area. MOUNT_OFFSET_M is
printed for reference only; it does not affect azimuths.

PHASE A (primary, proven pattern: walls exist BEFORE the LiDAR is created)
  Eight walls, one per commanded sensor-frame direction, each at a DIFFERENT
  radius. A single stationary scan identifies every direction by its range
  window, so no no-wall baseline and no scene geometry are involved.

PHASE B (A/B diagnostic: walls added/removed WHILE the LiDAR runs)
  Phase-A walls are removed, then one 2.0 m wall at a time is created at
  runtime (the method of the earlier failing tests). If A passes and B fails,
  runtime-created geometry is not reaching the RTX LiDAR.

FlatScan azimuth per ray = azimuthRange[0] + index * horizontalResolution; raw
indices are printed too. Nothing is saved; the USD is never modified.
"""
from __future__ import annotations

import math
from pathlib import Path


SCENE_PATH = Path("/home/g8/Desktop/FInalSceneDemo.usd")
ARTICULATION_PATH = "/World/turtlebot3_burger_02"
BASE_LINK_PATH = f"{ARTICULATION_PATH}/a__namespace_base_link"
TEMP_SENSOR_PATH = "/World/tb03_azimuth_cal_lidar"
TEMP_WALL_ROOT = "/World/tb03_azimuth_cal_walls"
LEFT_WHEEL_DOF = "a__namespace_wheel_left_joint"
RIGHT_WHEEL_DOF = "a__namespace_wheel_right_joint"
LIDAR_CONFIG = "Example_Rotary_2D"
POINT_CLOUD_ANNOTATOR = "IsaacExtractRTXSensorPointCloudNoAccumulator"
FLAT_SCAN_ANNOTATOR = "IsaacComputeRTXLidarFlatScan"
NEAR_RANGE_ATTRIBUTE = "omni:sensor:Core:nearRangeM"  # identified by isaac_lidar_range_test.py
NEAR_RANGE_M = 0.10
MOUNT_OFFSET_M = (0.10, 0.0, 0.22)  # validated on-robot mount; reference only in the clean area
WHEEL_RADIUS_M = 0.033
WHEEL_BASE_M = 0.160
SETTLE_FRAMES = 8
INVALID_RANGE_EPSILON_M = 1e-3

# Sensor at 1 m height (as in the passing scene-control test) so no ground plane
# can return inside the wall range windows.
CLEAN_CENTER_M = (50.0, 50.0, 1.0)
CLEAN_AREA_HALF_SIZE_M = 6.0
# Commanded sensor-frame direction -> Phase-A radius (distinct, 0.25 m apart).
PHASE_A_WALLS = {
    0.0: 1.50,
    45.0: 1.75,
    90.0: 2.00,
    135.0: 2.25,
    180.0: 2.50,
    -45.0: 2.75,
    -90.0: 3.00,
    -135.0: 3.25,
}
PHASE_B_RADIUS_M = 2.0
WALL_THICKNESS_M = 0.05
WALL_WIDTH_M = 0.60
WALL_HEIGHT_M = 1.00
WALL_RANGE_SLACK_M = 0.08       # window: [R - slack, R / cos(half_width) + slack]
CHANGED_RAY_THRESHOLD_M = 0.05  # Phase B: ray differs from the no-wall median
MIN_WALL_RAYS = 3
MIN_WALL_HIT_RANGE_M = 1.0
WALL_HIT_VOTE_FRACTION = 0.5    # ray must be a hit in at least half of the frames
WALL_SPAN_MARGIN_DEG = 10.0     # detected only if angular span <= geometric span + margin
SCAN_STABLE_FRAMES = 10
SCAN_WAIT_MAX_UPDATES = 600
SCAN_WAIT_LOG_EVERY = 30
MEASURE_FRAMES = 45
WALL_FRAMES = 45
WALL_FLUSH_FRAMES = 15
TAG = "[TB03_LIDAR_AZIMUTH]"


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


def _wrap_deg(angle: float) -> float:
    return (angle + 180.0) % 360.0 - 180.0


def _circular_mean_deg(angles) -> float:
    return math.degrees(
        math.atan2(
            sum(math.sin(math.radians(a)) for a in angles),
            sum(math.cos(math.radians(a)) for a in angles),
        )
    )


def _circular_spread_deg(angles) -> float:
    mean = _circular_mean_deg(angles)
    return max(abs(_wrap_deg(a - mean)) for a in angles)


# ---------------------------------------------------------------- helpers
# Pose / release helpers are copied from isaac_lidar_dynamic_follow_test.py.

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
    """FlatScan ranges with invalid rays as NaN."""
    import numpy as np

    if not isinstance(data, dict):
        return np.zeros(0)
    ranges = np.asarray(data.get("linearDepthData", []), dtype=np.float64).reshape(-1)
    return np.where(np.isfinite(ranges) & (ranges > INVALID_RANGE_EPSILON_M), ranges, np.nan)


def _azimuths_deg(data, size: int):
    import numpy as np

    return float(data["azimuthRange"][0]) + np.arange(size) * float(data["horizontalResolution"])


def _base_rigid_rotation(stage, UsdGeom):
    """Normalized base_link world rotation with inherited scale removed."""
    base_link = stage.GetPrimAtPath(BASE_LINK_PATH)
    if not base_link.IsValid():
        raise RuntimeError(f"Missing TurtleBot base link: {BASE_LINK_PATH}")
    base_world = UsdGeom.XformCache().GetLocalToWorldTransform(base_link)
    _, _, _, rotation_matrix, _, _ = base_world.Factor()
    return rotation_matrix.ExtractRotationQuat().GetNormalized()


def _synchronize_lidar(lidar, stage, UsdGeom) -> bool:
    """Pin the LiDAR in the clean area with the base rigid rotation; True if scale is unit."""
    import numpy as np

    mpu = UsdGeom.GetStageMetersPerUnit(stage)
    lidar.set_world_pose(
        position=np.asarray([v / mpu for v in CLEAN_CENTER_M], dtype=np.float64),
        orientation=np.asarray(_wxyz(_base_rigid_rotation(stage, UsdGeom)), dtype=np.float64),
    )
    lidar_world = UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath(TEMP_SENSOR_PATH))
    _, _, lidar_scale, _, _, _ = lidar_world.Factor()
    return all(abs(float(component) - 1.0) <= 1e-6 for component in lidar_scale)


def _release_lidar(lidar, stage) -> None:
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
    if stage.GetPrimAtPath(TEMP_SENSOR_PATH).IsValid():
        stage.RemovePrim(TEMP_SENSOR_PATH)


# ---------------------------------------------------------------- geometry

def _clean_area_report(stage, UsdGeom) -> int:
    """Count existing scene Gprims whose world bounds reach into the clean area."""
    from pxr import Gf, Usd

    mpu = UsdGeom.GetStageMetersPerUnit(stage)
    half = CLEAN_AREA_HALF_SIZE_M / mpu
    center = Gf.Vec3d(*(v / mpu for v in CLEAN_CENTER_M))
    area = Gf.Range3d(center - Gf.Vec3d(half, half, half), center + Gf.Vec3d(half, half, half))
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    foreign = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Gprim):
            continue
        box = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        if not box.IsEmpty() and not Gf.Range3d.GetIntersection(box, area).IsEmpty():
            foreign.append(str(prim.GetPath()))
    log(f"CLEAN_AREA center={CLEAN_CENTER_M} half_size={CLEAN_AREA_HALF_SIZE_M} m")
    log(f"CLEAN_AREA_FOREIGN_GPRIMS={len(foreign)}")
    for path in foreign[:20]:
        log(f"    FOREIGN_GPRIM {path}")
    return len(foreign)


def _make_wall(stage, UsdGeom, Gf, path: str, sensor_yaw_deg: float, direction_deg: float, radius_m: float):
    """Wall facing the clean-area sensor, front face radius_m away along a SENSOR-frame direction."""
    mpu = UsdGeom.GetStageMetersPerUnit(stage)
    world_yaw = sensor_yaw_deg + direction_deg
    direction = Gf.Vec3d(math.cos(math.radians(world_yaw)), math.sin(math.radians(world_yaw)), 0.0)
    sensor_position = Gf.Vec3d(*(v / mpu for v in CLEAN_CENTER_M))
    center = sensor_position + direction * ((radius_m + 0.5 * WALL_THICKNESS_M) / mpu)
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    xform = UsdGeom.XformCommonAPI(cube.GetPrim())
    xform.SetTranslate(Gf.Vec3d(center[0], center[1], center[2]))
    xform.SetRotate(Gf.Vec3f(0.0, 0.0, world_yaw))
    xform.SetScale(Gf.Vec3f(WALL_THICKNESS_M / mpu, WALL_WIDTH_M / mpu, WALL_HEIGHT_M / mpu))
    cube.CreateDisplayColorAttr([Gf.Vec3f(0.8, 0.2, 0.2)])
    return tuple(round(float(v) * mpu, 4) for v in center)


def _wall_path(direction_deg: float) -> str:
    return f"{TEMP_WALL_ROOT}/wall_{'m' if direction_deg < 0 else 'p'}{abs(int(direction_deg)):03d}"


# ---------------------------------------------------------------- measurement

def _step(ctx, frames: int) -> list:
    """Step the stationary robot, syncing the LiDAR before and after each step."""
    lidar, stage, UsdGeom = ctx["lidar"], ctx["stage"], ctx["UsdGeom"]
    samples = []
    for _ in range(frames):
        ctx["scale_ok"] &= _synchronize_lidar(lidar, stage, UsdGeom)
        ctx["robot"].apply_wheel_actions(ctx["stop_action"])
        ctx["simulation_context"].step(render=True)
        ctx["scale_ok"] &= _synchronize_lidar(lidar, stage, UsdGeom)
        samples.append((_point_count(ctx["point_annotator"].get_data()), ctx["flat_annotator"].get_data()))
    return samples


def _stack_scans(samples, size: int):
    """(frames x rays) matrix of FlatScan ranges (NaN = invalid) for scans of the given size."""
    import numpy as np

    scans = [_flat_ranges(flat) for _, flat in samples]
    scans = [scan for scan in scans if scan.size == size]
    return np.vstack(scans) if scans else np.zeros((0, size))


def _in_window(ranges, radius_m: float):
    """Rays at the expected distance of a wall at radius_m; nothing below MIN_WALL_HIT_RANGE_M."""
    import numpy as np

    far_edge = radius_m / math.cos(math.atan2(0.5 * WALL_WIDTH_M, radius_m))
    with np.errstate(invalid="ignore"):
        return (
            (ranges >= max(radius_m - WALL_RANGE_SLACK_M, MIN_WALL_HIT_RANGE_M))
            & (ranges <= far_edge + WALL_RANGE_SLACK_M)
        )


def _expected_span_deg(radius_m: float) -> float:
    return 2.0 * math.degrees(math.atan2(0.5 * WALL_WIDTH_M, radius_m))


def _voted_hits(mask_stack):
    """Rays whose mask is true in >= WALL_HIT_VOTE_FRACTION of frames."""
    import numpy as np

    if mask_stack.shape[0] == 0:
        return np.zeros(0, dtype=int)
    return np.nonzero(mask_stack.mean(axis=0) >= WALL_HIT_VOTE_FRACTION)[0]


def _report_wall(label, direction_deg, radius_m, wall_position, hits, stack, azimuths):
    """Print one wall result; return its mean FlatScan angle if detected, else None."""
    import numpy as np

    log(f"--- {label} wall_direction={direction_deg:.1f} deg (sensor frame) radius={radius_m:.2f} m")
    log(f"    wall_world_position={wall_position}")
    log(f"    frames={stack.shape[0]}")
    log(f"    detected_rays={int(hits.size)}")
    if hits.size == 0:
        log("    mean_flat_angle=n/a")
        log("    angle_span=n/a")
        log("    detected=no")
        return None
    hit_angles = azimuths[hits]
    mean_angle = _circular_mean_deg(hit_angles)
    offsets = [_wrap_deg(a - mean_angle) for a in hit_angles]
    span = max(offsets) - min(offsets)
    with np.errstate(all="ignore"):
        hit_distance = float(np.nanmedian(stack[:, hits]))
    detected = hits.size >= MIN_WALL_RAYS and span <= _expected_span_deg(radius_m) + WALL_SPAN_MARGIN_DEG
    log(f"    mean_flat_angle={mean_angle:.2f} deg index_range=[{hits.min()}, {hits.max()}]")
    log(f"    angle_span={span:.2f} deg (expected {_expected_span_deg(radius_m):.2f})")
    log(f"    median_hit_distance={hit_distance:.4f} m")
    log(f"    detected={'yes' if detected else 'no'}")
    return mean_angle if detected else None


def _fit_mapping(label: str, rows) -> None:
    """Fit flat_angle = sign * sensor_angle + offset over detected (sensor_angle, flat_angle) rows."""
    log(f"================ {label} MAPPING FIT ================")
    log(f"{label} DETECTED_WALLS={len(rows)}/{len(PHASE_A_WALLS)}")
    if len(rows) < 2:
        log(f"{label} BEST_MAPPING=insufficient detections")
        return
    fits = {}
    for sign in (+1, -1):
        offsets = [_wrap_deg(flat - sign * sensor) for sensor, flat in rows]
        fits[sign] = (_circular_mean_deg(offsets), _circular_spread_deg(offsets))
        log(
            f"{label} HYPOTHESIS flat = {'+' if sign > 0 else '-'}sensor_angle + offset: "
            f"offset={fits[sign][0]:.2f} deg max_residual={fits[sign][1]:.2f} deg"
        )
    sign = min(fits, key=lambda s: fits[s][1])
    offset, residual = fits[sign]
    log(
        f"{label} BEST_MAPPING flat_angle = {'+' if sign > 0 else '-'}sensor_angle "
        f"{'+' if offset >= 0 else '-'} {abs(offset):.2f} deg (max_residual={residual:.2f} deg)"
    )
    log(f"{label} SENSOR_FORWARD(+X)_FLAT_ANGLE={_wrap_deg(offset):.2f} deg")
    log(f"{label} SENSOR_LEFT(+Y)_FLAT_ANGLE={_wrap_deg(sign * 90.0 + offset):.2f} deg")
    log(f"{label} SENSOR_RIGHT(-Y)_FLAT_ANGLE={_wrap_deg(-sign * 90.0 + offset):.2f} deg")
    log(f"{label} SENSOR_BACK(-X)_FLAT_ANGLE={_wrap_deg(180.0 + offset):.2f} deg")
    for sensor_angle, flat in rows:
        predicted = _wrap_deg(sign * sensor_angle + offset)
        log(
            f"    sensor={sensor_angle:7.1f} flat={flat:8.2f} "
            f"predicted={predicted:8.2f} residual={_wrap_deg(flat - predicted):6.2f}"
        )


def _wait_for_scan(ctx) -> None:
    """Wait for SCAN_STABLE_FRAMES consecutive non-empty FlatScans of equal size."""
    update, stable, last_size, max_points = 0, 0, None, 0
    while stable < SCAN_STABLE_FRAMES:
        if update >= SCAN_WAIT_MAX_UPDATES:
            raise RuntimeError(
                f"FlatScan never produced {SCAN_STABLE_FRAMES} consecutive non-empty scans of equal size "
                f"within {SCAN_WAIT_MAX_UPDATES} updates (last samples={last_size}, max raw POINT_COUNT={max_points})"
            )
        count, flat = _step(ctx, 1)[0]
        update += 1
        max_points = max(max_points, count)
        size = _flat_ranges(flat).size
        stable = stable + 1 if size and size == last_size else (1 if size else 0)
        last_size = size
        if update % SCAN_WAIT_LOG_EVERY == 0 or stable == SCAN_STABLE_FRAMES:
            log(f"WAIT_FOR_VALID_SCAN update={update} samples={size} stable={stable} raw_points={count}")


def _scan_size(samples) -> int:
    sizes = {_flat_ranges(flat).size for _, flat in samples} - {0}
    if len(sizes) != 1:
        raise RuntimeError(f"No stable FlatScan size: {sizes}")
    return sizes.pop()


def main() -> bool:
    import gc
    import numpy as np
    import omni.usd
    from pxr import Gf, UsdGeom
    from isaacsim.core.api import SimulationContext
    from isaacsim.robot.wheeled_robots.controllers.differential_controller import DifferentialController
    from isaacsim.robot.wheeled_robots.robots.wheeled_robot import WheeledRobot
    from isaacsim.sensors.rtx import LidarRtx

    stage = None
    robot = None
    stop_action = None
    lidar = None
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
        for path in (TEMP_SENSOR_PATH, TEMP_WALL_ROOT):
            if stage.GetPrimAtPath(path).IsValid():
                raise RuntimeError(f"Temporary path already exists: {path}")

        simulation_context = SimulationContext()
        simulation_context.initialize_physics()
        simulation_context.play()
        for _ in range(SETTLE_FRAMES):
            simulation_context.step(render=True)

        robot = WheeledRobot(
            prim_path=ARTICULATION_PATH,
            name="tb03_azimuth_cal_robot",
            wheel_dof_names=[LEFT_WHEEL_DOF, RIGHT_WHEEL_DOF],
        )
        robot.initialize()
        controller = DifferentialController(
            name="tb03_azimuth_cal_controller",
            wheel_radius=WHEEL_RADIUS_M,
            wheel_base=WHEEL_BASE_M,
            max_linear_speed=0.22,
            max_angular_speed=1.0,
        )
        stop_action = controller.forward(np.array([0.0, 0.0], dtype=np.float64))
        robot.apply_wheel_actions(stop_action)

        log(f"MOUNT_OFFSET_M=({', '.join(f'{v:.3f}' for v in MOUNT_OFFSET_M)}) (reference; clean-area mode)")
        _clean_area_report(stage, UsdGeom)
        sensor_yaw = math.degrees(_yaw_from_quat(_base_rigid_rotation(stage, UsdGeom)))
        log(f"SENSOR_YAW_FROM_BASE_LINK_DEG={sensor_yaw:.3f}")

        # ================= PHASE A: static walls created BEFORE the LiDAR =================
        UsdGeom.Xform.Define(stage, TEMP_WALL_ROOT)
        wall_positions = {}
        for direction, radius in PHASE_A_WALLS.items():
            wall_positions[direction] = _make_wall(
                stage, UsdGeom, Gf, _wall_path(direction), sensor_yaw, direction, radius
            )

        lidar = LidarRtx(
            prim_path=TEMP_SENSOR_PATH,
            name="tb03_azimuth_cal_lidar",
            config_file_name=LIDAR_CONFIG,
            **{NEAR_RANGE_ATTRIBUTE: NEAR_RANGE_M},
        )
        lidar.initialize()
        _synchronize_lidar(lidar, stage, UsdGeom)
        lidar.attach_annotator(POINT_CLOUD_ANNOTATOR)
        lidar.attach_annotator(FLAT_SCAN_ANNOTATOR)
        annotators = lidar.get_annotators()
        ctx = {
            "stage": stage,
            "UsdGeom": UsdGeom,
            "robot": robot,
            "stop_action": stop_action,
            "simulation_context": simulation_context,
            "lidar": lidar,
            "point_annotator": annotators[POINT_CLOUD_ANNOTATOR],
            "flat_annotator": annotators[FLAT_SCAN_ANNOTATOR],
            "scale_ok": True,
        }
        log(f"CONFIGURED_MIN_RANGE={stage.GetPrimAtPath(lidar.prim_path).GetAttribute(NEAR_RANGE_ATTRIBUTE).Get()}")

        log("================ PHASE A: STATIC WALLS (created before LiDAR) ================")
        _wait_for_scan(ctx)
        samples = _step(ctx, MEASURE_FRAMES)
        size = _scan_size(samples)
        reference = next(flat for _, flat in reversed(samples) if _flat_ranges(flat).size == size)
        azimuths = _azimuths_deg(reference, size)
        stack = _stack_scans(samples, size)
        log(
            f"FLAT_numCols={size} FLAT_azimuthRange={[float(v) for v in reference['azimuthRange']]} "
            f"FLAT_horizontalResolution={reference['horizontalResolution']} "
            f"FLAT_depthRange={[float(v) for v in reference['depthRange']]}"
        )
        log(f"PHASE_A POINT_COUNT={max(count for count, _ in samples)}")
        with np.errstate(invalid="ignore"):
            valid_votes = np.isfinite(stack).mean(axis=0) >= WALL_HIT_VOTE_FRACTION
        any_window = np.zeros(size, dtype=bool)
        rows_a = []
        for direction, radius in PHASE_A_WALLS.items():
            hits = _voted_hits(_in_window(stack, radius))
            any_window[hits] = True
            flat = _report_wall("PHASE_A", direction, radius, wall_positions[direction], hits, stack, azimuths)
            if flat is not None:
                rows_a.append((direction, flat))
        log(f"PHASE_A valid_rays={int(valid_votes.sum())} unexplained_valid_rays={int((valid_votes & ~any_window).sum())}")
        _fit_mapping("PHASE_A", rows_a)

        # ================= PHASE B: walls created/removed WHILE the LiDAR runs =================
        log("================ PHASE B: RUNTIME WALLS (created while LiDAR runs) ================")
        stage.RemovePrim(TEMP_WALL_ROOT)
        _step(ctx, 2 * WALL_FLUSH_FRAMES)
        baseline = _step(ctx, MEASURE_FRAMES)
        baseline_stack = _stack_scans(baseline, size)
        with np.errstate(all="ignore"):
            baseline_median = np.nanmedian(baseline_stack, axis=0) if baseline_stack.shape[0] else np.full(size, np.nan)
        baseline_valid = int(np.isfinite(baseline_median).sum())
        log(f"PHASE_B no-wall valid_rays={baseline_valid} (≈0 expected if runtime removal is seen)")
        baseline_clutter = _in_window(baseline_stack, PHASE_B_RADIUS_M).any(axis=0)
        rows_b = []
        UsdGeom.Xform.Define(stage, TEMP_WALL_ROOT)
        for direction in PHASE_A_WALLS:
            path = _wall_path(direction)
            try:
                position = _make_wall(stage, UsdGeom, Gf, path, sensor_yaw, direction, PHASE_B_RADIUS_M)
                wall_samples = _step(ctx, WALL_FRAMES)
            finally:
                if stage.GetPrimAtPath(path).IsValid():
                    stage.RemovePrim(path)
            _step(ctx, WALL_FLUSH_FRAMES)
            wall_stack = _stack_scans(wall_samples, size)
            with np.errstate(invalid="ignore"):
                changed = np.isnan(baseline_median)[None, :] | (
                    np.abs(wall_stack - baseline_median[None, :]) > CHANGED_RAY_THRESHOLD_M
                )
            hits = _voted_hits(_in_window(wall_stack, PHASE_B_RADIUS_M) & changed)
            hits = hits[~baseline_clutter[hits]]
            flat = _report_wall("PHASE_B", direction, PHASE_B_RADIUS_M, position, hits, wall_stack, azimuths)
            if flat is not None:
                rows_b.append((direction, flat))
        _fit_mapping("PHASE_B", rows_b)

        log("================ SUMMARY ================")
        log(f"PHASE_A_STATIC_WALLS_DETECTED={len(rows_a)}/{len(PHASE_A_WALLS)}")
        log(f"PHASE_B_RUNTIME_WALLS_DETECTED={len(rows_b)}/{len(PHASE_A_WALLS)}")
        if len(rows_a) >= 2 and not rows_b:
            log("DIAGNOSIS=RUNTIME_CREATED_GEOMETRY_NOT_SEEN_BY_RTX_LIDAR")
        log(f"LIDAR_SCALE_UNIT={'PASS' if ctx['scale_ok'] else 'FAIL'}")
        return len(rows_a) == len(PHASE_A_WALLS)
    except Exception as error:
        log(f"FAIL: {error}")
        return False
    finally:
        if robot is not None and stop_action is not None:
            try:
                robot.apply_wheel_actions(stop_action)
            except Exception:
                pass
        if stage is not None and stage.GetPrimAtPath(TEMP_WALL_ROOT).IsValid():
            stage.RemovePrim(TEMP_WALL_ROOT)
        if lidar is not None and stage is not None:
            _release_lidar(lidar, stage)
        gc.collect()


if __name__ == "__main__":
    try:
        main()
    finally:
        log("closing SimulationApp without saving stage")
        simulation_app.close()
