#!/usr/bin/env python3
"""Isolated near-field experiment: one wall and a fresh static RTX sensor per case.

Uses an empty in-memory stage. No external scene is opened or saved.
Run with Isaac Sim's python.sh; ordinary Python is only for syntax checking.
"""
from copy import deepcopy

DISTANCES = (0.20, 0.30, 0.40, 0.50, 0.60, 1.20)
CENTER = (50.0, 50.0, 1.0)
UPDATES = 360
RAW = "IsaacExtractRTXSensorPointCloudNoAccumulator"
FLAT = "IsaacComputeRTXLidarFlatScan"
NEAR = "omni:sensor:Core:nearRangeM"
FAR = "omni:sensor:Core:farRangeM"
ECHO = "omni:sensor:Core:minDistBetweenEchosM"
# Two times the profile's 0.02 m range accuracy; no range setting is changed.
MATCH_TOLERANCE_M = 0.04


def log(message):
    print(f"[TB03_LIDAR_NEAR_CLEAN] {message}", flush=True)


def release(lidar, stage, sensor_path):
    """Release the same wrapper resources used by the existing diagnostics."""
    if lidar is not None:
        lidar.pause()
        for name, method in (("_acquisition_callback", "reset"),
                             ("_stage_open_callback", "reset"),
                             ("_timer_reset_callback", "unsubscribe")):
            callback = getattr(lidar, name, None)
            if callback is not None:
                getattr(callback, method)()
                setattr(lidar, name, None)
        lidar.detach_all_annotators()
        lidar.detach_all_writers()
        product = getattr(lidar, "_render_product", None)
        if product is not None:
            product.destroy()
            lidar._render_product = None
    if stage.GetPrimAtPath(sensor_path).IsValid():
        stage.RemovePrim(sensor_path)


def run_case(app, stage, timeline, index, distance):
    import numpy as np
    from pxr import Gf, Usd, UsdGeom
    from isaacsim.sensors.rtx import LidarRtx

    sensor_path = f"/World/near_sensor_{index}"
    wall_path = f"/World/near_wall_{index}"
    lidar = None
    timeline.stop()
    log(f"TARGET_DISTANCE={distance:.2f}")
    try:
        # The ONLY surface: a narrow front wall, authored BEFORE the sensor.
        # Its front face, not its center, is at the requested distance.
        # Narrow width prevents far oblique corners from disguising a near cutoff.
        cube = UsdGeom.Cube.Define(stage, wall_path)
        cube.CreateSizeAttr(1.0)
        transform = UsdGeom.XformCommonAPI(cube.GetPrim())
        transform.SetTranslate(Gf.Vec3d(CENTER[0] + distance + 0.01, CENTER[1], CENTER[2]))
        transform.SetScale(Gf.Vec3f(0.02, 0.30, 0.60))
        cube.CreateDisplayColorAttr([Gf.Vec3f(0.7, 0.7, 0.7)])
        cube.CreateDisplayOpacityAttr([1.0])
        app.update()

        lidar = LidarRtx(
            prim_path=sensor_path,
            name=f"near_sensor_{index}",
            translation=np.array(CENTER),
            orientation=np.array([1.0, 0.0, 0.0, 0.0]),
            config_file_name="Example_Rotary_2D",
            **{NEAR: 0.10},
        )
        prim = stage.GetPrimAtPath(sensor_path)
        near, far = prim.GetAttribute(NEAR).Get(), prim.GetAttribute(FAR).Get()
        log(f"nearRangeM={near} farRangeM={far}")
        log(f"minDistBetweenEchosM={prim.GetAttribute(ECHO).Get()} (unchanged)")
        if not np.allclose([near, far], [0.10, 200.0], atol=1e-6, rtol=0):
            raise RuntimeError("Live range attributes do not equal (0.10, 200.0)")

        lidar.initialize()
        product = lidar.get_render_product_path()
        log(f"render_product_path={product}")
        if not product:
            raise RuntimeError("Missing render product")
        lidar.attach_annotator(RAW)
        lidar.attach_annotator(FLAT)
        annotators = lidar.get_annotators()

        matrix = UsdGeom.XformCache().GetLocalToWorldTransform(prim)
        origin = np.array(matrix.ExtractTranslation(), dtype=float)
        bounds = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render]
        ).ComputeWorldBound(cube.GetPrim()).ComputeAlignedRange()
        lower, upper = np.array(bounds.GetMin()), np.array(bounds.GetMax())
        actual = float(lower[0] - origin[0])
        if actual <= 0 or not (lower[1] <= origin[1] <= upper[1]
                               and lower[2] <= origin[2] <= upper[2]):
            raise RuntimeError("Sensor is not facing the wall front face")
        log(f"SENSOR_POSITION={origin.tolist()}")
        log(f"WALL_WORLD_BOUNDS={lower.tolist()} .. {upper.tolist()}")
        log(f"ACTUAL_SENSOR_TO_WALL_DISTANCE={actual:.6f}")

        timeline.play()
        max_points = 0
        first_points = None
        snapshot = {}
        snapshot_update = None
        best_valid = -1
        latest_metadata = {}
        for update in range(1, UPDATES + 1):
            app.update()
            raw = annotators[RAW].get_data()
            points = raw.get("data") if isinstance(raw, dict) else None
            count = len(points) if points is not None else 0
            max_points = max(max_points, count)
            if count and first_points is None:
                first_points = update
            flat = annotators[FLAT].get_data()
            if isinstance(flat, dict):
                latest_metadata = flat
                ranges = np.asarray(flat.get("linearDepthData", [])).reshape(-1)
                cols = int(flat.get("numCols", 0))
                valid_count = int(np.count_nonzero(np.isfinite(ranges) & (ranges > 1e-3)))
                # Copy buffers: subsequent RTX updates may overwrite them.
                # Retain a real complete scan instead of requiring the last update
                # to coincide with the rotating sensor's full-scan emission.
                if cols > 0 and ranges.size == cols and valid_count >= best_valid:
                    snapshot = deepcopy(flat)
                    snapshot_update = update
                    best_valid = valid_count
            if update % 60 == 0:
                log(f"update={update} raw_points={count} time={timeline.get_current_time():.6f}")

        flat = snapshot or latest_metadata
        ranges = np.asarray(flat.get("linearDepthData", []), dtype=float).reshape(-1)
        cols = int(flat.get("numCols", 0))
        valid = np.isfinite(ranges) & (ranges > 1e-3)
        azimuth_bounds = np.asarray(flat.get("azimuthRange", []), dtype=float).reshape(-1)
        depth = np.asarray(flat.get("depthRange", []), dtype=float).reshape(-1)
        matched = np.zeros(ranges.size, dtype=bool)
        angles = np.array([], dtype=float)
        if cols > 0 and ranges.size == cols and azimuth_bounds.size == 2:
            # Same degree-valued endpoint convention as installed RTX tests.
            angles = np.linspace(azimuth_bounds[0], azimuth_bounds[1], cols)
            radians = np.deg2rad(angles)
            with np.errstate(invalid="ignore"):
                hit_x = ranges * np.cos(radians)
                hit_y = ranges * np.sin(radians)
            matched = (valid & (np.cos(radians) > 0)
                       & (np.abs(hit_x - actual) <= MATCH_TOLERANCE_M)
                       & (hit_y >= lower[1] - origin[1] - MATCH_TOLERANCE_M)
                       & (hit_y <= upper[1] - origin[1] + MATCH_TOLERANCE_M))
        matches = ranges[matched]
        depth_ok = depth.size == 2 and np.allclose(depth, [0.10, 200.0], atol=1e-6, rtol=0)
        detected = bool(matches.size > 0)
        passed = bool(detected and max_points > 0 and depth_ok and snapshot)
        log(f"POINT_COUNT={max_points} (maximum per-update raw count)")
        log(f"FIRST_NONEMPTY_POINT_UPDATE={first_points}")
        log(f"FLAT_SNAPSHOT_UPDATE={snapshot_update} (most populated complete scan)")
        log(f"FlatScan depthRange={depth.tolist()}")
        log(f"FLAT_numCols={cols}")
        log(f"FLAT_samples={ranges.size}")
        log(f"FLAT_finite_valid={int(valid.sum())}")
        log(f"FLAT_min={float(ranges[valid].min()) if valid.any() else None}")
        log(f"WALL_MATCHED_RAYS={matches.size}")
        log(f"MATCHED_AZIMUTHS_DEG={np.round(angles[matched], 4).tolist() if angles.size else []}")
        log(f"MIN_MATCHED_WALL_DISTANCE={float(matches.min()) if matches.size else None}")
        log(f"MAX_MATCHED_WALL_DISTANCE={float(matches.max()) if matches.size else None}")
        log(f"DETECTED={'yes' if detected else 'no'}")
        log(f"PASS={'PASS' if passed else 'FAIL'}")
        return distance, detected, passed
    finally:
        timeline.stop()
        release(lidar, stage, sensor_path)
        if stage.GetPrimAtPath(wall_path).IsValid():
            stage.RemovePrim(wall_path)
        app.update()


def main(app):
    import carb
    import omni.timeline
    import omni.usd
    from pxr import UsdGeom

    context = omni.usd.get_context()
    context.new_stage()
    app.update()
    stage = context.get_stage()
    if stage is None:
        raise RuntimeError("No in-memory stage")
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.Xform.Define(stage, "/World")
    setting = "/app/sensors/nv/lidar/outputBufferOnGPU"
    log(f"{setting}={carb.settings.get_settings().get(setting)} (unchanged)")
    log("Empty stage; one wall per fresh sensor; no imported geometry")
    log(f"MATCH_TOLERANCE_M={MATCH_TOLERANCE_M}")
    timeline = omni.timeline.get_timeline_interface()
    results = [run_case(app, stage, timeline, i, d) for i, d in enumerate(DISTANCES)]
    log("SUMMARY: DISTANCE | DETECTED | PASS")
    for distance, detected, passed in results:
        log(f"{distance:.2f} | {'yes' if detected else 'no'} | {'PASS' if passed else 'FAIL'}")
    if results[1][2]:
        log("RESULT=0.30 m detected in isolation: prior failure was scene/robot geometry "
            "or wall-placement/test context, not a hard 0.30 m LiDAR range limit")
    else:
        log("RESULT=0.30 m not validated in isolation; geometry alone does not explain it. "
            "Inspect reported scan validity and unchanged minDistBetweenEchosM next")
    overall = all(row[2] for row in results)
    log(f"OVERALL_PASS={'PASS' if overall else 'FAIL'}")
    return overall


if __name__ == "__main__":
    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": False, "disable_viewport_updates": False})
    try:
        success = main(simulation_app)
    except Exception as error:
        log(f"FAIL: {type(error).__name__}: {error}")
        raise
    finally:
        log("Closing without saving the in-memory stage")
        simulation_app.close()
    raise SystemExit(0 if success else 1)
