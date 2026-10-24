#!/usr/bin/env python3
"""Strict A/B near-field experiment: only minimum echo distance differs by mode.

Uses an empty in-memory stage. No external scene is opened or saved.
Run with Isaac Sim's python.sh; ordinary Python is only for syntax checking.
"""
from copy import deepcopy

DISTANCES = (0.15, 0.20, 0.30, 0.40, 0.60)
MODES = ("CONTROL", "OVERRIDE")
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
    print(f"[TB03_LIDAR_MIN_ECHO] {message}", flush=True)


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


def run_case(app, stage, timeline, index, distance, mode):
    import numpy as np
    from pxr import Gf, Usd, UsdGeom
    from isaacsim.sensors.rtx import LidarRtx

    sensor_path = f"/World/near_sensor_{index}"
    wall_path = f"/World/near_wall_{index}"
    lidar = None
    timeline.stop()
    log(f"MODE={mode}")
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

        # Both modes share the 0.10 m near range. Only OVERRIDE authors ECHO.
        # Installed LidarRtx passes these kwargs to the sensor creation command
        # before constructing its render product.
        overrides = {NEAR: 0.10}
        if mode == "OVERRIDE":
            overrides[ECHO] = 0.10
        lidar = LidarRtx(
            prim_path=sensor_path,
            name=f"near_sensor_{index}",
            translation=np.array(CENTER),
            orientation=np.array([1.0, 0.0, 0.0, 0.0]),
            config_file_name="Example_Rotary_2D",
            **overrides,
        )
        prim = stage.GetPrimAtPath(sensor_path)
        near, far = prim.GetAttribute(NEAR).Get(), prim.GetAttribute(FAR).Get()
        log(f"nearRangeM={near} farRangeM={far}")
        echo_attr = prim.GetAttribute(ECHO)
        if not echo_attr.IsValid() or str(echo_attr.GetTypeName()) != "float":
            raise RuntimeError("Missing or unexpected minimum echo distance attribute type")
        log(f"ECHO_ATTRIBUTE_TYPE={echo_attr.GetTypeName()}")
        log(f"ECHO_ATTRIBUTE_METADATA={echo_attr.GetAllMetadata()}")
        log(f"ECHO_HAS_AUTHORED_VALUE={echo_attr.HasAuthoredValueOpinion()}")
        echo_before = float(echo_attr.Get())
        log(f"minDistBetweenEchosM={echo_before}")
        expected_echo = 0.10 if mode == "OVERRIDE" else 0.40
        if not np.isclose(echo_before, expected_echo, atol=1e-6, rtol=0):
            raise RuntimeError(f"{mode} echo distance rejected/unexpected: {echo_before}; "
                               "no documented smaller valid bound exists; no guessed fallback")
        if not np.allclose([near, far], [0.10, 200.0], atol=1e-6, rtol=0):
            raise RuntimeError("Live range attributes do not equal (0.10, 200.0)")

        lidar.initialize()
        echo_initialized = float(echo_attr.Get())
        log(f"MIN_ECHO_AFTER_INITIALIZATION={echo_initialized}")
        if not np.isclose(echo_initialized, expected_echo, atol=1e-6, rtol=0):
            raise RuntimeError("Minimum echo distance did not survive initialization")
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
        echo_final = float(echo_attr.Get())
        log(f"MIN_ECHO_AFTER_COLLECTION={echo_final}")
        live_ranges = [prim.GetAttribute(NEAR).Get(), prim.GetAttribute(FAR).Get()]
        config_ok = bool(np.isclose(echo_final, expected_echo, atol=1e-6, rtol=0)
                         and np.allclose(live_ranges, [0.10, 200.0], atol=1e-6, rtol=0))
        # Missing scans alone never prove a cutoff. The final paired decision
        # also requires the CONTROL's farther walls to work and all OVERRIDE
        # walls to yield complete scans/raw returns.
        scan_ok = bool(max_points > 0 and depth_ok and snapshot)
        no_returns = bool(max_points == 0 and not valid.any())
        expected_detection = mode == "OVERRIDE" or distance >= 0.40
        passed = bool(config_ok and ((detected and scan_ok) if expected_detection
                                    else (not detected and no_returns)))
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
        return {"mode": mode, "distance": distance, "detected": detected,
                "passed": passed, "scan_ok": scan_ok, "config_ok": config_ok,
                "no_returns": no_returns, "echo": echo_final}
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
    results = []
    # Finish CONTROL first: its live attribute type/metadata are inspected before
    # any OVERRIDE sensor is constructed. Every case gets a unique prim and RP.
    for mode in MODES:
        for distance in DISTANCES:
            results.append(run_case(app, stage, timeline, len(results), distance, mode))
    control = [row for row in results if row["mode"] == "CONTROL"]
    override = [row for row in results if row["mode"] == "OVERRIDE"]
    log(f"CONTROL_MIN_ECHO_DISTANCE={[row['echo'] for row in control]}")
    log(f"OVERRIDE_MIN_ECHO_DISTANCE={[row['echo'] for row in override]}")
    for mode in MODES:
        log(f"{mode}:")
        for row in results:
            if row["mode"] == mode:
                log(f"{row['distance']:.2f} detected={'yes' if row['detected'] else 'no'} "
                    f"expected_behavior={'PASS' if row['passed'] else 'FAIL'}")
    applied = all(row["config_ok"] for row in override)
    configs_ok = applied and all(row["config_ok"] for row in control)
    # YES requires the entire specified A/B pattern, including functioning far
    # controls. NO is reserved for a working pipeline with no detection change.
    overall = configs_ok and all(row["passed"] for row in results)
    far_ok = all(row["scan_ok"] and row["detected"] for row in results
                 if row["distance"] >= 0.40)
    same_pattern = all(a["detected"] == b["detected"] for a, b in zip(control, override))
    observations_resolved = all((row["detected"] and row["scan_ok"]) or row["no_returns"]
                                for row in results)
    cause = "YES" if overall else (
        "NO" if configs_ok and far_ok and same_pattern and observations_resolved
        else "INCONCLUSIVE")
    log(f"OVERRIDE_APPLIED={'PASS' if applied else 'FAIL'}")
    log(f"MIN_ECHO_DISTANCE_CAUSES_NEAR_FIELD_CUTOFF={cause}")
    log(f"OVERALL_PASS={'PASS' if overall else 'FAIL'}")
    return overall


if __name__ == "__main__":
    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": False, "disable_viewport_updates": False})
    try:
        success = main(simulation_app)
    except Exception as error:
        log(f"FAIL: {type(error).__name__}: {error}")
        log("OVERRIDE_APPLIED=FAIL (test incomplete or configuration rejected)")
        log("MIN_ECHO_DISTANCE_CAUSES_NEAR_FIELD_CUTOFF=INCONCLUSIVE")
        log("OVERALL_PASS=FAIL")
        raise
    finally:
        log("Closing without saving the in-memory stage")
        simulation_app.close()
    raise SystemExit(0 if success else 1)
