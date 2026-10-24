#!/usr/bin/env python3
"""Runtime-only test: which obstacle-spawning patterns does the RTX LiDAR see?

Background: isaac_lidar_azimuth_calibration_test.py showed walls created BEFORE
LidarRtx are detected (8/8) while walls created AFTER it starts are not (0/8).
This test isolates the cause with one wall per case, each at its own
direction + radius so every case is identified by range window AND angle:

  C0 PRE_CREATED       created before the sensor                     (control)
  C1 POOL_TELEPORT     created before the sensor, parked out of range,
                       moved into place at runtime (Xform change only)
  C2 POOL_VISIBILITY   created before the sensor INVISIBLE, made visible at runtime
  C3 RUNTIME_NEW_PATH  brand-new, never-used prim path created at runtime
  C4 RUNTIME_PAUSED    brand-new prim created while the timeline is paused
  S2 SENSOR_RECREATE   after all runtime actions, the sensor is destroyed and a
                       fresh LidarRtx is created; every case is measured again

  --no-fsd   start Kit with /app/useFabricSceneDelegate=false (A/B for Fabric sync)

Sensor: Example_Rotary_2D, nearRangeM = 0.10, unit scale, pinned in the clean
area (50, 50, 1) with identity rotation, so FlatScan angle ~= world angle
(calibrated mapping: flat = sensor_angle - 0.07 deg).
Nothing is saved; the USD is never modified. No robot, PPO, RL or ROS code is used.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path


SCENE_PATH = Path("/home/g8/Desktop/FInalSceneDemo.usd")
SENSOR_PATHS = ("/World/tb03_runtime_geo_lidar_s1", "/World/tb03_runtime_geo_lidar_s2")
WALL_ROOT = "/World/tb03_runtime_geo_walls"
LIDAR_CONFIG = "Example_Rotary_2D"
POINT_CLOUD_ANNOTATOR = "IsaacExtractRTXSensorPointCloudNoAccumulator"
FLAT_SCAN_ANNOTATOR = "IsaacComputeRTXLidarFlatScan"
NEAR_RANGE_ATTRIBUTE = "omni:sensor:Core:nearRangeM"
NEAR_RANGE_M = 0.10
FSD_SETTING = "/app/useFabricSceneDelegate"
INVALID_RANGE_EPSILON_M = 1e-3

CLEAN_CENTER_M = (50.0, 50.0, 1.0)
PARK_POSITION_M = (1000.0, 1000.0, 1.0)  # beyond farRangeM (200 m)
FLAT_ANGLE_OFFSET_DEG = -0.07            # from azimuth calibration
# case -> (world direction deg, radius m)
CASES = {
    "C0_PRE_CREATED": (0.0, 1.50),
    "C1_POOL_TELEPORT": (90.0, 2.00),
    "C2_POOL_VISIBILITY": (180.0, 2.50),
    "C3_RUNTIME_NEW_PATH": (-90.0, 3.00),
    "C4_RUNTIME_PAUSED": (45.0, 1.75),
}
WALL_THICKNESS_M = 0.05
WALL_WIDTH_M = 0.60
WALL_HEIGHT_M = 1.00
WALL_RANGE_SLACK_M = 0.08
WALL_HIT_VOTE_FRACTION = 0.5
MIN_WALL_RAYS = 3
ANGLE_TOLERANCE_DEG = 10.0
SETTLE_FRAMES = 8
SCAN_STABLE_FRAMES = 10
SCAN_WAIT_MAX_UPDATES = 600
MEASURE_FRAMES = 45
PAUSED_UPDATES = 10
TAG = "[TB03_RUNTIME_GEO]"

_arg_parser = argparse.ArgumentParser(add_help=False)
_arg_parser.add_argument("--no-fsd", action="store_true")
_args = _arg_parser.parse_known_args()[0]


from isaacsim import SimulationApp


simulation_app = SimulationApp(
    {
        "headless": False,
        "open_usd": str(SCENE_PATH),
        "create_new_stage": False,
        "fast_shutdown": False,
        "disable_viewport_updates": False,
        "extra_args": [f"--{FSD_SETTING}=false"] if _args.no_fsd else [],
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


def _flat_ranges(data):
    """FlatScan ranges with invalid rays as NaN."""
    import numpy as np

    if not isinstance(data, dict):
        return np.zeros(0)
    ranges = np.asarray(data.get("linearDepthData", []), dtype=np.float64).reshape(-1)
    return np.where(np.isfinite(ranges) & (ranges > INVALID_RANGE_EPSILON_M), ranges, np.nan)


def _release_lidar(lidar, stage, sensor_path) -> None:
    """Copied from isaac_lidar_dynamic_follow_test.py, parameterized by path."""
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


# ---------------------------------------------------------------- walls

def _wall_pose(direction_deg: float, radius_m: float):
    rad = math.radians(direction_deg)
    distance = radius_m + 0.5 * WALL_THICKNESS_M
    return (
        CLEAN_CENTER_M[0] + distance * math.cos(rad),
        CLEAN_CENTER_M[1] + distance * math.sin(rad),
        CLEAN_CENTER_M[2],
    )


def _make_wall(stage, UsdGeom, Gf, path: str, direction_deg: float, position_m, visible: bool = True):
    mpu = UsdGeom.GetStageMetersPerUnit(stage)
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    xform = UsdGeom.XformCommonAPI(cube.GetPrim())
    xform.SetTranslate(Gf.Vec3d(*(v / mpu for v in position_m)))
    xform.SetRotate(Gf.Vec3f(0.0, 0.0, direction_deg))
    xform.SetScale(Gf.Vec3f(WALL_THICKNESS_M / mpu, WALL_WIDTH_M / mpu, WALL_HEIGHT_M / mpu))
    cube.CreateDisplayColorAttr([Gf.Vec3f(0.8, 0.2, 0.2)])
    if not visible:
        UsdGeom.Imageable(cube.GetPrim()).MakeInvisible()
    return cube


# ---------------------------------------------------------------- sensor

class Sensor:
    def __init__(self, ctx, sensor_path: str):
        import numpy as np
        from isaacsim.sensors.rtx import LidarRtx

        self.ctx = ctx
        self.path = sensor_path
        mpu = ctx["UsdGeom"].GetStageMetersPerUnit(ctx["stage"])
        self.lidar = LidarRtx(
            prim_path=sensor_path,
            name=sensor_path.rsplit("/", 1)[-1],
            config_file_name=LIDAR_CONFIG,
            **{NEAR_RANGE_ATTRIBUTE: NEAR_RANGE_M},
        )
        self.lidar.initialize()
        self.lidar.set_world_pose(
            position=np.asarray([v / mpu for v in CLEAN_CENTER_M], dtype=np.float64),
            orientation=np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        )
        self.lidar.attach_annotator(POINT_CLOUD_ANNOTATOR)
        self.lidar.attach_annotator(FLAT_SCAN_ANNOTATOR)
        self.flat_annotator = self.lidar.get_annotators()[FLAT_SCAN_ANNOTATOR]
        self.point_annotator = self.lidar.get_annotators()[POINT_CLOUD_ANNOTATOR]

    def step(self, frames: int) -> list:
        samples = []
        for _ in range(frames):
            self.ctx["simulation_context"].step(render=True)
            samples.append((_point_count(self.point_annotator.get_data()), self.flat_annotator.get_data()))
        return samples

    def wait_for_scan(self) -> None:
        update, stable, last_size = 0, 0, None
        while stable < SCAN_STABLE_FRAMES:
            if update >= SCAN_WAIT_MAX_UPDATES:
                raise RuntimeError(f"{self.path}: FlatScan never stabilized (last samples={last_size})")
            _, flat = self.step(1)[0]
            update += 1
            size = _flat_ranges(flat).size
            stable = stable + 1 if size and size == last_size else (1 if size else 0)
            last_size = size
        log(f"WAIT_FOR_VALID_SCAN sensor={self.path} update={update} samples={last_size}")

    def release(self) -> None:
        _release_lidar(self.lidar, self.ctx["stage"], self.path)
        self.lidar = None


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


def _measure(sensor: Sensor, label: str, case_names) -> dict:
    """Measure MEASURE_FRAMES frames; report each requested case; return {case: detected}."""
    import numpy as np

    samples = sensor.step(MEASURE_FRAMES)
    sizes = {_flat_ranges(flat).size for _, flat in samples} - {0}
    log(f"==== MEASUREMENT {label} (sensor={sensor.path}) ====")
    log(f"    POINT_COUNT_MAX={max(count for count, _ in samples)} FLAT_sizes={sorted(sizes)}")
    if len(sizes) != 1:
        log("    no stable FlatScan in this window")
        return {name: False for name in case_names}
    size = sizes.pop()
    stack = np.vstack([_flat_ranges(flat) for _, flat in samples if _flat_ranges(flat).size == size])
    reference = next(flat for _, flat in reversed(samples) if _flat_ranges(flat).size == size)
    azimuths = float(reference["azimuthRange"][0]) + np.arange(size) * float(reference["horizontalResolution"])
    log(f"    valid_rays={int((np.isfinite(stack).mean(axis=0) >= WALL_HIT_VOTE_FRACTION).sum())}/{size}")

    results = {}
    for name in case_names:
        direction, radius = CASES[name]
        far_edge = radius / math.cos(math.atan2(0.5 * WALL_WIDTH_M, radius))
        with np.errstate(invalid="ignore"):
            in_window = (stack >= radius - WALL_RANGE_SLACK_M) & (stack <= far_edge + WALL_RANGE_SLACK_M)
        hits = np.nonzero(in_window.mean(axis=0) >= WALL_HIT_VOTE_FRACTION)[0]
        expected_flat = _wrap_deg(direction + FLAT_ANGLE_OFFSET_DEG)
        mean_angle = _circular_mean_deg(azimuths[hits]) if hits.size else None
        detected = (
            hits.size >= MIN_WALL_RAYS
            and mean_angle is not None
            and abs(_wrap_deg(mean_angle - expected_flat)) <= ANGLE_TOLERANCE_DEG
        )
        results[name] = detected
        log(
            f"    {name:<20} dir={direction:7.1f} r={radius:.2f} rays={int(hits.size):3d} "
            f"mean_flat_angle={'n/a' if mean_angle is None else f'{mean_angle:8.2f}'} "
            f"expected={expected_flat:8.2f} DETECTED={'yes' if detected else 'no'}"
        )
    return results


def main() -> bool:
    import gc
    import carb
    import omni.usd
    from pxr import Gf, UsdGeom
    from isaacsim.core.api import SimulationContext

    stage = None
    sensor = None
    try:
        usd_context = omni.usd.get_context()
        while usd_context.get_stage_loading_status()[2] > 0:
            simulation_app.update()
        stage = usd_context.get_stage()
        if stage is None:
            raise RuntimeError("No active USD stage after opening the scene")
        for path in (*SENSOR_PATHS, WALL_ROOT):
            if stage.GetPrimAtPath(path).IsValid():
                raise RuntimeError(f"Temporary path already exists: {path}")
        log(f"{FSD_SETTING}={carb.settings.get_settings().get(FSD_SETTING)} (--no-fsd={_args.no_fsd})")

        simulation_context = SimulationContext()
        simulation_context.initialize_physics()
        simulation_context.play()
        for _ in range(SETTLE_FRAMES):
            simulation_context.step(render=True)
        ctx = {"stage": stage, "UsdGeom": UsdGeom, "simulation_context": simulation_context}
        UsdGeom.Xform.Define(stage, WALL_ROOT)

        # ---- geometry that exists BEFORE sensor S1
        d0, r0 = CASES["C0_PRE_CREATED"]
        _make_wall(stage, UsdGeom, Gf, f"{WALL_ROOT}/c0_pre", d0, _wall_pose(d0, r0))
        d1, r1 = CASES["C1_POOL_TELEPORT"]
        pool_teleport = _make_wall(stage, UsdGeom, Gf, f"{WALL_ROOT}/c1_pool", d1, PARK_POSITION_M)
        d2, r2 = CASES["C2_POOL_VISIBILITY"]
        pool_hidden = _make_wall(stage, UsdGeom, Gf, f"{WALL_ROOT}/c2_hidden", d2, _wall_pose(d2, r2), visible=False)

        sensor = Sensor(ctx, SENSOR_PATHS[0])
        sensor.wait_for_scan()
        timeline = {}
        timeline["M0_START"] = _measure(sensor, "M0_START", list(CASES))

        # C1: move a pre-existing prim into place (transform change only)
        mpu = UsdGeom.GetStageMetersPerUnit(stage)
        UsdGeom.XformCommonAPI(pool_teleport.GetPrim()).SetTranslate(
            Gf.Vec3d(*(v / mpu for v in _wall_pose(d1, r1)))
        )
        timeline["M1_AFTER_TELEPORT"] = _measure(sensor, "M1_AFTER_TELEPORT", ["C1_POOL_TELEPORT"])

        # C2: make a pre-existing invisible prim visible
        UsdGeom.Imageable(pool_hidden.GetPrim()).MakeVisible()
        timeline["M2_AFTER_VISIBLE"] = _measure(sensor, "M2_AFTER_VISIBLE", ["C2_POOL_VISIBILITY"])

        # C3: brand-new prim path while playing
        d3, r3 = CASES["C3_RUNTIME_NEW_PATH"]
        _make_wall(stage, UsdGeom, Gf, f"{WALL_ROOT}/c3_new_while_playing", d3, _wall_pose(d3, r3))
        timeline["M3_AFTER_NEW_PRIM"] = _measure(sensor, "M3_AFTER_NEW_PRIM", ["C3_RUNTIME_NEW_PATH"])

        # C4: brand-new prim while paused, then resume
        simulation_context.pause()
        d4, r4 = CASES["C4_RUNTIME_PAUSED"]
        _make_wall(stage, UsdGeom, Gf, f"{WALL_ROOT}/c4_new_while_paused", d4, _wall_pose(d4, r4))
        for _ in range(PAUSED_UPDATES):
            simulation_app.update()
        simulation_context.play()
        timeline["M4_AFTER_PAUSED_PRIM"] = _measure(sensor, "M4_AFTER_PAUSED_PRIM", ["C4_RUNTIME_PAUSED"])

        # S2: destroy sensor, create a fresh one; all walls now pre-exist it
        sensor.release()
        sensor = None
        gc.collect()
        sensor = Sensor(ctx, SENSOR_PATHS[1])
        sensor.wait_for_scan()
        timeline["M5_FRESH_SENSOR"] = _measure(sensor, "M5_FRESH_SENSOR", list(CASES))

        log("================ SUMMARY ================")
        log(f"{FSD_SETTING}={carb.settings.get_settings().get(FSD_SETTING)}")
        runtime_result = {
            "C0_PRE_CREATED": timeline["M0_START"]["C0_PRE_CREATED"],
            "C1_POOL_TELEPORT": timeline["M1_AFTER_TELEPORT"]["C1_POOL_TELEPORT"],
            "C2_POOL_VISIBILITY": timeline["M2_AFTER_VISIBLE"]["C2_POOL_VISIBILITY"],
            "C3_RUNTIME_NEW_PATH": timeline["M3_AFTER_NEW_PRIM"]["C3_RUNTIME_NEW_PATH"],
            "C4_RUNTIME_PAUSED": timeline["M4_AFTER_PAUSED_PRIM"]["C4_RUNTIME_PAUSED"],
        }
        log(f"{'CASE':<20} {'SEEN_BY_RUNNING_SENSOR':>24} {'SEEN_BY_FRESH_SENSOR':>22}")
        for name in CASES:
            log(
                f"{name:<20} {'yes' if runtime_result[name] else 'no':>24} "
                f"{'yes' if timeline['M5_FRESH_SENSOR'][name] else 'no':>22}"
            )
        # Sanity: parked/hidden pool walls must not be seen before their runtime action.
        leaks = [n for n in ("C1_POOL_TELEPORT", "C2_POOL_VISIBILITY", "C3_RUNTIME_NEW_PATH", "C4_RUNTIME_PAUSED")
                 if timeline["M0_START"][n]]
        log(f"M0_UNEXPECTED_DETECTIONS={leaks if leaks else 'none'}")
        return runtime_result["C0_PRE_CREATED"]
    except Exception as error:
        log(f"FAIL: {error}")
        return False
    finally:
        if sensor is not None and stage is not None:
            sensor.release()
        if stage is not None and stage.GetPrimAtPath(WALL_ROOT).IsValid():
            stage.RemovePrim(WALL_ROOT)
        gc.collect()


if __name__ == "__main__":
    try:
        main()
    finally:
        log("closing SimulationApp without saving stage")
        simulation_app.close()
