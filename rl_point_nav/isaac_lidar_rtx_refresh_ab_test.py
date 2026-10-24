#!/usr/bin/env python3
"""Strict single-method A/B: can a NEW USD prim become visible to a running RTX LiDAR?

Run ONE method per fresh SimulationApp process (no other method runs first):

  1. open scene, start timeline
  2. create LidarRtx (validated config: Example_Rotary_2D, nearRangeM = 0.10,
     unit scale, pinned in the clean area (50, 50, 1), identity rotation)
  3. wait for a stable FlatScan
  4. NO_WALL measurement: confirm the wall window is empty
  5. create ONE new wall (never-used prim path) while the LiDAR runs
  6. BEFORE_REFRESH measurement
  7. apply exactly ONE refresh method
  8. AFTER_REFRESH measurement

  --method {NONE, APP_UPDATE, USDRT_SYNC, TIMELINE_STOP_PLAY, SENSOR_RECREATE}
  --no-fsd  start Kit with /app/useFabricSceneDelegate=false (optional)

Nothing is saved; the USD is never modified. No robot, PPO, RL or ROS code is used.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path


SCENE_PATH = Path("/home/g8/Desktop/FInalSceneDemo.usd")
SENSOR_PATH_PREFIX = "/World/tb03_rtx_ab_lidar"
WALL_PATH = "/World/tb03_rtx_ab_wall"
LIDAR_CONFIG = "Example_Rotary_2D"
POINT_CLOUD_ANNOTATOR = "IsaacExtractRTXSensorPointCloudNoAccumulator"
FLAT_SCAN_ANNOTATOR = "IsaacComputeRTXLidarFlatScan"
NEAR_RANGE_ATTRIBUTE = "omni:sensor:Core:nearRangeM"
NEAR_RANGE_M = 0.10
FSD_SETTING = "/app/useFabricSceneDelegate"
INVALID_RANGE_EPSILON_M = 1e-3
METHODS = ("NONE", "APP_UPDATE", "USDRT_SYNC", "TIMELINE_STOP_PLAY", "SENSOR_RECREATE")

CLEAN_CENTER_M = (50.0, 50.0, 1.0)
FLAT_ANGLE_OFFSET_DEG = -0.07  # from azimuth calibration
WALL_DIRECTION_DEG = 0.0
WALL_RADIUS_M = 2.0
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
APP_UPDATE_COUNT = 30
TAG = "[TB03_RTX_AB]"

_arg_parser = argparse.ArgumentParser(add_help=False)
_arg_parser.add_argument("--method", choices=METHODS, required=True)
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


class Harness:
    def __init__(self, stage, simulation_context):
        self.stage = stage
        self.sim = simulation_context
        self.lidar = None
        self.sensor_path = None
        self.sensor_count = 0

    def step(self, frames: int) -> list:
        samples = []
        for _ in range(frames):
            self.sim.step(render=True)
            annotators = self.lidar.get_annotators()
            samples.append(
                (
                    _point_count(annotators[POINT_CLOUD_ANNOTATOR].get_data()),
                    annotators[FLAT_SCAN_ANNOTATOR].get_data(),
                )
            )
        return samples

    def wait_for_scan(self) -> int:
        update, stable, last_size = 0, 0, None
        while stable < SCAN_STABLE_FRAMES:
            if update >= SCAN_WAIT_MAX_UPDATES:
                raise RuntimeError(f"FlatScan never stabilized (last samples={last_size})")
            _, flat = self.step(1)[0]
            update += 1
            size = _flat_ranges(flat).size
            stable = stable + 1 if size and size == last_size else (1 if size else 0)
            last_size = size
        return update

    def create_sensor(self) -> None:
        import numpy as np
        from pxr import UsdGeom
        from isaacsim.sensors.rtx import LidarRtx

        self.sensor_count += 1
        self.sensor_path = f"{SENSOR_PATH_PREFIX}_{self.sensor_count}"
        mpu = UsdGeom.GetStageMetersPerUnit(self.stage)
        self.lidar = LidarRtx(
            prim_path=self.sensor_path,
            name=f"tb03_rtx_ab_lidar_{self.sensor_count}",
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
        log(f"LIDAR_CREATED path={self.sensor_path} sim_t={self.sim.current_time:.3f}s")
        log(f"WAIT_FOR_VALID_SCAN updates={self.wait_for_scan()}")

    def release_sensor(self) -> None:
        if self.lidar is not None:
            _release_lidar(self.lidar, self.stage, self.sensor_path)
            self.lidar = None

    def make_wall(self) -> None:
        from pxr import Gf, UsdGeom

        mpu = UsdGeom.GetStageMetersPerUnit(self.stage)
        rad = math.radians(WALL_DIRECTION_DEG)
        distance = WALL_RADIUS_M + 0.5 * WALL_THICKNESS_M
        position = (
            CLEAN_CENTER_M[0] + distance * math.cos(rad),
            CLEAN_CENTER_M[1] + distance * math.sin(rad),
            CLEAN_CENTER_M[2],
        )
        cube = UsdGeom.Cube.Define(self.stage, WALL_PATH)
        cube.CreateSizeAttr(1.0)
        xform = UsdGeom.XformCommonAPI(cube.GetPrim())
        xform.SetTranslate(Gf.Vec3d(*(v / mpu for v in position)))
        xform.SetRotate(Gf.Vec3f(0.0, 0.0, WALL_DIRECTION_DEG))
        xform.SetScale(Gf.Vec3f(WALL_THICKNESS_M / mpu, WALL_WIDTH_M / mpu, WALL_HEIGHT_M / mpu))
        cube.CreateDisplayColorAttr([Gf.Vec3f(0.8, 0.2, 0.2)])
        log(f"WALL_CREATED path={WALL_PATH} sim_t={self.sim.current_time:.3f}s")

    def measure(self, label: str) -> tuple[int, bool]:
        """Return (wall_rays, detected) over MEASURE_FRAMES frames."""
        import numpy as np

        samples = self.step(MEASURE_FRAMES)
        sizes = [_flat_ranges(flat).size for _, flat in samples]
        size = max(set(sizes) - {0}, key=sizes.count, default=0)
        rays, mean_angle, detected, finite_valid = 0, None, False, 0
        if size:
            stack = np.vstack([_flat_ranges(flat) for _, flat in samples if _flat_ranges(flat).size == size])
            reference = next(flat for _, flat in reversed(samples) if _flat_ranges(flat).size == size)
            azimuths = float(reference["azimuthRange"][0]) + np.arange(size) * float(reference["horizontalResolution"])
            finite_valid = int((np.isfinite(stack).mean(axis=0) >= WALL_HIT_VOTE_FRACTION).sum())
            far_edge = WALL_RADIUS_M / math.cos(math.atan2(0.5 * WALL_WIDTH_M, WALL_RADIUS_M))
            with np.errstate(invalid="ignore"):
                in_window = (stack >= WALL_RADIUS_M - WALL_RANGE_SLACK_M) & (stack <= far_edge + WALL_RANGE_SLACK_M)
            hits = np.nonzero(in_window.mean(axis=0) >= WALL_HIT_VOTE_FRACTION)[0]
            rays = int(hits.size)
            if hits.size:
                mean_angle = _circular_mean_deg(azimuths[hits])
                expected = _wrap_deg(WALL_DIRECTION_DEG + FLAT_ANGLE_OFFSET_DEG)
                detected = hits.size >= MIN_WALL_RAYS and abs(_wrap_deg(mean_angle - expected)) <= ANGLE_TOLERANCE_DEG
        log(
            f"{label}: POINT_COUNT={max(count for count, _ in samples)} FLAT_numCols={size} "
            f"FINITE_VALID={finite_valid} wall_rays={rays} "
            f"mean_flat_angle={'n/a' if mean_angle is None else f'{mean_angle:.2f}'} "
            f"detected={'yes' if detected else 'no'}"
        )
        return rays, detected

    def refresh(self, method: str) -> None:
        if method == "NONE":
            return
        if method == "APP_UPDATE":
            for _ in range(APP_UPDATE_COUNT):
                simulation_app.update()
        elif method == "USDRT_SYNC":
            import omni.usd
            from usdrt import Usd as RtUsd

            RtUsd.Stage.Attach(omni.usd.get_context().get_stage_id()).SynchronizeToFabric()
        elif method == "TIMELINE_STOP_PLAY":
            self.sim.stop()
            simulation_app.update()
            self.sim.play()
            self.wait_for_scan()
        elif method == "SENSOR_RECREATE":
            self.release_sensor()
            self.create_sensor()
        else:
            raise ValueError(method)


def main() -> bool:
    import gc
    import carb
    import omni.usd
    from isaacsim.core.api import SimulationContext

    stage = None
    harness = None
    method = _args.method
    try:
        usd_context = omni.usd.get_context()
        while usd_context.get_stage_loading_status()[2] > 0:
            simulation_app.update()
        stage = usd_context.get_stage()
        if stage is None:
            raise RuntimeError("No active USD stage after opening the scene")
        if stage.GetPrimAtPath(WALL_PATH).IsValid():
            raise RuntimeError(f"Temporary path already exists: {WALL_PATH}")
        log(f"METHOD={method}")
        log(f"{FSD_SETTING}={carb.settings.get_settings().get(FSD_SETTING)}")

        simulation_context = SimulationContext()
        simulation_context.initialize_physics()
        simulation_context.play()
        for _ in range(SETTLE_FRAMES):
            simulation_context.step(render=True)

        harness = Harness(stage, simulation_context)
        harness.create_sensor()
        _, no_wall_detected = harness.measure("NO_WALL")
        harness.make_wall()
        before_rays, before_detected = harness.measure("BEFORE_REFRESH")
        log(f"APPLY_REFRESH={method}")
        harness.refresh(method)
        after_rays, after_detected = harness.measure("AFTER_REFRESH")

        if no_wall_detected:
            verdict = "INVALID_NO_WALL_DETECTED"
        elif before_detected:
            verdict = "VISIBLE_WITHOUT_REFRESH"
        else:
            verdict = "PASS" if after_detected else "FAIL"
        log("================ RESULT ================")
        log(f"METHOD={method}")
        log(f"BEFORE_WALL_RAYS={before_rays}")
        log(f"AFTER_WALL_RAYS={after_rays}")
        log(f"BEFORE_DETECTED={'yes' if before_detected else 'no'}")
        log(f"AFTER_DETECTED={'yes' if after_detected else 'no'}")
        log(f"RESULT={verdict}")
        return verdict == "PASS"
    except Exception as error:
        log(f"METHOD={method} FAIL: {type(error).__name__}: {error}")
        return False
    finally:
        if harness is not None:
            harness.release_sensor()
        if stage is not None and stage.GetPrimAtPath(WALL_PATH).IsValid():
            stage.RemovePrim(WALL_PATH)
        gc.collect()


if __name__ == "__main__":
    try:
        main()
    finally:
        log("closing SimulationApp without saving stage")
        simulation_app.close()
