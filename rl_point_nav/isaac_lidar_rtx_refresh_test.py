#!/usr/bin/env python3
"""Runtime-only test: can RTX LiDAR be made to see geometry created after it starts?

Validated sensor (Example_Rotary_2D, nearRangeM = 0.10, unit scale) pinned in the
clean area (50, 50, 1) with identity rotation, so FlatScan angle = world angle - 0.07 deg.

Every test uses its OWN wall (unique prim path, direction and radius), so a
result can only come from that test's geometry:

  CONTROL_PRE_CREATED     wall created before LidarRtx                   (control)
  MOVE_EXISTING           wall created before LidarRtx, parked out of range,
                          moved into place after the sensor runs
  then, for walls created AFTER the sensor runs, one refresh method each:
  NONE                    only normal rendered steps                    (Experiment 1)
  APP_UPDATE              extra simulation_app.update() calls
  USDRT_SYNC              usdrt.Usd.Stage.Attach(stage_id).SynchronizeToFabric()
  USDRT_GET_PRIM          usdrt stage GetPrimAtPath(wall) (on-demand Fabric population)
  ANNOTATOR_RECREATE      detach + re-attach both annotators
  RENDER_PRODUCT_RECREATE destroy LidarRtx render product, create a new one, re-attach
  REPLICATOR_STEP         omni.replicator.core.orchestrator.step(pause_timeline=False)
  TIMELINE_STOP_PLAY      SimulationContext stop() + play()
  SENSOR_RECREATE         destroy LidarRtx, create a fresh one (all walls then pre-exist it)

Each runtime test measures BEFORE its refresh (must be invisible; otherwise it is
flagged CARRYOVER from an earlier method) and AFTER.

  --no-fsd   start Kit with /app/useFabricSceneDelegate=false

Nothing is saved; the USD is never modified. No robot, PPO, RL or ROS code is used.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path


SCENE_PATH = Path("/home/g8/Desktop/FInalSceneDemo.usd")
SENSOR_PATH_PREFIX = "/World/tb03_rtx_refresh_lidar"
WALL_ROOT = "/World/tb03_rtx_refresh_walls"
LIDAR_CONFIG = "Example_Rotary_2D"
POINT_CLOUD_ANNOTATOR = "IsaacExtractRTXSensorPointCloudNoAccumulator"
FLAT_SCAN_ANNOTATOR = "IsaacComputeRTXLidarFlatScan"
NEAR_RANGE_ATTRIBUTE = "omni:sensor:Core:nearRangeM"
NEAR_RANGE_M = 0.10
RENDER_PRODUCT_RESOLUTION = (128, 128)  # same as LidarRtx.__init__
FSD_SETTING = "/app/useFabricSceneDelegate"
INVALID_RANGE_EPSILON_M = 1e-3

CLEAN_CENTER_M = (50.0, 50.0, 1.0)
PARK_POSITION_M = (1000.0, 1000.0, 1.0)  # beyond farRangeM (200 m)
FLAT_ANGLE_OFFSET_DEG = -0.07            # from azimuth calibration
RUNTIME_METHODS = (
    "NONE",
    "APP_UPDATE",
    "USDRT_SYNC",
    "USDRT_GET_PRIM",
    "ANNOTATOR_RECREATE",
    "RENDER_PRODUCT_RECREATE",
    "REPLICATOR_STEP",
    "TIMELINE_STOP_PLAY",
    "SENSOR_RECREATE",
)
TEST_NAMES = ("CONTROL_PRE_CREATED", "MOVE_EXISTING", *RUNTIME_METHODS)
# One slot per test: 11 directions 360/11 deg apart, radii 0.25 m apart.
SLOTS = {
    name: (round(-180.0 + index * 360.0 / len(TEST_NAMES), 2), 1.50 + 0.25 * index)
    for index, name in enumerate(TEST_NAMES)
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
APP_UPDATE_COUNT = 30
REPLICATOR_STEP_COUNT = 5
TAG = "[TB03_RTX_REFRESH]"

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


# ---------------------------------------------------------------- harness

class Harness:
    def __init__(self, stage, simulation_context):
        self.stage = stage
        self.sim = simulation_context
        self.frame = 0
        self.lidar = None
        self.sensor_path = None
        self.sensor_count = 0
        self.lidar_created = None  # (frame, sim_time)

    # -- time / stepping
    def now(self) -> str:
        return f"frame={self.frame} sim_t={self.sim.current_time:.3f}s"

    def step(self, frames: int) -> list:
        samples = []
        for _ in range(frames):
            self.sim.step(render=True)
            self.frame += 1
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

    # -- sensor
    def create_sensor(self) -> None:
        import numpy as np
        from pxr import UsdGeom
        from isaacsim.sensors.rtx import LidarRtx

        self.sensor_count += 1
        self.sensor_path = f"{SENSOR_PATH_PREFIX}_{self.sensor_count}"
        mpu = UsdGeom.GetStageMetersPerUnit(self.stage)
        self.lidar = LidarRtx(
            prim_path=self.sensor_path,
            name=f"tb03_rtx_refresh_lidar_{self.sensor_count}",
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
        self.lidar_created = self.now()
        log(f"LIDAR_CREATED path={self.sensor_path} {self.lidar_created}")
        updates = self.wait_for_scan()
        log(f"WAIT_FOR_VALID_SCAN updates={updates}")

    def release_sensor(self) -> None:
        if self.lidar is not None:
            _release_lidar(self.lidar, self.stage, self.sensor_path)
            self.lidar = None

    # -- walls
    def make_wall(self, name: str, parked: bool = False) -> str:
        from pxr import Gf, UsdGeom

        direction, radius = SLOTS[name]
        path = f"{WALL_ROOT}/{name.lower()}"
        mpu = UsdGeom.GetStageMetersPerUnit(self.stage)
        cube = UsdGeom.Cube.Define(self.stage, path)
        cube.CreateSizeAttr(1.0)
        xform = UsdGeom.XformCommonAPI(cube.GetPrim())
        xform.SetTranslate(Gf.Vec3d(*(v / mpu for v in (PARK_POSITION_M if parked else _wall_pose(name)))))
        xform.SetRotate(Gf.Vec3f(0.0, 0.0, direction))
        xform.SetScale(Gf.Vec3f(WALL_THICKNESS_M / mpu, WALL_WIDTH_M / mpu, WALL_HEIGHT_M / mpu))
        cube.CreateDisplayColorAttr([Gf.Vec3f(0.8, 0.2, 0.2)])
        return path

    def move_wall_into_place(self, name: str) -> None:
        from pxr import Gf, UsdGeom

        mpu = UsdGeom.GetStageMetersPerUnit(self.stage)
        prim = self.stage.GetPrimAtPath(f"{WALL_ROOT}/{name.lower()}")
        UsdGeom.XformCommonAPI(prim).SetTranslate(Gf.Vec3d(*(v / mpu for v in _wall_pose(name))))

    # -- measurement
    def measure(self, names) -> tuple[dict, dict]:
        """Return (scan stats, {name: (wall_rays, mean_angle, detected)})."""
        import numpy as np

        samples = self.step(MEASURE_FRAMES)
        sizes = [_flat_ranges(flat).size for _, flat in samples]
        size = max(set(sizes) - {0}, key=sizes.count, default=0)
        stats = {"point_count": max(count for count, _ in samples), "num_cols": 0, "finite_valid": 0}
        if size == 0:
            return stats, {name: (0, None, False) for name in names}
        stack = np.vstack([_flat_ranges(flat) for _, flat in samples if _flat_ranges(flat).size == size])
        reference = next(flat for _, flat in reversed(samples) if _flat_ranges(flat).size == size)
        stats["num_cols"] = int(reference.get("numCols", size))
        stats["finite_valid"] = int((np.isfinite(stack).mean(axis=0) >= WALL_HIT_VOTE_FRACTION).sum())
        azimuths = float(reference["azimuthRange"][0]) + np.arange(size) * float(reference["horizontalResolution"])
        walls = {}
        for name in names:
            direction, radius = SLOTS[name]
            far_edge = radius / math.cos(math.atan2(0.5 * WALL_WIDTH_M, radius))
            with np.errstate(invalid="ignore"):
                in_window = (stack >= radius - WALL_RANGE_SLACK_M) & (stack <= far_edge + WALL_RANGE_SLACK_M)
            hits = np.nonzero(in_window.mean(axis=0) >= WALL_HIT_VOTE_FRACTION)[0]
            mean_angle = _circular_mean_deg(azimuths[hits]) if hits.size else None
            detected = (
                hits.size >= MIN_WALL_RAYS
                and mean_angle is not None
                and abs(_wrap_deg(mean_angle - _wrap_deg(direction + FLAT_ANGLE_OFFSET_DEG))) <= ANGLE_TOLERANCE_DEG
            )
            walls[name] = (int(hits.size), mean_angle, detected)
        return stats, walls

    # -- refresh methods
    def refresh(self, method: str) -> None:
        if method == "NONE":
            return
        if method == "APP_UPDATE":
            for _ in range(APP_UPDATE_COUNT):
                simulation_app.update()
        elif method in ("USDRT_SYNC", "USDRT_GET_PRIM"):
            import omni.usd
            from usdrt import Sdf as RtSdf, Usd as RtUsd

            rt_stage = RtUsd.Stage.Attach(omni.usd.get_context().get_stage_id())
            if method == "USDRT_SYNC":
                rt_stage.SynchronizeToFabric()
            else:
                prim = rt_stage.GetPrimAtPath(RtSdf.Path(f"{WALL_ROOT}/{method.lower()}"))
                log(f"    USDRT_PRIM_VALID={prim.IsValid() if prim else False}")
        elif method == "ANNOTATOR_RECREATE":
            self.lidar.detach_all_annotators()
            self.lidar.attach_annotator(POINT_CLOUD_ANNOTATOR)
            self.lidar.attach_annotator(FLAT_SCAN_ANNOTATOR)
            self.wait_for_scan()
        elif method == "RENDER_PRODUCT_RECREATE":
            import omni.replicator.core as rep

            self.lidar.detach_all_annotators()
            self.lidar._render_product.destroy()
            self.lidar._render_product = rep.create.render_product(self.sensor_path, resolution=RENDER_PRODUCT_RESOLUTION)
            self.lidar._render_product_path = self.lidar._render_product.path
            self.lidar.attach_annotator(POINT_CLOUD_ANNOTATOR)
            self.lidar.attach_annotator(FLAT_SCAN_ANNOTATOR)
            self.wait_for_scan()
        elif method == "REPLICATOR_STEP":
            import omni.replicator.core as rep

            for _ in range(REPLICATOR_STEP_COUNT):
                rep.orchestrator.step(pause_timeline=False)
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


def _wall_pose(name: str):
    direction, radius = SLOTS[name]
    rad = math.radians(direction)
    distance = radius + 0.5 * WALL_THICKNESS_M
    return (
        CLEAN_CENTER_M[0] + distance * math.cos(rad),
        CLEAN_CENTER_M[1] + distance * math.sin(rad),
        CLEAN_CENTER_M[2],
    )


def _print_result(test_name, geometry_time, lidar_time, method, stats, wall, before=None) -> str:
    rays, mean_angle, detected = wall
    carryover = before is not None and before[2]
    verdict = "CARRYOVER" if carryover else ("PASS" if detected else "FAIL")
    direction, radius = SLOTS[test_name]
    log(f"TEST_NAME={test_name}")
    log(f"    GEOMETRY_CREATED_TIME={geometry_time}")
    log(f"    LIDAR_CREATED_TIME={lidar_time}")
    log(f"    REFRESH_METHOD={method}")
    log(f"    wall dir={direction:.2f} deg r={radius:.2f} m")
    if before is not None:
        log(f"    BEFORE_REFRESH wall_rays={before[0]} detected={'yes' if before[2] else 'no'}")
    log(f"    POINT_COUNT={stats['point_count']}")
    log(f"    FLAT_numCols={stats['num_cols']}")
    log(f"    FINITE_VALID={stats['finite_valid']}")
    log(
        f"    WALL_RAYS={rays} WALL_MEAN_FLAT_ANGLE={'n/a' if mean_angle is None else f'{mean_angle:.2f}'}"
    )
    log(f"    RESULT={verdict}")
    return verdict


def main() -> bool:
    import gc
    import carb
    import omni.usd
    from pxr import UsdGeom
    from isaacsim.core.api import SimulationContext

    stage = None
    harness = None
    try:
        usd_context = omni.usd.get_context()
        while usd_context.get_stage_loading_status()[2] > 0:
            simulation_app.update()
        stage = usd_context.get_stage()
        if stage is None:
            raise RuntimeError("No active USD stage after opening the scene")
        if stage.GetPrimAtPath(WALL_ROOT).IsValid():
            raise RuntimeError(f"Temporary path already exists: {WALL_ROOT}")
        log(f"{FSD_SETTING}={carb.settings.get_settings().get(FSD_SETTING)} (--no-fsd={_args.no_fsd})")
        for name, (direction, radius) in SLOTS.items():
            log(f"SLOT {name:<24} dir={direction:8.2f} deg r={radius:.2f} m")

        simulation_context = SimulationContext()
        simulation_context.initialize_physics()
        simulation_context.play()
        for _ in range(SETTLE_FRAMES):
            simulation_context.step(render=True)
        harness = Harness(stage, simulation_context)
        UsdGeom.Xform.Define(stage, WALL_ROOT)

        # ---- geometry created BEFORE the sensor
        control_time = harness.now()
        harness.make_wall("CONTROL_PRE_CREATED")
        move_time = harness.now()
        harness.make_wall("MOVE_EXISTING", parked=True)
        harness.create_sensor()

        verdicts = {}
        stats, walls = harness.measure(["CONTROL_PRE_CREATED", "MOVE_EXISTING"])
        verdicts["CONTROL_PRE_CREATED"] = _print_result(
            "CONTROL_PRE_CREATED", control_time, harness.lidar_created, "NONE", stats, walls["CONTROL_PRE_CREATED"]
        )
        parked_seen = walls["MOVE_EXISTING"][2]
        log(f"MOVE_EXISTING while parked detected={'yes' if parked_seen else 'no'} (must be no)")

        # ---- movement of pre-existing geometry
        harness.move_wall_into_place("MOVE_EXISTING")
        stats, walls = harness.measure(["MOVE_EXISTING"])
        verdicts["MOVE_EXISTING"] = _print_result(
            "MOVE_EXISTING", f"{move_time} (moved at {harness.now()})", harness.lidar_created,
            "XFORM_TRANSLATE_AFTER_START", stats, walls["MOVE_EXISTING"],
        )

        # ---- geometry created AFTER the sensor, one refresh method each
        for method in RUNTIME_METHODS:
            geometry_time = harness.now()
            harness.make_wall(method)
            lidar_time = harness.lidar_created
            _, before = harness.measure([method])
            try:
                harness.refresh(method)
            except Exception as error:
                log(f"TEST_NAME={method} REFRESH_ERROR={type(error).__name__}: {error}")
                verdicts[method] = "ERROR"
                if harness.lidar is None:
                    harness.create_sensor()
                continue
            stats, walls = harness.measure([method])
            verdicts[method] = _print_result(
                method, geometry_time, lidar_time, method, stats, walls[method], before=before[method]
            )

        # ---- final: every wall against the (possibly recreated) current sensor
        stats, walls = harness.measure(list(TEST_NAMES))
        log("================ SUMMARY ================")
        log(f"{FSD_SETTING}={carb.settings.get_settings().get(FSD_SETTING)}")
        log(f"{'TEST_NAME':<24} {'RESULT':>10} {'SEEN_BY_FINAL_SENSOR':>22}")
        for name in TEST_NAMES:
            log(f"{name:<24} {verdicts[name]:>10} {'yes' if walls[name][2] else 'no':>22}")
        log(f"FINAL_SENSOR={harness.sensor_path} created {harness.lidar_created}")
        return verdicts["CONTROL_PRE_CREATED"] == "PASS"
    except Exception as error:
        log(f"FAIL: {error}")
        return False
    finally:
        if harness is not None:
            harness.release_sensor()
        if stage is not None and stage.GetPrimAtPath(WALL_ROOT).IsValid():
            stage.RemovePrim(WALL_ROOT)
        gc.collect()


if __name__ == "__main__":
    try:
        main()
    finally:
        log("closing SimulationApp without saving stage")
        simulation_app.close()
