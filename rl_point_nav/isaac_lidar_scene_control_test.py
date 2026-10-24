#!/usr/bin/env python3
"""RTX LiDAR scene-control diagnostic using temporary geometry only.

The scene is opened but never saved. Four temporary walls and one temporary
LiDAR are placed far from all existing geometry, then removed before shutdown.
"""
from __future__ import annotations

from pathlib import Path


SCENE_PATH = Path("/home/g8/Desktop/FInalSceneDemo.usd")
TEST_CENTER = (50.0, 50.0, 1.0)
TEMP_SENSOR_PATH = "/World/tb03_scene_control_lidar"
TEMP_WALLS = {
    "/World/tb03_scene_control_wall_pos_x": ((53.0, 50.0, 1.0), (0.2, 4.0, 2.0)),
    "/World/tb03_scene_control_wall_neg_x": ((47.0, 50.0, 1.0), (0.2, 4.0, 2.0)),
    "/World/tb03_scene_control_wall_pos_y": ((50.0, 53.0, 1.0), (4.0, 0.2, 2.0)),
    "/World/tb03_scene_control_wall_neg_y": ((50.0, 47.0, 1.0), (4.0, 0.2, 2.0)),
}
LIDAR_CONFIG = "Example_Rotary_2D"
POINT_CLOUD_ANNOTATOR = "IsaacExtractRTXSensorPointCloudNoAccumulator"
FLAT_SCAN_ANNOTATOR = "IsaacComputeRTXLidarFlatScan"
OUTPUT_BUFFER_SETTING = "/app/sensors/nv/lidar/outputBufferOnGPU"
NUM_UPDATES = 360
INVALID_RANGE_EPSILON_M = 1e-3


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


def _make_wall(stage, UsdGeom, Gf, path: str, position, scale) -> None:
    """Create a unit cube with opaque display color and requested world scale."""
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    xform = UsdGeom.XformCommonAPI(cube.GetPrim())
    xform.SetTranslate(Gf.Vec3d(*position))
    xform.SetScale(Gf.Vec3f(*scale))
    cube.CreateDisplayColorAttr([Gf.Vec3f(0.7, 0.7, 0.7)])


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


def _flat_stats(data):
    import numpy as np

    if not isinstance(data, dict):
        return 0, 0, 0, None, None
    ranges = np.asarray(data.get("linearDepthData", [])).reshape(-1)
    num_cols = int(data.get("numCols", 0))
    valid = ranges[np.isfinite(ranges) & (ranges > INVALID_RANGE_EPSILON_M)]
    return (
        num_cols,
        int(ranges.size),
        int(valid.size),
        float(valid.min()) if valid.size else None,
        float(valid.max()) if valid.size else None,
    )


def main() -> bool:
    import carb
    import numpy as np
    import omni.timeline
    import omni.usd
    from pxr import Gf, UsdGeom
    from isaacsim.sensors.rtx import LidarRtx

    lidar = None
    timeline = None
    stage = None
    try:
        if not SCENE_PATH.is_file():
            raise FileNotFoundError(f"Scene does not exist: {SCENE_PATH}")

        usd_context = omni.usd.get_context()
        while usd_context.get_stage_loading_status()[2] > 0:
            simulation_app.update()
        stage = usd_context.get_stage()
        if stage is None:
            raise RuntimeError("No active USD stage after opening the scene")
        if stage.GetPrimAtPath(TEMP_SENSOR_PATH).IsValid() or any(
            stage.GetPrimAtPath(path).IsValid() for path in TEMP_WALLS
        ):
            raise RuntimeError("One or more temporary scene-control paths already exist")

        for path, (position, scale) in TEMP_WALLS.items():
            _make_wall(stage, UsdGeom, Gf, path, position, scale)

        lidar = LidarRtx(
            prim_path=TEMP_SENSOR_PATH,
            name="scene_control_lidar",
            translation=np.array(TEST_CENTER, dtype=np.float64),
            orientation=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
            config_file_name=LIDAR_CONFIG,
        )
        lidar.initialize()
        render_product_path = lidar.get_render_product_path()
        if not render_product_path:
            raise RuntimeError("LidarRtx did not create a render product")
        lidar.attach_annotator(POINT_CLOUD_ANNOTATOR)
        lidar.attach_annotator(FLAT_SCAN_ANNOTATOR)
        point_annotator = lidar.get_annotators()[POINT_CLOUD_ANNOTATOR]
        flat_annotator = lidar.get_annotators()[FLAT_SCAN_ANNOTATOR]

        print(
            f"[TB03_LIDAR_SCENE_CONTROL] STAGE_METERS_PER_UNIT="
            f"{UsdGeom.GetStageMetersPerUnit(stage)}",
            flush=True,
        )
        print(f"[TB03_LIDAR_SCENE_CONTROL] STAGE_UP_AXIS={UsdGeom.GetStageUpAxis(stage)}", flush=True)
        print(f"[TB03_LIDAR_SCENE_CONTROL] TEST_CENTER={TEST_CENTER}", flush=True)
        print(f"[TB03_LIDAR_SCENE_CONTROL] render_product_path={render_product_path}", flush=True)
        print(
            f"[TB03_LIDAR_SCENE_CONTROL] {OUTPUT_BUFFER_SETTING}="
            f"{carb.settings.get_settings().get(OUTPUT_BUFFER_SETTING)}",
            flush=True,
        )

        cache = UsdGeom.XformCache()
        for path in TEMP_WALLS:
            wall = stage.GetPrimAtPath(path)
            print(
                f"[TB03_LIDAR_SCENE_CONTROL] wall_exists path={path} exists={wall.IsValid()} "
                f"world_transform={cache.GetLocalToWorldTransform(wall) if wall.IsValid() else 'n/a'}",
                flush=True,
            )

        timeline = omni.timeline.get_timeline_interface()
        timeline.play()
        first_nonempty_point_update = None
        max_point_count = 0
        best_flat_data = None
        best_flat_score = (-1, -1)
        for update_index in range(1, NUM_UPDATES + 1):
            simulation_app.update()
            point_count = _point_count(point_annotator.get_data())
            max_point_count = max(max_point_count, point_count)
            if point_count > 0 and first_nonempty_point_update is None:
                first_nonempty_point_update = update_index

            flat_data = flat_annotator.get_data()
            num_cols, samples, finite_valid, _, _ = _flat_stats(flat_data)
            score = (finite_valid, samples)
            if score > best_flat_score:
                best_flat_data = flat_data
                best_flat_score = score

        num_cols, samples, finite_valid, minimum, maximum = _flat_stats(best_flat_data)
        raw_pass = max_point_count > 0
        flat_pass = num_cols > 0 and samples > 0 and finite_valid > 0
        print(f"[TB03_LIDAR_SCENE_CONTROL] POINT_COUNT={max_point_count}", flush=True)
        print(
            f"[TB03_LIDAR_SCENE_CONTROL] FIRST_NONEMPTY_POINT_UPDATE={first_nonempty_point_update}",
            flush=True,
        )
        print(f"[TB03_LIDAR_SCENE_CONTROL] FLAT_numCols={num_cols}", flush=True)
        print(f"[TB03_LIDAR_SCENE_CONTROL] FLAT_samples={samples}", flush=True)
        print(f"[TB03_LIDAR_SCENE_CONTROL] FLAT_finite_valid={finite_valid}", flush=True)
        print(
            f"[TB03_LIDAR_SCENE_CONTROL] FLAT_min="
            f"{'n/a' if minimum is None else f'{minimum:.4f} m'}",
            flush=True,
        )
        print(
            f"[TB03_LIDAR_SCENE_CONTROL] FLAT_max="
            f"{'n/a' if maximum is None else f'{maximum:.4f} m'}",
            flush=True,
        )
        print(f"[TB03_LIDAR_SCENE_CONTROL] PASS_RAW={'PASS' if raw_pass else 'FAIL'}", flush=True)
        print(f"[TB03_LIDAR_SCENE_CONTROL] PASS_FLAT={'PASS' if flat_pass else 'FAIL'}", flush=True)
        return raw_pass or flat_pass
    except Exception as error:
        print(f"[TB03_LIDAR_SCENE_CONTROL] FAIL: {error}", flush=True)
        return False
    finally:
        if timeline is not None:
            timeline.stop()
        if lidar is not None:
            for annotator_name in (POINT_CLOUD_ANNOTATOR, FLAT_SCAN_ANNOTATOR):
                try:
                    lidar.detach_annotator(annotator_name)
                except Exception:
                    pass
        if stage is not None:
            if stage.GetPrimAtPath(TEMP_SENSOR_PATH).IsValid():
                stage.RemovePrim(TEMP_SENSOR_PATH)
            for path in TEMP_WALLS:
                if stage.GetPrimAtPath(path).IsValid():
                    stage.RemovePrim(path)


if __name__ == "__main__":
    try:
        main()
    finally:
        print("[TB03_LIDAR_SCENE_CONTROL] closing SimulationApp without saving stage", flush=True)
        simulation_app.close()
