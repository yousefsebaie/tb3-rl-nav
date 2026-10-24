#!/usr/bin/env python3
"""Runtime-only raw-versus-FlatScan RTX LiDAR diagnostic.

The test creates one temporary top-level Example_Rotary_2D LiDAR and poses it
once from the saved base_scan *world* transform.  It never moves or modifies
any saved USD prim, and it never reparents the temporary sensor.
"""
from __future__ import annotations

from pathlib import Path


SCENE_PATH = Path("/home/g8/Desktop/FInalSceneDemo.usd")
BASE_SCAN_PATH = "/World/turtlebot3_burger_02/a__namespace_base_link/base_scan"
TEMP_SENSOR_PATH = "/World/tb03_lidar_runtime_test"
LIDAR_CONFIG = "Example_Rotary_2D"
FLAT_SCAN_ANNOTATOR = "IsaacComputeRTXLidarFlatScan"
POINT_CLOUD_ANNOTATOR = "IsaacExtractRTXSensorPointCloudNoAccumulator"
MIN_RENDER_UPDATES = 180
MAX_RENDER_UPDATES = 360
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


def _quat_wxyz(quaternion) -> tuple[float, float, float, float]:
    normalized = quaternion.GetNormalized()
    imaginary = normalized.GetImaginary()
    return (
        float(normalized.GetReal()),
        float(imaginary[0]),
        float(imaginary[1]),
        float(imaginary[2]),
    )


def _format_vector(vector) -> str:
    return f"({float(vector[0]):.6f}, {float(vector[1]):.6f}, {float(vector[2]):.6f})"


def _rigid_pose_from_world(world_transform, Gf):
    """Extract world translation and normalized rotation while discarding scale."""
    _, _, scale, rotation_matrix, translation, _ = world_transform.Factor()
    rotation = rotation_matrix.ExtractRotationQuat().GetNormalized()
    rigid_world = Gf.Matrix4d(1.0)
    rigid_world.SetRotateOnly(Gf.Rotation(rotation))
    rigid_world.SetTranslateOnly(translation)
    return translation, rotation, scale, rigid_world


def _set_top_level_lidar_from_base_scan(stage, UsdGeom, Gf):
    """Pose the temporary LiDAR once at the saved base_scan rigid world pose."""
    cache = UsdGeom.XformCache()
    base_scan = stage.GetPrimAtPath(BASE_SCAN_PATH)
    lidar_prim = stage.GetPrimAtPath(TEMP_SENSOR_PATH)
    if not base_scan.IsValid() or not lidar_prim.IsValid():
        raise RuntimeError("Saved base_scan or temporary top-level LiDAR is absent")

    base_scan_world = cache.GetLocalToWorldTransform(base_scan)
    translation, rotation, scale, rigid_world = _rigid_pose_from_world(base_scan_world, Gf)
    parent_world = cache.GetLocalToWorldTransform(lidar_prim.GetParent())
    # The top-level sensor gets the same world translation and pure rotation as
    # base_scan, while its world transform deliberately contains no scale.
    UsdGeom.Xformable(lidar_prim).MakeMatrixXform().Set(parent_world.GetInverse() * rigid_world)
    return translation, rotation, scale


def _read_world_pose_and_scale(stage, UsdGeom, Gf, path):
    cache = UsdGeom.XformCache()
    world = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(path))
    return _rigid_pose_from_world(world, Gf)[:3]


def _flat_scan_from_data(data):
    """Extract completed FlatScan values from one annotator result mapping."""
    import numpy as np

    if not isinstance(data, dict):
        return None
    ranges = np.asarray(data.get("linearDepthData", [])).reshape(-1)
    num_cols = int(data.get("numCols", 0))
    if num_cols <= 0 or ranges.size <= 0:
        return None
    return {
        "num_cols": num_cols,
        "ranges": ranges,
        "azimuth_range": data.get("azimuthRange"),
        "horizontal_resolution": data.get("horizontalResolution"),
    }


def _scan_stats(scan):
    import numpy as np

    if scan is None:
        return 0, 0, 0, None, None
    ranges = np.asarray(scan["ranges"], dtype=np.float64).reshape(-1)
    valid = ranges[np.isfinite(ranges) & (ranges > INVALID_RANGE_EPSILON_M)]
    return (
        int(scan["num_cols"]),
        int(ranges.size),
        int(valid.size),
        float(valid.min()) if valid.size else None,
        float(valid.max()) if valid.size else None,
    )


def _scan_ready(scan) -> bool:
    num_cols, samples, finite_valid, _, _ = _scan_stats(scan)
    return num_cols > 0 and samples > 0 and finite_valid > 0


def _point_count(point_cloud_data) -> int:
    """Count raw point-cloud rows without printing or copying point arrays."""
    if isinstance(point_cloud_data, dict):
        points = point_cloud_data.get("data")
    else:
        points = point_cloud_data
    if points is None:
        return 0
    shape = getattr(points, "shape", None)
    if shape is not None and len(shape) >= 1:
        return int(shape[0])
    try:
        return len(points)
    except TypeError:
        return 0


def _mapping_keys(data):
    return sorted(data.keys()) if isinstance(data, dict) else "n/a"


def main() -> bool:
    import omni.kit.commands
    import omni.usd
    from pxr import Gf, UsdGeom
    from isaacsim.core.api import SimulationContext
    from isaacsim.sensors.rtx import LidarRtx

    lidar = None
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
        if not stage.GetPrimAtPath(BASE_SCAN_PATH).IsValid():
            raise RuntimeError(f"Missing saved base_scan prim: {BASE_SCAN_PATH}")
        if stage.GetPrimAtPath(TEMP_SENSOR_PATH).IsValid():
            raise RuntimeError(f"Temporary sensor path already exists: {TEMP_SENSOR_PATH}")

        # Preserve the proven top-level creation pattern. This temporary prim
        # remains a direct /World child until explicit in-memory cleanup.
        created, _ = omni.kit.commands.execute(
            "IsaacSensorCreateRtxLidar",
            path=TEMP_SENSOR_PATH,
            parent=None,
            config=LIDAR_CONFIG,
            translation=Gf.Vec3d(0.0, 0.0, 0.0),
            orientation=Gf.Quatd(1.0, 0.0, 0.0, 0.0),
        )
        lidar_prim = stage.GetPrimAtPath(TEMP_SENSOR_PATH)
        if not created or not lidar_prim.IsValid() or lidar_prim.GetTypeName() != "OmniLidar":
            raise RuntimeError("Failed to create temporary top-level Example_Rotary_2D OmniLidar")
        if lidar_prim.GetParent().GetPath().pathString != "/World":
            raise RuntimeError("Temporary LiDAR is not a direct /World child")

        simulation_context = SimulationContext()
        simulation_context.initialize_physics()
        simulation_context.play()

        # Set this pose exactly once. base_scan already contains the intended
        # mounted offset, so no additional (-0.032, 0, 0.172) is applied.
        base_translation, base_rotation, base_scale = _set_top_level_lidar_from_base_scan(
            stage, UsdGeom, Gf
        )

        lidar = LidarRtx(TEMP_SENSOR_PATH, name="tb03_raw_vs_flat_lidar")
        lidar.initialize()
        render_product_path = lidar.get_render_product_path()
        if not render_product_path:
            raise RuntimeError("LidarRtx initialization returned an empty render-product path")
        lidar.attach_annotator(POINT_CLOUD_ANNOTATOR)
        lidar.attach_annotator(FLAT_SCAN_ANNOTATOR)

        annotators = lidar.get_annotators()
        point_annotator = annotators.get(POINT_CLOUD_ANNOTATOR)
        flat_annotator = annotators.get(FLAT_SCAN_ANNOTATOR)
        if point_annotator is None or flat_annotator is None:
            raise RuntimeError("One or both requested RTX annotators did not attach")

        first_point_data = None
        first_point_update = None
        latest_point_data = None
        flat_scan = None
        for update_index in range(1, MAX_RENDER_UPDATES + 1):
            # The temporary sensor pose is intentionally not rewritten here.
            simulation_app.update()
            latest_point_data = point_annotator.get_data()
            point_count = _point_count(latest_point_data)
            if point_count > 0 and first_point_data is None:
                first_point_data = latest_point_data
                first_point_update = update_index

            flat_data = flat_annotator.get_data()
            flat_scan = _flat_scan_from_data(flat_data)
            if flat_scan is None:
                # This remains only a compatibility fallback; direct annotator
                # data is the primary source for the diagnostic.
                flat_scan = _flat_scan_from_data(lidar.get_current_frame().get(FLAT_SCAN_ANNOTATOR))
            if update_index >= MIN_RENDER_UPDATES and _scan_ready(flat_scan):
                break

        lidar_translation, lidar_rotation, lidar_scale = _read_world_pose_and_scale(
            stage, UsdGeom, Gf, TEMP_SENSOR_PATH
        )
        point_report_data = first_point_data if first_point_data is not None else latest_point_data
        point_count = _point_count(point_report_data)
        num_cols, samples, finite_valid, minimum, maximum = _scan_stats(flat_scan)
        raw_pass = point_count > 0
        flat_pass = _scan_ready(flat_scan)

        print(f"[TB03_LIDAR_RAW_VS_FLAT] BASE_SCAN_WORLD_TRANSLATION={_format_vector(base_translation)}", flush=True)
        print(f"[TB03_LIDAR_RAW_VS_FLAT] BASE_SCAN_WORLD_ROTATION={_quat_wxyz(base_rotation)}", flush=True)
        print(f"[TB03_LIDAR_RAW_VS_FLAT] BASE_SCAN_WORLD_SCALE={_format_vector(base_scale)}", flush=True)
        print(f"[TB03_LIDAR_RAW_VS_FLAT] LIDAR_WORLD_TRANSLATION={_format_vector(lidar_translation)}", flush=True)
        print(f"[TB03_LIDAR_RAW_VS_FLAT] LIDAR_WORLD_ROTATION={_quat_wxyz(lidar_rotation)}", flush=True)
        print(f"[TB03_LIDAR_RAW_VS_FLAT] LIDAR_WORLD_SCALE={_format_vector(lidar_scale)}", flush=True)
        print(f"[TB03_LIDAR_RAW_VS_FLAT] RENDER_PRODUCT_PATH={render_product_path}", flush=True)
        print(f"[TB03_LIDAR_RAW_VS_FLAT] POINT_CLOUD_DATA_TYPE={type(point_report_data).__name__}", flush=True)
        print(f"[TB03_LIDAR_RAW_VS_FLAT] POINT_CLOUD_KEYS={_mapping_keys(point_report_data)}", flush=True)
        print(f"[TB03_LIDAR_RAW_VS_FLAT] POINT_COUNT={point_count}", flush=True)
        print(f"[TB03_LIDAR_RAW_VS_FLAT] FIRST_NONEMPTY_POINT_UPDATE={first_point_update}", flush=True)
        print(f"[TB03_LIDAR_RAW_VS_FLAT] FLAT_numCols={num_cols}", flush=True)
        print(f"[TB03_LIDAR_RAW_VS_FLAT] FLAT_samples={samples}", flush=True)
        print(f"[TB03_LIDAR_RAW_VS_FLAT] FLAT_finite_valid={finite_valid}", flush=True)
        print(f"[TB03_LIDAR_RAW_VS_FLAT] min={'n/a' if minimum is None else f'{minimum:.4f} m'}", flush=True)
        print(f"[TB03_LIDAR_RAW_VS_FLAT] max={'n/a' if maximum is None else f'{maximum:.4f} m'}", flush=True)
        print(
            f"[TB03_LIDAR_RAW_VS_FLAT] azimuthRange="
            f"{'n/a' if flat_scan is None else flat_scan['azimuth_range']}",
            flush=True,
        )
        print(
            f"[TB03_LIDAR_RAW_VS_FLAT] horizontalResolution="
            f"{'n/a' if flat_scan is None else flat_scan['horizontal_resolution']}",
            flush=True,
        )
        print(f"[TB03_LIDAR_RAW_VS_FLAT] PASS_RAW={'PASS' if raw_pass else 'FAIL'}", flush=True)
        print(f"[TB03_LIDAR_RAW_VS_FLAT] PASS_FLAT={'PASS' if flat_pass else 'FAIL'}", flush=True)
        return raw_pass or flat_pass
    except Exception as error:
        print(f"[TB03_LIDAR_RAW_VS_FLAT] FAIL: {error}", flush=True)
        return False
    finally:
        if lidar is not None:
            for annotator_name in (POINT_CLOUD_ANNOTATOR, FLAT_SCAN_ANNOTATOR):
                try:
                    lidar.detach_annotator(annotator_name)
                except Exception:
                    pass
        if stage is not None and stage.GetPrimAtPath(TEMP_SENSOR_PATH).IsValid():
            stage.RemovePrim(TEMP_SENSOR_PATH)


if __name__ == "__main__":
    try:
        main()
    finally:
        print("[TB03_LIDAR_RAW_VS_FLAT] closing SimulationApp without saving stage", flush=True)
        simulation_app.close()
