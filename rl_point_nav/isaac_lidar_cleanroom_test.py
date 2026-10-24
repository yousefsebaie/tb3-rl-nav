#!/usr/bin/env python3
"""Pure Isaac Sim RTX 2D LiDAR clean-room diagnostic.

This program creates an empty in-memory stage containing only four opaque
walls and one RTX LiDAR. It opens no external USD and saves nothing.
"""
from __future__ import annotations


LIDAR_PATH = "/World/lidar"
POINT_CLOUD_ANNOTATOR = "IsaacExtractRTXSensorPointCloudNoAccumulator"
FLAT_SCAN_ANNOTATOR = "IsaacComputeRTXLidarFlatScan"
LIDAR_OUTPUT_BUFFER_SETTING = "/app/sensors/nv/lidar/outputBufferOnGPU"
NUM_UPDATES = 360
INVALID_RANGE_EPSILON_M = 1e-3


from isaacsim import SimulationApp


simulation_app = SimulationApp(
    {
        "headless": False,
        "disable_viewport_updates": False,
    }
)


def _flat_scan_stats(flat_data):
    import numpy as np

    if not isinstance(flat_data, dict):
        return 0, 0, 0, None, None
    ranges = np.asarray(flat_data.get("linearDepthData", [])).reshape(-1)
    num_cols = int(flat_data.get("numCols", 0))
    valid = ranges[np.isfinite(ranges) & (ranges > INVALID_RANGE_EPSILON_M)]
    return (
        num_cols,
        int(ranges.size),
        int(valid.size),
        float(valid.min()) if valid.size else None,
        float(valid.max()) if valid.size else None,
    )


def _point_count(point_data) -> int:
    """Return raw point rows without printing or copying point arrays."""
    if not isinstance(point_data, dict):
        return 0
    points = point_data.get("data")
    if points is None:
        return 0
    shape = getattr(points, "shape", None)
    if shape is not None and len(shape) >= 1:
        return int(shape[0])
    try:
        return len(points)
    except TypeError:
        return 0


def _make_wall(stage, UsdGeom, Gf, path: str, position, scale) -> None:
    """Create one opaque unit cube transformed into a wall."""
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    xform = UsdGeom.XformCommonAPI(cube.GetPrim())
    xform.SetTranslate(Gf.Vec3d(*position))
    xform.SetScale(Gf.Vec3f(*scale))
    # Display color is intentionally opaque; no transparency material is used.
    cube.CreateDisplayColorAttr([Gf.Vec3f(0.7, 0.7, 0.7)])


def main() -> bool:
    import carb
    import numpy as np
    import omni.timeline
    import omni.usd
    from pxr import Gf, UsdGeom
    from isaacsim.sensors.rtx import LidarRtx

    lidar = None
    timeline = None
    try:
        usd_context = omni.usd.get_context()
        usd_context.new_stage()
        simulation_app.update()
        stage = usd_context.get_stage()
        if stage is None:
            raise RuntimeError("Failed to create a new in-memory USD stage")

        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        UsdGeom.Xform.Define(stage, "/World")

        # Unit cube sizes are scaled to the requested physical wall dimensions.
        _make_wall(stage, UsdGeom, Gf, "/World/wall_pos_x", (3.0, 0.0, 1.0), (0.2, 4.0, 2.0))
        _make_wall(stage, UsdGeom, Gf, "/World/wall_neg_x", (-3.0, 0.0, 1.0), (0.2, 4.0, 2.0))
        _make_wall(stage, UsdGeom, Gf, "/World/wall_pos_y", (0.0, 3.0, 1.0), (4.0, 0.2, 2.0))
        _make_wall(stage, UsdGeom, Gf, "/World/wall_neg_y", (0.0, -3.0, 1.0), (4.0, 0.2, 2.0))

        # Construct directly: there is no pre-existing sensor prim or imported
        # scene content in this clean-room stage.
        lidar = LidarRtx(
            prim_path=LIDAR_PATH,
            name="cleanroom_lidar",
            translation=np.array([0.0, 0.0, 1.0]),
            orientation=np.array([1.0, 0.0, 0.0, 0.0]),
            config_file_name="Example_Rotary_2D",
        )
        lidar.initialize()
        render_product_path = lidar.get_render_product_path()
        print(f"[CLEANROOM_LIDAR] render_product_path={render_product_path}", flush=True)
        if not render_product_path:
            raise RuntimeError("LidarRtx did not create a render product")

        lidar.attach_annotator(POINT_CLOUD_ANNOTATOR)
        lidar.attach_annotator(FLAT_SCAN_ANNOTATOR)
        annotators = lidar.get_annotators()
        point_annotator = annotators[POINT_CLOUD_ANNOTATOR]
        flat_annotator = annotators[FLAT_SCAN_ANNOTATOR]

        output_buffer_on_gpu = carb.settings.get_settings().get(LIDAR_OUTPUT_BUFFER_SETTING)
        print(
            f"[CLEANROOM_LIDAR] {LIDAR_OUTPUT_BUFFER_SETTING}={output_buffer_on_gpu}",
            flush=True,
        )

        timeline = omni.timeline.get_timeline_interface()
        timeline.play()

        first_nonempty_point_update = None
        max_point_count = 0
        latest_flat_data = None
        for update_index in range(1, NUM_UPDATES + 1):
            simulation_app.update()

            point_data = point_annotator.get_data()
            point_count = _point_count(point_data)
            max_point_count = max(max_point_count, point_count)
            if point_count > 0 and first_nonempty_point_update is None:
                first_nonempty_point_update = update_index

            latest_flat_data = flat_annotator.get_data()
            if update_index % 30 == 0:
                print(
                    f"[CLEANROOM_LIDAR] update={update_index} "
                    f"timeline_time={timeline.get_current_time():.6f} "
                    f"raw_point_count={point_count}",
                    flush=True,
                )

        num_cols, samples, finite_valid, minimum, maximum = _flat_scan_stats(latest_flat_data)
        raw_pass = max_point_count > 0
        flat_pass = num_cols > 0 and samples > 0 and finite_valid > 0

        print(f"[CLEANROOM_LIDAR] POINT_COUNT={max_point_count}", flush=True)
        print(
            f"[CLEANROOM_LIDAR] FIRST_NONEMPTY_POINT_UPDATE={first_nonempty_point_update}",
            flush=True,
        )
        print(f"[CLEANROOM_LIDAR] FLAT_numCols={num_cols}", flush=True)
        print(f"[CLEANROOM_LIDAR] FLAT_samples={samples}", flush=True)
        print(f"[CLEANROOM_LIDAR] FLAT_finite_valid={finite_valid}", flush=True)
        print(
            f"[CLEANROOM_LIDAR] FLAT_min={'n/a' if minimum is None else f'{minimum:.4f} m'}",
            flush=True,
        )
        print(
            f"[CLEANROOM_LIDAR] FLAT_max={'n/a' if maximum is None else f'{maximum:.4f} m'}",
            flush=True,
        )
        print(
            f"[CLEANROOM_LIDAR] FLAT_depthRange="
            f"{'n/a' if not isinstance(latest_flat_data, dict) else latest_flat_data.get('depthRange')}",
            flush=True,
        )
        print(
            f"[CLEANROOM_LIDAR] FLAT_azimuthRange="
            f"{'n/a' if not isinstance(latest_flat_data, dict) else latest_flat_data.get('azimuthRange')}",
            flush=True,
        )
        print(
            f"[CLEANROOM_LIDAR] FLAT_horizontalResolution="
            f"{'n/a' if not isinstance(latest_flat_data, dict) else latest_flat_data.get('horizontalResolution')}",
            flush=True,
        )
        print(f"[CLEANROOM_LIDAR] PASS_RAW={'PASS' if raw_pass else 'FAIL'}", flush=True)
        print(f"[CLEANROOM_LIDAR] PASS_FLAT={'PASS' if flat_pass else 'FAIL'}", flush=True)
        return raw_pass or flat_pass
    except Exception as error:
        print(f"[CLEANROOM_LIDAR] FAIL: {error}", flush=True)
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


if __name__ == "__main__":
    try:
        main()
    finally:
        print("[CLEANROOM_LIDAR] closing SimulationApp without saving the in-memory stage", flush=True)
        simulation_app.close()
