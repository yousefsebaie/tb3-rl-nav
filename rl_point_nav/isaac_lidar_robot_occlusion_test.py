#!/usr/bin/env python3
"""Runtime-only robot-visibility RTX LiDAR occlusion diagnostic.

The opened scene is never saved. Each case gets a fresh top-level LiDAR, while
robot visibility is restored exactly after the hidden-case measurement.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


SCENE_PATH = Path("/home/g8/Desktop/FInalSceneDemo.usd")
ROBOT_ROOT_PATH = "/World/turtlebot3_burger_02"
BASE_SCAN_PATH = f"{ROBOT_ROOT_PATH}/a__namespace_base_link/base_scan"
VISIBLE_SENSOR_PATH = "/World/tb03_occlusion_visible_lidar"
HIDDEN_SENSOR_PATH = "/World/tb03_occlusion_hidden_lidar"
LIDAR_CONFIG = "Example_Rotary_2D"
POINT_CLOUD_ANNOTATOR = "IsaacExtractRTXSensorPointCloudNoAccumulator"
FLAT_SCAN_ANNOTATOR = "IsaacComputeRTXLidarFlatScan"
OUTPUT_BUFFER_SETTING = "/app/sensors/nv/lidar/outputBufferOnGPU"
UPDATES_PER_CASE = 180
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


@dataclass(frozen=True)
class CaseResult:
    name: str
    point_count: int
    num_cols: int
    samples: int
    finite_valid: int
    minimum: float | None
    maximum: float | None

    @property
    def raw_pass(self) -> bool:
        return self.point_count > 0

    @property
    def flat_pass(self) -> bool:
        return self.num_cols > 0 and self.samples > 0 and self.finite_valid > 0

    @property
    def any_data(self) -> bool:
        return self.raw_pass or self.flat_pass


def _quat_wxyz(quaternion) -> tuple[float, float, float, float]:
    normalized = quaternion.GetNormalized()
    imaginary = normalized.GetImaginary()
    return (
        float(normalized.GetReal()),
        float(imaginary[0]),
        float(imaginary[1]),
        float(imaginary[2]),
    )


def _format_position(position) -> str:
    return f"({float(position[0]):.6f}, {float(position[1]):.6f}, {float(position[2]):.6f})"


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


def _release_lidar(lidar, stage, sensor_path: str) -> None:
    """Release this case's runtime consumers before deleting its temporary prim."""
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
    simulation_app.update()


def _run_case(stage, sensor_path: str, position, orientation, name: str) -> CaseResult:
    """Run one fresh top-level LiDAR case at the captured base_scan pose."""
    import gc
    import numpy as np
    from isaacsim.sensors.rtx import LidarRtx

    lidar = None
    try:
        if stage.GetPrimAtPath(sensor_path).IsValid():
            raise RuntimeError(f"Temporary case path already exists: {sensor_path}")
        lidar = LidarRtx(
            prim_path=sensor_path,
            name=f"tb03_occlusion_{name.lower()}",
            position=np.asarray(position, dtype=np.float64),
            orientation=np.asarray(orientation, dtype=np.float64),
            config_file_name=LIDAR_CONFIG,
        )
        lidar.initialize()
        lidar.attach_annotator(POINT_CLOUD_ANNOTATOR)
        lidar.attach_annotator(FLAT_SCAN_ANNOTATOR)
        point_annotator = lidar.get_annotators()[POINT_CLOUD_ANNOTATOR]
        flat_annotator = lidar.get_annotators()[FLAT_SCAN_ANNOTATOR]

        max_point_count = 0
        best_flat_data = None
        best_flat_score = (-1, -1)
        for _ in range(UPDATES_PER_CASE):
            simulation_app.update()
            max_point_count = max(max_point_count, _point_count(point_annotator.get_data()))
            flat_data = flat_annotator.get_data()
            num_cols, samples, finite_valid, _, _ = _flat_stats(flat_data)
            score = (finite_valid, samples)
            if score > best_flat_score:
                best_flat_data = flat_data
                best_flat_score = score

        num_cols, samples, finite_valid, minimum, maximum = _flat_stats(best_flat_data)
        return CaseResult(name, max_point_count, num_cols, samples, finite_valid, minimum, maximum)
    finally:
        _release_lidar(lidar, stage, sensor_path)
        lidar = None
        gc.collect()


def _print_case(result: CaseResult) -> None:
    print(f"[TB03_LIDAR_OCCLUSION] CASE={result.name}", flush=True)
    print(f"[TB03_LIDAR_OCCLUSION] POINT_COUNT={result.point_count}", flush=True)
    print(f"[TB03_LIDAR_OCCLUSION] FLAT_numCols={result.num_cols}", flush=True)
    print(f"[TB03_LIDAR_OCCLUSION] FLAT_samples={result.samples}", flush=True)
    print(f"[TB03_LIDAR_OCCLUSION] FLAT_finite_valid={result.finite_valid}", flush=True)
    print(
        f"[TB03_LIDAR_OCCLUSION] FLAT_min="
        f"{'n/a' if result.minimum is None else f'{result.minimum:.4f} m'}",
        flush=True,
    )
    print(
        f"[TB03_LIDAR_OCCLUSION] FLAT_max="
        f"{'n/a' if result.maximum is None else f'{result.maximum:.4f} m'}",
        flush=True,
    )
    print(f"[TB03_LIDAR_OCCLUSION] PASS_RAW={'PASS' if result.raw_pass else 'FAIL'}", flush=True)
    print(f"[TB03_LIDAR_OCCLUSION] PASS_FLAT={'PASS' if result.flat_pass else 'FAIL'}", flush=True)


def main() -> bool:
    import carb
    import omni.timeline
    import omni.usd
    from pxr import UsdGeom

    timeline = None
    robot_visibility_attr = None
    original_visibility = None
    had_authored_visibility = False
    try:
        if not SCENE_PATH.is_file():
            raise FileNotFoundError(f"Scene does not exist: {SCENE_PATH}")

        usd_context = omni.usd.get_context()
        while usd_context.get_stage_loading_status()[2] > 0:
            simulation_app.update()
        stage = usd_context.get_stage()
        if stage is None:
            raise RuntimeError("No active USD stage after opening the scene")
        robot_root = stage.GetPrimAtPath(ROBOT_ROOT_PATH)
        base_scan = stage.GetPrimAtPath(BASE_SCAN_PATH)
        if not robot_root.IsValid() or not base_scan.IsValid():
            raise RuntimeError("Robot root or saved base_scan prim is absent")
        if stage.GetPrimAtPath(VISIBLE_SENSOR_PATH).IsValid() or stage.GetPrimAtPath(HIDDEN_SENSOR_PATH).IsValid():
            raise RuntimeError("A temporary occlusion-test sensor path already exists")

        timeline = omni.timeline.get_timeline_interface()
        timeline.play()
        for _ in range(4):
            simulation_app.update()

        base_scan_world = UsdGeom.XformCache().GetLocalToWorldTransform(base_scan)
        _, _, _, rotation_matrix, position, _ = base_scan_world.Factor()
        orientation = _quat_wxyz(rotation_matrix.ExtractRotationQuat())
        print(f"[TB03_LIDAR_OCCLUSION] BASE_SCAN_WORLD_POSITION={_format_position(position)}", flush=True)
        print(f"[TB03_LIDAR_OCCLUSION] BASE_SCAN_WORLD_ROTATION={orientation}", flush=True)
        print(
            f"[TB03_LIDAR_OCCLUSION] {OUTPUT_BUFFER_SETTING}="
            f"{carb.settings.get_settings().get(OUTPUT_BUFFER_SETTING)}",
            flush=True,
        )

        visible_result = _run_case(stage, VISIBLE_SENSOR_PATH, position, orientation, "ROBOT_VISIBLE")
        _print_case(visible_result)

        imageable = UsdGeom.Imageable(robot_root)
        robot_visibility_attr = imageable.GetVisibilityAttr()
        had_authored_visibility = robot_visibility_attr.HasAuthoredValueOpinion()
        original_visibility = robot_visibility_attr.Get()
        robot_visibility_attr.Set(UsdGeom.Tokens.invisible)
        simulation_app.update()

        hidden_result = _run_case(stage, HIDDEN_SENSOR_PATH, position, orientation, "ROBOT_HIDDEN")
        _print_case(hidden_result)

        if not visible_result.any_data and hidden_result.any_data:
            outcome = "ROBOT_GEOMETRY_OCCLUSION"
        else:
            outcome = "NOT_ROBOT_OCCLUSION"
        print("[TB03_LIDAR_OCCLUSION] SUMMARY", flush=True)
        print("[TB03_LIDAR_OCCLUSION] CASE | POINTS | COLS | SAMPLES | FINITE | RAW | FLAT", flush=True)
        for result in (visible_result, hidden_result):
            print(
                f"[TB03_LIDAR_OCCLUSION] {result.name} | {result.point_count} | "
                f"{result.num_cols} | {result.samples} | {result.finite_valid} | "
                f"{'PASS' if result.raw_pass else 'FAIL'} | {'PASS' if result.flat_pass else 'FAIL'}",
                flush=True,
            )
        print(f"[TB03_LIDAR_OCCLUSION] RESULT={outcome}", flush=True)
        return visible_result.any_data or hidden_result.any_data
    except Exception as error:
        print(f"[TB03_LIDAR_OCCLUSION] FAIL: {error}", flush=True)
        return False
    finally:
        if robot_visibility_attr is not None:
            if had_authored_visibility:
                robot_visibility_attr.Set(original_visibility)
            else:
                robot_visibility_attr.Clear()
        if timeline is not None:
            timeline.stop()


if __name__ == "__main__":
    try:
        main()
    finally:
        print("[TB03_LIDAR_OCCLUSION] closing SimulationApp without saving stage", flush=True)
        simulation_app.close()
