#!/usr/bin/env python3
"""Runtime-only RTX LiDAR height diagnostic for the saved scene.

Only a temporary top-level sensor is created and removed between trials. The
opened USD stage is never saved, and no saved robot or sensor prim is changed.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


SCENE_PATH = Path("/home/g8/Desktop/FInalSceneDemo.usd")
BASE_SCAN_PATH = "/World/turtlebot3_burger_02/a__namespace_base_link/base_scan"
TEMP_SENSOR_PATH = "/World/tb03_lidar_height_runtime_test"
LIDAR_CONFIG = "Example_Rotary_2D"
POINT_CLOUD_ANNOTATOR = "IsaacExtractRTXSensorPointCloudNoAccumulator"
FLAT_SCAN_ANNOTATOR = "IsaacComputeRTXLidarFlatScan"
UPDATES_PER_TEST = 120
INVALID_RANGE_EPSILON_M = 1e-3
HEIGHT_OFFSETS_M = (
    ("TEST_0", 0.00),
    ("TEST_025", 0.25),
    ("TEST_050", 0.50),
    ("TEST_100", 1.00),
)


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
class HeightResult:
    name: str
    position: tuple[float, float, float]
    point_count: int
    num_cols: int
    samples: int
    finite_valid: int
    minimum: float | None
    maximum: float | None
    pass_raw: bool
    pass_flat: bool


def _point_count(data) -> int:
    """Return raw point rows without printing or copying point arrays."""
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


def _format_position(position: tuple[float, float, float]) -> str:
    return f"({position[0]:.6f}, {position[1]:.6f}, {position[2]:.6f})"



def _run_height_test(stage, timeline, name: str, position: tuple[float, float, float]) -> HeightResult:
    """Create, sample, and remove one independent top-level LiDAR trial."""
    import gc
    import numpy as np
    from isaacsim.sensors.rtx import LidarRtx

    lidar = None
    try:
        lidar = LidarRtx(
            prim_path=TEMP_SENSOR_PATH,
            name=f"tb03_height_{name.lower()}",
            translation=np.asarray(position, dtype=np.float64),
            orientation=np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
            config_file_name=LIDAR_CONFIG,
        )
        lidar.initialize()
        lidar.attach_annotator(POINT_CLOUD_ANNOTATOR)
        lidar.attach_annotator(FLAT_SCAN_ANNOTATOR)
        annotators = lidar.get_annotators()
        point_annotator = annotators[POINT_CLOUD_ANNOTATOR]
        flat_annotator = annotators[FLAT_SCAN_ANNOTATOR]

        max_point_count = 0
        best_flat_data = None
        best_flat_score = (-1, -1)
        for _ in range(UPDATES_PER_TEST):
            simulation_app.update()
            max_point_count = max(max_point_count, _point_count(point_annotator.get_data()))
            flat_data = flat_annotator.get_data()
            num_cols, samples, finite_valid, _, _ = _flat_stats(flat_data)
            score = (finite_valid, samples)
            if score > best_flat_score:
                best_flat_data = flat_data
                best_flat_score = score

        num_cols, samples, finite_valid, minimum, maximum = _flat_stats(best_flat_data)
        return HeightResult(
            name=name,
            position=position,
            point_count=max_point_count,
            num_cols=num_cols,
            samples=samples,
            finite_valid=finite_valid,
            minimum=minimum,
            maximum=maximum,
            pass_raw=max_point_count > 0,
            pass_flat=num_cols > 0 and samples > 0 and finite_valid > 0,
        )
    finally:
        if lidar is not None:
            for annotator_name in (POINT_CLOUD_ANNOTATOR, FLAT_SCAN_ANNOTATOR):
                try:
                    lidar.detach_annotator(annotator_name)
                except Exception:
                    pass
        if stage.GetPrimAtPath(TEMP_SENSOR_PATH).IsValid():
            stage.RemovePrim(TEMP_SENSOR_PATH)
        # Drop the wrapper before the next trial. Its destructor destroys the
        # trial render product; collection plus an update releases it before the
        # same top-level prim path is reused.
        lidar = None
        gc.collect()
        simulation_app.update()


def _print_result(result: HeightResult) -> None:
    print(f"[TB03_LIDAR_HEIGHT] TEST={result.name}", flush=True)
    print(f"[TB03_LIDAR_HEIGHT] position={_format_position(result.position)}", flush=True)
    print(f"[TB03_LIDAR_HEIGHT] POINT_COUNT={result.point_count}", flush=True)
    print(f"[TB03_LIDAR_HEIGHT] FLAT_numCols={result.num_cols}", flush=True)
    print(f"[TB03_LIDAR_HEIGHT] FLAT_samples={result.samples}", flush=True)
    print(f"[TB03_LIDAR_HEIGHT] FLAT_finite_valid={result.finite_valid}", flush=True)
    print(
        f"[TB03_LIDAR_HEIGHT] FLAT_min="
        f"{'n/a' if result.minimum is None else f'{result.minimum:.4f} m'}",
        flush=True,
    )
    print(
        f"[TB03_LIDAR_HEIGHT] FLAT_max="
        f"{'n/a' if result.maximum is None else f'{result.maximum:.4f} m'}",
        flush=True,
    )
    print(f"[TB03_LIDAR_HEIGHT] PASS_RAW={'PASS' if result.pass_raw else 'FAIL'}", flush=True)
    print(f"[TB03_LIDAR_HEIGHT] PASS_FLAT={'PASS' if result.pass_flat else 'FAIL'}", flush=True)


def main() -> bool:
    import omni.timeline
    import omni.usd
    from pxr import UsdGeom

    timeline = None
    try:
        if not SCENE_PATH.is_file():
            raise FileNotFoundError(f"Scene does not exist: {SCENE_PATH}")

        usd_context = omni.usd.get_context()
        while usd_context.get_stage_loading_status()[2] > 0:
            simulation_app.update()
        stage = usd_context.get_stage()
        if stage is None:
            raise RuntimeError("No active USD stage after opening the scene")
        base_scan = stage.GetPrimAtPath(BASE_SCAN_PATH)
        if not base_scan.IsValid():
            raise RuntimeError(f"Missing saved base_scan prim: {BASE_SCAN_PATH}")
        if stage.GetPrimAtPath(TEMP_SENSOR_PATH).IsValid():
            raise RuntimeError(f"Temporary sensor path already exists: {TEMP_SENSOR_PATH}")

        timeline = omni.timeline.get_timeline_interface()
        timeline.play()
        for _ in range(4):
            simulation_app.update()

        base_scan_translation = UsdGeom.XformCache().GetLocalToWorldTransform(base_scan).ExtractTranslation()
        base_x = float(base_scan_translation[0])
        base_y = float(base_scan_translation[1])
        base_z = float(base_scan_translation[2])
        print(
            f"[TB03_LIDAR_HEIGHT] saved_base_scan_translation="
            f"{_format_position((base_x, base_y, base_z))}",
            flush=True,
        )

        results = []
        for name, offset_z in HEIGHT_OFFSETS_M:
            position = (base_x, base_y, base_z + offset_z)
            result = _run_height_test(stage, timeline, name, position)
            _print_result(result)
            results.append(result)

        print("[TB03_LIDAR_HEIGHT] SUMMARY", flush=True)
        print("[TB03_LIDAR_HEIGHT] test | z_m | points | cols | samples | finite | raw | flat", flush=True)
        for result in results:
            print(
                f"[TB03_LIDAR_HEIGHT] {result.name} | {result.position[2]:.6f} | "
                f"{result.point_count} | {result.num_cols} | {result.samples} | "
                f"{result.finite_valid} | {'PASS' if result.pass_raw else 'FAIL'} | "
                f"{'PASS' if result.pass_flat else 'FAIL'}",
                flush=True,
            )
        return any(result.pass_raw or result.pass_flat for result in results)
    except Exception as error:
        print(f"[TB03_LIDAR_HEIGHT] FAIL: {error}", flush=True)
        return False
    finally:
        if timeline is not None:
            timeline.stop()


if __name__ == "__main__":
    try:
        main()
    finally:
        print("[TB03_LIDAR_HEIGHT] closing SimulationApp without saving stage", flush=True)
        simulation_app.close()
