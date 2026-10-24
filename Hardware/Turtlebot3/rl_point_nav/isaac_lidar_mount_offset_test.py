#!/usr/bin/env python3
"""Runtime-only top-level RTX LiDAR candidate mount-offset diagnostic.

Each candidate uses a fresh top-level LiDAR. The LiDAR follows only the rigid
translation and normalized rotation of base_link; inherited robot scale is not
propagated into either its pose offset or world scale. Nothing is saved.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


SCENE_PATH = Path("/home/g8/Desktop/FInalSceneDemo.usd")
BASE_LINK_PATH = "/World/turtlebot3_burger_02/a__namespace_base_link"
LIDAR_CONFIG = "Example_Rotary_2D"
POINT_CLOUD_ANNOTATOR = "IsaacExtractRTXSensorPointCloudNoAccumulator"
FLAT_SCAN_ANNOTATOR = "IsaacComputeRTXLidarFlatScan"
UPDATES_PER_OFFSET = 180
INVALID_RANGE_EPSILON_M = 1e-3
CANDIDATE_OFFSETS_M = (
    ("OFFSET_080_000_200", (0.08, 0.0, 0.20)),
    ("OFFSET_100_000_220", (0.10, 0.0, 0.22)),
    ("OFFSET_120_000_250", (0.12, 0.0, 0.25)),
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
class OffsetResult:
    name: str
    offset: tuple[float, float, float]
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
    def passed(self) -> bool:
        return self.raw_pass and self.flat_pass


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


def _rigid_base_world(stage, UsdGeom, Gf):
    """Return base_link world translation/rotation with inherited scale removed."""
    base_link = stage.GetPrimAtPath(BASE_LINK_PATH)
    if not base_link.IsValid():
        raise RuntimeError(f"Missing TurtleBot base link: {BASE_LINK_PATH}")
    base_world = UsdGeom.XformCache().GetLocalToWorldTransform(base_link)
    _, _, inherited_scale, rotation_matrix, translation, _ = base_world.Factor()
    rotation = rotation_matrix.ExtractRotationQuat().GetNormalized()
    rigid_world = Gf.Matrix4d(1.0)
    rigid_world.SetRotateOnly(Gf.Rotation(rotation))
    rigid_world.SetTranslateOnly(translation)
    return translation, rotation, inherited_scale, rigid_world


def _synchronize_top_level_lidar(stage, sensor_path: str, offset, UsdGeom, Gf) -> None:
    """Set the top-level sensor to rigid(base_link) × candidate local offset."""
    lidar_prim = stage.GetPrimAtPath(sensor_path)
    if not lidar_prim.IsValid() or lidar_prim.GetParent().GetPath().pathString != "/World":
        raise RuntimeError(f"Temporary LiDAR is not a direct /World child: {sensor_path}")
    _, _, _, rigid_base_world = _rigid_base_world(stage, UsdGeom, Gf)
    local_offset = Gf.Matrix4d(1.0)
    local_offset.SetTranslateOnly(Gf.Vec3d(*offset))
    desired_world = rigid_base_world * local_offset
    parent_world = UsdGeom.XformCache().GetLocalToWorldTransform(lidar_prim.GetParent())
    UsdGeom.Xformable(lidar_prim).MakeMatrixXform().Set(parent_world.GetInverse() * desired_world)


def _release_lidar(lidar, stage, sensor_path: str) -> None:
    """Release callbacks/render products and remove exactly one temporary prim."""
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


def _run_offset_test(stage, UsdGeom, Gf, name: str, offset: tuple[float, float, float]) -> OffsetResult:
    """Collect one fresh-sensor measurement while synchronizing each update."""
    import gc
    from isaacsim.sensors.rtx import LidarRtx

    sensor_path = f"/World/tb03_mount_offset_{name.lower()}"
    lidar = None
    try:
        if stage.GetPrimAtPath(sensor_path).IsValid():
            raise RuntimeError(f"Temporary LiDAR path already exists: {sensor_path}")
        lidar = LidarRtx(
            prim_path=sensor_path,
            name=f"tb03_mount_{name.lower()}",
            config_file_name=LIDAR_CONFIG,
        )
        _synchronize_top_level_lidar(stage, sensor_path, offset, UsdGeom, Gf)
        lidar.initialize()
        lidar.attach_annotator(POINT_CLOUD_ANNOTATOR)
        lidar.attach_annotator(FLAT_SCAN_ANNOTATOR)
        point_annotator = lidar.get_annotators()[POINT_CLOUD_ANNOTATOR]
        flat_annotator = lidar.get_annotators()[FLAT_SCAN_ANNOTATOR]

        max_point_count = 0
        best_flat_data = None
        best_flat_score = (-1, -1)
        for _ in range(UPDATES_PER_OFFSET):
            _synchronize_top_level_lidar(stage, sensor_path, offset, UsdGeom, Gf)
            simulation_app.update()
            _synchronize_top_level_lidar(stage, sensor_path, offset, UsdGeom, Gf)

            max_point_count = max(max_point_count, _point_count(point_annotator.get_data()))
            flat_data = flat_annotator.get_data()
            num_cols, samples, finite_valid, _, _ = _flat_stats(flat_data)
            score = (finite_valid, samples)
            if score > best_flat_score:
                best_flat_data = flat_data
                best_flat_score = score

        num_cols, samples, finite_valid, minimum, maximum = _flat_stats(best_flat_data)
        return OffsetResult(name, offset, max_point_count, num_cols, samples, finite_valid, minimum, maximum)
    finally:
        _release_lidar(lidar, stage, sensor_path)
        lidar = None
        gc.collect()


def _print_result(result: OffsetResult) -> None:
    print(f"[TB03_LIDAR_OFFSET] TEST={result.name}", flush=True)
    print(f"[TB03_LIDAR_OFFSET] OFFSET={result.offset}", flush=True)
    print(f"[TB03_LIDAR_OFFSET] POINT_COUNT={result.point_count}", flush=True)
    print(f"[TB03_LIDAR_OFFSET] FLAT_numCols={result.num_cols}", flush=True)
    print(f"[TB03_LIDAR_OFFSET] FLAT_samples={result.samples}", flush=True)
    print(f"[TB03_LIDAR_OFFSET] FLAT_finite_valid={result.finite_valid}", flush=True)
    print(
        f"[TB03_LIDAR_OFFSET] FLAT_min="
        f"{'n/a' if result.minimum is None else f'{result.minimum:.4f} m'}",
        flush=True,
    )
    print(
        f"[TB03_LIDAR_OFFSET] FLAT_max="
        f"{'n/a' if result.maximum is None else f'{result.maximum:.4f} m'}",
        flush=True,
    )
    print(f"[TB03_LIDAR_OFFSET] PASS={'PASS' if result.passed else 'FAIL'}", flush=True)


def main() -> bool:
    import omni.timeline
    import omni.usd
    from pxr import Gf, UsdGeom

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
        if not stage.GetPrimAtPath(BASE_LINK_PATH).IsValid():
            raise RuntimeError(f"Missing TurtleBot base link: {BASE_LINK_PATH}")

        timeline = omni.timeline.get_timeline_interface()
        timeline.play()
        for _ in range(4):
            simulation_app.update()
        _, _, base_scale, _ = _rigid_base_world(stage, UsdGeom, Gf)
        print(f"[TB03_LIDAR_OFFSET] BASE_LINK_INHERITED_SCALE={base_scale}", flush=True)

        results = []
        for name, offset in CANDIDATE_OFFSETS_M:
            result = _run_offset_test(stage, UsdGeom, Gf, name, offset)
            _print_result(result)
            results.append(result)

        first_passing = next((result for result in results if result.passed), None)
        print("[TB03_LIDAR_OFFSET] SUMMARY", flush=True)
        print("[TB03_LIDAR_OFFSET] TEST | OFFSET | POINTS | COLS | SAMPLES | FINITE | PASS", flush=True)
        for result in results:
            print(
                f"[TB03_LIDAR_OFFSET] {result.name} | {result.offset} | {result.point_count} | "
                f"{result.num_cols} | {result.samples} | {result.finite_valid} | "
                f"{'PASS' if result.passed else 'FAIL'}",
                flush=True,
            )
        print(
            f"[TB03_LIDAR_OFFSET] FIRST_PASSING_OFFSET="
            f"{'NONE' if first_passing is None else first_passing.offset}",
            flush=True,
        )
        return first_passing is not None
    except Exception as error:
        print(f"[TB03_LIDAR_OFFSET] FAIL: {error}", flush=True)
        return False
    finally:
        if timeline is not None:
            timeline.stop()


if __name__ == "__main__":
    try:
        main()
    finally:
        print("[TB03_LIDAR_OFFSET] closing SimulationApp without saving stage", flush=True)
        simulation_app.close()
