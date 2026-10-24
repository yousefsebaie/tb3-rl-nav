#!/usr/bin/env python3
"""Runtime-only dynamic top-level RTX LiDAR follow diagnostic.

A temporary top-level LiDAR follows the TurtleBot's scale-free base-link pose
through stationary, forward, turn, and forward phases. The USD is never saved.
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path


SCENE_PATH = Path("/home/g8/Desktop/FInalSceneDemo.usd")
ARTICULATION_PATH = "/World/turtlebot3_burger_02"
BASE_LINK_PATH = f"{ARTICULATION_PATH}/a__namespace_base_link"
TEMP_SENSOR_PATH = "/World/tb03_dynamic_follow_lidar"
LEFT_WHEEL_DOF = "a__namespace_wheel_left_joint"
RIGHT_WHEEL_DOF = "a__namespace_wheel_right_joint"
LIDAR_CONFIG = "Example_Rotary_2D"
NEAR_RANGE_ATTRIBUTE = "omni:sensor:Core:nearRangeM"  # identified by isaac_lidar_range_test.py
POINT_CLOUD_ANNOTATOR = "IsaacExtractRTXSensorPointCloudNoAccumulator"
FLAT_SCAN_ANNOTATOR = "IsaacComputeRTXLidarFlatScan"
# base_scan frame expressed in the scale-free base_link frame (isaac_robot_geometry_inspect.py).
# Override without editing:  --offset X Y Z   (e.g. --offset 0.10 0.0 0.22 for the previous mount)
DEFAULT_MOUNT_OFFSET_M = (-0.143, 0.0, 0.685)
_arg_parser = argparse.ArgumentParser(add_help=False)
_arg_parser.add_argument("--offset", type=float, nargs=3, metavar=("X", "Y", "Z"), default=DEFAULT_MOUNT_OFFSET_M)
_arg_parser.add_argument("--near-range", type=float, default=0.10)
_args = _arg_parser.parse_known_args()[0]
MOUNT_OFFSET_M = tuple(_args.offset)
WHEEL_RADIUS_M = 0.033
WHEEL_BASE_M = 0.160
FORWARD_SPEED_MPS = 0.05
TURN_SPEED_RADPS = 0.80
STATIONARY_FRAMES = 180
FORWARD_FRAMES = 30
TURN_FRAMES = 15
SETTLE_FRAMES = 8
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
class PhaseResult:
    name: str
    robot_x: float
    robot_y: float
    robot_yaw: float
    lidar_position: tuple[float, float, float]
    point_count: int
    num_cols: int
    samples: int
    finite_valid: int
    minimum: float | None
    offset_error_m: float
    lidar_scale_is_unit: bool

    @property
    def passed(self) -> bool:
        return (
            self.point_count > 0
            and self.num_cols > 0
            and self.samples > 0
            and self.finite_valid > 0
            and self.lidar_scale_is_unit
        )


def _wxyz(quaternion) -> tuple[float, float, float, float]:
    normalized = quaternion.GetNormalized()
    imaginary = normalized.GetImaginary()
    return (
        float(normalized.GetReal()),
        float(imaginary[0]),
        float(imaginary[1]),
        float(imaginary[2]),
    )


def _yaw_from_quat(quaternion) -> float:
    w, x, y, z = _wxyz(quaternion)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _format_vector(vector) -> str:
    return f"({float(vector[0]):.6f}, {float(vector[1]):.6f}, {float(vector[2]):.6f})"


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
        return 0, 0, 0, None
    ranges = np.asarray(data.get("linearDepthData", [])).reshape(-1)
    num_cols = int(data.get("numCols", 0))
    valid = ranges[np.isfinite(ranges) & (ranges > INVALID_RANGE_EPSILON_M)]
    return (
        num_cols,
        int(ranges.size),
        int(valid.size),
        float(valid.min()) if valid.size else None,
    )


def _base_rigid_pose(stage, UsdGeom, Gf):
    """Return scale-free base pose and stage-unit mount offset."""
    base_link = stage.GetPrimAtPath(BASE_LINK_PATH)
    if not base_link.IsValid():
        raise RuntimeError(f"Missing TurtleBot base link: {BASE_LINK_PATH}")
    base_world = UsdGeom.XformCache().GetLocalToWorldTransform(base_link)
    _, _, inherited_scale, rotation_matrix, translation, _ = base_world.Factor()
    rotation = rotation_matrix.ExtractRotationQuat().GetNormalized()
    meters_per_unit = UsdGeom.GetStageMetersPerUnit(stage)
    if meters_per_unit <= 0.0:
        raise RuntimeError(f"Invalid stage meters-per-unit: {meters_per_unit}")
    offset_stage = Gf.Vec3d(*(component / meters_per_unit for component in MOUNT_OFFSET_M))
    return translation, rotation, inherited_scale, offset_stage, meters_per_unit


def _synchronize_lidar(lidar, stage, UsdGeom, Gf) -> tuple[float, bool]:
    """Synchronize top-level LiDAR from rigid base pose; return offset error in m."""
    import numpy as np

    base_translation, base_rotation, _, offset_stage, meters_per_unit = _base_rigid_pose(stage, UsdGeom, Gf)
    world_offset = Gf.Rotation(base_rotation).TransformDir(offset_stage)
    lidar_position = base_translation + world_offset
    lidar.set_world_pose(
        position=np.asarray(lidar_position, dtype=np.float64),
        orientation=np.asarray(_wxyz(base_rotation), dtype=np.float64),
    )

    # Re-read the runtime world pose and measure its rigid-coordinate offset.
    lidar_world = UsdGeom.XformCache().GetLocalToWorldTransform(
        stage.GetPrimAtPath(TEMP_SENSOR_PATH)
    )
    _, _, lidar_scale, lidar_rotation_matrix, measured_position, _ = lidar_world.Factor()
    del lidar_rotation_matrix
    relative_stage = Gf.Rotation(base_rotation).GetInverse().TransformDir(measured_position - base_translation)
    error_stage = relative_stage - offset_stage
    error_m = math.sqrt(sum(float(component) ** 2 for component in error_stage)) * meters_per_unit
    scale_is_unit = all(abs(float(component) - 1.0) <= 1e-6 for component in lidar_scale)
    return error_m, scale_is_unit


def _phase_pose(stage, UsdGeom, Gf):
    base_translation, base_rotation, _, _, _ = _base_rigid_pose(stage, UsdGeom, Gf)
    lidar_world = UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath(TEMP_SENSOR_PATH))
    lidar_position = lidar_world.ExtractTranslation()
    return (
        float(base_translation[0]),
        float(base_translation[1]),
        _yaw_from_quat(base_rotation),
        (float(lidar_position[0]), float(lidar_position[1]), float(lidar_position[2])),
    )


def _release_lidar(lidar, stage) -> None:
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
    if stage.GetPrimAtPath(TEMP_SENSOR_PATH).IsValid():
        stage.RemovePrim(TEMP_SENSOR_PATH)


def _run_phase(
    name: str,
    action,
    drive_frames: int,
    settle_frames: int,
    robot,
    stop_action,
    simulation_context,
    lidar,
    point_annotator,
    flat_annotator,
    stage,
    UsdGeom,
    Gf,
) -> PhaseResult:
    """Drive exactly the requested frames while sampling LiDAR each update."""
    max_point_count = 0
    best_flat_data = None
    best_flat_score = (-1, -1)
    max_offset_error_m = 0.0
    scale_is_unit = True

    def step_once(wheel_action) -> None:
        nonlocal max_point_count, best_flat_data, best_flat_score, max_offset_error_m, scale_is_unit
        before_error, before_scale_ok = _synchronize_lidar(lidar, stage, UsdGeom, Gf)
        robot.apply_wheel_actions(wheel_action)
        simulation_context.step(render=True)
        after_error, after_scale_ok = _synchronize_lidar(lidar, stage, UsdGeom, Gf)
        max_offset_error_m = max(max_offset_error_m, before_error, after_error)
        scale_is_unit = scale_is_unit and before_scale_ok and after_scale_ok
        max_point_count = max(max_point_count, _point_count(point_annotator.get_data()))
        flat_data = flat_annotator.get_data()
        num_cols, samples, finite_valid, _ = _flat_stats(flat_data)
        score = (finite_valid, samples)
        if score > best_flat_score:
            best_flat_data = flat_data
            best_flat_score = score

    for _ in range(drive_frames):
        step_once(action)
    robot.apply_wheel_actions(stop_action)
    for _ in range(settle_frames):
        step_once(stop_action)

    num_cols, samples, finite_valid, minimum = _flat_stats(best_flat_data)
    robot_x, robot_y, robot_yaw, lidar_position = _phase_pose(stage, UsdGeom, Gf)
    return PhaseResult(
        name,
        robot_x,
        robot_y,
        robot_yaw,
        lidar_position,
        max_point_count,
        num_cols,
        samples,
        finite_valid,
        minimum,
        max_offset_error_m,
        scale_is_unit,
    )


def _print_phase(result: PhaseResult) -> None:
    print(f"[TB03_LIDAR_DYNAMIC] PHASE={result.name}", flush=True)
    print(f"[TB03_LIDAR_DYNAMIC] robot_x={result.robot_x:.6f}", flush=True)
    print(f"[TB03_LIDAR_DYNAMIC] robot_y={result.robot_y:.6f}", flush=True)
    print(f"[TB03_LIDAR_DYNAMIC] robot_yaw={result.robot_yaw:.6f}", flush=True)
    print(f"[TB03_LIDAR_DYNAMIC] lidar_world_position={result.lidar_position}", flush=True)
    print(f"[TB03_LIDAR_DYNAMIC] POINT_COUNT={result.point_count}", flush=True)
    print(f"[TB03_LIDAR_DYNAMIC] FLAT_numCols={result.num_cols}", flush=True)
    print(f"[TB03_LIDAR_DYNAMIC] FLAT_samples={result.samples}", flush=True)
    print(f"[TB03_LIDAR_DYNAMIC] FLAT_finite_valid={result.finite_valid}", flush=True)
    print(
        f"[TB03_LIDAR_DYNAMIC] FLAT_min="
        f"{'n/a' if result.minimum is None else f'{result.minimum:.4f} m'}",
        flush=True,
    )
    print(f"[TB03_LIDAR_DYNAMIC] PHASE_PASS={'PASS' if result.passed else 'FAIL'}", flush=True)


def main() -> bool:
    import gc
    import numpy as np
    import omni.timeline
    import omni.usd
    from pxr import Gf, UsdGeom
    from isaacsim.core.api import SimulationContext
    from isaacsim.robot.wheeled_robots.controllers.differential_controller import DifferentialController
    from isaacsim.robot.wheeled_robots.robots.wheeled_robot import WheeledRobot
    from isaacsim.sensors.rtx import LidarRtx

    lidar = None
    robot = None
    timeline = None
    stop_action = None
    try:
        if not SCENE_PATH.is_file():
            raise FileNotFoundError(f"Scene does not exist: {SCENE_PATH}")
        usd_context = omni.usd.get_context()
        while usd_context.get_stage_loading_status()[2] > 0:
            simulation_app.update()
        stage = usd_context.get_stage()
        if stage is None:
            raise RuntimeError("No active USD stage after opening the scene")
        if not stage.GetPrimAtPath(ARTICULATION_PATH).IsValid() or not stage.GetPrimAtPath(BASE_LINK_PATH).IsValid():
            raise RuntimeError("TurtleBot articulation or base link is absent")
        if stage.GetPrimAtPath(TEMP_SENSOR_PATH).IsValid():
            raise RuntimeError(f"Temporary LiDAR path already exists: {TEMP_SENSOR_PATH}")

        simulation_context = SimulationContext()
        simulation_context.initialize_physics()
        simulation_context.play()
        for _ in range(SETTLE_FRAMES):
            simulation_context.step(render=True)

        robot = WheeledRobot(
            prim_path=ARTICULATION_PATH,
            name="tb03_dynamic_follow_robot",
            wheel_dof_names=[LEFT_WHEEL_DOF, RIGHT_WHEEL_DOF],
        )
        robot.initialize()
        if LEFT_WHEEL_DOF not in robot.dof_names or RIGHT_WHEEL_DOF not in robot.dof_names:
            raise RuntimeError(f"Expected wheel DOFs absent: {list(robot.dof_names)}")
        controller = DifferentialController(
            name="tb03_dynamic_follow_controller",
            wheel_radius=WHEEL_RADIUS_M,
            wheel_base=WHEEL_BASE_M,
            max_linear_speed=0.22,
            max_angular_speed=1.0,
        )
        stop_action = controller.forward(np.array([0.0, 0.0], dtype=np.float64))
        forward_action = controller.forward(np.array([FORWARD_SPEED_MPS, 0.0], dtype=np.float64))
        left_action = controller.forward(np.array([0.0, TURN_SPEED_RADPS], dtype=np.float64))
        robot.apply_wheel_actions(stop_action)

        lidar = LidarRtx(
            prim_path=TEMP_SENSOR_PATH,
            name="tb03_dynamic_follow_lidar",
            config_file_name=LIDAR_CONFIG,
            **{NEAR_RANGE_ATTRIBUTE: _args.near_range},
        )
        lidar.initialize()
        print(f"[TB03_LIDAR_DYNAMIC] MOUNT_OFFSET_M={MOUNT_OFFSET_M}", flush=True)
        print(
            f"[TB03_LIDAR_DYNAMIC] CONFIGURED_MIN_RANGE="
            f"{stage.GetPrimAtPath(lidar.prim_path).GetAttribute(NEAR_RANGE_ATTRIBUTE).Get()}",
            flush=True,
        )
        # Establish the initial fixed-offset pose after BaseSensor is ready.
        _synchronize_lidar(lidar, stage, UsdGeom, Gf)
        lidar.attach_annotator(POINT_CLOUD_ANNOTATOR)
        lidar.attach_annotator(FLAT_SCAN_ANNOTATOR)
        point_annotator = lidar.get_annotators()[POINT_CLOUD_ANNOTATOR]
        flat_annotator = lidar.get_annotators()[FLAT_SCAN_ANNOTATOR]

        stationary = _run_phase(
            "STATIONARY", stop_action, STATIONARY_FRAMES, 0, robot, stop_action,
            simulation_context, lidar, point_annotator, flat_annotator, stage, UsdGeom, Gf,
        )
        _print_phase(stationary)
        forward_1 = _run_phase(
            "FORWARD_1", forward_action, FORWARD_FRAMES, SETTLE_FRAMES, robot, stop_action,
            simulation_context, lidar, point_annotator, flat_annotator, stage, UsdGeom, Gf,
        )
        _print_phase(forward_1)
        turn_left = _run_phase(
            "TURN_LEFT", left_action, TURN_FRAMES, SETTLE_FRAMES, robot, stop_action,
            simulation_context, lidar, point_annotator, flat_annotator, stage, UsdGeom, Gf,
        )
        _print_phase(turn_left)
        forward_2 = _run_phase(
            "FORWARD_2", forward_action, FORWARD_FRAMES, SETTLE_FRAMES, robot, stop_action,
            simulation_context, lidar, point_annotator, flat_annotator, stage, UsdGeom, Gf,
        )
        _print_phase(forward_2)
        results = (stationary, forward_1, turn_left, forward_2)

        max_error_m = max(result.offset_error_m for result in results)
        overall = all(result.passed for result in results)
        print(f"[TB03_LIDAR_DYNAMIC] STATIONARY_PASS={'PASS' if stationary.passed else 'FAIL'}", flush=True)
        print(f"[TB03_LIDAR_DYNAMIC] FORWARD_1_PASS={'PASS' if forward_1.passed else 'FAIL'}", flush=True)
        print(f"[TB03_LIDAR_DYNAMIC] TURN_LEFT_PASS={'PASS' if turn_left.passed else 'FAIL'}", flush=True)
        print(f"[TB03_LIDAR_DYNAMIC] FORWARD_2_PASS={'PASS' if forward_2.passed else 'FAIL'}", flush=True)
        print(f"[TB03_LIDAR_DYNAMIC] FOLLOW_OFFSET_ERROR_MAX={max_error_m:.8f} m", flush=True)
        print(f"[TB03_LIDAR_DYNAMIC] OVERALL_PASS={'PASS' if overall else 'FAIL'}", flush=True)
        return overall
    except Exception as error:
        print(f"[TB03_LIDAR_DYNAMIC] FAIL: {error}", flush=True)
        return False
    finally:
        if robot is not None and stop_action is not None:
            try:
                robot.apply_wheel_actions(stop_action)
            except Exception:
                pass
        if lidar is not None:
            _release_lidar(lidar, stage)
            lidar = None
            gc.collect()
        if timeline is not None:
            timeline.stop()


if __name__ == "__main__":
    try:
        main()
    finally:
        print("[TB03_LIDAR_DYNAMIC] closing SimulationApp without saving stage", flush=True)
        simulation_app.close()
