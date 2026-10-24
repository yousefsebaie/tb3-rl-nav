#!/usr/bin/env python3
"""Runtime-only TurtleBot geometry inspection for choosing the LiDAR mount.

Enumerates every geometric prim (Gprim: Mesh, Cube, Cylinder, ...) under the
TurtleBot and reports, in meters:

  prim path, type, purpose, collision/visual role, world scale,
  world axis-aligned bounding box, center, size,
  distance from the current LiDAR position (0 = LiDAR inside the box),
  center expressed in the scale-free base_link frame

The "current LiDAR position" is computed exactly like the frozen follow
architecture: base_link translation + normalized rotation applied to
MOUNT_OFFSET_M, no inherited scale. No LiDAR is created.

Poses are reported twice: as authored in the USD, and after the same physics
settle used by the LiDAR tests. Nothing is saved; the USD is never modified.
"""
from __future__ import annotations

import math
from pathlib import Path


SCENE_PATH = Path("/home/g8/Desktop/FInalSceneDemo.usd")
ARTICULATION_PATH = "/World/turtlebot3_burger_02"
BASE_LINK_PATH = f"{ARTICULATION_PATH}/a__namespace_base_link"
MOUNT_OFFSET_M = (0.10, 0.0, 0.22)
SETTLE_FRAMES = 8
MOUNT_CLEARANCE_M = 0.02
LIDAR_NAME_HINTS = ("scan", "lidar", "lds", "laser")
TAG = "[TB03_GEOMETRY]"


from isaacsim import SimulationApp


simulation_app = SimulationApp(
    {
        "headless": True,
        "open_usd": str(SCENE_PATH),
        "create_new_stage": False,
        "fast_shutdown": False,
    }
)


def log(message: str) -> None:
    print(f"{TAG} {message}", flush=True)


def _v(vector, scale: float = 1.0, digits: int = 4) -> tuple:
    return tuple(round(float(component) * scale, digits) for component in vector)


def _base_rigid_pose(stage, UsdGeom, Gf):
    """Scale-free base pose (same decomposition as isaac_lidar_dynamic_follow_test.py)."""
    base_world = UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath(BASE_LINK_PATH))
    _, _, inherited_scale, rotation_matrix, translation, _ = base_world.Factor()
    rotation = rotation_matrix.ExtractRotationQuat().GetNormalized()
    return translation, rotation, inherited_scale


def _box_distance(box, point) -> float:
    lo, hi = box.GetMin(), box.GetMax()
    delta = [max(lo[i] - point[i], 0.0, point[i] - hi[i]) for i in range(3)]
    return math.sqrt(sum(d * d for d in delta))


def _inspect(stage, label: str) -> None:
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    mpu = UsdGeom.GetStageMetersPerUnit(stage)
    base_translation, base_rotation, base_scale = _base_rigid_pose(stage, UsdGeom, Gf)
    to_base = Gf.Rotation(base_rotation).GetInverse()
    offset_stage = Gf.Vec3d(*(component / mpu for component in MOUNT_OFFSET_M))
    lidar_position = base_translation + Gf.Rotation(base_rotation).TransformDir(offset_stage)

    def in_base_m(point) -> tuple:
        return _v(to_base.TransformDir(Gf.Vec3d(point) - base_translation), mpu)

    log(f"================ STATE={label} ================")
    log(f"STAGE_METERS_PER_UNIT={mpu}")
    log(f"BASE_LINK_WORLD_POSITION={_v(base_translation)}")
    log(f"BASE_LINK_INHERITED_SCALE={_v(base_scale, digits=6)}")
    log(f"CURRENT_LIDAR_WORLD_POSITION={_v(lidar_position)} (offset {MOUNT_OFFSET_M} m)")

    xform_cache = UsdGeom.XformCache()
    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy]
    )
    robot_root = stage.GetPrimAtPath(ARTICULATION_PATH)

    # Links / frames: every Xformable that is not a Gprim (base_scan etc. would show here).
    log("---- XFORM FRAMES (non-geometry) ----")
    for prim in Usd.PrimRange(robot_root):
        if not prim.IsA(UsdGeom.Xformable) or prim.IsA(UsdGeom.Gprim):
            continue
        world = xform_cache.GetLocalToWorldTransform(prim)
        _, _, scale, _, translation, _ = world.Factor()
        hint = any(h in prim.GetName().lower() for h in LIDAR_NAME_HINTS)
        log(
            f"FRAME path={prim.GetPath()} type={prim.GetTypeName()} "
            f"world={_v(translation)} in_base_m={in_base_m(translation)} "
            f"world_scale={_v(scale, digits=4)}{'  <-- LIDAR NAME HINT' if hint else ''}"
        )

    log("---- GEOMETRY (Gprims) ----")
    rows = []
    for prim in Usd.PrimRange(robot_root):
        if not prim.IsA(UsdGeom.Gprim):
            continue
        imageable = UsdGeom.Imageable(prim)
        purpose = imageable.ComputePurpose()
        box = bbox_cache.ComputeWorldBound(prim).ComputeAlignedRange()
        if box.IsEmpty():
            log(f"MESH path={prim.GetPath()} EMPTY_BBOX purpose={purpose}")
            continue
        world = xform_cache.GetLocalToWorldTransform(prim)
        _, _, scale, _, _, _ = world.Factor()
        center = (box.GetMin() + box.GetMax()) * 0.5
        rows.append(
            {
                "path": str(prim.GetPath()),
                "type": prim.GetTypeName(),
                "purpose": str(purpose),
                "collision": prim.HasAPI(UsdPhysics.CollisionAPI),
                "visible": imageable.ComputeVisibility() != UsdGeom.Tokens.invisible,
                "scale": _v(scale, digits=4),
                "min": _v(box.GetMin()),
                "max": _v(box.GetMax()),
                "center": _v(center),
                "size_m": _v(box.GetSize(), mpu),
                "center_in_base_m": in_base_m(center),
                "dist_m": _box_distance(box, lidar_position) * mpu,
                "center_dist_m": (center - lidar_position).GetLength() * mpu,
                "box": box,
                "hint": any(h in str(prim.GetPath()).lower() for h in LIDAR_NAME_HINTS),
            }
        )
    rows.sort(key=lambda r: (r["dist_m"], r["center_dist_m"]))
    for r in rows:
        log(
            f"MESH path={r['path']} type={r['type']} purpose={r['purpose']} "
            f"role={'COLLISION' if r['collision'] else 'VISUAL'} visible={r['visible']}"
            f"{'  <-- LIDAR NAME HINT' if r['hint'] else ''}"
        )
        log(f"    bbox_min={r['min']} bbox_max={r['max']} size_m={r['size_m']}")
        log(f"    center={r['center']} center_in_base_m={r['center_in_base_m']} world_scale={r['scale']}")
        log(
            f"    dist_to_lidar={r['dist_m']:.4f} m center_dist_to_lidar={r['center_dist_m']:.4f} m"
            f"{'  <-- LIDAR INSIDE' if r['dist_m'] == 0.0 else ''}"
        )
    log(f"GPRIM_COUNT={len(rows)}")
    log(f"GPRIMS_CONTAINING_LIDAR={[r['path'] for r in rows if r['dist_m'] == 0.0]}")

    # Summaries per role, in the scale-free base frame.
    def extent_in_base(selected):
        corners = []
        for r in selected:
            lo, hi = r["box"].GetMin(), r["box"].GetMax()
            for x in (lo[0], hi[0]):
                for y in (lo[1], hi[1]):
                    for z in (lo[2], hi[2]):
                        corners.append(in_base_m((x, y, z)))
        if not corners:
            return None
        return tuple(min(c[i] for c in corners) for i in range(3)), tuple(max(c[i] for c in corners) for i in range(3))

    log("---- EXTENTS IN BASE FRAME (m, scale-free; from world AABB corners) ----")
    for role, selected in (
        ("ALL", rows),
        ("VISUAL", [r for r in rows if not r["collision"]]),
        ("COLLISION", [r for r in rows if r["collision"]]),
        ("LIDAR_NAME_HINT", [r for r in rows if r["hint"]]),
    ):
        extent = extent_in_base(selected)
        if extent is None:
            log(f"EXTENT_{role}=none")
            continue
        lo, hi = extent
        size = tuple(round(hi[i] - lo[i], 4) for i in range(3))
        log(f"EXTENT_{role} min={lo} max={hi} size={size} count={len(selected)}")

    # Candidate: directly above all geometry, at the current x/y mount offset.
    all_extent = extent_in_base(rows)
    if all_extent is not None:
        top_z = all_extent[1][2]
        log(
            f"CANDIDATE_ABOVE_ALL_GEOMETRY_OFFSET_M="
            f"({MOUNT_OFFSET_M[0]:.3f}, {MOUNT_OFFSET_M[1]:.3f}, {top_z + MOUNT_CLEARANCE_M:.3f}) "
            f"(top_z_in_base={top_z:.4f} + clearance {MOUNT_CLEARANCE_M})"
        )


def main() -> bool:
    import omni.usd
    from isaacsim.core.api import SimulationContext

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

        _inspect(stage, "USD_AUTHORED")

        simulation_context = SimulationContext()
        simulation_context.initialize_physics()
        simulation_context.play()
        for _ in range(SETTLE_FRAMES):
            simulation_context.step(render=False)
        _inspect(stage, "AFTER_SETTLE")
        simulation_context.stop()
        return True
    except Exception as error:
        log(f"FAIL: {error}")
        return False


if __name__ == "__main__":
    try:
        main()
    finally:
        log("closing SimulationApp without saving stage")
        simulation_app.close()
