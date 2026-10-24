#!/usr/bin/env python3
"""Diagnostic-only normal/hidden visual comparison; restores session opinions.

One minimal environment, same fixed reset and goal, no drive actions. Only a
fresh anonymous session sublayer contains the temporary visibility changes.
"""
from __future__ import annotations

import argparse
import math
import numpy as np

try:
    from .isaac_point_nav_lidar_minimal_env import IsaacPointNavLidarMinimalEnv
    from .isaac_point_nav_env import ARTICULATION_PATH
except ImportError:
    from isaac_point_nav_lidar_minimal_env import IsaacPointNavLidarMinimalEnv
    from isaac_point_nav_env import ARTICULATION_PATH

GOAL = (-2.118, 0.444)


def log(message):
    print(f"[TB03_SELF_RETURN] {message}", flush=True)


def inspect_visuals(stage):
    from pxr import Usd, UsdGeom, UsdPhysics

    robot = stage.GetPrimAtPath(ARTICULATION_PATH)
    prims = list(Usd.PrimRange(robot, Usd.TraverseInstanceProxies()))

    def protected(prim):
        return (prim.HasAPI(UsdPhysics.CollisionAPI)
                or prim.HasAPI(UsdPhysics.RigidBodyAPI)
                or prim.HasAPI(UsdPhysics.ArticulationRootAPI)
                or prim.IsA(UsdPhysics.Joint)
                or any("collision" in str(part).lower() for part in prim.GetPath().pathString.split("/")))

    def visible(prim):
        imageable = UsdGeom.Imageable(prim)
        return (imageable.ComputeVisibility() != UsdGeom.Tokens.invisible
                and imageable.ComputePurpose() in (UsdGeom.Tokens.default_, UsdGeom.Tokens.render))

    # Require an explicitly named visual subtree, never a whole link/root.
    candidates = []
    for prim in prims:
        if prim.GetName().lower() not in ("visual", "visuals") or prim.IsInstanceProxy():
            continue
        subtree = list(Usd.PrimRange(prim, Usd.TraverseInstanceProxies()))
        if any(protected(child) for child in subtree):
            log(f"SKIP_UNSAFE_VISUAL_ROOT={prim.GetPath()}")
            continue
        if not any(child.IsA(UsdGeom.Gprim) for child in subtree):
            continue
        candidates.append(prim)
    roots = []
    for prim in sorted(candidates, key=lambda item: len(item.GetPath().pathString)):
        if not any(prim.GetPath().HasPrefix(parent.GetPath()) for parent in roots):
            roots.append(prim)
    for prim in roots:
        log(f"HIDE_VISUAL_ROOT={prim.GetPath()} instance={prim.IsInstance()}")
    visual_meshes = [prim for prim in prims if prim.IsA(UsdGeom.Gprim) and not protected(prim) and visible(prim)]
    uncovered = [str(prim.GetPath()) for prim in visual_meshes
                 if not any(prim.GetPath().HasPrefix(root.GetPath()) for root in roots)]
    for path in uncovered:
        log(f"UNHIDDEN_VISUAL_OR_UNCLASSIFIED_GEOMETRY={path}")
    return roots, uncovered


def invariant_snapshot(stage):
    """Check collision visibility and authored physics/drive settings, not motion."""
    from pxr import Usd, UsdGeom, UsdPhysics

    result = {}
    root = stage.GetPrimAtPath(ARTICULATION_PATH)
    for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        path = str(prim.GetPath())
        result[path + ":schemas"] = tuple(prim.GetAppliedSchemas())
        for attr in prim.GetAttributes():
            if attr.GetName().startswith(("physics:", "physx", "drive:")):
                result[path + ":" + attr.GetName()] = repr(attr.Get())
        if prim.HasAPI(UsdPhysics.CollisionAPI) and prim.IsA(UsdGeom.Imageable):
            result[path + ":collision_visibility"] = str(UsdGeom.Imageable(prim).ComputeVisibility())
    return result


def capture(env, label):
    # Minimal reset reattaches FlatScan and renders >=180 stationary warm-up
    # frames, so the hidden case cannot reuse our cached normal-case scan.
    observation, info = env.reset(seed=42, options={"goal": GOAL})
    scan = info["lidar"]["scan"]
    if scan is None or scan["num_cols"] <= 0 or scan["finite_valid"] <= 0:
        raise RuntimeError(f"{label}: no real FlatScan; cannot infer absence of self-returns")
    ranges, angles = scan["ranges_m"], scan["azimuth_deg"]
    valid = scan["valid"]
    counts = {}
    log(f"CASE={label}")
    log(f"POSE={info['pose']} YAW={info['yaw']} GOAL={info['goal']}")
    log(f"OBS={observation.tolist()} FLAT_numCols={scan['num_cols']} "
        f"FLAT_finite_valid={scan['finite_valid']}")
    for name, sector in info["lidar_sectors"].items():
        log(f"{name.upper()}_RAW_M={sector['raw_min_m']}")
        log(f"VALID_RAYS_{name.upper()}={sector['valid_rays']}")
    for threshold in (0.30, 0.40, 0.60):
        counts[threshold] = int(np.count_nonzero(valid & (ranges < threshold)))
        log(f"COUNT_RAYS_LT_{threshold:.2f}M={counts[threshold]}".replace("0.", "0_", 1))
    log(f"FLAT_MIN={float(ranges[valid].min())}")
    indices = np.flatnonzero(valid & (ranges < 0.30))
    log("NEAR_RETURNS_FORMAT=(azimuth_deg, range_m)")
    for start in range(0, len(indices), 12):
        chunk = indices[start:start + 12]
        log(f"NEAR_RETURNS={[(round(float(angles[i]), 4), round(float(ranges[i]), 6)) for i in chunk]}")
    return info, counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true", help="Visible renderer by default")
    args = parser.parse_args()
    env = IsaacPointNavLidarMinimalEnv(headless=args.headless)
    layer = None
    original_visibility = {}
    session = None
    try:
        from pxr import Sdf, Usd, UsdGeom

        stage = env._lidar_stage
        roots, uncovered = inspect_visuals(stage)
        if not roots:
            raise RuntimeError("No safely editable visual roots identified")
        original_visibility = {str(p.GetPath()): str(UsdGeom.Imageable(p).ComputeVisibility()) for p in roots}
        normal, normal_counts = capture(env, "NORMAL_ROBOT_VISUALS")
        physics_before = invariant_snapshot(stage)
        # Layer is memory-only. Removing it restores the exact prior authored
        # visibility composition, including references and instance roots.
        session = stage.GetSessionLayer()
        layer = Sdf.Layer.CreateAnonymous("tb03_self_return_visibility.usda")
        session.subLayerPaths = [layer.identifier] + list(session.subLayerPaths)
        with Usd.EditContext(stage, layer):
            for prim in roots:
                if not UsdGeom.Imageable(prim).GetVisibilityAttr().Set(UsdGeom.Tokens.invisible):
                    raise RuntimeError(f"Visibility authoring failed: {prim.GetPath()}")
        for prim in roots:
            if UsdGeom.Imageable(prim).ComputeVisibility() != UsdGeom.Tokens.invisible:
                raise RuntimeError(f"Stronger opinion prevents hiding: {prim.GetPath()}")
        if invariant_snapshot(stage) != physics_before:
            raise RuntimeError("Physics/drive settings or collision visibility changed")
        hidden, hidden_counts = capture(env, "ROBOT_VISUALS_HIDDEN")
        if invariant_snapshot(stage) != physics_before:
            raise RuntimeError("Physics/drive settings or collision visibility changed during hidden case")
        position_error = math.hypot(normal["pose"]["x"] - hidden["pose"]["x"],
                                    normal["pose"]["y"] - hidden["pose"]["y"])
        yaw_delta = normal["yaw"] - hidden["yaw"]
        yaw_error = abs(math.atan2(math.sin(yaw_delta), math.cos(yaw_delta)))
        same_pose = position_error <= 0.01 and yaw_error <= 0.02
        same_sensor = (normal["lidar"]["settings"] == hidden["lidar"]["settings"]
                       and np.allclose(normal["lidar"]["relative_offset_m"], hidden["lidar"]["relative_offset_m"], atol=1e-4))
        side_before = [normal["lidar_sectors"][s]["raw_min_m"] for s in ("left", "right")]
        side_after = [hidden["lidar_sectors"][s]["raw_min_m"] for s in ("left", "right")]
        baseline = all(v is not None and 0.15 <= v < 0.30 for v in side_before)
        # No return is reported as None, never claimed to be a measured large range.
        sides_removed = baseline and all(b is None or b >= a + 0.10 for a, b in zip(side_before, side_after))
        strong_drop = normal_counts[0.30] > 0 and hidden_counts[0.30] <= 0.25 * normal_counts[0.30]
        matched_conditions = same_pose and same_sensor and normal["goal"] == hidden["goal"]
        if matched_conditions and not uncovered and sides_removed and strong_drop:
            conclusion = "YES"
        elif (matched_conditions and baseline and not uncovered
              and all(b is not None and abs(b - a) <= 0.05 for a, b in zip(side_before, side_after))
              and hidden_counts[0.30] >= 0.75 * normal_counts[0.30]):
            conclusion = "NO"
        else:
            conclusion = "INCONCLUSIVE"
        log(f"RESET_POSITION_DIFFERENCE_M={position_error} RESET_YAW_DIFFERENCE_RAD={yaw_error}")
        log(f"SAME_POSE_AND_SENSOR={'PASS' if matched_conditions else 'FAIL'}")
        log("SUMMARY: CASE | LEFT_RAW_M | RIGHT_RAW_M | RAYS_LT_0_30M")
        log(f"NORMAL | {side_before[0]} | {side_before[1]} | {normal_counts[0.30]}")
        log(f"HIDDEN | {side_after[0]} | {side_after[1]} | {hidden_counts[0.30]}")
        log(f"SELF_RETURN_FROM_ROBOT_VISUALS={conclusion}")
        log("PHYSICS_AND_COLLISION_SETTINGS_UNCHANGED=PASS")
    except Exception as error:
        log(f"ERROR={type(error).__name__}: {error}")
        log("SELF_RETURN_FROM_ROBOT_VISUALS=INCONCLUSIVE")
        raise
    finally:
        try:
            if layer is not None and session is not None:
                session.subLayerPaths = [p for p in session.subLayerPaths if p != layer.identifier]
                restored = all(str(UsdGeom.Imageable(env._lidar_stage.GetPrimAtPath(path)).ComputeVisibility()) == value
                               for path, value in original_visibility.items())
                log(f"VISIBILITY_RESTORED={'PASS' if restored else 'FAIL'}")
                if not restored:
                    raise RuntimeError("Original visual visibility did not restore")
        finally:
            env.close()


if __name__ == "__main__":
    main()
