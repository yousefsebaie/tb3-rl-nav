#!/usr/bin/env python3
"""Build a real-size copy of the TurtleBot3 scene with a robot-mounted RTX LiDAR on ROS 2.

FInalSceneDemo.usd scales the Burger by about (4.47, 6.83, 3.98). A LiDAR parented to
that robot inherits the non-uniform scale and sits inside the enlarged LDS housing,
so it returns nothing. This script writes a NEW scene (the original is never saved):

* robot root scale -> 1 (true Burger size), wheels placed on the floor;
* the LiDAR as a child of the robot at the real base_scan mount (-0.032, 0, 0.172) m,
  like the Isaac Sim "add a LiDAR to your robot" tutorial, with nearRangeM = 0.12 m and
  farRangeM = 3.5 m (TurtleBot3 LDS-01);
* stray and duplicate LiDAR prims deactivated, and the unwired ROS2PublishLaserScan
  node removed;
* drive fixes for the imported Burger, which on its own covers only ~60% of the
  commanded motion: the fixed rear caster box gets a frictionless material (it
  dragged on the floor) and each wheel's cylinder collider is replaced by an
  invisible sphere of the wheel radius (the cylinder contact made wheel speed jitter);
* a ROS 2 graph: OnPlaybackTick -> RunOneSimulationFrame -> CreateRenderProduct ->
  ROS2RtxLidarHelper publishing sensor_msgs/LaserScan on /tb03/scan
  (frame tb03/base_scan).

It then plays the scene and checks the FlatScan output. If almost every ray is
invalid (the sensor sits inside the LDS housing mesh), it hides only the LDS visual
mesh (collision untouched), saves again and re-checks.

  ./isaac_run.sh make_realsize_scene.py [--out PATH] [--gui]   (log: ~/Desktop/FInalSceneDemo_realsize.log)

Train on it with:  TB3_SCENE_USD=<out> ~/isaac-sim/python.sh train_isaac_point_nav_lidar_ppo_v2.py \
                       --output-name isaac_point_nav_lidar_ppo_v3_realsize
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

SOURCE_SCENE = Path("/home/g8/Desktop/FInalSceneDemo.usd")
# Same folder as the source so the robot's relative asset reference (../isaac-sim/...) resolves.
DEFAULT_OUT = SOURCE_SCENE.with_name("FInalSceneDemo_realsize.usd")

ROBOT = "/World/turtlebot3_burger_02"
LIDAR = f"{ROBOT}/a__namespace_base_link/base_scan/Example_Rotary_2D"
LIDAR_MOUNT = f"{ROBOT}/a__namespace_base_link/base_scan"
LDS_VISUALS = f"{ROBOT}/a__namespace_base_scan/visuals"
FLOOR = "/SimpleRoom/GroundPlane/CollisionPlane"
DEACTIVATE = (
    "/SimpleRoom/Towel_Room01_window1_290/World",            # two stray LiDARs inside a window prop
    f"{ROBOT}/a__namespace_base_link/Example_Rotary_2D_01",   # duplicate robot LiDAR
    f"{ROBOT}/a__namespace_base_link/base_scan/World",        # empty leftover
)
OLD_SCAN_GRAPH = "/World/ActionGraph_robot02"
OLD_SCAN_NODE = "ros2_publish_laser_scan"
LIDAR_GRAPH = "/World/ActionGraph_lidar"

BASE_SCAN_XYZ_M = (-0.032, 0.0, 0.172)   # TurtleBot3 Burger URDF scan_joint
LIDAR_SETTINGS = {
    "omni:sensor:Core:nearRangeM": 0.12,
    "omni:sensor:Core:minDistBetweenEchosM": 0.10,
    "omni:sensor:Core:farRangeM": 3.5,
}
SCAN_TOPIC = "/tb03/scan"
SCAN_FRAME = "tb03/base_scan"
FLOOR_CLEARANCE_M = 0.002
WHEEL_RADIUS_M = 0.033
CASTER_COLLISIONS = f"{ROBOT}/a__namespace_caster_back_link/collisions"
WHEEL_LINKS = (f"{ROBOT}/a__namespace_wheel_left_link", f"{ROBOT}/a__namespace_wheel_right_link")
CASTER_MATERIAL = "/World/PhysicsMaterials/caster_frictionless"
MIN_VALID_FRACTION = 0.10
TAG = "[TB3_REALSIZE]"


LOG_FILE = DEFAULT_OUT.with_suffix(".log")


def log(message: str) -> None:
    # Kit can swallow stdout, so every line also goes to LOG_FILE.
    line = f"{TAG} {message}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as handle:
        handle.write(line + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--gui", action="store_true")
    args = parser.parse_args()
    out = args.out.resolve()
    if out == SOURCE_SCENE.resolve():
        raise SystemExit("Refusing to overwrite the original training scene.")

    from isaacsim import SimulationApp

    LOG_FILE.write_text("")
    app = SimulationApp({"headless": not args.gui})
    try:
        build(app, out)
    except Exception:
        import traceback

        log("FAILED:\n" + traceback.format_exc())
        raise
    finally:
        app.close()


def build(app, out: Path) -> None:
    import omni.graph.core as og
    import omni.timeline
    import omni.usd
    from isaacsim.core.utils.extensions import enable_extension
    from pxr import Gf, Sdf, UsdGeom

    # Enable the bridge BEFORE opening the scene: enabling it afterwards replaces the stage.
    enable_extension("isaacsim.ros2.bridge")
    for _ in range(10):
        app.update()
    context = omni.usd.get_context()
    if not context.open_stage(str(SOURCE_SCENE)):
        raise RuntimeError(f"Could not open {SOURCE_SCENE}")
    # Enabling the bridge can restart stage loading; wait until the robot is really there.
    for _ in range(600):
        app.update()
        stage = context.get_stage()
        if (context.get_stage_loading_status()[2] == 0 and stage is not None
                and stage.GetPrimAtPath(ROBOT).IsValid()):
            break
    else:
        root = stage.GetRootLayer().identifier if stage is not None else None
        raise RuntimeError(f"Scene did not finish loading (root layer {root})")
    log(f"loaded {stage.GetRootLayer().identifier}")
    for path in (ROBOT, LIDAR, LIDAR_MOUNT, FLOOR):
        if not stage.GetPrimAtPath(path).IsValid():
            raise RuntimeError(f"Expected prim is missing: {path}")
    meters = UsdGeom.GetStageMetersPerUnit(stage)

    # 1) True-size robot standing on the floor.
    cache = UsdGeom.XformCache()
    floor_z = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(FLOOR)).ExtractTranslation()[2]
    robot = UsdGeom.Xformable(stage.GetPrimAtPath(ROBOT))
    ops = {op.GetOpName(): op for op in robot.GetOrderedXformOps()}
    if "xformOp:scale" not in ops or "xformOp:translate" not in ops:
        raise RuntimeError(f"Robot root has unexpected xform ops: {list(ops)}")
    old_scale = ops["xformOp:scale"].Get()
    ops["xformOp:scale"].Set(type(old_scale)(1.0, 1.0, 1.0))
    translate = ops["xformOp:translate"].Get()
    ops["xformOp:translate"].Set(type(translate)(translate[0], translate[1], floor_z + FLOOR_CLEARANCE_M / meters))
    log(f"robot scale {tuple(round(v, 3) for v in old_scale)} -> (1, 1, 1); "
        f"root z {translate[2]:.3f} -> {floor_z + FLOOR_CLEARANCE_M / meters:.3f} (floor {floor_z:.3f})")

    # 2) Drive fixes: frictionless caster, sphere wheel colliders.
    fix_drive(stage, meters)

    # 3) LiDAR on the robot at the real base_scan mount.
    for path in DEACTIVATE:
        prim = stage.GetPrimAtPath(path)
        if prim.IsValid():
            prim.SetActive(False)
            log(f"deactivated {path}")
    mount = UsdGeom.Xformable(stage.GetPrimAtPath(LIDAR_MOUNT))
    mount_ops = {op.GetOpName(): op for op in mount.GetOrderedXformOps()}
    mount_ops["xformOp:translate"].Set(Gf.Vec3d(*(v / meters for v in BASE_SCAN_XYZ_M)))
    lidar_prim = stage.GetPrimAtPath(LIDAR)
    for name, value in LIDAR_SETTINGS.items():
        attribute = lidar_prim.GetAttribute(name)
        if not attribute:
            attribute = lidar_prim.CreateAttribute(name, Sdf.ValueTypeNames.Float)
        attribute.Set(value)
    log(f"LiDAR {LIDAR} at base_scan {BASE_SCAN_XYZ_M} m, settings {LIDAR_SETTINGS}")

    # 4) ROS 2 graph (replaces the unwired publisher in the robot graph).
    if stage.GetPrimAtPath(f"{OLD_SCAN_GRAPH}/{OLD_SCAN_NODE}").IsValid():
        og.Controller.edit(OLD_SCAN_GRAPH, {og.Controller.Keys.DELETE_NODES: [OLD_SCAN_NODE]})
        log(f"removed unwired {OLD_SCAN_GRAPH}/{OLD_SCAN_NODE}")
    if stage.GetPrimAtPath(LIDAR_GRAPH).IsValid():
        stage.RemovePrim(LIDAR_GRAPH)
    keys = og.Controller.Keys
    og.Controller.edit(
        {"graph_path": LIDAR_GRAPH, "evaluator_name": "execution"},
        {
            keys.CREATE_NODES: [
                ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                ("RunOnce", "isaacsim.core.nodes.OgnIsaacRunOneSimulationFrame"),
                ("RenderProduct", "isaacsim.core.nodes.IsaacCreateRenderProduct"),
                ("Context", "isaacsim.ros2.bridge.ROS2Context"),
                ("ScanPublish", "isaacsim.ros2.bridge.ROS2RtxLidarHelper"),
            ],
            keys.SET_VALUES: [
                ("RenderProduct.inputs:cameraPrim", [Sdf.Path(LIDAR)]),
                ("ScanPublish.inputs:type", "laser_scan"),
                ("ScanPublish.inputs:topicName", SCAN_TOPIC),
                ("ScanPublish.inputs:frameId", SCAN_FRAME),
                ("ScanPublish.inputs:resetSimulationTimeOnStop", True),
            ],
            keys.CONNECT: [
                ("OnPlaybackTick.outputs:tick", "RunOnce.inputs:execIn"),
                ("RunOnce.outputs:step", "RenderProduct.inputs:execIn"),
                ("RenderProduct.outputs:execOut", "ScanPublish.inputs:execIn"),
                ("RenderProduct.outputs:renderProductPath", "ScanPublish.inputs:renderProductPath"),
                ("Context.outputs:context", "ScanPublish.inputs:context"),
            ],
        },
    )
    log(f"graph {LIDAR_GRAPH}: LaserScan on {SCAN_TOPIC} frame {SCAN_FRAME}")

    save(context, out)
    timeline = omni.timeline.get_timeline_interface()
    if not verify(app, timeline):
        log("most rays invalid: hiding only the LDS visual mesh (collision untouched) and retrying")
        UsdGeom.Imageable(stage.GetPrimAtPath(LDS_VISUALS)).MakeInvisible()
        save(context, out)
        if not verify(app, timeline):
            raise RuntimeError("LiDAR still returns almost no valid rays; scene saved for inspection")
    log(f"DONE: {out}")


def fix_drive(stage, meters: float) -> None:
    from pxr import Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

    material = UsdShade.Material.Define(stage, CASTER_MATERIAL)
    api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    api.CreateStaticFrictionAttr(0.0)
    api.CreateDynamicFrictionAttr(0.0)
    api.CreateRestitutionAttr(0.0)
    material.GetPrim().CreateAttribute("physxMaterial:frictionCombineMode", Sdf.ValueTypeNames.Token).Set("min")
    # The collision subtrees are instanced; un-instance them so the edits apply.
    caster = stage.GetPrimAtPath(CASTER_COLLISIONS)
    caster.SetInstanceable(False)
    bound = 0
    for prim in Usd.PrimRange(caster):
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(material, UsdShade.Tokens.strongerThanDescendants, "physics")
            bound += 1
    log(f"caster: frictionless material on {bound} collider(s)")

    for link in WHEEL_LINKS:
        collisions = stage.GetPrimAtPath(f"{link}/collisions")
        collisions.SetInstanceable(False)
        disabled = 0
        for prim in Usd.PrimRange(collisions):
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Set(False)
                disabled += 1
        sphere = UsdGeom.Sphere.Define(stage, f"{link}/sphere_collider")
        sphere.CreateRadiusAttr(WHEEL_RADIUS_M / meters)
        UsdPhysics.CollisionAPI.Apply(sphere.GetPrim())
        sphere.MakeInvisible()
        log(f"{link.rsplit('/', 1)[-1]}: {disabled} cylinder collider(s) off, sphere r={WHEEL_RADIUS_M} m on")


def save(context, out: Path) -> None:
    import omni.timeline

    omni.timeline.get_timeline_interface().stop()
    result = context.save_as_stage(str(out))
    if result is False:
        raise RuntimeError(f"Could not save {out}")
    log(f"saved {out}")


def verify(app, timeline, frames: int = 240) -> bool:
    """Play, read the FlatScan of the robot LiDAR, report valid-ray statistics."""
    import numpy as np
    import omni.graph.core as og
    import omni.replicator.core as rep

    timeline.play()
    render_product = None
    annotator = None
    best = None
    for frame in range(frames):
        app.update()
        if annotator is None:
            node = og.Controller.node(f"{LIDAR_GRAPH}/RenderProduct")
            path = og.Controller.attribute("outputs:renderProductPath", node).get() if node else None
            if path:
                render_product = path
                annotator = rep.AnnotatorRegistry.get_annotator("IsaacComputeRTXLidarFlatScan")
                annotator.attach([render_product])
            continue
        data = annotator.get_data()
        ranges = np.asarray(data.get("linearDepthData", []) if isinstance(data, dict) else [], dtype=float).reshape(-1)
        if ranges.size == 0:
            continue
        valid = np.isfinite(ranges) & (ranges >= LIDAR_SETTINGS["omni:sensor:Core:nearRangeM"]) \
            & (ranges <= LIDAR_SETTINGS["omni:sensor:Core:farRangeM"])
        stats = (valid.mean(), ranges.size, float(np.nanmin(np.where(valid, ranges, np.nan))) if valid.any() else math.nan)
        if best is None or stats[0] > best[0]:
            best = stats
    if annotator is not None:
        annotator.detach()
    timeline.stop()
    app.update()
    if best is None:
        log(f"VERIFY: no FlatScan data after {frames} frames (render product {render_product})")
        return False
    fraction, rays, nearest = best
    log(f"VERIFY: rays={rays} valid={fraction:.1%} nearest={nearest:.3f} m")
    return fraction >= MIN_VALID_FRACTION


if __name__ == "__main__":
    main()
