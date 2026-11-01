#!/usr/bin/env python3
"""Check the real-size scene: robot stands still, robot LiDAR sees a box 0.40 m ahead.

Opens FInalSceneDemo_realsize.usd (never saves), puts a 0.12 m box 0.40 m in front
of the LiDAR, plays for --seconds and reports the robot pose drift and the FlatScan
range straight ahead. While it runs, /tb03/scan is published over ROS 2, so it can be
checked from another terminal with `ros2 topic echo /tb03/scan --once`.

  ./isaac_run.sh isaac_realsize_lidar_check.py [--seconds 60] [--gui]
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

SCENE = Path("/home/g8/Desktop/FInalSceneDemo_realsize.usd")
LOG_FILE = SCENE.with_name("FInalSceneDemo_realsize_check.log")
ROBOT = "/World/turtlebot3_burger_02"
BASE_LINK = f"{ROBOT}/a__namespace_base_link"
GRAPH_RENDER_PRODUCT = "/World/ActionGraph_lidar/RenderProduct"
BOX = "/World/realsize_check_box"
BOX_AHEAD_M = 0.40          # from the LiDAR
BOX_SIZE_M = 0.12
LIDAR_X_M = -0.032          # base_scan in base_link
FRONT_HALF_DEG = 5.0
TAG = "[TB3_REALSIZE_CHECK]"


def log(message: str) -> None:
    line = f"{TAG} {message}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as handle:
        handle.write(line + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--gui", action="store_true")
    args = parser.parse_args()
    LOG_FILE.write_text("")

    from isaacsim import SimulationApp

    app = SimulationApp({"headless": not args.gui})
    try:
        run(app, args.seconds)
    except Exception:
        import traceback

        log("FAILED:\n" + traceback.format_exc())
        raise
    finally:
        app.close()


def run(app, seconds: float) -> None:
    import time

    import numpy as np
    import omni.graph.core as og
    import omni.replicator.core as rep
    import omni.timeline
    import omni.usd
    from isaacsim.core.utils.extensions import enable_extension
    from pxr import Gf, UsdGeom

    enable_extension("isaacsim.ros2.bridge")
    for _ in range(10):
        app.update()
    context = omni.usd.get_context()
    if not context.open_stage(str(SCENE)):
        raise RuntimeError(f"Could not open {SCENE}")
    for _ in range(600):
        app.update()
        stage = context.get_stage()
        if context.get_stage_loading_status()[2] == 0 and stage and stage.GetPrimAtPath(ROBOT).IsValid():
            break
    else:
        raise RuntimeError("Scene did not finish loading")

    def base_pose():
        m = UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath(BASE_LINK))
        t = m.ExtractTranslation()
        fwd = m.TransformDir(Gf.Vec3d(1, 0, 0))
        return float(t[0]), float(t[1]), float(t[2]), math.atan2(fwd[1], fwd[0])

    x, y, z, yaw = base_pose()
    floor_z = z - 0.010 - 0.002   # base_link is 0.010 m above base_footprint
    d = LIDAR_X_M + BOX_AHEAD_M
    box = UsdGeom.Cube.Define(stage, BOX)
    box.CreateSizeAttr(1.0)
    api = UsdGeom.XformCommonAPI(box.GetPrim())
    api.SetScale(Gf.Vec3f(BOX_SIZE_M, BOX_SIZE_M, 0.5))
    api.SetTranslate(Gf.Vec3d(x + d * math.cos(yaw) + 0.5 * BOX_SIZE_M * math.cos(yaw),
                              y + d * math.sin(yaw) + 0.5 * BOX_SIZE_M * math.sin(yaw), floor_z + 0.25))
    log(f"start base=({x:.3f}, {y:.3f}, {z:.3f}) yaw={math.degrees(yaw):.1f} deg; "
        f"box face {BOX_AHEAD_M:.2f} m ahead of the LiDAR")

    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    annotator = None
    fronts = []
    end = time.monotonic() + seconds
    frames = 0
    while time.monotonic() < end:
        app.update()
        frames += 1
        if annotator is None:
            node = og.Controller.node(GRAPH_RENDER_PRODUCT)
            path = og.Controller.attribute("outputs:renderProductPath", node).get() if node else None
            if path:
                annotator = rep.AnnotatorRegistry.get_annotator("IsaacComputeRTXLidarFlatScan")
                annotator.attach([path])
                log(f"render product {path}")
            continue
        data = annotator.get_data()
        if not isinstance(data, dict):
            continue
        ranges = np.asarray(data.get("linearDepthData", []), dtype=float).reshape(-1)
        bounds = np.asarray(data.get("azimuthRange", []), dtype=float).reshape(-1)
        if ranges.size == 0 or bounds.size != 2:
            continue
        angles = (np.linspace(bounds[0], bounds[1], ranges.size) + 180.0) % 360.0 - 180.0
        front = ranges[(np.abs(angles) <= FRONT_HALF_DEG) & np.isfinite(ranges) & (ranges > 1e-3)]
        if front.size:
            fronts.append(float(front.min()))
    if annotator is not None:
        annotator.detach()
    x2, y2, z2, yaw2 = base_pose()
    timeline.stop()

    drift = math.hypot(x2 - x, y2 - y)
    log(f"after {frames} frames: base=({x2:.3f}, {y2:.3f}, {z2:.3f}) drift={drift:.4f} m "
        f"dz={z2 - z:+.4f} m dyaw={math.degrees(yaw2 - yaw):+.2f} deg")
    if not fronts:
        log("LIDAR: no front readings")
        return
    median = float(np.median(fronts))
    log(f"LIDAR front (+/-{FRONT_HALF_DEG:.0f} deg) min range: median={median:.3f} m "
        f"over {len(fronts)} scans (expected ~{BOX_AHEAD_M:.2f})")
    ok = abs(median - BOX_AHEAD_M) < 0.05 and drift < 0.02 and abs(z2 - z) < 0.02
    log("RESULT " + ("PASS" if ok else "CHECK"))


if __name__ == "__main__":
    main()
