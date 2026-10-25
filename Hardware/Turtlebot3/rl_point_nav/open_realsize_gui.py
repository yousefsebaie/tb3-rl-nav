#!/usr/bin/env python3
"""Open the real-size TurtleBot3 scene in the Isaac Sim window and press Play.

The robot LiDAR's scan is drawn in the viewport (showDebugView on the ROS 2 LiDAR
helper) and published on /tb03/scan. Drive the robot from another terminal:

  ros2 run teleop_twist_keyboard teleop_twist_keyboard --ros-args -r cmd_vel:=/tb03/cmd_vel

Close the window to quit. Nothing is saved.

  ./isaac_run.sh open_realsize_gui.py
"""

from pathlib import Path

SCENE = Path("/home/g8/Desktop/FInalSceneDemo_realsize.usd")
ROBOT = "/World/turtlebot3_burger_02"
SCAN_HELPER = "/World/ActionGraph_lidar/ScanPublish"


def main() -> None:
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": False})
    import omni.timeline
    import omni.usd
    from isaacsim.core.utils.extensions import enable_extension
    from isaacsim.core.utils.viewports import set_camera_view

    # Enable the bridge before opening the scene (enabling it later replaces the stage).
    enable_extension("isaacsim.ros2.bridge")
    for _ in range(10):
        app.update()
    context = omni.usd.get_context()
    if not context.open_stage(str(SCENE)):
        raise RuntimeError(f"Could not open {SCENE}; build it with make_realsize_scene.py")
    for _ in range(600):
        app.update()
        stage = context.get_stage()
        if context.get_stage_loading_status()[2] == 0 and stage and stage.GetPrimAtPath(ROBOT).IsValid():
            break

    helper = stage.GetPrimAtPath(SCAN_HELPER)
    if helper.IsValid():
        helper.GetAttribute("inputs:showDebugView").Set(True)
    set_camera_view(eye=[-4.6, -0.9, 0.4], target=[-3.0, 0.6, -0.7])
    omni.timeline.get_timeline_interface().play()
    print("[TB3_GUI] Playing. LiDAR on /tb03/scan, drive with /tb03/cmd_vel. Close the window to quit.", flush=True)
    while app.is_running():
        app.update()
    app.close()


if __name__ == "__main__":
    main()
