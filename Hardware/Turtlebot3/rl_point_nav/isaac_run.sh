#!/usr/bin/env bash
# Run an Isaac Sim Python script with Isaac's internal ROS 2 Humble.
# A sourced system ROS (/opt/ros/humble, Python 3.10) breaks Isaac's ROS 2
# bridge (Python 3.11), so its Python and library paths are removed here.
# ROS_DOMAIN_ID is kept, so `ros2 topic list` in a normal terminal still sees Isaac.
#   ./isaac_run.sh make_realsize_scene.py
#   ./isaac_run.sh train_isaac_point_nav_lidar_ppo_v2.py --timesteps 200000
ISAAC_SIM="${ISAAC_SIM:-$HOME/isaac-sim}"
unset PYTHONPATH AMENT_PREFIX_PATH CMAKE_PREFIX_PATH COLCON_PREFIX_PATH
export ROS_DISTRO=humble
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export LD_LIBRARY_PATH="$ISAAC_SIM/exts/isaacsim.ros2.bridge/humble/lib"
exec "$ISAAC_SIM/python.sh" "$@"
