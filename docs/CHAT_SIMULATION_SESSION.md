# Chat log: simulation session (Claude Code, 2026-10-24)

## User: "is this the rl version code"

**Answer:** Yes. The RL code is in `Hardware/Turtlebot3/rl_point_nav/`. It trains TurtleBot3 point-navigation policies with PPO in Isaac Sim. The other files in `Hardware/Turtlebot3/` (move_turtlebot3.py, safety_filter_node.py, burger_comm_node.py, ...) are the non-RL ROS 2 nodes.

| Version | Observation | Train | Model |
|---|---|---|---|
| V1 | odometry only | train_isaac_point_nav_ppo.py | models/isaac_point_nav_ppo_v1/ |
| V1B | odometry only, retuned | train_isaac_point_nav_ppo_v1b.py | models/isaac_point_nav_ppo_v1b/ (15k steps) |
| V2 (latest) | 6D with RTX LiDAR sectors + pooled obstacles | train_isaac_point_nav_lidar_ppo_v2.py | models/isaac_point_nav_lidar_ppo_v2/ (checkpoints up to 170k) |

The other files:
- **Environments:** isaac_point_nav_env.py (V1/V1B), isaac_point_nav_lidar_env.py (V2) and ros_point_nav_env.py (ROS, odometry only).
- **Comparison:** eval_lidar_vs_nolidar.py (V2 vs V1B vs rules).
- **Tests:** the isaac_lidar_*_test.py files are sensor validation tests.

## User: asked for everything done to make RL work in simulation (code, how it's done, knowledge, how Overcooked relates), for the hardware chat, and whether the code can run the hardware

**Answer:** I wrote `RL_HANDOFF.md` (included in this bundle).

**Use the right V2 model.** `final_model.zip` in the V2 folder is from a short test run. The 200k-step run stopped at 170k without saving a final model, so the trained policy is `checkpoint_170000_steps.zip`. No evaluation results are recorded yet; run `eval_lidar_vs_nolidar.py` before hardware.

**Can this code run the hardware? Only partly.**
- **Isaac files:** none of them can (isaac_*, train_*, eval_*, demo_*). They need Isaac Sim.
- **ros_point_nav_env.py:** hardware-ready. It sends commands to /cmd_vel_raw through the safety filter, reads /odom and always stops after each action. But it only supports the V1B model (no LiDAR), and there's no script that loads a model and runs it.
- **Needed for V2 on the robot:**
  1. A ROS environment that reads /scan and builds the same 6 values.
  2. A runner script.
  3. SB3, PyTorch and Gymnasium installed in the system Python 3.10. They're currently only in Isaac's Python.
- **Biggest sim-to-real catch:** the simulated LiDAR sat 10 cm ahead of the robot's centre, but the real one is 3.2 cm behind it. Without a correction, front obstacles read about 13 cm too far. The doc has an untested conversion function.

**Can I work on both on one PC?** Yes. The machine has Isaac Sim (~/isaac-sim) and ROS 2 Humble. Train and evaluate in simulation there, and drive the real Burger over the network (same ROS_DOMAIN_ID, bringup on the Pi). Keep the two Pythons separate: ~/isaac-sim/python.sh for simulation, system python3 with ROS sourced for the robot.

## User: wanted to move the files to another PC, along with the hardware request and this chat

**Answer:** I made this bundle.
