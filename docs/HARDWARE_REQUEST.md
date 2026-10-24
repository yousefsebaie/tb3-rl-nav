# Request for the hardware chat (paste this first)

I trained a PPO point-navigation policy for the TurtleBot3 Burger in Isaac Sim. Now I want to run it on the real Burger. Everything about the simulation side is in `RL_HANDOFF.md` in `docs/`; read it first, especially sections 2 (the exact MDP) and 5 (hardware gaps).

## My setup
- Real TurtleBot3 **Burger** (Raspberry Pi + OpenCR, LDS LiDAR), ROS 2 **Humble**.
- Laptop (Ubuntu, ROS 2 Humble, Python 3.10) on the same network as the robot.
- Existing nodes in `hardware/`: `safety_filter_node.py` (`/cmd_vel_raw` → `/cmd_vel`, blocks forward motion if something is < 0.30 m in front) and `burger_comm_node.py` (current non-RL mission executor).
- Trained models in `rl_point_nav/models/`:
  - `isaac_point_nav_lidar_ppo_v2/checkpoint_170000_steps.zip`: **main target**, 6D observation (goal + LiDAR sectors)
  - `isaac_point_nav_ppo_v1b/checkpoint_15000_steps.zip`: 3D observation (goal only), simpler first test

## What I want you to build
1. **`ros_point_nav_lidar_env.py`**: extend `rl_point_nav/ros_point_nav_env.py` so it also subscribes to `/scan` and produces the **same 6D observation** as training: `[d_norm, sin(bearing), cos(bearing), front, left, right]`.
   - Use the same sectors, normalization and edge cases as `rl_point_nav/isaac_point_nav_lidar_env.py`.
   - Correct for the LiDAR mount difference: the sim sensor was +0.10 m ahead of base_link, the real one is −0.032 m behind it. A sketch is in RL_HANDOFF.md §5.
   - Handle real LDS invalid readings (0 / inf).
   - Add a short zero-velocity settle (about 0.13 s) after each action to match the sim.
2. **`run_policy_on_hardware.py`**: loads a model with Stable-Baselines3 (CPU), takes a goal **relative to the start pose** (odometry starts at 0,0), runs deterministic actions until success, collision or timeout, logs every step, and always stops the robot on exit or Ctrl-C.
3. **Install steps** for `stable-baselines3`, `gymnasium` and CPU `torch` in the laptop's system Python 3.10. If the model zip won't load, give me a NumPy-only fallback that exports the 64×64 MLP weights.
4. **A step-by-step test plan**:
   1. Calibrate one FORWARD and one TURN on the real robot (sim: 2.5 cm and 0.2 rad).
   2. Run V1B with no obstacles.
   3. Run V2 with no obstacles.
   4. Run V2 with one box obstacle (≈0.12 m wide, ≥0.25 m tall).

   Include exact terminal commands and where each one runs (Pi vs laptop).
5. Tell me what metrics to record (success / collision / timeout, steps, final distance) so I can compare them with the simulation results.

## Rules
- Commands must go to `/cmd_vel_raw` (through the safety filter), never straight to `/cmd_vel`.
- Keep the action timings, reward constants and observation exactly as in `rl_point_nav/config.py`. Don't change them to make hardware "work"; tell me instead.
- Goals must stay in the trained range: 0.5–2.0 m away, within about ±60° of straight ahead.
