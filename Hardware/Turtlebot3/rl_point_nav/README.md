# rl_point_nav: TurtleBot3 RL point navigation

PPO point navigation for the TurtleBot3 Burger: trained in Isaac Sim, then deployed on the real robot over ROS 2 Humble.

**New here? Read [docs/RL_HANDOFF.md](../../../docs/RL_HANDOFF.md) first.** It explains the trained policy, the exact observation/action/reward, and what is still needed for hardware. The hardware team's task list is [docs/HARDWARE_REQUEST.md](../../../docs/HARDWARE_REQUEST.md).

## Files (paths relative to this folder)

| Path | What | Who |
|---|---|---|
| `config.py` | Actions, timings, reward, ROS topics. **Shared: the policy depends on it. Don't change it without telling the other team.** | both |
| `isaac_*.py`, `train_*`, `eval_*`, `demo_*` | Isaac Sim environments, training, evaluation, GUI demos, sensor tests. Need Isaac Sim. | simulation PC |
| `ros_point_nav_env.py` | ROS 2 environment (odometry, `/cmd_vel_raw`). Starting point for hardware. New hardware RL files go next to it. | hardware PC |
| `run_policy_on_hardware.py`, `ros_point_nav_lidar_env.py`, `lidar_sectors.py`, `policy_numpy.py` | Run a trained policy on the real Burger (V1B or V2). Need only ROS 2 + NumPy, so they run on the Pi. | hardware PC |
| `export_policy_numpy.py` | Converts a `.zip` checkpoint to the `.npz` the robot uses, and checks it picks the same actions. Needs SB3 + torch. | either |
| `models/` | Trained PPO policies. **Use `isaac_point_nav_lidar_ppo_v2/checkpoint_170000_steps.zip` (V2)** and `isaac_point_nav_ppo_v1b/checkpoint_15000_steps.zip` (V1B). V2's `final_model.zip` is an old short test run. | both |
| `results/` | Evaluation CSVs (`eval_lidar_vs_nolidar.py` writes here; put hardware results here too) | both |
| `../safety_filter_node.py`, `../burger_comm_node.py` | ROS 2 nodes for the real Burger | hardware PC |
| `../../Isaac/tb03_reset_controller/` | Isaac Kit extension: ROS 2–triggered robot reset inside Isaac | simulation PC |
| `scenes/FInalSceneDemo.usd` (repo root) | Isaac scene. The code expects it at `/home/g8/Desktop/FInalSceneDemo.usd` (`SCENE_PATH` in `isaac_point_nav_env.py`). | simulation PC |

## Simulation PC

```bash
cd Hardware/Turtlebot3/rl_point_nav
~/isaac-sim/python.sh train_isaac_point_nav_lidar_ppo_v2.py --timesteps 200000
~/isaac-sim/python.sh demo_isaac_point_nav_lidar_v2_gui.py --model models/isaac_point_nav_lidar_ppo_v2/checkpoint_170000_steps.zip
~/isaac-sim/python.sh eval_lidar_vs_nolidar.py --v2-model models/isaac_point_nav_lidar_ppo_v2/checkpoint_170000_steps.zip
```

## Hardware PC: run the policy on the real Burger

The policy runs **on the Burger's Pi** and needs only ROS 2 Humble and NumPy, with no PyTorch.
`export_policy_numpy.py` turned the two checkpoints into `.npz` files; they pick the same action as SB3 on 10,000/10,000 test inputs:
`models/isaac_point_nav_ppo_v1b/checkpoint_15000_steps.npz` (V1B) and `models/isaac_point_nav_lidar_ppo_v2/checkpoint_170000_steps.npz` (V2).
If you retrain, re-export: `python3 export_policy_numpy.py <checkpoint>.zip` (needs SB3 + torch).

`ros_point_nav_lidar_env.py` reproduces the training loop on the robot:
- **LiDAR:** reads `/scan` and builds the same 6 values as V2 training. Each ray is re-measured from the simulated LiDAR position (+0.10 m) instead of the real one (-0.032 m); see `lidar_sectors.py`.
- **Settle:** holds zero velocity for 0.13 s after each action, then reads fresh odometry and a fresh scan.
- **Contact guard:** ends the episode as `COLLISION` if anything is within 0.13 m of the robot's centre.
Actions, timings and reward come unchanged from `config.py`.

**1. Copy the files to the Pi** (from the repo root, on the hardware PC):
```bash
ssh ubuntu@<BURGER_IP> "mkdir -p ~/rl_point_nav/models"
cd Hardware/Turtlebot3
scp safety_filter_node.py rl_point_nav/{config.py,ros_point_nav_env.py,ros_point_nav_lidar_env.py,lidar_sectors.py,policy_numpy.py,run_policy_on_hardware.py} ubuntu@<BURGER_IP>:~/rl_point_nav/
scp rl_point_nav/models/isaac_point_nav_ppo_v1b/checkpoint_15000_steps.npz ubuntu@<BURGER_IP>:~/rl_point_nav/models/v1b.npz
scp rl_point_nav/models/isaac_point_nav_lidar_ppo_v2/checkpoint_170000_steps.npz ubuntu@<BURGER_IP>:~/rl_point_nav/models/v2.npz
```

**2. On the Pi, open three SSH terminals.** Run `source /opt/ros/humble/setup.bash` in each one first.

| Terminal | Command |
|---|---|
| T1 bringup | `export TURTLEBOT3_MODEL=burger && ros2 launch turtlebot3_bringup robot.launch.py` |
| T2 safety filter | `python3 ~/rl_point_nav/safety_filter_node.py` |
| T3 policy | `cd ~/rl_point_nav`, then the tests below |

Check the LiDAR with `ros2 topic hz /scan` (about 9.4 Hz). Our Burger's LiDAR is labelled LDS-03 but needs `export LDS_MODEL=LDS-02` (set in the Pi's `~/.bashrc`); don't start a second LiDAR driver next to bringup.

Stop `teleop_keyboard` before running the policy: it keeps publishing to `/cmd_vel` and fights the safety filter.

**3. Test in this order.** Goals are relative to the robot: `--goal-x` is metres ahead, `--goal-y` is metres to the left. The trained range is 0.5–2.0 m away, within ±60° of straight ahead.
```bash
# a) No motion: print what the policy sees and would do. Put a box ~0.5 m ahead: "front" should drop to ~0.4
python3 run_policy_on_hardware.py --model models/v2.npz --goal-x 1.0 --goal-y 0 --dry-run
# b) One action, compare with sim (FORWARD ~0.025 m, TURN ~0.2 rad)
python3 ros_point_nav_env.py --goal-x 1.0 --goal-y 0.0 --action FORWARD
# c) V1B, no obstacles
python3 run_policy_on_hardware.py --model models/v1b.npz --goal-x 1.0 --goal-y 0
# d) V2, no obstacles, then with one box (~12 cm wide, >= 25 cm tall) on the path
python3 run_policy_on_hardware.py --model models/v2.npz --goal-x 1.2 --goal-y 0
```
`Ctrl-C` always stops the robot. Each run prints `RESULT SUCCESS / COLLISION / TIMEOUT`, the steps and the final distance, and saves every step to `logs/*.csv`. Copy those CSVs into `results/` to compare them with simulation.

**What to expect**
- The robot moves stop-and-go, exactly as trained: 2.5 cm forward or an 11.5° turn, then a pause. A 1 m goal takes about 40–60 steps (about 30–45 s).
- An offline check with ideal motion found:
  - With no obstacles, both models reached all 7 trained goals.
  - With one blocking box, V2 got past it 29 times out of 40 and hit it 11 times.
  - V2 usually starts with a short left turn, then corrects.
- The safety filter blocks forward motion when something is closer than 0.30 m, but in sim the policy sometimes passed about 0.17 m from boxes. Near obstacles the real robot may stall and time out instead of colliding.

Robot commands always go to `/cmd_vel_raw` (through the safety filter), never directly to `/cmd_vel`.
