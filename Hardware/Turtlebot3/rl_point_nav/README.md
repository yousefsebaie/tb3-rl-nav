# rl_point_nav: TurtleBot3 RL point navigation

PPO point navigation for the TurtleBot3 Burger: trained in Isaac Sim, then deployed on the real robot over ROS 2 Humble.

**New here? Read [docs/RL_HANDOFF.md](../../../docs/RL_HANDOFF.md) first.** It explains the trained policy, the exact observation/action/reward, and what is still needed for hardware. The hardware team's task list is [docs/HARDWARE_REQUEST.md](../../../docs/HARDWARE_REQUEST.md).

## Files (paths relative to this folder)

| Path | What | Who |
|---|---|---|
| `config.py` | Actions, timings, reward, ROS topics. **Shared: the policy depends on it. Don't change it without telling the other team.** | both |
| `isaac_*.py`, `train_*`, `eval_*`, `demo_*` | Isaac Sim environments, training, evaluation, GUI demos, sensor tests. Need Isaac Sim. | simulation PC |
| `ros_point_nav_env.py` | ROS 2 environment (odometry, `/cmd_vel_raw`). Starting point for hardware. New hardware RL files go next to it. | hardware PC |
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

## Hardware PC (ROS 2 Humble, system Python 3.10)

```bash
source /opt/ros/humble/setup.bash
pip install stable-baselines3 gymnasium torch --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple
cd Hardware/Turtlebot3
python3 safety_filter_node.py
python3 rl_point_nav/ros_point_nav_env.py --goal-x 0.5 --goal-y 0.0 --action FORWARD   # one-action smoke test
```
Robot commands always go to `/cmd_vel_raw` (through the safety filter), never directly to `/cmd_vel`.
