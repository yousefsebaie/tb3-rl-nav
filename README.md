# tb3-rl-nav

Reinforcement-learning point navigation for the TurtleBot3 Burger: trained with PPO in Isaac Sim, then deployed on the real robot over ROS 2 Humble. Part of the CyberScape thesis (LLM mission planning + RL/MAPPO execution).

**New here? Read [docs/RL_HANDOFF.md](docs/RL_HANDOFF.md) first.** It explains the trained policy, the exact observation/action/reward, and what is still needed for hardware.

## Layout

| Path | What | Who |
|---|---|---|
| `rl_point_nav/config.py` | Actions, timings, reward, ROS topics. **Shared: the policy depends on it. Don't change it without telling the other team.** | both |
| `rl_point_nav/isaac_*.py`, `train_*`, `eval_*`, `demo_*` | Isaac Sim environments, training, evaluation, GUI demos, sensor tests. Need Isaac Sim. | simulation PC |
| `rl_point_nav/ros_point_nav_env.py` | ROS 2 environment (odometry, `/cmd_vel_raw`). Starting point for hardware. New hardware RL files go next to it. | hardware PC |
| `rl_point_nav/models/` | Trained PPO policies. **Use `isaac_point_nav_lidar_ppo_v2/checkpoint_170000_steps.zip` (V2)** and `isaac_point_nav_ppo_v1b/checkpoint_15000_steps.zip` (V1B). V2's `final_model.zip` is an old short test run. | both |
| `hardware/` | ROS 2 nodes for the real Burger: `safety_filter_node.py`, `burger_comm_node.py` | hardware PC |
| `isaac_extensions/tb03_reset_controller/` | Isaac Kit extension: ROS 2–triggered robot reset inside Isaac | simulation PC |
| `scenes/FInalSceneDemo.usd` | Isaac scene. The code expects it at `/home/g8/Desktop/FInalSceneDemo.usd` (`SCENE_PATH` in `isaac_point_nav_env.py`). | simulation PC |
| `docs/` | Handoff, hardware request, chat log | both |
| `results/` | Evaluation CSVs from sim and hardware runs | both |

## Simulation PC

```bash
cd rl_point_nav
~/isaac-sim/python.sh train_isaac_point_nav_lidar_ppo_v2.py --timesteps 200000
~/isaac-sim/python.sh demo_isaac_point_nav_lidar_v2_gui.py --model models/isaac_point_nav_lidar_ppo_v2/checkpoint_170000_steps.zip
~/isaac-sim/python.sh eval_lidar_vs_nolidar.py --v2-model models/isaac_point_nav_lidar_ppo_v2/checkpoint_170000_steps.zip
```

## Hardware PC (ROS 2 Humble, system Python 3.10)

```bash
source /opt/ros/humble/setup.bash
pip install stable-baselines3 gymnasium torch --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple
python3 hardware/safety_filter_node.py
python3 rl_point_nav/ros_point_nav_env.py --goal-x 0.5 --goal-y 0.0 --action FORWARD   # one-action smoke test
```
Robot commands always go to `/cmd_vel_raw` (through the safety filter), never directly to `/cmd_vel`.

## How we work

- `main` = works and is tested. Nobody pushes half-finished work to `main`.
- Each team works on its own branch: **`sim`** (simulation PC) and **`hardware`** (hardware PC).
- Every session:
  ```bash
  git pull                         # get the latest first
  # ... work ...
  git add <files> && git commit -m "what and why"
  git push
  ```
- When something works, open a Pull Request into `main` on GitHub; the other team glances at it and merges.
- New model? Commit only the checkpoint you want people to use, and say in the commit message which one is best.
- Changing `config.py`, the observation, or the reward means the policy must be retrained. Agree on it first.
