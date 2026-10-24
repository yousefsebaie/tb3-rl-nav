# TurtleBot3 RL point navigation: simulation work so far, and what hardware needs

Handoff for the hardware side of the CyberScape project. It covers what was built to get RL working in Isaac Sim, how it works, the exact numbers the policy was trained on, how it fits the Overcooked-AI / MAPPO thesis plan, and what is still missing to run it on the real TurtleBot3 Burger.

Code lives in `rl_point_nav/` of this repo (originally `Hardware/Turtlebot3/rl_point_nav/` in `cyberspace-main`). Hardware nodes are in `hardware/`; in this repo `../safety_filter_node.py` means `hardware/safety_filter_node.py`.

---

## 1. What the RL does

One TurtleBot3 Burger learns **point navigation**: drive from its start pose to a goal (x, y) 0.5–2.0 m away. In V2 it also avoids small box obstacles using LiDAR.

- Algorithm: **PPO** (Stable-Baselines3 2.9.0, PyTorch 2.7, Gymnasium 1.3), MLP policy, CPU, a single environment.
- Simulator: **Isaac Sim** (`~/isaac-sim`). The scene is `/home/g8/Desktop/FInalSceneDemo.usd`, robot prim `/World/turtlebot3_burger_02`.
- Training talks to Isaac **directly through its Python API, not through ROS 2**. That is faster, deterministic and lets us teleport the robot to reset. The ROS 2 path (`ros_point_nav_env.py`) exists for hardware and was not used for training.

### Versions

| Version | Observation | Obstacles | Train script | Model to use |
|---|---|---|---|---|
| V1 | 3D (goal only) | none | `train_isaac_point_nav_ppo.py` (2,048 steps, pipeline test) | `models/isaac_point_nav_ppo_v1/final_model.zip` |
| V1B | 3D (goal only) | none | `train_isaac_point_nav_ppo_v1b.py` | `models/isaac_point_nav_ppo_v1b/checkpoint_15000_steps.zip` (frozen; never overwrite) |
| **V2** | **6D (goal + 3 LiDAR sectors)** | 0–3 boxes per episode | `train_isaac_point_nav_lidar_ppo_v2.py` | **`models/isaac_point_nav_lidar_ppo_v2/checkpoint_170000_steps.zip`** |

**Important:** in the V2 folder, `final_model.zip` (saved 2026-09-29 01:19) comes from an earlier short test run, *not* the long run. The long run (`--timesteps 200000`) wrote checkpoints every 10k steps from 03:18 to 14:12 and stopped at 170k without saving a final model. **Use `checkpoint_170000_steps.zip`**, not `final_model.zip`.

**No evaluation numbers are recorded yet.** Before going to hardware, run the comparison (section 6) to get success, collision and timeout rates for V1B vs V2 vs the rule-based baseline.

---

## 2. The MDP (exactly as trained)

### Actions: discrete, 4 (`config.py`)

```python
class PointNavAction(IntEnum):
    FORWARD = 0      # v = 0.05 m/s, w = 0          for 30 physics frames
    TURN_LEFT = 1    # v = 0,        w = +0.80 rad/s for 15 frames
    TURN_RIGHT = 2   # v = 0,        w = -0.80 rad/s for 15 frames
    STAY = 3         # v = 0,        w = 0          for 15 frames
```

Every action is **stop-and-go**. The command is held for N frames, then the wheels are held at zero for **8 settle frames**, then the observation is read. At the assumed 60 Hz physics rate (check the `physics_dt=` line in the env startup log) that works out to:

- FORWARD = 0.50 s, about **2.5 cm** per step
- TURN = 0.25 s, about **0.2 rad (11.5°)** per step
- settle = about 0.13 s at zero velocity

`config.py` already contains the matching hardware timings (`forward_duration_s=0.50`, `turn_duration_s=0.25`, `stay_duration_s=0.25`).

The wheel commands go through Isaac's `DifferentialController` with wheel radius 0.033 m and wheel base 0.160 m (the real Burger values), clamped to max 0.22 m/s and 1.0 rad/s.

### Observation

V1/V1B (3D):
```
[ d_norm, sin(bearing), cos(bearing) ]
d_norm  = clip(distance_to_goal / 2.0 m, 0, 1)
bearing = wrap_to_pi(atan2(goal_y - y, goal_x - x) - yaw)   # goal angle in robot frame, CCW +
```

V2 (6D) appends three LiDAR sector clearances:
```
[ d_norm, sin(b), cos(b), front, left, right ]
front = clip(min valid range in sensor angles [-30°, +30°]  / 1.0 m, 0, 1)
left  = clip(min valid range in           [+30°, +120°] / 1.0 m, 0, 1)
right = clip(min valid range in           [-120°, -30°] / 1.0 m, 0, 1)
```
- Angles are in the sensor frame: 0° = robot forward, CCW positive (+90° = left).
- Clearances saturate at 1.0 m, so anything farther than 1 m reads as 1.0.
- Rear rays (|angle| > 120°) are not used.
- Sector edge cases in the simulator:
  - valid rays in the sector → clearance = minimum range
  - no valid rays, but the sector normally has rays → obstacle is closer than the 0.10 m near range → **clearance 0.0 (blocked)**
  - no valid rays and the baseline had none either → open space → inf, which clips to 1.0

**Pose source:** in simulation the pose (x, y, yaw) is **ground truth** from the articulation root, not odometry.

### Reward (identical code in the sim and ROS environments)

```python
progress = dist_before - dist_after
reward  = 5.0 * clip(progress, -0.10, +0.10) - 0.01   # progress shaping + step penalty
if dist_after <= 0.05:  reward += 10.0; terminated   # success, 5 cm tolerance
# V2 only:
if collision:           reward -= 10.0; terminated
truncated if steps >= 150 (V1B) / 250 (V2)  or  sim_time >= 60 s (V1B) / 120 s (V2)
```

Collision in V2 is **geometric, not physical**. It is a check of a circle of radius 0.105 m (the Burger footprint) against each 0.12 × 0.12 m box. The policy never experienced real physical contact.

### Episodes / reset

- The start pose is always the same: about (-3.318, 0.444) in the world frame, heading +X.
- Reset teleports the articulation back to the cached pose, zeroes all velocities, settles for 8 frames and checks the result (≤ 1 cm, ≤ 0.02 rad).
- The goal is drawn uniformly from three banks: LEFT, RIGHT and STRAIGHT (`isaac_point_nav_env.py`). In robot-relative terms (start = origin, forward = +x):
  - straight: +0.6, +1.2, +1.6 m ahead
  - left: about (+0.5, +0.87) and (+0.6, +1.04)
  - right: about (+0.5, -0.87) and (+0.6, -1.04)
- V2 obstacles, per episode:
  - with probability 0.7, one "blocking" box on the start–goal line (±10 cm lateral), at least 0.40 m from the start and 0.30 m from the goal
  - plus 0–2 random distractor boxes in the start–goal bounding box ±0.40 m
  - boxes at least 0.30 m apart
  - box size: 0.12 m square, 0.50 m tall

### PPO hyperparameters (V1B and V2 are identical)

```python
PPO("MlpPolicy", env, policy_kwargs={"net_arch": [64, 64]},
    learning_rate=3e-4, gamma=0.99, gae_lambda=0.95, ent_coef=0.01,
    n_steps=256, batch_size=64, n_epochs=5, seed=42, device="cpu")
# CheckpointCallback every 10k steps (V2) / 5k (V1B), wrapped in Monitor
```
V2 trained at about 10k steps per 40 minutes, headless.

---

## 3. How the simulation side was made to work (problems solved)

1. **Direct Isaac Gym env** (`isaac_point_nav_env.py`). `SimulationApp` starts once and stays alive across episodes. The robot is driven with `WheeledRobot` + `DifferentialController` using the DOFs `a__namespace_wheel_left_joint` / `a__namespace_wheel_right_joint`. Reset uses `Articulation.set_world_poses` plus zeroed velocities. Each piece was proven first with `isaac_direct_drive_smoke_test.py`, `isaac_direct_turn_smoke_test.py` and `isaac_articulation_reset_smoke_test.py`.
2. **RTX LiDAR set up for RL** (validated by the `isaac_lidar_*_test.py` files):
   - **Mount.** The LiDAR prim is top-level at `/World/tb03_v2_lidar` (config `Example_Rotary_2D`). It is *not* parented to the robot, because the robot USD is scaled about 4–7× and the sensor would inherit that scale. Before and after every physics step it is moved to follow `base_link` (translation + normalized rotation only, unit scale) at offset **(+0.10, 0, +0.22) m** from `base_link`. The real `base_scan` offset (-0.143, 0, 0.685) in that scaled asset put the sensor inside the LDS housing mesh, so it returned no data.
   - **Range.** `nearRangeM` and `minDistBetweenEchosM` were overridden from 1.0 to **0.10 m**; `farRangeM` is 200 m.
   - **Angles.** FlatScan angle = sensor angle − 0.07°, with angle `i` = `azimuthRange[0] + i*res`. The returned azimuth endpoints are used directly.
   - **Self-returns.** The robot's visual meshes (only the four `.../visuals` roots; collision and physics are untouched) are hidden **before physics starts**, so the LiDAR does not see the oversized robot body. `--show-robot` in the demo is for presentation only.
   - **Rendering in headless mode.** Rendering is forced on every step and the sensor render product is explicitly kept updating, otherwise the RTX LiDAR produces nothing.
3. **Obstacle pooling.** Four box prims are created once under `/World/obstacles`. Reset only **teleports** them; unused boxes are parked at (1000, 1000). Never delete and re-create a prim at the same path while the LiDAR is running: the earlier "runtime walls are invisible to LiDAR" bug traced back to doing exactly that.
4. **Floor height** is measured with a PhysX downward raycast, never from a rendered bounding box (hidden visuals give a bogus box).
5. **Reset controller extension** (`Hardware/Isaac/tb03_reset_controller/`). This is a separate Isaac Kit extension that lets a **ROS 2** request (decoded by the ROS2 Bridge into custom OmniGraph nodes) teleport the robot inside a running Isaac session. It belongs to the ROS-in-the-loop sim path, not to the direct training path.

---

## 4. How this relates to Overcooked-AI / MAPPO (thesis plan)

The thesis question: *how can LLM-driven mission planning be integrated with MAPPO-based execution to improve multi-robot coordination?* The plan validates in stages:

1. RL foundations (MDP → returns/value → policy gradient → actor–critic)
2. **PPO baseline, single agent.** ← The TurtleBot3 point-nav work in this document is here: one robot, PPO, a real robot model in Isaac.
3. **Overcooked-AI with MAPPO.** A cooperative two-agent benchmark (kitchen: divide tasks, navigate, time actions, adapt to a partner). It checks that the MAPPO pipeline trains, that the agents cooperate, and that the reward design and metrics work. The code is a separate repo (`github.com/saraabdelfatta/MAPPO-agent-`: env, networks, buffer, mappo, train, test_env_reward). **None of it is in this repo.** Overcooked is a controlled testbed, not the robot problem itself.
4. Robot simulation: map MAPPO observations, actions and rewards onto the multi-robot digital twin (Isaac).
5. Physical validation: deploy carefully and compare against the existing heuristic baseline.

How the pieces connect:
- MAPPO uses **CTDE** (centralized training, decentralized execution). Each robot has an *actor* that acts only on its **local observation**. A centralized critic uses the global state, but only during training.
- The V2 TurtleBot policy has exactly the shape of one such decentralized actor:
  - local observation = goal distance and bearing plus LiDAR sectors
  - discrete navigation actions
  - reward = progress + success − time − collision

  Going multi-robot means adding other robots' relative positions and task status to the observation, sharing a team reward, and adding a centralized critic.
- The Overcooked mapping: agents ↔ robots, kitchen subtasks ↔ mission subtasks from the LLM planner, discrete moves ↔ discrete navigation actions, shared dish reward ↔ mission success reward.
- In the full system, the **LLM layer** decides *what* to do (decomposes the mission into goals/tasks) and the **RL layer** decides *how* to execute each step. Concretely, the planner's `move_to_point(x, y)` (today a hand-written controller in `burger_comm_node.py`) could be replaced by the PPO policy.

---

## 5. Hardware: can this code run the real robot?

**Short answer: partly. The Isaac files cannot run on hardware; the ROS environment can, but only for the 3D (no-LiDAR) policy. Nothing yet runs V2 on hardware.**

| File | Runs on hardware? |
|---|---|
| `isaac_*.py`, `train_*`, `eval_*`, `demo_*` | **No.** They need Isaac Sim. |
| `config.py` | Yes (shared constants; topics `/cmd_vel_raw`, `/odom`) |
| `ros_point_nav_env.py` | **Yes, for V1B only.** ROS 2, publishes Twist to `/cmd_vel_raw`, reads `/odom`, times each action from odom stamps, always sends a zero Twist after each step, and has a 2 s stale-odom watchdog. It gives the 3D observation only, reset does *not* move the robot (you place it by hand), and there is no script that loads a policy and runs it. |
| `../safety_filter_node.py` | Yes. Relays `/cmd_vel_raw` → `/cmd_vel` and zeroes forward motion if any `/scan` ray within ±18° is < 0.30 m. |
| `../burger_comm_node.py` | Yes. The current mission executor (non-RL `move_to_point`, rotate, scan). It already publishes to `/cmd_vel_raw`. |

### What has to be built to run V2 on the Burger

1. **`ros_point_nav_lidar_env.py`.** Extend `RosPointNavEnv`:
   - subscribe to `/scan` (`sensor_msgs/LaserScan`, BEST_EFFORT QoS)
   - build the same 6D observation
   - add about 0.13 s of zero-velocity settle after each action, to match the sim's 8 settle frames
2. **LiDAR sim-to-real conversion (most important).**
   - **Mount mismatch.** The sim sensor sits **+0.10 m ahead** of `base_link`. The real Burger `base_scan` sits **-0.032 m behind** it (URDF `xyz="-0.032 0 0.172"`). A front obstacle therefore reads about 13 cm farther on the real robot than in training. Fix: convert each real ray to an (x, y) point in `base_link`, shift it into a virtual sensor at (+0.10, 0), recompute range and angle, then bin into the sectors.
   - **Angle convention.** Real `/scan` has `angle_min = 0` at the front, CCW, running up to about 2π. Wrap the angles to [-180°, 180°) before applying the sector masks.
   - **Invalid values.** The real LDS reports `0.0` or `inf` for no return.
     - Range limits: LDS-01 is 0.12–3.5 m; LDS-02 is 0.16–8 m.
     - Treat `inf` or > `range_max` as open (clearance 1.0).
     - Ignore `0` / < `range_min`.
     - If a sector has no valid rays, it is ambiguous: open space or something closer than `range_min`. Safest is to use the last valid value, or 0.0 if the previous reading was already close (< 0.25 m).
   - Sketch, **untested**:
     ```python
     def sim_matched_sectors(scan, mount_real=(-0.032, 0.0), mount_sim=(0.10, 0.0), norm=1.0):
         sectors = {"front": (-30, 30), "left": (30, 120), "right": (-120, -30)}
         best = {k: math.inf for k in sectors}
         for i, r in enumerate(scan.ranges):
             if not math.isfinite(r) or r < scan.range_min:
                 continue                                  # no return / too close: skip
             r = min(r, scan.range_max)
             a = scan.angle_min + i * scan.angle_increment
             px = mount_real[0] + r * math.cos(a) - mount_sim[0]   # point relative to sim sensor
             py = mount_real[1] + r * math.sin(a) - mount_sim[1]
             ang = math.degrees(math.atan2(py, px)); rng = math.hypot(px, py)
             for k, (lo, hi) in sectors.items():
                 if lo <= ang <= hi:
                     best[k] = min(best[k], rng)
         return [min(best[k] / norm, 1.0) for k in ("front", "left", "right")]
     ```
3. **Runner script** (`run_policy_on_hardware.py`):
   - load the checkpoint with `PPO.load(path, device="cpu")`
   - `obs = env.reset(goal=...)`, then loop `action = model.predict(obs, deterministic=True)` → `env.step(action)` until terminated or truncated
   - stop on Ctrl-C (the env already sends zero Twist in `finally`)
   - **Goals are relative to where the robot starts.** Odometry is (0, 0, 0) at bringup, so give goals in the trained range: 0.5–2.0 m away, within about ±60° of forward (the trained banks).
4. **Run order on the robot side:**
   1. TB3 bringup on the Pi (`ros2 launch turtlebot3_bringup robot.launch.py`, `TURTLEBOT3_MODEL=burger`)
   2. `safety_filter_node.py`
   3. the runner

   Start with V1B, no obstacles, and the robot on blocks or in open floor space.
5. **Python environment.** Run the policy on the **laptop** (ROS 2 Humble, Python 3.10), not on the Pi. Use the same `ROS_DOMAIN_ID` on both, on the same network.
   - Install `stable-baselines3` (2.x), `gymnasium` and CPU `torch` into the system Python 3.10. They are currently installed **only** in Isaac's Python 3.11.
   - If the model zip will not load across versions, export the 64×64 MLP weights to NumPy and run the forward pass by hand. That needs no torch, and could even run on the Pi.

### Known sim-to-real gaps to watch

- **Pose.** Sim used ground-truth pose; hardware uses wheel odometry, which drifts. The 5 cm goal tolerance is checked against odometry, so the real robot may stop a few cm off.
- **Safety filter interplay.** The filter blocks forward motion at < 0.30 m in front, but in sim the policy could legally pass about 0.17 m from a box. Near obstacles the real robot may stall (FORWARD blocked → repeated FORWARD → timeout). That is safe but lowers the success rate. You can lower the threshold slightly or log how often it triggers.
- **Obstacles.** Real obstacles must cross the LiDAR plane, which is about 0.18 m above the floor on the Burger. Use boxes about 0.12 m wide and ≥ 0.25 m tall, like the training boxes.
- **Dynamics.** The sim has no motor ramp and no latency. The real OpenCR ramps velocity, and 0.5 s at 0.05 m/s may cover a bit less than 2.5 cm. Measure one FORWARD and one TURN on the real robot (`python ros_point_nav_env.py --goal-x 1 --goal-y 0 --action FORWARD` prints progress) and compare with the sim figures (2.5 cm, 0.2 rad).
- **Training distribution.** The policy only ever started from one pose with goals in the listed banks, so it may not generalize to arbitrary goals. If hardware tests show that, the next step is training with randomized start poses and goals.

---

## 6. Commands (simulation, on this machine)

```bash
cd ~/Downloads/cyberspace-main/Hardware/Turtlebot3/rl_point_nav
# train V2 (headless)
~/isaac-sim/python.sh train_isaac_point_nav_lidar_ppo_v2.py --timesteps 200000
# GUI demo of a trained V2 checkpoint (no --model = rule-based controller)
~/isaac-sim/python.sh demo_isaac_point_nav_lidar_v2_gui.py --model models/isaac_point_nav_lidar_ppo_v2/checkpoint_170000_steps.zip
# quantitative comparison V1B vs V2 vs rules -> results/*.csv
~/isaac-sim/python.sh eval_lidar_vs_nolidar.py --v2-model models/isaac_point_nav_lidar_ppo_v2/checkpoint_170000_steps.zip --episodes 30
```

Hardware (ROS 2 Humble, system Python, after `source /opt/ros/humble/setup.bash`):
```bash
python3 ../safety_filter_node.py
python3 ros_point_nav_env.py --goal-x 0.5 --goal-y 0.0 --action FORWARD   # one-action smoke test
```

## 7. Key files

- `config.py`: actions, hardware topics, timings and reward constants (shared by sim and ROS)
- `isaac_point_nav_env.py`: V1/V1B Isaac Gym env (actions, observation, reward, reset, goal banks)
- `isaac_point_nav_lidar_env.py`: V2 env (LiDAR mount and sectors, obstacle pool, collision)
- `train_isaac_point_nav_lidar_ppo_v2.py`: V2 PPO training
- `eval_lidar_vs_nolidar.py`: fair comparison on seeded scenarios
- `demo_isaac_point_nav_lidar_v2_gui.py`: GUI demo plus the `ReactiveController` rule baseline
- `ros_point_nav_env.py`: ROS 2 hardware/sim env (3D observation, odometry-timed actions)
- `../safety_filter_node.py`: `/cmd_vel_raw` → `/cmd_vel` front-obstacle filter
