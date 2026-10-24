# Project status and next steps (2026-10-24)

For both PCs. Read this first after a break.

## Where everything is

- **Repo:** https://github.com/yousefsebaie/tb3-rl-nav. It is the only copy we change. The old `cyberspace-main` folders are backups.
- **Branches:**
  - `hardware`: hardware PC (Mac, real robots)
  - `sim`: simulation PC (Ubuntu, Isaac Sim)
  - `main`: work that is tested on both sides
- **Secrets:** `config.json` holds the API keys and is not in git. Copy `config.example.json` to `config.json` and fill in the keys.
- **On macOS:** use `git switch hardware`, not `git checkout hardware`. The `Hardware/` folder has the same name, and macOS ignores case.

## Done so far

### Real robots (LLM mission → approval → robots move)
Both robots completed a full mission from a curl request: plan → approve → move forward, turn 180°, come back.
- **Burger** (`Hardware/Turtlebot3/burger_comm_node.py`):
  - `move_forward(metres)` drives straight, with no turn at the start.
  - If something blocks the path, it turns 90° left or right at random (50/50).
  - Distances are in real metres, measured by odometry.
- **PuppyPi** (`Hardware/PuppyPi/puppy_comm_node.py`):
  - Movement is timed in seconds, because the dog has no odometry.
  - Tuned settings: speed 5.0, turn rate 0.18, height -10, x_shift -0.5, walking gait 0.35 / 0.4 / 7.0.
  - Turning uses a separate gait (0.1 / 0.3 / 5.0) so it stops at 180° instead of overshooting.
- **`Manager.py`** now runs `High_Level_Plan_Generator.py`, then `Low_Level_Planner.py` for each robot in `config.json` `robots_in_curr_mission`. It used to run `generate_multi_turtlebot_plan.py`. **Sim PC: this changes your `/generate_plan`.**
- **`SharedAutonomy/hitl_endpoints.py`:** the sender log path is now relative to the project. The old fixed `/home/g8` path caused HTTP 500 on approve on any other PC.
- **Missions:** `Plans/final_*_low_level_plan.json` are the real missions. Burger: 2 m forward, turn 180°, return to base. Dog: walk 30 s, turn 180°, walk 30 s back.

### RL on the real Burger (code ready, not yet run on the robot)
All of it is in `Hardware/Turtlebot3/rl_point_nav/`; its README has the full run steps.
- **Model export:** `export_policy_numpy.py` turned the V1B and V2 checkpoints into `.npz` files. They pick the same action as SB3 on 10,000/10,000 test inputs, and the robot needs only NumPy.
- **Robot environment:** `ros_point_nav_lidar_env.py` reads `/scan` and builds V2's six inputs.
  - `lidar_sectors.py` corrects for the LiDAR position: the sim sensor sat 0.10 m ahead of the robot's centre, the real one sits 0.032 m behind it.
  - After each action it holds zero velocity for 0.13 s.
  - It ends the episode as COLLISION if anything comes within 0.13 m of the robot's centre.
- **Runner:** `run_policy_on_hardware.py`.
  - Goals are relative to the robot.
  - `--dry-run` reads the sensors without moving.
  - It logs a CSV per run and always stops the robot.
- **Offline results** (ideal motion, not Isaac):
  - No obstacles: V1B and V2 reached all 7 trained goals.
  - One blocking box: V2 succeeded 29 of 40 times and collided 11 times.
  - V2 usually starts with a short left turn.
  - Run against a fake ROS 2 robot: V1B succeeded at 0.6 m and V2 at 1 m to the left.

## Not done / unknown
- No RL run on the real Burger yet.
- No Isaac evaluation numbers for V2 yet.
- The safety filter blocks forward motion under 0.30 m, but in sim the policy passed about 0.17 m from boxes, so the real robot may stall near obstacles.
- **API keys must be replaced:** Cohere, SambaNova (LLaMA) and OpenAI. They were exposed in a chat and in the old `ismaileldahshan1/Cyberspace` repo.

## Hardware PC: next steps
1. Charge the Burger and SSH in with `ssh ubuntu@<IP>`. The ROBOTIS default password is `turtlebot`; find the IP with `arp -a`.
2. Follow the hardware section of `Hardware/Turtlebot3/rl_point_nav/README.md`:
   1. dry run
   2. measure one FORWARD and one TURN
   3. V1B, no obstacles
   4. V2, no obstacles
   5. V2 with one box
3. Put the run CSVs in `rl_point_nav/results/`, commit and push to `hardware`.
4. When it works, open a Pull Request from `hardware` into `main`.

## Simulation PC: next steps
1. **Get the hardware work.** Review and merge the `hardware` → `main` Pull Request, then run `git switch sim && git pull origin main`. Or merge it directly: `git fetch && git merge origin/hardware`.
2. **Run the evaluation and push the CSV:**
   ```bash
   ~/isaac-sim/python.sh eval_lidar_vs_nolidar.py --v2-model models/isaac_point_nav_lidar_ppo_v2/checkpoint_170000_steps.zip --episodes 30
   ```
   Commit `results/*.csv` to `sim`. The hardware results get compared against these numbers.
3. **Check `/generate_plan` with the new `Manager.py`.** It now uses High → Low level planning per robot in `config.json`. If Isaac still needs the old multi-TurtleBot generator, run `python3 generate_multi_turtlebot_plan.py` directly, or tell the hardware side before changing `Manager.py`.
4. **If you retrain**, check the training for these real-robot issues:
   - Train with the LiDAR at the real position (-0.032 m) so no correction is needed.
   - Randomize the start pose and the goals.
   - Penalize getting closer than 0.30 m in front, to match the safety filter.
   - Add small motion delays and noise.

   Then export the new model with `python3 export_policy_numpy.py <checkpoint>.zip` and push both the `.zip` and the `.npz`.
5. **Agree with the hardware side before changing `rl_point_nav/config.py`**, the observation or the reward. Either one changes what the robot code must reproduce.

## Daily routine (both PCs)
```bash
cd ~/tb3-rl-nav && git switch <hardware|sim> && git pull
# work ... when something works:
git add <files> && git commit -m "what and why" && git push
# get the other side's finished work:
git pull origin main
```

## Quick reference: running the robots (hardware PC = Mac)

**IP addresses** (these can change on the hotspot):

| Device | IP |
|---|---|
| Mac | 192.168.137.126 (check with `ipconfig getifaddr en0`) |
| Burger | 192.168.137.69 |
| PuppyPi | 192.168.137.136 |

- **Mac server:** `cd ~/tb3-rl-nav && source .venv/bin/activate && python3 hitl-comm-server.py`
- **Burger Pi:**
  - T1: `export TURTLEBOT3_MODEL=burger && ros2 launch turtlebot3_bringup robot.launch.py`
  - T2, only if `/scan` is missing: `ros2 launch hls_lfcd_lds_driver hlds_laser.launch.py`
  - T3: `source /opt/ros/humble/setup.bash && python3 ~/burger_comm_node.py --server http://<MAC_IP>:5001 --cmd_vel_topic /cmd_vel`
- **PuppyPi:** `ssh pi@192.168.137.136`, then:
  ```bash
  sudo docker exec -it puppypi bash
  source /opt/ros/noetic/setup.bash && source /home/ubuntu/puppypi/devel/setup.bash
  python3 /puppy_comm_node.py --server http://<MAC_IP>:5001 --puppy_ip 127.0.0.1
  ```
  After changing `puppy_comm_node.py`, copy it in again from the Mac:
  ```bash
  scp Hardware/PuppyPi/puppy_comm_node.py pi@192.168.137.136:/tmp/ && \
    ssh pi@192.168.137.136 "sudo docker cp /tmp/puppy_comm_node.py puppypi:/puppy_comm_node.py"
  ```
- **Mission:**
  1. Generate a plan:
     ```bash
     curl -X POST http://127.0.0.1:5001/generate_plan -H "Content-Type: application/json" \
       -d '{"mission_title":"multi_robot_mission","mission_text":"..."}'
     ```
  2. Review it: `curl http://127.0.0.1:5001/hitl/pending_plans`
  3. Approve it: `curl -X POST http://127.0.0.1:5001/hitl/approve_plan`
