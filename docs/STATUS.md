# Project status and next steps (2026-11-01)

For both PCs. Read this first after a break.

## Where everything is

- **Repo:** https://github.com/yousefsebaie/tb3-rl-nav. It is the only copy we change. The old `cyberspace-main` folders are backups.
- **Branches:**
  - `hardware`: hardware PC (Mac, real robots)
  - `sim`: simulation PC (Ubuntu, Isaac Sim)
  - `main`: work that is tested on both sides
- **Secrets:** `config.json` holds the API keys and is not in git. Copy `config.example.json` to `config.json` and fill in the keys.
- **On macOS:** use `git switch hardware`, not `git checkout hardware`. The `Hardware/` folder has the same name, and macOS ignores case.

## Simulation PC update (2026-11-23): V4, MAPPO for two robots, two-robot CyberScape missions
**Hardware side: read [MAPPO_HARDWARE_HANDOFF.md](MAPPO_HARDWARE_HANDOFF.md).** In short:
- **V4** (`models/isaac_point_nav_lidar_ppo_v3_random/checkpoint_1570000_steps.npz`, profile `v3`, 6 inputs) is a drop-in for V3 on one Burger. Sim benchmark: 83% / 80% / 85% success vs B0's 30% / 35% / 25%.
- **MAPPO V5** (`models/mappo_v5_kl05/best_model.npz`, **15 inputs**) is the final two-robot policy. In sim it reached 68% both-reached with 12% collisions, using the gap controller. It needs 5 extra LiDAR sectors and the **other robot's position** in a shared frame.
- In Isaac, CyberScape plans for TURTLEBOT3_1 + TURTLEBOT3_2 run phase by phase, including a live LLM plan (5/5 phases, 4/4 survivors). On hardware this needs a **phase barrier** and **pose exchange** between the two Pis.
- `config.py` (actions, timings, reward) is unchanged. Nothing new has run on a real robot yet.

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

### RL on the real Burger (first runs done 2026-10-25, see "Real-robot RL results" below)
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
- V2/V3 alone do not avoid reliably on the real Burger; with the bridge + route re-planning the patrol mission completes (see "CyberScape → RL on the real Burger").
- No Isaac evaluation numbers for V2 yet.
- The safety filter blocks forward motion under 0.30 m, but in sim the policy passed about 0.17 m from boxes, so the real robot may stall near obstacles.
- **API keys must be replaced:** Cohere, SambaNova (LLaMA) and OpenAI. They were exposed in a chat and in the old `ismaileldahshan1/Cyberspace` repo.

## Real-robot RL results (2026-10-25)
Logs: `Hardware/Turtlebot3/rl_point_nav/results/`. The CSV names start with `20260506`, because the Pi's clock is wrong (it has no internet time), so that date doesn't mean anything.

| Run | Result |
|---|---|
| Single actions | TURN ≈ 9–11° (sim 11.5°). FORWARD 0.6–2.1 cm alone, ≈ 2.8 cm inside a run (sim 2.5 cm). |
| V1B, 1.0 m, clear | SUCCESS ×2 (34–35 steps, ~4.7 cm from goal, straight line). The first run bumped a chair: V1B is blind. |
| V2, 1.2 m, clear | SUCCESS (51 steps). Always turns left first, then drifts ~20 cm left and curves back. |
| V2, narrow box centred ahead | Mostly SUCCESS, but the clear-floor route also misses a centred narrow box. |
| V2, box in its usual path | It turns further left than without the box, so it does react to the LiDAR, but late (~15–20 cm). |
| V2, black fabric case | COLLISION stop: the LiDAR barely sees matte black, curved objects about as tall as the scan plane. |
| V2, wide wooden box (~35 cm) | COLLISION stop: it got within ~15 cm, kept choosing FORWARD, and the safety filter blocked it. |

"COLLISION" here means the safety filter and the 0.13 m contact guard stopped the robot, not a hard crash.

What the real robot showed, for retraining:
- Training only punishes actual contact (−10), so V2 cuts as close as it can. **Penalise getting closer than 0.30 m.**
- The real LDS-02 drops points (dark or curved objects, anything under ~0.15 m), so `front` jumps between 0.0 and 1.0. **Add LiDAR dropout and noise.**
- **Train with varied obstacle widths** (12–40 cm), not only 12 cm boxes.
- V2's habitual first left turn shows up on every run.
- The safety filter checks ±18°, but the policy's `front` sector is ±30°. Consider matching them.

## V3 training (simulation PC, started 2026-10-25)
V3 is training now: 1M steps, about 3 days. Watch it with `cd Hardware/Turtlebot3/rl_point_nav && ./train_long.sh`. Models go to `models/isaac_point_nav_lidar_ppo_v3/`.

Changes from V2. They come from the real-robot results above and are all in `v3_lidar_config()`; V2 is unchanged:
- **Real-size scene:** `FInalSceneDemo_realsize.usd`, built by `make_realsize_scene.py`. The Burger is at true scale, and the drive is fixed: frictionless caster, sphere wheels. One FORWARD is 1.8 cm and one TURN is 8.4°; the real robot does about 2.8 cm and 9–11°. The training LiDAR sits at the real `base_scan` (-0.032, 0, 0.172) m.
- **Proximity penalty:** up to −0.30 per step when the robot centre is within 0.30 m of an obstacle surface, on top of −10 for contact.
- **LiDAR model:** ±1.5 cm noise, 10% of rays dropped, no returns under 0.15 m. Each sector also drops out 5% of the time, and then reads like the hardware's empty-sector rule: open (1.0), or 0.0 if it was already under 0.25 m.
- **Obstacles:** 12–40 cm wide per episode.
- **Front sector ±18°,** matching the safety filter. Left is 18–120°, right is −120…−18°.

**To run a V3 model on the Burger, `lidar_sectors.py` needs V3 settings.** V2 settings will give wrong inputs:
- `SIM_MOUNT_M = REAL_MOUNT_M`. The sim LiDAR is at the real mount now, so no +0.10 m shift.
- `SECTORS_DEG = {"front": (-18, 18), "left": (18, 120), "right": (-120, -18)}`.
- Keep `SIM_NEAR_RANGE_M = 0.10` and the empty-sector rule.

`profile.txt` in each model folder says which settings a model needs. Export to `.npz` with `export_policy_numpy.py`, as before.

## CyberScape → RL on the real Burger (hardware, 2026-11-01)
**Working:** an LLM-generated CyberScape plan (the sim's patrol mission: A (1.5, 0.75) → scan → B (1.5, −0.75) → scan → rotate 90° → report) ran to the end on the real Burger with the sim demo's 3-box layout.

How it runs (`Hardware/Turtlebot3/rl_point_nav/hardware_rl_bridge.py`, on the Pi):
- It reuses `cyberscape_rl_bridge.Bridge` from the sim demo unchanged (goal alignment, waypoint legs, stall recovery, spin guard) through a small adapter for the real robot.
- **Route re-planning (default, hardware only so far):** `route_planner.py` maps every LiDAR scan into a grid; A* plans a route to each checkpoint around everything seen; V3 drives 0.5 m legs along it and the route is re-planned after each leg. In the real run the route changed 1× on the way to A and 3× to B. `--no-replan` gives the sim demo's straight legs.
- Plans come from the Mac's HITL server (`--server http://<MAC_IP>:5001`, same `/get_instruction` flow as `burger_comm_node.py`) or a plan JSON (`--plan`).
- `--no-safety-filter` publishes straight to `/cmd_vel` (then stop `safety_filter_node.py`). The real run used it.
- On the Pi, the Isaac env module is stubbed (no gymnasium there); `Bridge` does not need it.

```bash
python3 hardware_rl_bridge.py --plan rl_live_burger_plan.json --no-safety-filter     # a plan file
python3 hardware_rl_bridge.py --server http://192.168.137.171:5001 --no-safety-filter # approved CyberScape plans
```
Server on the Mac for the hardware: run it from a copy of the repo with its own `config.json` (`robots_in_curr_mission: ["TURTLEBOT3"]`, `plan_parser_llm: cohere`, `final_low: Plans/rl_live_burger_plan.json`), like `run_cyberscape_rl_demo.sh` does, and set `CYBERSCAPE_ISAAC_SENDER` to a no-op script so approval does not start the Isaac sender. The planner needs `langchain-google-genai` and `python-dotenv` in `.venv`.

**V3 alone on the real robot (no bridge):** clear path 3/3; one box on the direct line 3/3; box in its left-swing path: stuck (it keeps choosing FORWARD; the safety filter holds it); two boxes close together: spins ~3.5 turns, then STAY until timeout. Clear goals beyond ~1.5 m: turns left and then STAYs. The sim demo hides this with the bridge's helpers, which is why the bridge is needed on hardware too.

**For V3 retraining (sim side):** obstacles anywhere, including in the policy's own swing path (not only on the direct line); mirrored episodes against the left-turn habit; a penalty for no progress; goals up to the scene size.

**Battery:** below ~11.1 V the wheel node crashes (`There is no status packet!`, then `stack smashing detected`) and after a restart the motors are off (`torque: false` on `/sensor_state`). Charge first. Turn the motors on with `ros2 service call /motor_power std_srvs/srv/SetBool "{data: true}"`.

## Hardware PC: next steps
MAPPO handoff ([MAPPO_HARDWARE_HANDOFF.md](MAPPO_HARDWARE_HANDOFF.md) section 7), status on 2026-11-23:
- **Built and unit-tested on the Mac** (`python3 -m unittest test_mappo_hardware`, 10 tests against the simulator's formulas):
  - `lidar_sectors.py`: profile `v5` (the 8 sectors, each computed on its own rays, overlaps included). `REAL_MOUNT_M = (+0.032, 0)` for the caster-first Burgers; V3/V4/V5 are re-measured from the training mount `STOCK_MOUNT_M = (-0.032, 0)`, so an obstacle 0.50 m from the LDS now reads 0.564, as in training. This changes V3 on hardware too (it used to read 0.50). The route planner takes the mount as a parameter (sim keeps the stock one).
  - `mappo_hardware.py`: shared mission frame, UDP pose exchange between the Pis (port 47110, 10 Hz, poses older than 0.5 s refused), teammate inputs, the 0.105 m teammate circle merged into the scan.
  - `ros_mappo_env.py`: the 15 V5 inputs on the robot. `mappo_dry_run.py`: steps 3 and 4 without driving (dummy command topic), logs a CSV per robot.
  - With the teammate "far and idle", V5 picks V4's action in 93.7% of 2000 random states.
- **To do on the robots, in order:**
  1. Measure the LDS offset with a ruler (wheel-axle line → LDS centre, along the driving direction) and set `REAL_MOUNT_M`.
  2. Step 1: copy V4 to `~/rl_point_nav/models/v4/` (with `profile.txt` = `v3`), run the loop plan with V3 and with V4 on Burger 1.
  3. Step 2: the same on Burger 2.
  4. Step 3: `mappo_dry_run.py` on one robot (no `--peer`).
  5. Step 4: `mappo_dry_run.py` on both robots with `--peer`, robots still, checked with a tape measure.
  6. Steps 5–7: **built and run, see below.** Steps 1–4 and 5 (single encounters) were skipped on purpose for the demo.

**CyberScape → MAPPO V5 on both real Burgers: first run COMPLETE (2026-11-23).** Hand-written `Plans/rl_mappo_hardware_plan.json` (5 phases: both cross to the far station at the same time, Burger 1 searches east while Burger 2 holds, Burger 2 searches west while Burger 1 holds, both return to base at the same time with crossing routes, report), approved in the HITL dashboard. Result: 5/5 phases, 2/2 survivor stand-ins found, no collision, no contact stop, 0 gap hand-overs (MAPPO drove every step after the align-first turns), closest approach 0.36 m centre to centre (0.15 m between bodies) during the crossings. Record: `Hardware/Turtlebot3/rl_point_nav/results/missions/mappo_two_burgers_20261123/` (sender log, both agent logs, per-step CSVs with all 15 inputs).
- Run it: `START1=0,0,0 START2=1.8,0,180 SURVIVORS="1.8,-1.2 0.6,-1.2" ./run_cyberscape_mappo_hardware.sh <BURGER1_IP> <BURGER2_IP>` (or `... --live` for LLM planning from `mission_files/rl_mappo_hardware_mission.txt`), then approve at http://localhost:5001/dashboard/index.html. Stop: `curl -X POST http://<IP>:8765/stop`.
- Layout on the 0.60 m floor tiles: Burger 1 centred in a tile, Burger 2 three tiles ahead in the same column facing back, cone 3 ahead / 2 right, box 1 ahead / 2 right (right = Burger 1's right).
- **The LDS-02 does not see the black Burgers at all** (no returns at 2 m in either scan), so the LiDAR placement step (`mappo_find_teammate.py`) does not work and the starts are given from the tiles. Robot-robot separation rests entirely on the pose exchange (as in the simulator, where robot bodies were hidden from the LiDAR); `mappo_drive.py` stops a move when the centres come within 0.24 m.
- Survivor detection is the simulator's stand-in (known positions within 1.5 m), not a camera detector.
- Next: more runs for numbers (the sim reported 72% both-reached on its benchmark), the doorway layout (0.8 m opening, walls >= 25 cm tall), and a `--live` LLM-planned run.
- Older items: the full single-robot cycle in server mode (LLM plans → approve → `hardware_rl_bridge.py --server ...`), and a Pull Request from `hardware` into `main`.

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
| Mac | 192.168.137.171 (check with `ipconfig getifaddr en0`) |
| Burger 1 (hostname `ubuntu`, `ROS_DOMAIN_ID=30`) | 192.168.137.65 (was .146; find it in the hotspot device list) |
| Burger 2 (hostname `burger2`, `ROS_DOMAIN_ID=31`, no camera) | 192.168.137.143 (was .127, .191; changes often, scan the hotspot for SSH) |

**Burger 2** runs a byte-for-byte copy of Burger 1's SD card (2026-11-21; image at `~/burger1_card.img` on the Mac, also a backup of Burger 1). After the copy: new hostname, new machine ID (which also gave it a new hotspot IP), `ROS_DOMAIN_ID=31`. Same LDS-02 LiDAR; the OpenCR firmware worked as is (`/scan` 9.3 Hz, `/odom` 20 Hz). At first the wheels didn't turn (not even with the OpenCR SW1/SW2 buttons): the Dynamixel chain cable to motor ID 1 was half out, so that motor's LED never blinked at power-on. Re-seated it; first test 2026-11-23: 0.201 m forward, 43.5° turn as commanded. If a Burger doesn't move with torque on, check that both motor LEDs blink once at power-on.

**Both Burgers drive ball-first** (the caster leads, the camera on Burger 1 faces the caster side, the LiDAR's 0° faces the caster side). That is the reverse of the stock TurtleBot3 and of the simulation, where the caster trails; it is consistent within each robot, so missions work. Seen from the caster side, the motors must be **ID 2 on your left, ID 1 on your right** on both robots. Burger 2 had them the other way round, so it drove toward its LiDAR's back. Fixed 2026-11-23 with ROBOTIS `turtlebot3_setup_motor` (OpenCR core 1.5.3 via `arduino-cli`, one motor connected at a time; a read-only scan option was added to the sketch): ID 1 = drive mode 0, ID 2 = drive mode 1, 1 Mbps, velocity mode. Then the Burger firmware was re-flashed from the Pi (`~/opencr_update/update.sh /dev/ttyACM0 burger.opencr`, V230127R1). Check after any motor work: put a box in front of the LiDAR (0°), drive 10 cm forward, the box must get 10 cm closer. Burger 2 passed (0.510 → 0.408 m). With the caster leading, the LiDAR sits ~3 cm ahead of the wheel axle instead of behind it: `lidar_sectors.REAL_MOUNT_M` (see "Hardware PC: next steps"). The two robots use different domain IDs so a `/cmd_vel` meant for one never moves the other; each runs its own bridge and talks to CyberScape over HTTP. `sudo` on the robots needs the password (not stored here).
| PuppyPi | 192.168.137.136 |

- **Mac server:** `cd ~/tb3-rl-nav && source .venv/bin/activate && python3 hitl-comm-server.py`
- **Burger Pi:**
  - T1: `LDS_MODEL=LDS-02 TURTLEBOT3_MODEL=burger ros2 launch turtlebot3_bringup robot.launch.py`
    - The LiDAR label says LDS-03, but it sends LDS-02 (LD08) data at 115200 baud. With `LDS-01` `/scan` stays empty; with `LDS-03` the driver logs "Detected lidar stuck". `/scan` should be about 9.4 Hz.
    - Don't also start `hlds_laser.launch.py`: bringup already starts the LiDAR driver, and two drivers block each other on `/dev/ttyUSB0`.
    - Stop bringup with Ctrl+C before closing its window, or it keeps running and holds the ports.
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
