# Simulation → hardware handoff: V4, MAPPO (two robots) and CyberScape two-robot missions

**For:** the hardware PC (Mac + Burger 1 + Burger 2).
**From:** the simulation PC, 2026-11-23.
**Covers:** everything the sim side did since the last handoff (STATUS.md, 2026-11-01). That includes:
- what each model is and which one to run;
- the exact inputs the final two-robot policy expects;
- everything that runs around the policy in simulation;
- how CyberScape drives two robots;
- what has to be built to do the same on the two real Burgers, as a step-by-step test plan.

Read `docs/RL_HANDOFF.md` (V1–V3 MDP) and `docs/STATUS.md` (V3 profile, the hardware bridge) first if you have not. This document only adds what is new.

All paths are relative to `Hardware/Turtlebot3/rl_point_nav/` unless they start with `docs/`, `Plans/`, `mission_files/` or `Hardware/`.

---

## 0. Short version

1. **Nothing changed in `config.py`.** The actions, their timings, the reward constants and the V3 LiDAR profile are exactly what you already run on the Burger.
2. **V4 is a drop-in replacement for V3 on one robot.** It has the same 6 inputs and the same `v3` profile. It's exported as `models/isaac_point_nav_lidar_ppo_v3_random/checkpoint_1570000_steps.npz`. It is better than V3 on the sim benchmark, but it collided in both full single-robot CyberScape missions in sim, which V3 completed (§2).
3. **The final two-robot policy is MAPPO V5:** `models/mappo_v5_kl05/best_model.pt`, exported for NumPy as `best_model.npz`.
   - Each robot runs its own copy. It's the same network shape as V4 (Tanh 64-64, 4 actions).
   - **It takes 15 inputs, not 6.** The extra 9 are 5 finer LiDAR sectors and 4 numbers about the other robot (role, its position relative to me, whether it is still driving).
   - **So each Burger must know where the other one is, in its own frame.** That is the one genuinely new hardware requirement (§4, §6).
4. **In simulation, the policy never drives alone.** The mission executor first turns each robot to face its goal (until within 15°). A follow-the-gap controller takes over when a robot is stuck. A robot with nothing to do in a phase holds still. These rules are part of the results, so the hardware needs them too (§5).
5. **CyberScape → two robots works end to end in sim**, including a live run where the LLM planned both robots from a plain-language mission and the operator approved it in the dashboard. Result: 5/5 phases and 4/4 survivors found.
   - Phases with the same number run at the same time, and the next phase starts only when both robots are done.
   - **On hardware that needs a phase barrier** across the two robots (§6).
6. **Nothing here has run on a real robot yet.** Every number is from Isaac Sim.

---

## 1. What the sim side did since 2026-11-01, in order

| When | What | Where |
|---|---|---|
| Nov 8 – 14 | **V4**: V3's 1.0M checkpoint fine-tuned on random clutter (up to 6 boxes, 10–45 cm) for +1.09M steps. Best window 84.5% success, 11.7% collisions at 1.5–1.6M steps. Chosen checkpoint: 1.57M. | `train_isaac_point_nav_lidar_ppo_v2.py --random-obstacles`, `models/isaac_point_nav_lidar_ppo_v3_random/` |
| Nov 1, Nov 16 | **Comparison study:** B0 (the original CyberScape controller, no RL) vs B1 (a heuristic with PPO's inputs and actions) vs PPO, on 70 matched seeded scenarios. V3 was run on Nov 1 and V4 on the same seeds on Nov 16. Results in §2. | `eval_timing.py`, `run_comparison_study.sh`, `results/timing/`, `results/comparison_report_*.pdf` |
| Nov 15 – 16 | **MAPPO run 1:** two robots, a shared actor warm-started from V4, and a centralized critic. Encounters that force interaction: head-on, crossing, doorway, corridor. | `isaac_multi_point_nav_env.py`, `train_mappo.py`, `models/mappo_v4/` |
| Nov 16 – 19 | **MAPPO v2/v3:** cooperative team reward (`coop`) and heavier crash penalties (`coop3`). Then `coop_tm`, in which the **actor sees the teammate** (role + relative position + active). Plus a passage curriculum (`passage_p15`/`p20`). | `models/mappo_v4_coop*`, `models/mappo_passage_*` |
| Nov 19 – 21 | **MAPPO V5 (final):** `coop_tm`, plus 5 finer LiDAR sectors, wider doorway and corridor training (`wide` task), and a KL penalty toward V4 so it keeps V4's obstacle skill. Two variants: KL weight 0.02 and 0.05. **kl05 is the one we use.** | `models/mappo_v5_kl05/` |
| Nov 22 | **Follow-the-gap hand-over controller** (`--gap-assist`): a scripted fallback when a robot is stuck, for doorways and corridors that the 3-sector policy will not enter. | `train_mappo.py` `gap_assist()` |
| Nov 22 – 23 | **CyberScape → MAPPO missions**: two-robot plans are executed one phase at a time by MAPPO in Isaac. There are three layouts. The hardest is the earthquake scenario: a rubble corridor, a fallen wall with two doorways, two rooms and 4 survivors. | `cyberscape_mappo_mission.py`, `Hardware/Isaac/cyberscape_mappo_plan_sender.py`, `run_cyberscape_mappo_demo.sh` |
| Nov 23 | **Live LLM planning for two robots**: CyberScape plans TURTLEBOT3_1 and TURTLEBOT3_2 separately from `mission_files/rl_mappo_disaster_mission.txt`. The operator approves, and MAPPO executes. Planning took 16.5 min. | `run_cyberscape_mappo_full_cycle.sh` |
| Nov 23 | **NumPy export** of V4 and V5 for the Pi. Both agree with torch/SB3 on 10,000/10,000 random inputs. | `export_mappo_numpy.py`, `export_policy_numpy.py` |

---

## 2. The models, and which one to use

| Model | Inputs | What it is | File for the Pi |
|---|---|---|---|
| V3 | 6 | What Burger 1 runs today | `models/isaac_point_nav_lidar_ppo_v3/checkpoint_1000000_steps.npz` |
| **V4** | 6 | V3 + random clutter fine-tune. **Single-robot option.** | `models/isaac_point_nav_lidar_ppo_v3_random/checkpoint_1570000_steps.npz` (+ `profile.txt` = `v3`) |
| MAPPO run 1 | 6 | First two-robot run; best checkpoint 110k | `models/mappo_v4/checkpoint_110000_steps.pt` |
| coop / coop3 / coop_tm | 6 / 6 / 10 | Intermediate runs (history only) | `models/mappo_v4_coop*/best_model.pt` |
| **MAPPO V5 kl05** | **15** | **Final two-robot policy** | `models/mappo_v5_kl05/best_model.npz` |

Every `.npz` loads with the existing `policy_numpy.NumpyPolicy`. It checks the input count and raises an error if you pass the wrong number. V5's `.npz` also holds `inputs`, the input names in order: `np.load(path)["inputs"]`.

### Sim results you will be compared against

**Single robot, 70 matched scenarios**, from the comparison study (`results/comparison_report_*.pdf` and the paper):

| | B0 (original CyberScape) | B1 (heuristic) | V4 PPO |
|---|---|---|---|
| Success, nominal / new layouts / degraded sensing | 30% / 35% / 25% | 20–30% | **83% / 80% / 85%** |
| Over 70 scenarios: reached / collided / stuck | 21 / 0 / 49 | 17 / 1 / 52 | **58** / 10 / 2 |
| Median time to goal | 12.9 s | – | 64.2 s (0.05 m/s stop-and-go) |

V3 reached 57% / 75% / 75% on the same benchmark.

In the two full single-robot CyberScape missions in sim (patrol and disaster), **V3 completed both and V4 collided in both**. So V4 is better at reaching goals but not a sure upgrade for missions. Try it on Burger 1 next to V3 (§7 step 1).

**Two robots, 100 deterministic episodes** of crossing, head-on and random-goal encounters with scattered boxes (`results/mappo_eval_summary.txt`):

| Policy | Both reached | Any collision | Per-robot success | Robot-robot contact |
|---|---|---|---|---|
| Two V4 copies (no MAPPO) | 57% | 23% | 72% | 6% |
| coop_tm (teammate inputs) | 62% | 20% | 74% | 2% |
| V5 kl02 | 64% | 17% | 77.5% | 4% |
| **V5 kl05** | **72%** | 15% | 81% | 4% |
| **V5 kl05 + gap controller** | 68% | **12%** | 79.5% | 4% |

On a second scenario set (seed 4000), V5 kl05 without the gap controller scored 56% / 22%, against 57% / 15% for two V4 copies. The MAPPO gain is therefore not robust across scenario sets. Report both if you compare.

**One robot through a doorway or corridor** (`solo`, 30 episodes, teammate parked):
- V4: **0%**. It refuses to enter.
- V5 kl05 + gap controller: **47%**, with 0–10% collisions.

That is why the gap controller exists.

**Full CyberScape missions in sim** (MAPPO V5 + gap controller, both robots together):
- Warehouse (station swap + doorway into a storage room): 3/3 after the doorway was widened from 0.65 to 0.80 m. The run before that collided in the 0.65 m doorway.
- Simple disaster (one doorway, 3 survivors): 2/2, all phases, 3/3 survivors.
- Hard disaster (rubble corridor, two doorways, partition, 4 survivors, 7 phases):
  - first layout: one run completed 7/7; a second collided with the wall in a 0.55 m strip;
  - the wall was then moved to leave an 0.8 m strip, and the next run completed 7/7 with 4/4 survivors.
- Hard disaster with the **LLM-generated** plan: 5/5 phases, 4/4 survivors.

**Doorways in these layouts are 0.8 m wide.** The policy was trained on 0.5–0.9 m doorways, but anything under ~0.7 m failed in practice. Build real test layouts with 0.8 m openings.

---

## 3. What did not change (you already have it)

These are the same in V3, V4 and V5 (`config.py`, `v3_lidar_config()`):

- **Actions, 4 discrete:**
  - FORWARD: 0.05 m/s for 30 frames (0.5 s);
  - TURN_LEFT and TURN_RIGHT: ±0.8 rad/s for 15 frames (0.25 s);
  - STAY: 15 frames.

  Then 8 settle frames at zero velocity, about 0.13 s. Physics runs at 1/60 s.
- **The first 6 inputs**, `[d_norm, sin(bearing), cos(bearing), front, left, right]`:
  - `d_norm = clip(distance_to_goal / 2.0, 0, 1)`;
  - bearing is the goal angle in the robot frame, counter-clockwise positive;
  - sectors (v3 profile): front ±18°, left 18…120°, right −120…−18°;
  - each sector value = `clip(min range / 1.0 m, 0, 1)`.
- **LiDAR rules:** the mount is the real `base_scan` (−0.032, 0, 0.172), so no shift. The empty-sector rule and the 0.10 m near range are what `lidar_sectors.py` profile `v3` already does.
- **Noise during training:** range noise 0.015 m, ray dropout 10%, sector dropout 5%, minimum valid range 0.15 m. These were training noise only. **Do not add them on the robot.**
- **Goal tolerance and collision:** goal tolerance 0.05 m. Footprint radius 0.105 m for collision.

---

## 4. The MAPPO V5 actor: exactly what it needs

### 4.1 Network

`15 → Linear 64 → tanh → Linear 64 → tanh → Linear 4 → argmax`, deterministic. Action indices are the same as `PointNavAction`: 0 = FORWARD, 1 = TURN_LEFT, 2 = TURN_RIGHT, 3 = STAY.

Both robots run **the same weights** (parameter sharing). They differ only in the `role` input.

### 4.2 The 15 inputs, in order

| # | Name | How to compute it on the robot | Range |
|---|---|---|---|
| 0 | `d_norm` | `clip(dist_to_goal_m / 2.0, 0, 1)`. Goals more than 2 m away read 1.0. | 0..1 |
| 1 | `sin_bearing` | `sin(wrap(atan2(gy − y, gx − x) − yaw))` | −1..1 |
| 2 | `cos_bearing` | `cos(...)` | −1..1 |
| 3 | `front` | v3 sector −18…+18° | 0..1 |
| 4 | `left` | v3 sector +18…+120° | 0..1 |
| 5 | `right` | v3 sector −120…−18° | 0..1 |
| 6 | `narrow_front` | sector −6…+6° | 0..1 |
| 7 | `front_left` | sector +18…+45° | 0..1 |
| 8 | `front_right` | sector −45…−18° | 0..1 |
| 9 | `left_side` | sector +45…+100° | 0..1 |
| 10 | `right_side` | sector −100…−45° | 0..1 |
| 11 | `role` | **1.0 on TurtleBot3_1 (Burger 1), 0.0 on TurtleBot3_2 (Burger 2).** Fixed for the whole mission. | 0 / 1 |
| 12 | `teammate_x` | other robot's position **ahead of me**: `clip(( cos(yaw)·dx + sin(yaw)·dy) / 2.0, −1, 1)` | −1..1 |
| 13 | `teammate_y` | other robot's position **to my left**: `clip((−sin(yaw)·dx + cos(yaw)·dy) / 2.0, −1, 1)` | −1..1 |
| 14 | `teammate_active` | 1.0 while the other robot is driving to a goal in the current phase. 0.0 once it reached it, collided, or has nothing to do this phase. | 0 / 1 |

Notes:
- `dx = x_other − x_me` and `dy = y_other − y_me`, with both robots' `base_link` positions in **one shared frame**. `yaw` is my heading in that same frame.
- **The sectors overlap**, e.g. `narrow_front` is inside `front`. Compute each of the 11 sectors **independently**: take the minimum valid range over its own rays, with the same v3 rules (near-range block, empty-sector rule, `clip(/1.0 m)`). `_sector_of()` in `lidar_sectors.py` returns only the first match, so the 5 new sectors need their own loop, or a second call with the V5 bounds. Angles are inclusive at both ends, in degrees, 0° = forward, counter-clockwise positive.
- The source of truth is `isaac_multi_point_nav_env.py`: `_observation()` (inputs 0–10) and `global_state()` (teammate geometry), plus `train_mappo.py` `actor_input()` (how 11–14 are picked).
- **Robot that is holding still:** in sim, a robot with no move in a phase gets its own position as its goal. It finishes on the first step, so its teammate sees `teammate_active = 0` for the rest of the phase.

### 4.3 How the other robot shows up in the LiDAR

In sim, robot bodies are **hidden** from the RTX LiDAR, and the teammate is **added** as an exact circle of radius 0.105 m at its centre. That happens before the noise and dropout model, so the policy learned that "the other robot" looks like a 21 cm round obstacle.

The real Burger at the scan plane (about 0.17 m) is mostly the other robot's LDS turret and top plate, which is a smaller target.

**Recommendation:** on each robot, merge a virtual 0.105 m circle at the teammate's shared-frame position into the scan before computing the sectors. That is the same ray–circle intersection as `_teammate_ranges()`. Keep the real ranges too, i.e. take the minimum of real and virtual. Then the policy sees the teammate the way it was trained to, even if the LDS misses it.

### 4.4 Reference code (what we would write; not run on a robot)

```python
import math, numpy as np
from policy_numpy import NumpyPolicy

V3 = {"front": (-18, 18), "left": (18, 120), "right": (-120, -18)}
V5 = {"narrow_front": (-6, 6), "front_left": (18, 45), "front_right": (-45, -18),
      "left_side": (45, 100), "right_side": (-100, -45)}
policy = NumpyPolicy("models/mappo_v5_kl05/best_model.npz")   # 15 inputs

def teammate_local(me, other):
    dx, dy = other[0] - me[0], other[1] - me[1]
    c, s = math.cos(me[2]), math.sin(me[2])
    return (np.clip((c * dx + s * dy) / 2.0, -1, 1), np.clip((-s * dx + c * dy) / 2.0, -1, 1))

def v5_input(me, goal, other, other_active, role, sector):
    """me/other = (x, y, yaw) in the shared frame; sector(lo_deg, hi_deg) -> 0..1 with the v3 rules."""
    d = math.hypot(goal[0] - me[0], goal[1] - me[1])
    b = math.atan2(goal[1] - me[1], goal[0] - me[0]) - me[2]
    b = math.atan2(math.sin(b), math.cos(b))
    tx, ty = teammate_local(me, other)
    return np.array([min(d / 2.0, 1.0), math.sin(b), math.cos(b),
                     *(sector(*V3[k]) for k in ("front", "left", "right")),
                     *(sector(*V5[k]) for k in V5),
                     role, tx, ty, 1.0 if other_active else 0.0])

action = policy.act(v5_input(me, goal, other, other_active, role=1.0, sector=my_sector_fn))
```

---

## 5. What runs around the policy in simulation (the hardware must do the same)

All of this is in `train_mappo.py`: `drive_phase()`, `gap_assist()` and `run_phase()`.

### 5.1 One phase = one episode

- Both robots start the phase from **where they are now**.
- Each robot with a `move_to_point(x, y)` (or `return_to_base()`) in this phase drives to it.
- The other robot sends **STAY** every step. It still counts as an obstacle for the mover.
- The phase ends when every mover has reached its goal (≤ 0.05 m), collided, or hit the step limit.
  - The step limit for missions is **400 steps**, raised from 250 because a doorway leg with a gap hand-over can take ~250 steps.
  - The sim-time limit is 800 s.
- **The phase fails if any mover did not reach its goal.** The mission stops there.
- After the moves, the phase's other commands run in plan order: `scan_with_lidar`, `detect_with_camera`, `communicate_with_apm`.

### 5.2 Align first (the "Goal Interface")

At the start of each phase, each mover turns in place, TURN_LEFT or TURN_RIGHT toward the goal, until `|bearing| ≤ 15°`. Only then does MAPPO drive.

Without this, a robot that starts facing away spins in place, the stuck detector (§5.3) sees "no movement" and hands over to the gap controller at step ~17.

### 5.3 Follow-the-gap hand-over (`--gap-assist`)

Per robot, every step:

1. **Stuck test.** Over the last 18 steps the robot moved less than 6 cm, it is more than 0.3 m from its goal, and it is not in the 25-step cool-down. → hand over.
2. **While handed over:**
   - If `|goal bearing| > 60°`: turn toward the goal.
   - Otherwise find the free gap nearest the goal on the **full scan**:
     - use rays within ±75° of forward; no return counts as 5 m;
     - apply a 5-ray median filter, to remove dropouts;
     - a ray is "free" if its range is more than `min(range in cone) + 0.35 m`;
     - free runs narrower than 6 rays are ignored;
     - pick the run whose centre is closest to the goal bearing.
   - If that gap is more than 8° off: turn toward it. Otherwise: FORWARD. If no gap is found, keep the policy's action.
3. **Hand back** after the robot has moved 1.2 m from where the hand-over started, or after 150 steps. Then a 25-step cool-down.

**Hardware note:** the sim scan has **1066 rays over 360°** (0.34° per ray), so "5 rays" ≈ 1.7° and "6 rays" ≈ 2°. The LDS-02 has far fewer rays. Convert both thresholds to **degrees**, not ray counts.

Across the 100 benchmark episodes it took over 53 times. In the mission runs it mostly took over in the rubble corridor legs. The doorway passes in the missions were driven by MAPPO itself.

### 5.4 Survivor detection is simulated

`detect_with_camera('survivor')` in sim reports the nearest survivor within **1.5 m** of the robot, using the simulator's known positions, and lights a green beacon in the GUI. There is no camera model.

On hardware this is whatever detection you use: Burger 1 has a camera, Burger 2 has none.

### 5.5 Things that are presentation-only

- `--show-robots` makes the robot meshes visible, so the LiDAR then sees them. Use it for the GUI only.
- The disaster props (rubble, people, lights) are below 0.09 m or above 0.27 m near routes, so they never enter the scan plane.

---

## 6. CyberScape with two robots: how it flows

### 6.1 In sim (`run_cyberscape_mappo_demo.sh disaster --live`)

```
mission text ──► /generate_plan (Mac or sim PC, port 5001)
                   Manager.py: High_Level_Plan_Generator.py, then for EACH robot
                   Low_Level_Planner.py → Verification_Module.py → Plan_Parser.py (Cohere)
               ──► Plans/rl_mappo_live_tb3_1_plan.json, Plans/rl_mappo_live_tb3_2_plan.json
               ──► HITL dashboard: Approve / Reject / Modify
               ──► Hardware/Isaac/cyberscape_mappo_plan_sender.py
                     merge_plans(): both robots' plans merged BY phase_number
                     for each phase: write {"phase": n, "lines": [...both robots...]} to the command file,
                                     wait for DONE / FAILED in the status file
               ──► train_mappo.py --mission-listen: run_phase() (§5) → DONE / FAILED: <summary>
```

- **Robot names.** The script writes its own `config.json` into `~/cyberscape_mappo_demo/`, so the project's `config.json` is untouched. In it, `robots_in_curr_mission = ["TURTLEBOT3_1", "TURTLEBOT3_2"]`. Each robot is a copy of the `TURTLEBOT3` entry with its own `high_plan_key`, `instruction_format` (`TurtleBot3_k.instruction()`) and plan files, and `plan_parser_llm = "cohere"`. Copy that block for the Mac server.
- **The mission text matters.** `mission_files/rl_mappo_disaster_mission.txt` is the brief we used. Its rules section tells the LLM:
  - use only the numeric coordinates given;
  - at most one `move_to_point` per phase;
  - phases with the same number run together, so number each robot's phases from 1;
  - at each inspection point, run scan → detect → report.
- **Plan quality.** The LLM plan was usable but not perfect:
  - TurtleBot3_1 checked for the west-sector survivor one phase before driving there. It still found it, because the survivor was 1.44 m away and the camera range is 1.5 m.
  - TurtleBot3_2 never drove back to base.

  The human approval step is where this gets caught.
- **Timing.** Planning took **16.5 min** for two robots. Generate the plan before a demo.
- `merge_plans()` (`cyberscape_mappo_mission.py`) accepts either one `MULTI_TURTLEBOT` plan or the per-robot plans. It readdresses every line to its robot, whatever prefix the LLM wrote (`Turtlebot31.`, `TURTLEBOT3_1.`, …).

### 6.2 What is different on hardware

1. **Two ROS domains (30 and 31) and two Pis.** Each Burger runs its own bridge, which is correct and should stay that way. But a MAPPO phase needs:
   - **Phase barrier:** neither robot starts phase n+1 until both finished phase n. In sim, the sender does this centrally. Recommended: run a hardware version of `cyberscape_mappo_plan_sender.py` on the Mac. It hands each Pi its own lines for phase n over HTTP and waits for both to report done. Each Pi bridge reports done or failed per phase. The current `/get_instruction` flow gives each robot its next command independently, so on its own it does not synchronize the two robots.
   - **Pose exchange:** each robot needs the other's pose about 10 times a second (§4.2). Cheapest option: each bridge sends its shared-frame `(x, y, yaw, active)` as a small UDP JSON packet to the other Pi (or through the Mac). The alternative is a ROS 2 `domain_bridge` for one pose topic.
2. **Shared frame.** In sim, both robots live in the Isaac map frame, and the plan coordinates are in that frame.
   - On hardware, each robot's odometry starts at (0, 0, 0) where it was switched on.
   - Measure both start poses on the floor in one mission frame, e.g. the tape-marked bases.
   - Convert with `to_world` / `to_mission`, as `hardware_rl_bridge.py` already does for one robot with its start pose.
   - Odometry drift then shows up as error in `teammate_x` and `teammate_y`. That's fine at 2 m scale, but keep missions short.
3. **The real scan already contains the other robot.** Merge in the virtual circle anyway (§4.3).
4. **Caster-first driving.** Both Burgers drive ball-first (see STATUS.md). The policy only cares that "forward" in the inputs is the direction FORWARD actually drives, and that LiDAR 0° is that direction, which you have already checked.
   - You note the LiDAR now sits about 3 cm **ahead** of the axle instead of 3.2 cm behind it.
   - In `lidar_sectors.py`, the `v3` profile's `sim_mount_m` is currently the variable `REAL_MOUNT_M`. **Hard-code it to `(-0.032, 0.0)`** (where the training LiDAR was) and set `REAL_MOUNT_M` to the measured caster-first offset, about `(+0.03, 0.0)`. The existing mount-shift code then gives the policy the distances it was trained on.
5. **The safety filter** blocks forward motion under 0.30 m within ±18°. In sim, robots pass each other and doorway edges closer than that. Expect stalls in doorways and in head-on passes. Log how often it triggers, rather than changing the policy.

---

## 7. Step-by-step plan for the hardware

Each step builds on the previous one. Stop at the first one that fails, and send us the logs.

1. **V4 on Burger 1 (single robot, no new code).**
   1. Copy `models/isaac_point_nav_lidar_ppo_v3_random/checkpoint_1570000_steps.npz` and `profile.txt` (it says `v3`) to the Pi, e.g. into `~/rl_point_nav/models/v4/`.
   2. Run `python3 hardware_rl_bridge.py --model models/v4/checkpoint_1570000_steps.npz --plan <plan>` next to the same run with V3.
   3. Compare: success, stuck, collision stop, time.
2. **Burger 2 runs V3/V4 alone**, to check its drive and LiDAR direction with the policy. Use the same commands with its own IP and domain 31.
3. **Build the V5 observation** (§4) as a new profile `v5` in `lidar_sectors.py` / `ros_point_nav_lidar_env.py`, so it gives 11 sector inputs.
   - First run `run_policy_on_hardware.py --dry-run` style **with the teammate inputs fixed**: `role = 1`, teammate far away (`x = y = 1.0`), `teammate_active = 0`.
   - Check that inputs 0–10 look like V4's and that the actions are sensible.
   - With the teammate far and inactive, V5 should behave like a single robot.
4. **Pose exchange + shared frame** (§6.2). With both robots standing still, print each robot's computed `teammate_x` and `teammate_y` and check them against a tape measure. Example: Burger 2 is 1.0 m directly ahead of Burger 1, so Burger 1 should read (0.5, 0.0).
5. **Two robots, one encounter at a time**, each with a fixed goal and no CyberScape:
   1. **Crossing:** routes at 90°, each 1.5 m, crossing in the middle.
   2. **Head-on:** 1.8 m apart, swapping places.
   3. **Doorway:** a wall with one 0.8 m gap, robots on opposite sides swapping.

   Record per run: both reached / collision stop / timeout, steps, and gap-controller hand-overs. Compare with §2.
6. **Add the align-first rule and the gap controller** (§5.2, §5.3), and repeat step 5.
7. **Phase barrier + CyberScape**:
   1. Start with a two-phase `MULTI_TURTLEBOT` plan written by hand. Copy the format of `Plans/rl_mappo_demo_plan.json`, using coordinates in your room's mission frame.
   2. Then do a live LLM plan with a mission text written like `mission_files/rl_mappo_disaster_mission.txt`, but for your room.

### Logs we need back

Per step: one CSV per robot with these columns:
- time;
- shared-frame pose;
- all 15 inputs;
- policy action and final action (after align / gap / safety filter);
- safety-filter state.

Also the mission log. With the 15 inputs logged, we can replay any real run in sim.

---

## 8. Running the sim side yourself (sim PC only, needs Isaac Sim)

```bash
cd ~/tb3-rl-nav/Hardware/Turtlebot3/rl_point_nav
./run_mappo_demo.sh                                  # GUI: crossing, head-on, random, doorway, corridor
./run_cyberscape_mappo_demo.sh disaster              # pre-written 7-phase disaster plan; approve in the dashboard
./run_cyberscape_mappo_full_cycle.sh                 # LLM plans both robots from the mission text, you approve
# headless benchmark (100 episodes)
./isaac_run.sh train_mappo.py --resume models/mappo_v5_kl05/best_model.pt --task bench --eval-episodes 100 --gap-assist
# export for the Pi
./isaac_run.sh export_mappo_numpy.py models/mappo_v5_kl05/best_model.pt
```

The dashboard is at `http://localhost:5001/dashboard/index.html`. The bridge log is at `results/mappo_bridge.log`; filter it with `grep CYBERSCAPE_MAPPO`.

## 9. Files

| File | What it does | Runs on |
|---|---|---|
| `isaac_multi_point_nav_env.py` | Two-robot Isaac env: encounters, teammate in the LiDAR, observations, critic state | sim |
| `train_mappo.py` | MAPPO training and evaluation, gap controller, demo, mission bridge | sim |
| `cyberscape_mappo_mission.py` | Plan parsing, `merge_plans()`, mission layouts (`SCENARIOS`), command-file protocol | sim (pure Python, reusable) |
| `Hardware/Isaac/cyberscape_mappo_plan_sender.py` | Sends approved plans phase by phase and waits for each phase. **Template for the hardware phase barrier.** | sim / Mac |
| `run_cyberscape_mappo_demo.sh`, `run_cyberscape_mappo_full_cycle.sh`, `run_mappo_demo.sh` | Launchers | sim |
| `export_mappo_numpy.py` | MAPPO actor → `.npz` | sim |
| `policy_numpy.py` | NumPy forward pass, unchanged; runs V5's `.npz` | Pi |
| `mappo_overlay.py`, `mappo_report.py`, `disaster_scene.py` | GUI overlay and camera, log report, disaster props | sim |
| `results/mappo_eval_summary.txt` | Every number in §2, and per-phase mission results | – |
| `Plans/rl_mappo_*_plan.json`, `mission_files/rl_mappo_disaster_mission.txt` | Plans and the mission brief | – |

## 10. Open questions for the hardware side

- How well does the LDS-02 see another Burger at the scan plane? (§4.3)
- Does the safety filter (0.30 m) block the doorway passes? (§6.2)
- How large does the teammate pose error get after a 3–4 phase mission, with odometry only?
- The measured caster-first LiDAR offset (§6.2).
