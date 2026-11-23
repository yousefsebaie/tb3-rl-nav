#!/usr/bin/env python3
"""MAPPO (and IPPO) for several TurtleBots in Isaac Sim, warm-started from the V4 PPO policy.

  ./isaac_run.sh train_mappo.py --timesteps 300000                 # MAPPO, centralized critic
  ./isaac_run.sh train_mappo.py --critic local --output-name ippo  # IPPO baseline, same code
  ./isaac_run.sh train_mappo.py --eval-episodes 50                 # no training: V4 copies (or --resume)
  ./isaac_run.sh train_mappo.py --task coop --resume models/mappo_v4/checkpoint_160000_steps.pt \
      --output-name mappo_v4_coop --timesteps 600000 --learning-rate 5e-5   # MAPPO v2

Actor: ONE policy shared by every robot (parameter sharing), the V4 network (6 -> 64 -> 64 -> 4,
Tanh) loaded from the SB3 checkpoint, same 6D observation, so it runs unchanged on a real robot.
Critic:
  central (MAPPO)  global_state(i) of IsaacMultiPointNavEnv (17D for two robots). The first layer's
                   weights for robot i's own 6D observation are V4's value network; the weights for
                   the teammate inputs start at zero, so at step 0 the critic equals V4's.
  local   (IPPO)   robot i's own 6D observation, V4's value network copied whole.
The only difference between the two runs is what the critic sees.

PPO settings are V4's: gamma 0.99, GAE lambda 0.95, clip 0.2, 5 epochs, minibatch 64,
vf_coef 0.5, max_grad_norm 0.5, Adam eps 1e-5, advantages normalized per minibatch,
learning rate 1e-4 and entropy 0.005 (the V4 fine-tune values). n_steps = 256 team steps
per update (up to 256 x robots samples). --timesteps counts team steps; checkpoints every
10,000 team steps (resume with --resume; --timesteps is then the TOTAL).

--task base is the first run's task (V4 reward per robot); --task coop is MAPPO v2
(cooperative_multi_config(): team-shared reward, harder teammate hits, wider doorways);
--task coop3 is MAPPO v3 (cooperative_v3_config(): v2 plus crash penalties 30/40 and a capped
proximity penalty, so crashing out of a deadlock no longer pays).
With --eval-every N, every N team steps training pauses for --eval-size deterministic
episodes on a fixed scenario set (seed 2000); the best score (both reached - collision)
is saved as best_model.pt, so a run that later degrades keeps its best policy.

Each finished team episode prints a [TB03_MAPPO] line (per-robot outcomes) and a
[TB03_PROGRESS] line in the single-robot format (team success = every robot reached its
goal; collision = any robot collided), so training_progress.py works unchanged.
"""

from __future__ import annotations

import argparse
from collections import deque
import io
import json
import math
import os
from pathlib import Path
import signal
import time
import zipfile

import numpy as np

HERE = Path(__file__).resolve().parent
MODELS_DIR = HERE / "models"
DEFAULT_INIT = MODELS_DIR / "isaac_point_nav_lidar_ppo_v3_random" / "checkpoint_1570000_steps.zip"
SEED = 42
CHECKPOINT_FREQ = 10_000
N_STEPS = 256
BATCH_SIZE = 64
N_EPOCHS = 5
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_RANGE = 0.2
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5


TEAMMATE_INPUTS = 4   # --actor-teammate: role flag, teammate local x / y, teammate still active
# --demo: one episode per entry, in order, as (task, encounter); the corridor is a head_on on the solo task.
DEMO_PLAYLIST = (("bench", "crossing"), ("bench", "head_on"), ("bench", "random"),
                 ("solo", "doorway"), ("solo", "head_on"))


def build_networks(state_dim: int, critic: str, init: Path, extra: int = 0):
    """Actor and critic as torch modules, initialized from the V4 SB3 checkpoint.

    extra > 0 widens the actor's input; the new weights start at zero, so the actor starts as V4.
    """
    import torch
    from torch import nn

    with zipfile.ZipFile(init) as archive:
        sd = torch.load(io.BytesIO(archive.read("policy.pth")), map_location="cpu")
    obs_dim = sd["mlp_extractor.policy_net.0.weight"].shape[1]

    actor = nn.Sequential(nn.Linear(obs_dim + extra, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 4))
    critic_in = state_dim if critic == "central" else obs_dim
    value = nn.Sequential(nn.Linear(critic_in, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 1))
    with torch.no_grad():
        actor[0].weight.zero_()
        actor[0].weight[:, :obs_dim].copy_(sd["mlp_extractor.policy_net.0.weight"])
        actor[0].bias.copy_(sd["mlp_extractor.policy_net.0.bias"])
        for dst, src in ((2, "mlp_extractor.policy_net.2"), (4, "action_net")):
            actor[dst].weight.copy_(sd[f"{src}.weight"])
            actor[dst].bias.copy_(sd[f"{src}.bias"])
        value[0].weight.zero_()
        value[0].weight[:, :obs_dim].copy_(sd["mlp_extractor.value_net.0.weight"])
        value[0].bias.copy_(sd["mlp_extractor.value_net.0.bias"])
        for dst, src in ((2, "mlp_extractor.value_net.2"), (4, "value_net")):
            value[dst].weight.copy_(sd[f"{src}.weight"])
            value[dst].bias.copy_(sd[f"{src}.bias"])
    return actor, value, obs_dim


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=300_000)
    parser.add_argument("--critic", choices=("central", "local"), default="central")
    parser.add_argument("--output-name", default=None, help="default: mappo_v4 / ippo_v4")
    parser.add_argument("--init", type=Path, default=DEFAULT_INIT, help="V4 SB3 checkpoint (.zip)")
    parser.add_argument("--resume", type=Path, default=None, help="MAPPO checkpoint (.pt)")
    parser.add_argument("--robots", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--ent-coef", type=float, default=0.005)
    parser.add_argument("--eval-episodes", type=int, default=0,
                        help="no training: run N episodes with deterministic actions and report")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--task", choices=("base", "coop", "coop3", "solo", "passage", "wide", "bench"), default="base")
    parser.add_argument("--eval-every", type=int, default=0, help="team steps between in-training evals (0 = off)")
    parser.add_argument("--eval-size", type=int, default=40)
    parser.add_argument("--actor-teammate", action="store_true",
                        help=f"actor also sees {TEAMMATE_INPUTS} teammate inputs (role, relative position, active)")
    parser.add_argument("--v5-sectors", action="store_true",
                        help="actor and critic also see the 5 finer V5 LiDAR sectors (start at zero weight)")
    parser.add_argument("--kl-coef", type=float, default=0.0,
                        help="weight of KL(V4 || actor) in the loss, keeps the obstacle skill (0 = off)")
    parser.add_argument("--right-of-way", action="store_true",
                        help="eval only: near the teammate, the robot farther from its goal waits (STAY)")
    parser.add_argument("--row-radius", type=float, default=1.0, help="right-of-way trigger distance (m)")
    parser.add_argument("--passage-assist", action="store_true",
                        help="eval only: walls close on both sides, front open, goal ahead -> FORWARD")
    parser.add_argument("--proximity-threshold", type=float, default=None,
                        help="override V4's 0.30 m obstacle proximity-penalty start (m)")
    parser.add_argument("--gap-assist", action="store_true",
                        help="eval only: a stuck robot hands over to a follow-the-gap controller on the full scan")
    parser.add_argument("--invalid-clear", action="store_true",
                        help="a LiDAR sector with no returns reads max range instead of 0 m (V4 rule)")
    parser.add_argument("--debug-obs", action="store_true", help="eval: print robot 0's observation every 10 steps")
    parser.add_argument("--demo", action="store_true",
                        help="eval: cycle DEMO_PLAYLIST (two-robot encounters, then a solo doorway and corridor)")
    parser.add_argument("--demo-seeds", default=None,
                        help="with --demo: comma-separated reset seed per playlist entry (default: --seed + round)")
    parser.add_argument("--mission", type=Path, default=None,
                        help="run a CyberScape MULTI_TURTLEBOT plan (cyberscape_mappo_mission.py) and exit")
    parser.add_argument("--mission-listen", action="store_true",
                        help="CyberScape bridge: execute phases written to the command file by the plan sender")
    parser.add_argument("--scenario", choices=("warehouse", "disaster", "disaster_simple"), default="warehouse",
                        help="mission layout (cyberscape_mappo_mission.SCENARIOS); disaster adds the set dressing")
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--show-robots", action="store_true",
                        help="with --gui: show the robot meshes; the LiDAR then sees them, so PRESENTATION ONLY")
    args = parser.parse_args()
    name = args.output_name or ("mappo_v4" if args.critic == "central" else "ippo_v4")
    output_dir = MODELS_DIR / name

    import torch
    from torch.distributions import Categorical

    try:
        from .isaac_multi_point_nav_env import IsaacMultiPointNavEnv, MultiNavConfig, cooperative_multi_config, cooperative_v3_config, solo_passage_config, passage_training_config, v4_lidar_config, \
            wide_passage_config, V5_SECTORS, benchmark_config
    except ImportError:
        from isaac_multi_point_nav_env import IsaacMultiPointNavEnv, MultiNavConfig, cooperative_multi_config, cooperative_v3_config, solo_passage_config, passage_training_config, v4_lidar_config, \
            wide_passage_config, V5_SECTORS, benchmark_config

    torch.manual_seed(args.seed)
    tasks = {"base": MultiNavConfig, "coop": cooperative_multi_config, "coop3": cooperative_v3_config,
             "solo": solo_passage_config, "passage": passage_training_config, "wide": wide_passage_config,
             "bench": benchmark_config}
    ckpt = torch.load(args.resume, map_location="cpu") if args.resume is not None else None
    if ckpt is not None and ckpt.get("extra_sectors"):
        args.v5_sectors = True          # a V5 checkpoint always runs with its sectors
    multi_config = tasks[args.task](robot_count=args.robots,
                                    extra_sectors=V5_SECTORS if args.v5_sectors else (),
                                    invalid_as_clear=args.invalid_clear)
    lidar_overrides = {}
    if args.show_robots:
        lidar_overrides["hide_robot_visuals"] = False
    if args.proximity_threshold is not None:
        lidar_overrides["proximity_threshold_m"] = args.proximity_threshold
    if args.mission or args.mission_listen:   # walls and corridors need more pooled boxes than V4's 6
        from cyberscape_mappo_mission import SCENARIOS
        lidar_overrides["obstacle_count"] = max(6, len(SCENARIOS[args.scenario]["obstacles"]))
    lidar_config = v4_lidar_config(**lidar_overrides) if lidar_overrides else None
    env = IsaacMultiPointNavEnv(multi_config=multi_config, lidar_config=lidar_config,
                                headless=not args.gui, seed=args.seed)
    n_robots = len(env.robots)
    obs_len = env.obs_dim                                 # 6, or 6 + V5 sectors
    if ckpt is not None and ckpt.get("teammate_inputs", ckpt.get("actor_extra", 0)):
        args.actor_teammate = True      # a teammate-input checkpoint always runs with its inputs
    teammate = TEAMMATE_INPUTS if args.actor_teammate else 0
    extra = (obs_len - 6) + teammate
    if extra and n_robots != 2:
        raise ValueError("--actor-teammate supports two robots")
    actor, value, obs_dim = build_networks(env.state_dim, args.critic, args.init, extra)
    reference = build_networks(env.state_dim, args.critic, args.init)[0].requires_grad_(False) if args.kl_coef else None
    params = list(actor.parameters()) + list(value.parameters())
    optimizer = torch.optim.Adam(params, lr=args.learning_rate, eps=1e-5)
    num_timesteps, episodes = 0, 0
    best_score = -float("inf")
    if ckpt is not None:
        if ckpt["critic"] != args.critic or ckpt["robots"] != n_robots:
            raise ValueError(f"{args.resume} is a {ckpt['critic']} critic for {ckpt['robots']} robots")
        weights, critic_weights = dict(ckpt["actor"]), dict(ckpt["value"])
        old_len = ckpt.get("obs_len", 6)
        old_teammate = weights["0.weight"].shape[1] - old_len
        widened = weights["0.weight"].shape[1] != obs_dim + extra
        if widened:     # new inputs start at zero; old columns keep their meaning
            if old_len > obs_len or old_teammate > teammate:
                raise ValueError(f"{args.resume} has inputs this run does not provide")
            old = weights["0.weight"]
            first = torch.zeros(64, obs_dim + extra)
            first[:, :old_len] = old[:, :old_len]
            first[:, obs_len:obs_len + old_teammate] = old[:, old_len:]
            weights["0.weight"] = first
            if args.critic == "central":   # [own obs | teammate obs | 5 relative] per teammate
                old = critic_weights["0.weight"]
                first = torch.zeros(64, env.state_dim)
                first[:, :old_len] = old[:, :old_len]
                first[:, obs_len:obs_len + old_len] = old[:, old_len:2 * old_len]
                first[:, 2 * obs_len:] = old[:, 2 * old_len:]
                critic_weights["0.weight"] = first
            elif old_len != obs_len:
                old = critic_weights["0.weight"]
                first = torch.zeros(64, obs_len)
                first[:, :old_len] = old
                critic_weights["0.weight"] = first
        actor.load_state_dict(weights)
        value.load_state_dict(critic_weights)
        if not widened:
            optimizer.load_state_dict(ckpt["optimizer"])
        for group in optimizer.param_groups:
            group["lr"] = args.learning_rate
        num_timesteps, episodes = ckpt["num_timesteps"], ckpt["episodes"]
        if ckpt.get("task", "base") == args.task and not widened:
            best_score = ckpt.get("best_score", best_score)
        print(f"[TB03_MAPPO] resumed {args.resume} at {num_timesteps} steps", flush=True)
    print(f"[TB03_MAPPO] critic={args.critic} robots={n_robots} obs_dim={obs_dim} state_dim={env.state_dim} "
          f"init={args.init.name} task={args.task} lr={args.learning_rate} ent_coef={args.ent_coef} "
          f"actor_extra={extra} kl_coef={args.kl_coef} right_of_way={args.right_of_way}", flush=True)

    def actor_input(observations) -> "torch.Tensor":
        """Each robot's 6D observation, plus (with --actor-teammate) its role and the teammate's
        position in its own frame, taken from the critic state."""
        rows = []
        for i, obs in enumerate(observations):
            if teammate:   # critic state = [own obs | teammate obs | x, y, sin, cos, active]
                g = env.global_state(i, observations)
                obs = np.concatenate([obs, [1.0 if i == 0 else 0.0, g[2 * obs_len], g[2 * obs_len + 1],
                                            g[2 * obs_len + 4]]])
            rows.append(obs)
        return torch.as_tensor(np.stack(rows), dtype=torch.float32)

    row_wait = [0] * n_robots

    def passage_assist(actions: list, observations) -> list:
        """Rule override for gaps the 3-sector policy will not enter: walls within 0.5 m on BOTH
        sides, front clear beyond 0.25 m and the goal within ~45 deg ahead -> drive forward."""
        if not args.passage_assist:
            return actions
        actions = list(actions)
        for i, (dist, sin_b, cos_b, front, left, right) in enumerate(o[:6] for o in observations):
            if left < 0.5 and right < 0.5 and front > 0.25 and cos_b > 0.7:
                actions[i] = 0   # PointNavAction.FORWARD
        return actions

    # Follow-the-gap hand-over (Tidd et al. 2021 style hybrid): the policy drives until a robot is
    # stuck (< 6 cm in 18 steps), then a scripted controller steers through the free gap nearest
    # the goal on the full 1066-ray scan and hands back after 1.2 m (or 150 steps); it first turns to face the goal if it is > 60 deg off.
    gap_hist = [deque(maxlen=18) for _ in range(n_robots)]
    gap_mode = [None] * n_robots     # (entry_x, entry_y, steps) while active
    gap_cool = [0] * n_robots
    gap_stats = {"handovers": 0}

    def free_gap_heading(i: int, bearing_deg: float) -> "float | None":
        robot = env.robots[i]
        ranges = np.fmin(env._current_ranges(robot), env._teammate_ranges(robot))
        az = env._sensor_azimuth_deg
        r = np.where(np.isfinite(ranges), ranges, 5.0)   # no return = far (0.0 here tested worse: 14/30 vs 21/30)
        order = np.argsort(az)
        az, r = az[order], r[order]
        k = 5
        r = np.array([np.median(r[max(0, j - k // 2):j + k // 2 + 1]) for j in range(r.size)])   # drops ray dropouts
        cone = np.abs(az) <= 75.0
        if not cone.any():
            return None
        free = cone & (r > np.min(r[cone]) + 0.35)
        best, best_cost, j = None, math.inf, 0
        while j < az.size:
            if free[j]:
                start = j
                while j < az.size and free[j]:
                    j += 1
                if j - start >= 6:   # ignore slivers
                    centre = 0.5 * (az[start] + az[j - 1])
                    cost = abs(centre - bearing_deg)
                    if cost < best_cost:
                        best, best_cost = centre, cost
            j += 1
        return best

    def gap_assist(actions: list, info: dict) -> list:
        if not args.gap_assist:
            return actions
        actions = list(actions)
        for i, r in enumerate(info["robots"]):
            x, y = r["pose"]["x"], r["pose"]["y"]
            if not r["active"]:
                gap_hist[i].clear(); gap_mode[i] = None
                continue
            gap_hist[i].append((x, y))
            gap_cool[i] = max(0, gap_cool[i] - 1)
            if gap_mode[i] is None:
                h = gap_hist[i]
                stuck = (len(h) == h.maxlen and math.hypot(h[-1][0] - h[0][0], h[-1][1] - h[0][1]) < 0.06
                         and r["distance_to_goal_m"] > 0.3 and gap_cool[i] == 0)
                if not stuck:
                    continue
                gap_mode[i] = (x, y, 0)
                gap_stats["handovers"] += 1
                print(f"[TB03_GAP] handover robot={i} {info['encounter']} step={info['episode_steps']} "
                      f"goal_dist={r['distance_to_goal_m']:.2f} bearing={math.degrees(r['relative_goal_bearing_rad']):.0f}",
                      flush=True)
            ex, ey, n = gap_mode[i]
            if n >= 150 or math.hypot(x - ex, y - ey) >= 1.2:
                print(f"[TB03_GAP] handback robot={i} steps={n} moved={math.hypot(x - ex, y - ey):.2f} "
                      f"goal_dist={r['distance_to_goal_m']:.2f}", flush=True)
                gap_mode[i] = None; gap_cool[i] = 25; gap_hist[i].clear()
                continue
            gap_mode[i] = (ex, ey, n + 1)
            bearing = math.degrees(r["relative_goal_bearing_rad"])
            if abs(bearing) > 60.0:   # spun away from the goal: face it first, then look for the gap
                actions[i] = 1 if bearing > 0 else 2
                continue
            target = free_gap_heading(i, bearing)
            if n % 10 == 0:
                print(f"[TB03_GAP] robot={i} n={n} target={target} goal_dist={r['distance_to_goal_m']:.2f} "
                      f"front={r['clearances_m'].get('front', float('nan')):.2f}", flush=True)
            if target is None:
                continue
            if abs(target) > 8.0:
                actions[i] = 1 if target > 0 else 2   # TURN_LEFT / TURN_RIGHT
            else:
                actions[i] = 0                         # FORWARD
        return actions

    def right_of_way(actions: list, info: dict) -> list:
        """Near the teammate, the robot farther from its goal STAYs; after 40 waiting steps they swap."""
        robots = info["robots"]
        if not args.right_of_way or not all(r["active"] for r in robots):
            row_wait[:] = [0] * n_robots
            return actions
        a, b = robots
        if math.hypot(a["pose"]["x"] - b["pose"]["x"], a["pose"]["y"] - b["pose"]["y"]) > args.row_radius:
            row_wait[:] = [0] * n_robots
            return actions
        yielder = 0 if a["distance_to_goal_m"] > b["distance_to_goal_m"] else 1
        if row_wait[yielder] >= 40:
            yielder = 1 - yielder
            if row_wait[yielder] >= 40:
                row_wait[:] = [0] * n_robots
        row_wait[yielder] += 1
        actions = list(actions)
        actions[yielder] = 3   # PointNavAction.STAY
        return actions

    def critic_input(observations, index):
        state = env.global_state(index, observations) if args.critic == "central" else observations[index]
        return torch.as_tensor(state, dtype=torch.float32)

    def save(path: Path) -> None:
        torch.save({"actor": actor.state_dict(), "value": value.state_dict(), "optimizer": optimizer.state_dict(),
                    "num_timesteps": num_timesteps, "episodes": episodes, "critic": args.critic,
                    "robots": n_robots, "obs_dim": obs_dim, "state_dim": env.state_dim,
                    "init": str(args.init), "task": args.task, "best_score": best_score,
                    "actor_extra": extra, "kl_coef": args.kl_coef, "obs_len": obs_len,
                    "extra_sectors": bool(args.v5_sectors), "teammate_inputs": teammate}, path)

    def quick_eval() -> tuple[float, float, float]:
        """--eval-size deterministic episodes on a fixed scenario set: both_reached, collision, robot-robot."""
        reached = collided = robot_hits = 0
        observations, _ = env.reset(seed=2000)
        with torch.no_grad():
            for k in range(args.eval_size):
                info = {"episode_done": False}
                while not info["episode_done"]:
                    logits = actor(actor_input(observations))
                    observations, _, _, _, info = env.step(logits.argmax(-1).tolist())
                outcomes = [r["outcome"] for r in info["robots"]]
                reached += all(o == "SUCCESS" for o in outcomes)
                collided += "COLLISION" in outcomes
                robot_hits += any(r["collision_kind"] == "robot" for r in info["robots"])
                if k + 1 < args.eval_size:
                    observations, _ = env.reset()
        n = args.eval_size
        return reached / n, collided / n, robot_hits / n

    recent = deque(maxlen=100)          # team outcome
    recent_robot = deque(maxlen=200)    # per-robot outcome
    recent_robot_hits = deque(maxlen=100)
    by_encounter: dict[str, deque] = {}    # team outcome per encounter type
    start_wall, start_steps = time.monotonic(), num_timesteps

    def report_episode(info) -> None:
        nonlocal episodes
        episodes += 1
        outcomes = [r["outcome"] for r in info["robots"]]
        kinds = [r["collision_kind"] for r in info["robots"]]
        team = ("SUCCESS" if all(o == "SUCCESS" for o in outcomes)
                else "COLLISION" if "COLLISION" in outcomes else "TIMEOUT")
        recent.append(team)
        recent_robot.extend(outcomes)
        recent_robot_hits.append("robot" in kinds)
        by_encounter.setdefault(info["encounter"], deque(maxlen=100)).append(team)
        n, m = len(recent), len(recent_robot)
        print(f"[TB03_MAPPO] episode={episodes} encounter={info['encounter']} outcomes={','.join(outcomes)} "
              f"collisions={','.join(k or '-' for k in kinds)} length={info['episode_steps']} "
              f"robot_success={recent_robot.count('SUCCESS') / m:.3f} "
              f"robot_robot_collision={sum(recent_robot_hits) / len(recent_robot_hits):.3f}", flush=True)
        elapsed = time.monotonic() - start_wall
        rate = (num_timesteps - start_steps) / elapsed if elapsed > 0 else 0.0
        total = args.timesteps if not args.eval_episodes else num_timesteps
        print(f"[TB03_PROGRESS] steps={num_timesteps} total={total} elapsed_s={elapsed:.0f} "
              f"steps_per_s={rate:.2f} eta_s={max(total - num_timesteps, 0) / rate if rate > 0 else -1:.0f} "
              f"episode={episodes} last{n}_success={recent.count('SUCCESS') / n:.3f} "
              f"last{n}_collision={recent.count('COLLISION') / n:.3f} "
              f"last{n}_timeout={recent.count('TIMEOUT') / n:.3f} outcome={team} length={info['episode_steps']}",
              flush=True)

    demo_configs = {task: tasks[task](robot_count=args.robots, extra_sectors=multi_config.extra_sectors,
                                      invalid_as_clear=args.invalid_clear) for task, _ in DEMO_PLAYLIST}
    demo_seeds = [int(v) for v in args.demo_seeds.split(",")] if args.demo_seeds else None

    def reset(episode: int):
        if not args.demo:
            return env.reset(seed=args.seed) if episode == 0 else env.reset()
        task, encounter = DEMO_PLAYLIST[episode % len(DEMO_PLAYLIST)]
        env.multi_config = demo_configs[task]
        seed = (demo_seeds[episode % len(DEMO_PLAYLIST)] if demo_seeds
                else args.seed + episode // len(DEMO_PLAYLIST))
        print(f"[TB03_DEMO] episode={episode + 1} task={task} encounter={encounter} seed={seed}", flush=True)
        return env.reset(seed=seed, options={"encounter": encounter})

    def run_phase(number: int, lines: list, overlay) -> tuple[bool, str]:
        """One CyberScape phase = one MAPPO episode from the robots' current poses; robots
        without a move_to_point hold still (goal = own pose, STAY)."""
        from cyberscape_mappo_mission import ROBOT_NAMES, SCENARIOS, parse_line
        scenario = SCENARIOS[args.scenario]
        info = env._info()
        goals = [(r["pose"]["x"], r["pose"]["y"]) for r in info["robots"]]
        movers, after = [], []
        for line in lines:
            index, action, nums = parse_line(line)
            if action == "move_to_point" and index is not None and len(nums) >= 2:
                goals[index] = (nums[0], nums[1])
                movers.append(index)
            elif action == "return_to_base" and index is not None:
                goals[index] = tuple(scenario["bases"][index][:2])
                movers.append(index)
            else:
                after.append((index, action, line))
        ok, summary = True, "no moves"
        if movers:
            ok, summary = drive_phase(number, movers, goals, scenario["obstacles"], overlay)
        for index, action, line in after:   # scans, camera checks and reports run after the moves
            result = "done"
            if action == "detect_with_camera" and index is not None:
                result = detect_survivor(index)
            print(f"[CYBERSCAPE_MAPPO] phase {number}: {line} -> {result}", flush=True)
        return ok, summary

    found_survivors: set = set()

    def detect_survivor(index: int) -> str:
        """Camera stand-in: the nearest survivor within CAMERA_RANGE_M (simulation ground truth)."""
        from cyberscape_mappo_mission import CAMERA_RANGE_M, SCENARIOS
        survivors = SCENARIOS[args.scenario]["survivors"]
        pose = env._info()["robots"][index]["pose"]
        near = sorted((math.hypot(x - pose["x"], y - pose["y"]), k, x, y) for k, (x, y, _) in enumerate(survivors))
        near = [n for n in near if n[0] <= CAMERA_RANGE_M]
        if not near:
            return "no survivor in camera range"
        distance, k, x, y = near[0]
        if k not in found_survivors:
            found_survivors.add(k)
            if k < len(beacons):
                from disaster_scene import show_beacon
                show_beacon(env._stage, beacons[k])
                for _ in range(10):
                    env._simulation_app.update()
        return (f"SURVIVOR DETECTED at ({x:.2f}, {y:.2f}), {distance:.2f} m away "
                f"({len(found_survivors)}/{len(survivors)} found)")

    beacons: list = []

    def drive_phase(number: int, movers: list, goals: list, obstacles, overlay) -> tuple[bool, str]:
        from cyberscape_mappo_mission import ROBOT_NAMES
        info = env._info()
        for i in movers:
            print(f"[CYBERSCAPE_MAPPO] phase {number}: {ROBOT_NAMES[i]} -> ({goals[i][0]:.2f}, {goals[i][1]:.2f})",
                  flush=True)
        starts = [(r["pose"]["x"], r["pose"]["y"], r["yaw"]) for r in info["robots"]]
        observations, info = env.reset(options={"encounter": f"phase_{number}", "starts": starts,
                                                "goals": goals, "obstacles": list(obstacles)})
        # Goal Interface: each mover first turns to face its goal (an in-place turn would otherwise
        # look "stuck" to the gap hand-over), then MAPPO drives with a fresh stuck history.
        aligning = set(movers)
        with torch.no_grad():
            while not info["episode_done"]:
                actions = actor(actor_input(observations)).argmax(-1).tolist()
                actions = gap_assist(actions, info)
                for i in list(aligning):
                    bearing = math.degrees(info["robots"][i]["relative_goal_bearing_rad"])
                    gap_hist[i].clear(); gap_mode[i] = None
                    if abs(bearing) <= 15.0:
                        aligning.discard(i)
                    else:
                        actions[i] = 1 if bearing > 0 else 2   # TURN_LEFT / TURN_RIGHT
                actions = [a if i in movers else 3 for i, a in enumerate(actions)]   # 3 = STAY
                observations, _, _, _, info = env.step(right_of_way(actions, info))
                if overlay:
                    overlay.update(info)
        outcomes = {ROBOT_NAMES[i]: info["robots"][i]["outcome"] for i in movers}
        summary = f"steps={info['episode_steps']} sim_s={info['episode_elapsed_sim_s']:.0f} " + \
            " ".join(f"{n}={o}" for n, o in outcomes.items())
        print(f"[CYBERSCAPE_MAPPO] phase {number} {summary}", flush=True)
        if overlay:
            for _ in range(60):
                env._simulation_app.update()
        return all(o == "SUCCESS" for o in outcomes.values()), summary

    def run_mission() -> None:
        import dataclasses
        from cyberscape_mappo_mission import COMMAND_FILE, MISSION_STEP_LIMIT, SCENARIOS, STATUS_FILE, load_plan
        scenario = SCENARIOS[args.scenario]
        env.point_nav_config = dataclasses.replace(env.point_nav_config, max_episode_steps=MISSION_STEP_LIMIT,
                                                   max_episode_sim_time_s=2 * MISSION_STEP_LIMIT)
        if args.scenario.startswith("disaster"):   # visual props, LiDAR-safe (disaster_scene.py rules)
            from disaster_scene import build_mission
            beacons.extend(build_mission(env, scenario["obstacles"], scenario["survivors"], scenario["routes"],
                                         scenario["base_xy"], scenario["wing"]))
        bases = scenario["bases"]
        env.reset(options={"encounter": "mission", "starts": list(bases), "goals": [b[:2] for b in bases],
                           "obstacles": list(scenario["obstacles"])})
        print(f"[CYBERSCAPE_MAPPO] scenario {args.scenario}: {len(scenario['obstacles'])} obstacles, "
              f"{len(scenario['survivors'])} survivors", flush=True)
        overlay = None
        if args.gui:
            from mappo_overlay import MultiRobotOverlay
            overlay = MultiRobotOverlay(env)
            if os.environ.get("VIEW", "top") == "angled":
                overlay.frame_camera()
            else:   # whole mission: bases, stations, doorway and storage room
                overlay.frame_overhead(*scenario["view"])
            overlay.update(env._info())
        if args.mission:
            results = []
            for number, lines in load_plan(args.mission):
                ok, summary = run_phase(number, lines, overlay)
                results.append(ok)
                if not ok:
                    break
            print(f"[CYBERSCAPE_MAPPO] MISSION {'COMPLETE' if all(results) else 'FAILED'} "
                  f"phases_ok={sum(results)} survivors_found={len(found_survivors)}", flush=True)
            return
        last = COMMAND_FILE.stat().st_mtime if COMMAND_FILE.exists() else 0.0
        print(f"[CYBERSCAPE_MAPPO] waiting for CyberScape commands in {COMMAND_FILE} "
              f"(approve a plan in the HITL dashboard)", flush=True)
        while env._simulation_app.is_running():
            env._simulation_app.update()
            if not COMMAND_FILE.exists() or COMMAND_FILE.stat().st_mtime == last:
                continue
            last = COMMAND_FILE.stat().st_mtime
            try:
                command = json.loads(COMMAND_FILE.read_text())
            except json.JSONDecodeError:
                command = {"phase": 0, "lines": [COMMAND_FILE.read_text().strip()]}
            STATUS_FILE.write_text("BUSY")
            ok, summary = run_phase(int(command["phase"]), command["lines"], overlay)
            STATUS_FILE.write_text("DONE" if ok else f"FAILED: {summary}")

    try:
        if args.mission or args.mission_listen:
            run_mission()
            return
        observations, info = reset(0) if args.eval_episodes else env.reset(seed=args.seed)

        if args.eval_episodes:
            overlay = None
            if args.gui:   # robots are hidden from the LiDAR; draw them as a viewport overlay
                from mappo_overlay import MultiRobotOverlay
                overlay = MultiRobotOverlay(env)
                overlay.frame_camera(info)
                print(f"[TB03_MAPPO] GUI episode 1 encounter={info['encounter']}", flush=True)
            done_episodes = 0
            with torch.no_grad():
                while done_episodes < args.eval_episodes:
                    logits = actor(actor_input(observations))
                    actions = passage_assist(logits.argmax(-1).tolist(), observations)
                    if args.debug_obs and info["episode_steps"] % 10 == 0:
                        print(f"[TB03_DEBUG] {info['encounter']} step={info['episode_steps']} obs0="
                              + ",".join(f"{v:.2f}" for v in observations[0]) + f" action0={actions[0]}", flush=True)
                    actions = gap_assist(actions, info)
                    observations, _, _, _, info = env.step(right_of_way(actions, info))
                    num_timesteps += 1
                    if overlay:
                        overlay.update(info)
                    if info["episode_done"]:
                        report_episode(info)
                        done_episodes += 1
                        if overlay:   # hold the final frame so the outcome is visible
                            for _ in range(120):
                                env._simulation_app.update()
                            overlay.new_episode()
                        if done_episodes < args.eval_episodes:
                            observations, info = reset(done_episodes)
                        if overlay and done_episodes < args.eval_episodes:
                            overlay.frame_camera(info)
                            print(f"[TB03_MAPPO] GUI episode {done_episodes + 1} encounter={info['encounter']}",
                                  flush=True)
            print(f"[TB03_MAPPO] EVAL {done_episodes} episodes: team_success={recent.count('SUCCESS') / len(recent):.3f} "
                  f"episode_collision={recent.count('COLLISION') / len(recent):.3f} "
                  f"robot_success={recent_robot.count('SUCCESS') / len(recent_robot):.3f} "
                  f"robot_robot_collision={sum(recent_robot_hits) / len(recent_robot_hits):.3f} "
                  f"invalid_sector_reads={env.invalid_sector_reads} gap_handovers={gap_stats['handovers']}", flush=True)
            for kind, outcomes in sorted(by_encounter.items()):
                print(f"[TB03_MAPPO] EVAL encounter={kind} episodes={len(outcomes)} "
                      f"team_success={outcomes.count('SUCCESS') / len(outcomes):.3f} "
                      f"collision={outcomes.count('COLLISION') / len(outcomes):.3f}", flush=True)
            return

        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "config.json").write_text(json.dumps(
            {"critic": args.critic, "robots": n_robots, "init": str(args.init), "learning_rate": args.learning_rate,
             "ent_coef": args.ent_coef, "seed": args.seed, "task": args.task,
             "resume": str(args.resume) if args.resume else None}, indent=2) + "\n")
        if args.timesteps <= num_timesteps:
            raise ValueError(f"Already at {num_timesteps} steps >= --timesteps {args.timesteps}")
        next_checkpoint = (num_timesteps // CHECKPOINT_FREQ + 1) * CHECKPOINT_FREQ
        next_eval = (num_timesteps // args.eval_every + 1) * args.eval_every if args.eval_every else None

        # SIGINT (e.g. the run-length timer) stops after the current team step with a checkpoint.
        stop = []
        signal.signal(signal.SIGINT, lambda *_: stop.append(True))

        while num_timesteps < args.timesteps and not stop:
            # ---------------- rollout: N_STEPS team steps
            T = N_STEPS
            buf_obs = torch.zeros(T, n_robots, obs_dim + extra)
            buf_state = torch.zeros(T, n_robots, env.state_dim if args.critic == "central" else obs_dim)
            buf_act = torch.zeros(T, n_robots, dtype=torch.long)
            buf_logp = torch.zeros(T, n_robots)
            buf_val = torch.zeros(T, n_robots)
            buf_rew = torch.zeros(T, n_robots)
            valid = torch.zeros(T, n_robots, dtype=torch.bool)
            ended = torch.zeros(T, n_robots, dtype=torch.bool)
            boot = torch.zeros(T, n_robots)    # value after a truncation (0 after a termination)
            for t in range(T):
                active = [r.active for r in env.robots]
                with torch.no_grad():
                    obs_t = actor_input(observations)
                    states = torch.stack([critic_input(observations, i) for i in range(n_robots)])
                    dist = Categorical(logits=actor(obs_t))
                    actions = dist.sample()
                    values = value(states).squeeze(-1)
                buf_obs[t], buf_state[t], buf_act[t] = obs_t, states, actions
                buf_logp[t], buf_val[t] = dist.log_prob(actions), values
                valid[t] = torch.as_tensor(active)
                next_obs, rewards, terminated, truncated, info = env.step(actions.tolist())
                num_timesteps += 1
                buf_rew[t] = torch.as_tensor(rewards)
                for i in range(n_robots):
                    if active[i] and (terminated[i] or truncated[i]):
                        ended[t, i] = True
                        if truncated[i]:
                            with torch.no_grad():
                                boot[t, i] = value(critic_input(next_obs, i)).squeeze()
                observations = next_obs
                if info["episode_done"]:
                    report_episode(info)
                    observations, info = env.reset()
                if num_timesteps >= next_checkpoint:
                    save(output_dir / f"checkpoint_{next_checkpoint}_steps.pt")
                    next_checkpoint += CHECKPOINT_FREQ
                if stop:
                    break
            if stop:
                save(output_dir / f"checkpoint_{num_timesteps}_steps.pt")
                print(f"[TB03_MAPPO] stopped by signal at {num_timesteps} steps; checkpoint saved", flush=True)
                break

            # ---------------- GAE per robot
            with torch.no_grad():
                last = torch.stack([value(critic_input(observations, i)).squeeze() for i in range(n_robots)])
            advantages = torch.zeros(T, n_robots)
            for i in range(n_robots):
                next_value = last[i] if env.robots[i].active else torch.tensor(0.0)
                next_adv = torch.tensor(0.0)
                for t in reversed(range(T)):
                    if not valid[t, i]:
                        continue
                    if ended[t, i]:
                        next_value, next_adv = boot[t, i], torch.tensor(0.0)
                    delta = buf_rew[t, i] + GAMMA * next_value - buf_val[t, i]
                    advantages[t, i] = delta + GAMMA * GAE_LAMBDA * next_adv
                    next_value, next_adv = buf_val[t, i], advantages[t, i]
            returns = advantages + buf_val

            mask = valid.reshape(-1)
            b_obs = buf_obs.reshape(-1, obs_dim + extra)[mask]
            b_state = buf_state.reshape(-1, buf_state.shape[-1])[mask]
            b_act, b_logp = buf_act.reshape(-1)[mask], buf_logp.reshape(-1)[mask]
            b_adv, b_ret = advantages.reshape(-1)[mask], returns.reshape(-1)[mask]

            # ---------------- PPO update (SB3 semantics)
            stats = []
            for _ in range(N_EPOCHS):
                for idx in torch.randperm(b_obs.shape[0]).split(BATCH_SIZE):
                    dist = Categorical(logits=actor(b_obs[idx]))
                    logp = dist.log_prob(b_act[idx])
                    adv = b_adv[idx]
                    if len(idx) > 1:
                        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                    ratio = torch.exp(logp - b_logp[idx])
                    policy_loss = -torch.min(adv * ratio, adv * torch.clamp(ratio, 1 - CLIP_RANGE, 1 + CLIP_RANGE)).mean()
                    value_loss = torch.nn.functional.mse_loss(b_ret[idx], value(b_state[idx]).squeeze(-1))
                    entropy = dist.entropy().mean()
                    loss = policy_loss - args.ent_coef * entropy + VF_COEF * value_loss
                    if reference is not None:   # KL(V4 || actor) on V4's own 6 inputs
                        with torch.no_grad():
                            ref = Categorical(logits=reference(b_obs[idx][:, :obs_dim]))
                        loss = loss + args.kl_coef * torch.distributions.kl_divergence(ref, dist).mean()
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(params, MAX_GRAD_NORM)
                    optimizer.step()
                    stats.append((policy_loss.item(), value_loss.item(), entropy.item(),
                                  ((ratio - 1).abs() > CLIP_RANGE).float().mean().item()))
            p, v, e, c = np.mean(stats, axis=0)
            print("[TB03_MAPPO] team success last 100 by encounter: " + " ".join(
                f"{k}={v.count('SUCCESS') / len(v):.2f}/coll {v.count('COLLISION') / len(v):.2f}({len(v)})"
                for k, v in sorted(by_encounter.items())), flush=True)
            print(f"[TB03_MAPPO] update steps={num_timesteps} samples={int(mask.sum())} policy_loss={p:.4f} "
                  f"value_loss={v:.4f} entropy={e:.4f} clip_fraction={c:.3f}", flush=True)

            if next_eval is not None and num_timesteps >= next_eval and not stop:
                reached, collided, robot_hits = quick_eval()
                score = reached - collided
                if score > best_score:
                    best_score = score
                    save(output_dir / "best_model.pt")
                print(f"[TB03_MAPPO] EVAL@{num_timesteps} episodes={args.eval_size} both_reached={reached:.3f} "
                      f"episode_collision={collided:.3f} robot_robot={robot_hits:.3f} score={score:.3f} "
                      f"best={best_score:.3f}", flush=True)
                next_eval += args.eval_every
                # A fresh scenario stream, not a replay of the one after the last eval.
                observations, info = env.reset(seed=args.seed + num_timesteps)

        if not stop:
            save(output_dir / "final_model.pt")
        print(f"[TB03_MAPPO] complete: timesteps={num_timesteps} episodes={episodes} -> {output_dir}", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
