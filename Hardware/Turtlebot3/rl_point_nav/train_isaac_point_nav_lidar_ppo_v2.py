#!/usr/bin/env python3
"""V2 PPO training: point navigation with RTX LiDAR sectors and pooled obstacles.

Same PPO hyperparameters as V1B; new 6D observation, so this trains a NEW
policy into models/isaac_point_nav_lidar_ppo_v2. The V1B model is never loaded
or written.

  --timesteps N   total PPO timesteps (default 100000)
  --gui           train with the Isaac GUI (slower; default headless)
  --output-name D model folder under models/ (default isaac_point_nav_lidar_ppo_v2);
                  use a new name when training on another scene (TB3_SCENE_USD)
  --profile P     v2 (default) or v3: real-robot lessons (proximity penalty, LiDAR
                  noise/dropout, 12-40 cm obstacles, +/-18 deg front); see v3_lidar_config()
  --resume PATH   continue from a checkpoint .zip; --timesteps is then the TOTAL
                  to reach (steps already in the checkpoint count)

Each finished episode prints one [TB03_PROGRESS] line (steps, %, speed, ETA and
success/collision rates over the last 100 episodes); training_progress.py turns
the log into a live terminal view.
"""

from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path
import time

try:
    from .isaac_point_nav_lidar_env import IsaacPointNavLidarEnv, IsaacLidarNavConfig, v3_lidar_config
except ImportError:  # pragma: no cover - direct-script convenience only
    from isaac_point_nav_lidar_env import IsaacPointNavLidarEnv, IsaacLidarNavConfig, v3_lidar_config


DEFAULT_TIMESTEPS = 100_000
SEED = 42
CHECKPOINT_FREQ = 10_000
MODELS_DIR = Path(__file__).resolve().parent / "models"
OUTPUT_DIR = MODELS_DIR / "isaac_point_nav_lidar_ppo_v2"
V1B_DIR = MODELS_DIR / "isaac_point_nav_ppo_v1b"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=DEFAULT_TIMESTEPS)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--output-name", default=OUTPUT_DIR.name)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--profile", choices=("v2", "v3"), default="v2")
    args = parser.parse_args()
    output_dir = MODELS_DIR / args.output_name

    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
        from stable_baselines3.common.monitor import Monitor
    except ImportError as exc:
        raise RuntimeError("Stable-Baselines3 is required for V2 PPO training.") from exc

    if output_dir.resolve() == V1B_DIR.resolve() or V1B_DIR.resolve() in output_dir.resolve().parents:
        raise RuntimeError("Refusing to write V2 models into the frozen V1B model directory.")

    class EpisodeStatsCallback(BaseCallback):
        """Print per-episode outcome and running success / collision rates."""

        def __init__(self, goal_tolerance_m: float, total_timesteps: int) -> None:
            super().__init__()
            self.goal_tolerance_m = goal_tolerance_m
            self.total_timesteps = total_timesteps
            self.episodes = 0
            self.successes = 0
            self.collisions = 0
            self.recent = deque(maxlen=100)
            self.start_wall = time.monotonic()
            self.start_steps = 0

        def _on_training_start(self) -> None:
            self.start_wall = time.monotonic()
            self.start_steps = self.num_timesteps

        def _on_step(self) -> bool:
            for done, info in zip(self.locals["dones"], self.locals["infos"]):
                if not done:
                    continue
                self.episodes += 1
                collision = bool(info.get("collision", False))
                success = not collision and info.get("distance_to_goal_m", float("inf")) <= self.goal_tolerance_m
                self.successes += int(success)
                self.collisions += int(collision)
                episode = info.get("episode", {})
                outcome = "SUCCESS" if success else ("COLLISION" if collision else "TIMEOUT")
                print(
                    f"[TB03_PPO_V2] episode={self.episodes} outcome={outcome} "
                    f"return={episode.get('r', float('nan')):.3f} length={episode.get('l', -1)} "
                    f"obstacles={len(info.get('obstacles', []))} "
                    f"success_rate={self.successes / self.episodes:.3f} "
                    f"collision_rate={self.collisions / self.episodes:.3f}",
                    flush=True,
                )
                self.recent.append(outcome)
                elapsed = time.monotonic() - self.start_wall
                rate = (self.num_timesteps - self.start_steps) / elapsed if elapsed > 0 else 0.0
                remaining = max(self.total_timesteps - self.num_timesteps, 0)
                n = len(self.recent)
                print(
                    f"[TB03_PROGRESS] steps={self.num_timesteps} total={self.total_timesteps} "
                    f"elapsed_s={elapsed:.0f} steps_per_s={rate:.2f} "
                    f"eta_s={remaining / rate if rate > 0 else -1:.0f} episode={self.episodes} "
                    f"last{n}_success={self.recent.count('SUCCESS') / n:.3f} "
                    f"last{n}_collision={self.recent.count('COLLISION') / n:.3f} "
                    f"last{n}_timeout={self.recent.count('TIMEOUT') / n:.3f} "
                    f"outcome={outcome} length={episode.get('l', -1)}",
                    flush=True,
                )
            return True

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[TB03_PPO_V2] Creating {'GUI' if args.gui else 'headless'} V2 environment", flush=True)
    lidar_config = (v3_lidar_config(hide_robot_visuals=True) if args.profile == "v3"
                    else IsaacLidarNavConfig(hide_robot_visuals=True))
    print(f"[TB03_PPO_V2] profile={args.profile} {lidar_config}", flush=True)
    # The policy only works with the profile it was trained on (sector angles differ).
    (output_dir / "profile.txt").write_text(args.profile + "\n")
    env = IsaacPointNavLidarEnv(headless=not args.gui, lidar_config=lidar_config)
    print("[TB03_PPO_V2] Environment ready; constructing PPO", flush=True)
    stats_callback = EpisodeStatsCallback(env.isaac_config.goal_tolerance_m, args.timesteps)
    try:
        monitored_env = Monitor(env)
        if args.resume is not None:
            model = PPO.load(str(args.resume), env=monitored_env, device="cpu")
            print(f"[TB03_PPO_V2] resumed {args.resume} at {model.num_timesteps} steps", flush=True)
        else:
            model = PPO(
                "MlpPolicy",
                monitored_env,
                policy_kwargs={"net_arch": [64, 64]},
                learning_rate=3e-4,
                gamma=0.99,
                gae_lambda=0.95,
                ent_coef=0.01,
                n_steps=256,
                batch_size=64,
                n_epochs=5,
                seed=SEED,
                device="cpu",
                verbose=0,
            )
        checkpoints = CheckpointCallback(
            save_freq=CHECKPOINT_FREQ,
            save_path=str(output_dir),
            name_prefix="checkpoint",
        )
        remaining = args.timesteps - model.num_timesteps
        if remaining <= 0:
            raise ValueError(f"Checkpoint already has {model.num_timesteps} steps >= --timesteps {args.timesteps}")
        print(f"[TB03_PPO_V2] training {remaining} more steps to reach {args.timesteps}", flush=True)
        model.learn(
            total_timesteps=remaining,
            callback=CallbackList([checkpoints, stats_callback]),
            progress_bar=False,
            reset_num_timesteps=args.resume is None,
        )
        final_model = output_dir / "final_model"
        model.save(str(final_model))
        print(
            f"[TB03_PPO_V2] complete: timesteps={args.timesteps} episodes={stats_callback.episodes} "
            f"successes={stats_callback.successes} collisions={stats_callback.collisions} "
            f"model={final_model}.zip",
            flush=True,
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
