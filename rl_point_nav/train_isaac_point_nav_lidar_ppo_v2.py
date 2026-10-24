#!/usr/bin/env python3
"""V2 PPO training: point navigation with RTX LiDAR sectors and pooled obstacles.

Same PPO hyperparameters as V1B; new 6D observation, so this trains a NEW
policy into models/isaac_point_nav_lidar_ppo_v2. The V1B model is never loaded
or written.

  --timesteps N   total PPO timesteps (default 100000)
  --gui           train with the Isaac GUI (slower; default headless)
"""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from .isaac_point_nav_lidar_env import IsaacPointNavLidarEnv, IsaacLidarNavConfig
except ImportError:  # pragma: no cover - direct-script convenience only
    from isaac_point_nav_lidar_env import IsaacPointNavLidarEnv, IsaacLidarNavConfig


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
    args = parser.parse_args()

    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
        from stable_baselines3.common.monitor import Monitor
    except ImportError as exc:
        raise RuntimeError("Stable-Baselines3 is required for V2 PPO training.") from exc

    if OUTPUT_DIR.resolve() == V1B_DIR.resolve() or V1B_DIR.resolve() in OUTPUT_DIR.resolve().parents:
        raise RuntimeError("Refusing to write V2 models into the frozen V1B model directory.")

    class EpisodeStatsCallback(BaseCallback):
        """Print per-episode outcome and running success / collision rates."""

        def __init__(self, goal_tolerance_m: float) -> None:
            super().__init__()
            self.goal_tolerance_m = goal_tolerance_m
            self.episodes = 0
            self.successes = 0
            self.collisions = 0

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
            return True

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[TB03_PPO_V2] Creating {'GUI' if args.gui else 'headless'} V2 environment", flush=True)
    env = IsaacPointNavLidarEnv(headless=not args.gui, lidar_config=IsaacLidarNavConfig(hide_robot_visuals=True))
    print("[TB03_PPO_V2] Environment ready; constructing PPO", flush=True)
    stats_callback = EpisodeStatsCallback(env.isaac_config.goal_tolerance_m)
    try:
        monitored_env = Monitor(env)
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
            verbose=1,
        )
        checkpoints = CheckpointCallback(
            save_freq=CHECKPOINT_FREQ,
            save_path=str(OUTPUT_DIR),
            name_prefix="checkpoint",
        )
        model.learn(
            total_timesteps=args.timesteps,
            callback=CallbackList([checkpoints, stats_callback]),
            progress_bar=False,
        )
        final_model = OUTPUT_DIR / "final_model"
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
