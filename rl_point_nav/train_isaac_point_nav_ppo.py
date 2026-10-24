#!/usr/bin/env python3
"""First single-environment PPO experiment for direct Isaac point navigation.

This script intentionally starts exactly one IsaacPointNavEnv. Do not use it
with SubprocVecEnv or other multi-environment launchers.
"""

from __future__ import annotations

from pathlib import Path

try:
    from .isaac_point_nav_env import IsaacPointNavEnv
except ImportError:  # pragma: no cover - direct-script convenience only
    from isaac_point_nav_env import IsaacPointNavEnv


TOTAL_TIMESTEPS = 2_048
SEED = 42
OUTPUT_DIR = Path(__file__).resolve().parent / "models" / "isaac_point_nav_ppo_v1"


def main() -> None:
    """Train PPO in the one persistent direct-Isaac environment."""
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
        from stable_baselines3.common.monitor import Monitor
    except ImportError as exc:
        raise RuntimeError(
            "Stable-Baselines3 is required for this training script but is not "
            "available in Isaac Sim's Python environment."
        ) from exc

    class EpisodeStatsCallback(BaseCallback):
        """Print concise terminal episode and success statistics."""

        def __init__(self) -> None:
            super().__init__()
            self.episodes = 0
            self.successes = 0

        def _on_step(self) -> bool:
            for done, info in zip(self.locals["dones"], self.locals["infos"]):
                if not done:
                    continue
                self.episodes += 1
                success = info.get("distance_to_goal_m", float("inf")) <= 0.10
                self.successes += int(success)
                episode = info.get("episode", {})
                print(
                    f"[TB03_PPO] episode={self.episodes} success={success} "
                    f"return={episode.get('r', float('nan')):.3f} "
                    f"length={episode.get('l', -1)} "
                    f"success_rate={self.successes / self.episodes:.3f}",
                    flush=True,
                )
            return True

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    env = IsaacPointNavEnv()
    stats_callback = EpisodeStatsCallback()
    try:
        # Monitor and PPO's internal one-item DummyVecEnv keep the same env
        # instance; neither creates a second Isaac SimulationApp.
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
            verbose=1,
        )
        checkpoints = CheckpointCallback(
            save_freq=512,
            save_path=str(OUTPUT_DIR),
            name_prefix="checkpoint",
        )
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=CallbackList([checkpoints, stats_callback]),
            progress_bar=False,
        )
        final_model = OUTPUT_DIR / "final_model"
        model.save(str(final_model))
        print(
            f"[TB03_PPO] complete: timesteps={TOTAL_TIMESTEPS} "
            f"episodes={stats_callback.episodes} successes={stats_callback.successes} "
            f"model={final_model}.zip",
            flush=True,
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
