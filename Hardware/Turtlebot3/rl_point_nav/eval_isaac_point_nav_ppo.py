#!/usr/bin/env python3
"""Deterministic evaluation for the first direct-Isaac PPO point-nav policy."""

from __future__ import annotations

from pathlib import Path
from statistics import mean

try:
    from .isaac_point_nav_env import IsaacPointNavEnv
except ImportError:  # pragma: no cover - direct-script convenience only
    from isaac_point_nav_env import IsaacPointNavEnv


EPISODES = 10
MODEL_PATH = Path(__file__).resolve().parent / "models" / "isaac_point_nav_ppo_v1" / "final_model.zip"


def main() -> None:
    """Evaluate ten sampled-goal episodes in one persistent Isaac process."""
    try:
        from stable_baselines3 import PPO
    except ImportError as exc:
        raise RuntimeError("Stable-Baselines3 is required to evaluate the saved PPO model.") from exc

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Saved PPO model not found: {MODEL_PATH}")

    env = IsaacPointNavEnv()
    try:
        model = PPO.load(str(MODEL_PATH), env=env)
        successes: list[bool] = []
        returns: list[float] = []
        lengths: list[int] = []
        final_distances: list[float] = []

        for episode_number in range(1, EPISODES + 1):
            observation, info = env.reset()
            starting_distance = float(info["distance_to_goal_m"])
            episode_return = 0.0
            episode_length = 0

            while True:
                action, _ = model.predict(observation, deterministic=True)
                observation, reward, terminated, truncated, info = env.step(action)
                episode_return += float(reward)
                episode_length += 1
                if terminated or truncated:
                    break

            success = bool(terminated)
            final_distance = float(info["distance_to_goal_m"])
            successes.append(success)
            returns.append(episode_return)
            lengths.append(episode_length)
            final_distances.append(final_distance)
            print(
                f"[TB03_EVAL] episode={episode_number} success={success} "
                f"return={episode_return:.3f} length={episode_length} "
                f"start_distance={starting_distance:.3f}m final_distance={final_distance:.3f}m",
                flush=True,
            )

        print(
            f"[TB03_EVAL] total_successes={sum(successes)}/{EPISODES} "
            f"success_rate={mean(successes):.3f} "
            f"mean_return={mean(returns):.3f} "
            f"mean_length={mean(lengths):.1f} "
            f"mean_final_distance={mean(final_distances):.3f}m",
            flush=True,
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
