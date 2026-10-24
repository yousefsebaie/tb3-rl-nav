#!/usr/bin/env python3
"""Deterministic V1B PPO checkpoint evaluation on approved goal families."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from statistics import mean

try:
    from .isaac_point_nav_env import (
        LEFT_GOALS,
        RIGHT_GOALS,
        STRAIGHT_GOALS,
        IsaacPointNavEnv,
    )
except ImportError:  # pragma: no cover - direct-script convenience only
    from isaac_point_nav_env import LEFT_GOALS, RIGHT_GOALS, STRAIGHT_GOALS, IsaacPointNavEnv


EPISODES_PER_GOAL = 5
MODEL_PATH = Path(__file__).resolve().parent / "models" / "isaac_point_nav_ppo_v1b" / "checkpoint_15000_steps.zip"
GOAL_FAMILIES = {
    "LEFT": LEFT_GOALS,
    "RIGHT": RIGHT_GOALS,
    "STRAIGHT": STRAIGHT_GOALS,
}


def main() -> None:
    """Evaluate each approved goal repeatedly without changing the PPO policy."""
    try:
        from stable_baselines3 import PPO
    except ImportError as exc:
        raise RuntimeError("Stable-Baselines3 is required to evaluate the V1B PPO checkpoint.") from exc

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Saved V1B checkpoint not found: {MODEL_PATH}")

    env = IsaacPointNavEnv()
    try:
        model = PPO.load(str(MODEL_PATH), env=env, device="cpu")
        successes: dict[str, list[bool]] = defaultdict(list)
        final_distances: list[float] = []

        for family, goals in GOAL_FAMILIES.items():
            for goal in goals:
                for repetition in range(1, EPISODES_PER_GOAL + 1):
                    observation, _ = env.reset(options={"goal": goal})
                    episode_length = 0
                    while True:
                        action, _ = model.predict(observation, deterministic=True)
                        observation, _, terminated, truncated, info = env.step(action)
                        episode_length += 1
                        if terminated or truncated:
                            break

                    success = bool(terminated)
                    final_distance = float(info["distance_to_goal_m"])
                    successes[family].append(success)
                    final_distances.append(final_distance)
                    print(
                        f"[TB03_EVAL_V1B] family={family} goal={goal} repeat={repetition} "
                        f"success={success} length={episode_length} "
                        f"final_distance={final_distance:.3f}m",
                        flush=True,
                    )

        total_episodes = sum(len(result) for result in successes.values())
        total_successes = sum(sum(result) for result in successes.values())
        for family in GOAL_FAMILIES:
            family_results = successes[family]
            print(
                f"[TB03_EVAL_V1B] {family}_success_rate="
                f"{sum(family_results)}/{len(family_results)}={mean(family_results):.3f}",
                flush=True,
            )
        print(
            f"[TB03_EVAL_V1B] overall_success_rate={total_successes}/{total_episodes}="
            f"{total_successes / total_episodes:.3f} "
            f"mean_final_distance={mean(final_distances):.3f}m",
            flush=True,
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
