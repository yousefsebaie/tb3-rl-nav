#!/usr/bin/env python3
"""Deterministic left, right, and straight PPO probes in the Isaac Sim GUI."""

from __future__ import annotations

from pathlib import Path
import time

try:
    from .config import PointNavAction
    from .isaac_point_nav_env import IsaacPointNavEnv
except ImportError:  # pragma: no cover - direct-script convenience only
    from config import PointNavAction
    from isaac_point_nav_env import IsaacPointNavEnv


MODEL_PATH = Path(__file__).resolve().parent / "models" / "isaac_point_nav_ppo_v1" / "final_model.zip"
ACTION_PAUSE_S = 0.35

# The known reset pose is approximately (-3.318, 0.444), with its initial
# heading approximately along +X. Each target is 0.5--2.0 m away. The saved
# USD is a binary crate and the available scene logs verify only the eastbound
# corridor; left/right targets are therefore explicit visual probes, not
# collision-clearance assertions.
TEST_GOALS = (
    ("LEFT", (-2.718, 1.483)),
    ("RIGHT", (-2.718, -0.595)),
    ("STRAIGHT", (-2.118, 0.444)),
)


def _run_episode(env: IsaacPointNavEnv, model, label: str, goal: tuple[float, float]) -> None:
    """Run one deterministic explicit-goal episode and print each transition."""
    observation, info = env.reset(options={"goal": goal})
    print(
        f"[TB03_GUI] {label} test goal=({info['goal']['x']:.3f}, {info['goal']['y']:.3f}) "
        f"robot=({info['pose']['x']:.3f}, {info['pose']['y']:.3f}) "
        f"yaw={info['yaw']:.3f} distance={info['distance_to_goal_m']:.3f}m",
        flush=True,
    )

    while True:
        action, _ = model.predict(observation, deterministic=True)
        action_name = PointNavAction(int(action)).name
        observation, _, terminated, truncated, info = env.step(action)
        print(
            f"[TB03_GUI] {label} action={action_name} "
            f"robot=({info['pose']['x']:.3f}, {info['pose']['y']:.3f}) "
            f"yaw={info['yaw']:.3f} distance={info['distance_to_goal_m']:.3f}m",
            flush=True,
        )
        if terminated:
            print(f"[TB03_GUI] {label} SUCCESS", flush=True)
            return
        if truncated:
            print(f"[TB03_GUI] {label} FAILURE: episode truncated", flush=True)
            return
        time.sleep(ACTION_PAUSE_S)


def main() -> None:
    """Run three visually inspectable deterministic policy episodes."""
    try:
        from stable_baselines3 import PPO
    except ImportError as exc:
        raise RuntimeError("Stable-Baselines3 is required to run the PPO GUI demo.") from exc

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Saved PPO model not found: {MODEL_PATH}")

    env = IsaacPointNavEnv(headless=False)
    try:
        model = PPO.load(str(MODEL_PATH), env=env)
        for index, (label, goal) in enumerate(TEST_GOALS):
            _run_episode(env, model, label, goal)
            if index < len(TEST_GOALS) - 1:
                input(f"[TB03_GUI] {label} test complete. Press Enter for the next test... ")
        input("[TB03_GUI] All tests complete. Press Enter to close Isaac Sim... ")
    finally:
        env.close()


if __name__ == "__main__":
    main()
