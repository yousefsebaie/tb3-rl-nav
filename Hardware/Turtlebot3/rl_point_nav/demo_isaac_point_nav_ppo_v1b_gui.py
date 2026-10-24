#!/usr/bin/env python3
"""Visual deterministic demo of the V1B 15k PPO checkpoint in Isaac Sim."""

from __future__ import annotations

from pathlib import Path
import time

try:
    from .config import PointNavAction
    from .isaac_point_nav_env import IsaacPointNavEnv
except ImportError:  # pragma: no cover - direct-script convenience only
    from config import PointNavAction
    from isaac_point_nav_env import IsaacPointNavEnv


MODEL_PATH = Path(__file__).resolve().parent / "models" / "isaac_point_nav_ppo_v1b" / "checkpoint_15000_steps.zip"
ACTION_PAUSE_S = 0.35
TEST_GOALS = (
    ("LEFT", (-2.718, 1.483)),
    ("RIGHT", (-2.718, -0.595)),
    ("STRAIGHT", (-2.118, 0.444)),
)


def _run_episode(env: IsaacPointNavEnv, model, label: str, goal: tuple[float, float]) -> None:
    """Reset and visually execute one deterministic explicit-goal episode."""
    observation, info = env.reset(options={"goal": goal})
    print(
        f"[TB03_GUI_V1B] {label} goal=({goal[0]:.3f}, {goal[1]:.3f}) "
        f"robot=({info['pose']['x']:.3f}, {info['pose']['y']:.3f}) "
        f"yaw={info['yaw']:.3f} distance={info['distance_to_goal_m']:.3f}m",
        flush=True,
    )

    while True:
        action, _ = model.predict(observation, deterministic=True)
        action_name = PointNavAction(int(action)).name
        observation, _, terminated, truncated, info = env.step(action)
        print(
            f"[TB03_GUI_V1B] {label} action={action_name} "
            f"robot=({info['pose']['x']:.3f}, {info['pose']['y']:.3f}) "
            f"yaw={info['yaw']:.3f} distance={info['distance_to_goal_m']:.3f}m",
            flush=True,
        )
        if terminated:
            print(f"[TB03_GUI_V1B] {label} SUCCESS", flush=True)
            return
        if truncated:
            print(f"[TB03_GUI_V1B] {label} FAILURE: episode truncated", flush=True)
            return
        time.sleep(ACTION_PAUSE_S)


def main() -> None:
    """Run the requested LEFT, RIGHT, and STRAIGHT GUI demonstrations."""
    try:
        from stable_baselines3 import PPO
    except ImportError as exc:
        raise RuntimeError("Stable-Baselines3 is required to run the V1B PPO GUI demo.") from exc

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Saved V1B checkpoint not found: {MODEL_PATH}")

    env = IsaacPointNavEnv(headless=False)
    try:
        model = PPO.load(str(MODEL_PATH), env=env, device="cpu")
        for index, (label, goal) in enumerate(TEST_GOALS):
            _run_episode(env, model, label, goal)
            if index < len(TEST_GOALS) - 1:
                input(f"[TB03_GUI_V1B] {label} test complete. Press Enter for the next test... ")
        input("[TB03_GUI_V1B] All tests complete. Press Enter to close Isaac Sim... ")
    finally:
        env.close()


if __name__ == "__main__":
    main()
