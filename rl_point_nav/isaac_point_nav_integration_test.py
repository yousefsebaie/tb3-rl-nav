#!/usr/bin/env python3
"""Manual persistent-process integration test for IsaacPointNavEnv.

Run only after approving robot motion.  This is not PPO training.
"""

from __future__ import annotations

try:
    from .config import PointNavAction
    from .isaac_point_nav_env import IsaacPointNavEnv
except ImportError:  # pragma: no cover - direct-script convenience only
    from config import PointNavAction
    from isaac_point_nav_env import IsaacPointNavEnv


def _print_state(label: str, observation, info: dict) -> None:
    print(
        f"[TB03_ENV_TEST] {label}: observation={observation.tolist()} "
        f"pose={info['pose']} yaw={info['yaw']:.6f} goal={info['goal']} "
        f"distance={info['distance_to_goal_m']:.6f} "
        f"elapsed={info['episode_elapsed_sim_s']:.6f}s",
        flush=True,
    )


def _step(env: IsaacPointNavEnv, action: PointNavAction) -> None:
    observation, reward, terminated, truncated, info = env.step(action)
    _print_state(f"after {action.name}", observation, info)
    print(
        f"[TB03_ENV_TEST] reward={reward:.6f} terminated={terminated} truncated={truncated}",
        flush=True,
    )
    if terminated or truncated:
        raise RuntimeError(f"Unexpected episode end during integration sequence after {action.name}")


def main() -> None:
    """Exercise repeated reset → step → reset cycles in one Isaac process."""
    env = IsaacPointNavEnv()
    try:
        observation, info = env.reset(options={"goal": (0.50, 0.00)})
        _print_state("cycle 1 reset", observation, info)
        _step(env, PointNavAction.FORWARD)
        _step(env, PointNavAction.TURN_LEFT)
        _step(env, PointNavAction.FORWARD)

        observation, info = env.reset(options={"goal": (-0.40, 0.20)})
        _print_state("cycle 2 reset", observation, info)
        _step(env, PointNavAction.TURN_RIGHT)
        _step(env, PointNavAction.FORWARD)

        observation, info = env.reset(options={"goal": (0.25, -0.25)})
        _print_state("cycle 3 reset", observation, info)
        print("[TB03_ENV_TEST] PASS: repeated reset/step/reset used one IsaacPointNavEnv instance", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
