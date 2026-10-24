#!/usr/bin/env python3
"""Export a Stable-Baselines3 PPO checkpoint's actor network to a NumPy .npz.

The exported file runs with policy_numpy.py, which needs only NumPy, so the
policy can run on the TurtleBot3's Raspberry Pi without PyTorch or SB3.

Needs stable-baselines3 + torch (run it where those are installed):
    python3 export_policy_numpy.py models/isaac_point_nav_lidar_ppo_v2/checkpoint_170000_steps.zip
Writes models/isaac_point_nav_lidar_ppo_v2/checkpoint_170000_steps.npz next to the zip and checks
that the NumPy forward pass picks the same action as SB3 on random inputs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

KEYS = (
    ("w0", "mlp_extractor.policy_net.0.weight"),
    ("b0", "mlp_extractor.policy_net.0.bias"),
    ("w1", "mlp_extractor.policy_net.2.weight"),
    ("b1", "mlp_extractor.policy_net.2.bias"),
    ("w2", "action_net.weight"),
    ("b2", "action_net.bias"),
)
CHECK_SAMPLES = 10_000


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path, help="SB3 PPO .zip checkpoint")
    args = parser.parse_args()

    from stable_baselines3 import PPO

    try:
        from .policy_numpy import NumpyPolicy
    except ImportError:  # pragma: no cover - direct-script convenience only
        from policy_numpy import NumpyPolicy

    model = PPO.load(str(args.model), device="cpu")
    state = model.policy.state_dict()
    arrays = {name: state[key].detach().cpu().numpy().astype(np.float64) for name, key in KEYS}
    obs_space = model.observation_space
    arrays["obs_low"] = np.asarray(obs_space.low, dtype=np.float64)
    arrays["obs_high"] = np.asarray(obs_space.high, dtype=np.float64)

    out_path = args.model.with_suffix(".npz")
    np.savez(out_path, **arrays)

    # Verify: same deterministic action as SB3 on random in-range observations.
    policy = NumpyPolicy(out_path)
    rng = np.random.default_rng(0)
    obs = rng.uniform(arrays["obs_low"], arrays["obs_high"], size=(CHECK_SAMPLES, arrays["obs_low"].size))
    sb3_actions, _ = model.predict(obs.astype(np.float32), deterministic=True)
    np_actions = np.array([policy.act(o) for o in obs])
    mismatches = int((sb3_actions != np_actions).sum())
    print(f"wrote {out_path}  obs_dim={policy.obs_dim}  "
          f"action agreement with SB3: {CHECK_SAMPLES - mismatches}/{CHECK_SAMPLES}")
    if mismatches:
        raise SystemExit("NumPy policy disagrees with SB3; do not use this export.")


if __name__ == "__main__":
    main()
