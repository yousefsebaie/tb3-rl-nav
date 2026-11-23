#!/usr/bin/env python3
"""Export a MAPPO checkpoint's shared actor (train_mappo.py .pt) to a NumPy .npz for the robot.

The .npz has the same keys as export_policy_numpy.py (w0, b0, w1, b1, w2, b2, obs_low, obs_high),
so policy_numpy.NumpyPolicy runs it unchanged with NumPy only (no PyTorch on the Pi). The actor is
the same network shape as V4 (Linear -> tanh -> Linear(64) -> tanh -> Linear(4) -> argmax); only the
input is wider. `inputs` in the .npz lists the input order (see docs/MAPPO_HARDWARE_HANDOFF.md).

Needs torch (run it with Isaac's Python):
    ./isaac_run.sh export_mappo_numpy.py models/mappo_v5_kl05/best_model.pt
Writes models/mappo_v5_kl05/best_model.npz and checks that the NumPy forward pass picks the same
action as the torch actor on random inputs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

V4_INPUTS = ("d_norm", "sin_bearing", "cos_bearing", "front", "left", "right")
V5_SECTOR_INPUTS = ("narrow_front", "front_left", "front_right", "left_side", "right_side")
TEAMMATE_INPUTS = ("role", "teammate_x", "teammate_y", "teammate_active")
CHECK_SAMPLES = 10_000


def input_layout(ckpt: dict) -> tuple[list[str], np.ndarray, np.ndarray]:
    names = list(V4_INPUTS) + (list(V5_SECTOR_INPUTS) if ckpt.get("extra_sectors") else [])
    if ckpt.get("teammate_inputs"):
        names += list(TEAMMATE_INPUTS)
    low = np.array([-1.0 if n in ("sin_bearing", "cos_bearing", "teammate_x", "teammate_y") else 0.0 for n in names])
    return names, low, np.ones(len(names))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path, help="train_mappo.py checkpoint (.pt)")
    args = parser.parse_args()

    import torch

    try:
        from .policy_numpy import NumpyPolicy
    except ImportError:  # pragma: no cover - direct-script convenience only
        from policy_numpy import NumpyPolicy

    ckpt = torch.load(args.model, map_location="cpu", weights_only=False)
    actor = ckpt["actor"]
    arrays = {name: actor[key].detach().cpu().numpy().astype(np.float64)
              for name, key in (("w0", "0.weight"), ("b0", "0.bias"), ("w1", "2.weight"),
                                ("b1", "2.bias"), ("w2", "4.weight"), ("b2", "4.bias"))}
    names, low, high = input_layout(ckpt)
    if len(names) != arrays["w0"].shape[1]:
        raise SystemExit(f"checkpoint actor takes {arrays['w0'].shape[1]} inputs, layout has {len(names)}")
    arrays.update(obs_low=low, obs_high=high, inputs=np.array(names))

    out_path = args.model.with_suffix(".npz")
    np.savez(out_path, **arrays)

    # Verify: same deterministic action as the torch actor (argmax of its logits).
    policy = NumpyPolicy(out_path)
    net = torch.nn.Sequential(torch.nn.Linear(len(names), 64), torch.nn.Tanh(), torch.nn.Linear(64, 64),
                              torch.nn.Tanh(), torch.nn.Linear(64, 4))
    net.load_state_dict(actor)
    rng = np.random.default_rng(0)
    obs = rng.uniform(low, high, size=(CHECK_SAMPLES, len(names)))
    with torch.no_grad():
        torch_actions = net(torch.as_tensor(obs, dtype=torch.float32)).argmax(dim=1).numpy()
    np_actions = np.array([policy.act(o) for o in obs])
    mismatches = int((torch_actions != np_actions).sum())
    print(f"wrote {out_path}  inputs={len(names)} ({', '.join(names)})  "
          f"action agreement with torch: {CHECK_SAMPLES - mismatches}/{CHECK_SAMPLES}")
    if mismatches:
        raise SystemExit("NumPy policy disagrees with torch; do not use this export.")


if __name__ == "__main__":
    main()
