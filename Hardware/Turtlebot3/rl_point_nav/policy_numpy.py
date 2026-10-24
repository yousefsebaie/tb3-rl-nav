"""NumPy-only forward pass of the trained PPO actor (no PyTorch / SB3 needed).

Network (SB3 MlpPolicy, net_arch=[64, 64]):
    obs -> Linear -> tanh -> Linear(64) -> tanh -> action_net(4) -> argmax
argmax of the action logits is SB3's deterministic action.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


class NumpyPolicy:
    def __init__(self, npz_path: str | Path) -> None:
        data = np.load(str(npz_path))
        self.w0, self.b0 = data["w0"], data["b0"]
        self.w1, self.b1 = data["w1"], data["b1"]
        self.w2, self.b2 = data["w2"], data["b2"]
        self.obs_low, self.obs_high = data["obs_low"], data["obs_high"]
        self.obs_dim = int(self.w0.shape[1])

    def logits(self, observation) -> np.ndarray:
        obs = np.asarray(observation, dtype=np.float64).reshape(-1)
        if obs.size != self.obs_dim:
            raise ValueError(f"policy expects a {self.obs_dim}D observation, got {obs.size}D")
        # SB3 clips Box observations to the space bounds only for actions;
        # inputs are fed raw. Our env already produces values inside the bounds.
        h = np.tanh(self.w0 @ obs + self.b0)
        h = np.tanh(self.w1 @ h + self.b1)
        return self.w2 @ h + self.b2

    def act(self, observation) -> int:
        return int(np.argmax(self.logits(observation)))
