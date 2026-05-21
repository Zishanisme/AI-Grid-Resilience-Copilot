"""
grid_env.py
===========
Sequential decision-support environment for grid resilience policy learning.

Design framing (H5 / W13 fix)
------------------------------
This environment wraps pre-recorded temporal graph sequences from the trained
STGAT_GRU model.  It is a SEQUENTIAL DECISION-SUPPORT environment, not a
closed-loop grid control simulator.

What the agent does:
  - At each timestep it observes a compact state derived from GNN-predicted
    risk probabilities and the graph embedding of the current snapshot.
  - It selects an action (a mitigation recommendation) and receives a reward
    based on whether that recommendation matches the dominant risk type.
  - It learns WHEN to recommend each action given temporal risk patterns.

What the agent does NOT do:
  - It does not modify the power-flow state of the grid.
  - Actions do not alter the next state (which is the next pre-recorded snapshot).
  - This is a policy-learning problem over a fixed operating-condition trajectory,
    analogous to imitation learning over a risk landscape.

Framing for the paper:
  "The DQN policy learns a temporal action-recommendation strategy over
   pre-recorded operating sequences.  It outperforms the static rule-based
   threshold policy by learning adaptive action timing under varying renewable
   stress conditions, without requiring closed-loop power-flow feedback."

Fix (H6 / W14): reward key names in info dict corrected from "fault_mean" /
"congestion_mean" / "voltage_mean" (which never existed) to "fault_risk" /
"congestion_risk" / "voltage_risk" (matching what step() actually computes).
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

ACTION_NAMES = {
    0: "monitor",
    1: "shed_load",
    2: "reroute_power",
    3: "voltage_support",
    4: "isolate_section",
}
N_ACTIONS = len(ACTION_NAMES)


class GridResilienceEnv:
    """
    Sequential decision-support environment built on trained GNN risk predictions.

    State  = mean graph embedding (hidden_dim) + [fault, congestion, voltage, renewable_proxy]
    Action = one of N_ACTIONS discrete recommendations
    Reward = risk-penalty + action-match bonus (see step())

    The environment is non-reactive: actions do not change future states.
    This makes it a risk-informed sequential recommendation problem.
    """

    def __init__(
        self,
        sequences:  List[list],
        model,
        device:     str = "cpu",
        max_steps:  int = 12,
    ):
        if not sequences:
            raise ValueError("GridResilienceEnv needs at least one graph sequence.")

        self.sequences  = sequences
        self.model      = model.to(device).eval()
        self.device     = device
        self.max_steps  = max_steps
        self.hidden_dim = int(getattr(model, "hidden_dim", 128))
        self.obs_dim    = self.hidden_dim + 4

        self._episode: list = []
        self._idx: int      = 0

    # ------------------------------------------------------------------
    def reset(self) -> np.ndarray:
        start = random.randint(0, max(0, len(self.sequences) - self.max_steps - 1))
        self._episode = self.sequences[start : start + self.max_steps]
        self._idx     = 0
        return self._get_state(self._episode[self._idx])

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _get_state(self, seq: list) -> np.ndarray:
        seq_dev = [g.to(self.device) for g in seq]
        logits  = self.model(seq_dev)
        probs   = torch.sigmoid(logits).detach().cpu().numpy()
        risk    = probs.mean(axis=0).astype(np.float32)  # [fault, congestion, voltage]

        # Mean-pooled graph embedding from the last snapshot
        emb = (
            self.model.encode_graph(seq_dev[-1])
            .mean(dim=0)
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

        # Renewable proxy: mean of last feature column across all buses
        x_last         = seq_dev[-1].x.detach().cpu()
        renewable_proxy = float(x_last[:, -1].mean().item()) if x_last.shape[1] > 1 else 0.0

        return np.concatenate([emb, risk, np.array([renewable_proxy], dtype=np.float32)])

    # ------------------------------------------------------------------
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        seq     = self._episode[self._idx]
        seq_dev = [g.to(self.device) for g in seq]

        with torch.no_grad():
            logits = self.model(seq_dev)
            probs  = torch.sigmoid(logits).detach().cpu().numpy()

        risks          = probs.mean(axis=0)
        fault_risk     = float(risks[0])
        congestion_risk = float(risks[1])
        voltage_risk   = float(risks[2])

        # Base reward: negative sum weighted by operational severity
        reward = -(
            5.0 * fault_risk +
            3.0 * congestion_risk +
            2.0 * voltage_risk
        )

        # Action-match bonus / mismatch penalty
        if action == 1:   # shed_load → appropriate for congestion
            reward += 1.5 if congestion_risk > 0.5 else -0.5

        elif action == 2:  # reroute_power → appropriate for congestion
            reward += 2.0 if congestion_risk > 0.5 else -0.5

        elif action == 3:  # voltage_support → appropriate for voltage risk
            reward += 1.5 if voltage_risk > 0.5 else -0.5

        elif action == 4:  # isolate_section → appropriate for fault risk
            reward += 2.5 if fault_risk > 0.5 else -1.0

        elif action == 0:  # monitor → correct only when all risks are low
            if max(fault_risk, congestion_risk, voltage_risk) < 0.4:
                reward += 0.5
            else:
                reward -= 0.2

        self._idx += 1
        done = self._idx >= len(self._episode)

        next_state = (
            np.zeros(self.obs_dim, dtype=np.float32)
            if done
            else self._get_state(self._episode[self._idx])
        )

        # FIX (H6 / action_log key fix): return correctly named risk keys
        info = {
            "fault_risk":      fault_risk,        # was "fault_mean" — FIXED
            "congestion_risk": congestion_risk,   # was "congestion_mean" — FIXED
            "voltage_risk":    voltage_risk,      # was "voltage_mean" — FIXED
            "action_name":     ACTION_NAMES.get(action, "unknown"),
        }

        return next_state, float(reward), done, info
