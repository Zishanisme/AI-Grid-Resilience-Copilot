from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def forward(self, x):
        return self.net(x)


@dataclass
class Transition:
    s: np.ndarray
    a: int
    r: float
    ns: np.ndarray
    done: bool


class DQNAgent:
    def __init__(self, obs_dim, n_actions, lr=1e-3, gamma=0.95,
                 epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=600,
                 batch_size=64, target_update=50, memory_size=10000,
                 device="cpu"):
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update
        self.device = device
        self.steps = 0

        self.q_net = QNetwork(obs_dim, n_actions).to(device)
        self.target_net = QNetwork(obs_dim, n_actions).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.opt = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.memory = deque(maxlen=memory_size)

    @property
    def epsilon(self):
        frac = min(1.0, self.steps / max(1, self.epsilon_decay))
        return self.epsilon_start + frac * (self.epsilon_end - self.epsilon_start)

    def select_action(self, state):
        self.steps += 1
        if random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            return int(self.q_net(s).argmax(dim=1).item())

    def store(self, s, a, r, ns, done):
        self.memory.append(Transition(s, a, r, ns, done))

    def update(self):
        if len(self.memory) < self.batch_size:
            return None
        batch = random.sample(self.memory, self.batch_size)
        s = torch.tensor(np.stack([b.s for b in batch]), dtype=torch.float32, device=self.device)
        a = torch.tensor([b.a for b in batch], dtype=torch.long, device=self.device)
        r = torch.tensor([b.r for b in batch], dtype=torch.float32, device=self.device)
        ns = torch.tensor(np.stack([b.ns for b in batch]), dtype=torch.float32, device=self.device)
        done = torch.tensor([b.done for b in batch], dtype=torch.float32, device=self.device)

        q = self.q_net(s).gather(1, a.view(-1, 1)).squeeze(1)
        with torch.no_grad():
            target = r + self.gamma * (1.0 - done) * self.target_net(ns).max(dim=1).values
        loss = F.smooth_l1_loss(q, target)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
        self.opt.step()

        if self.steps % self.target_update == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
        return float(loss.item())
