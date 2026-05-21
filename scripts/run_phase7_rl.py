#!/usr/bin/env python3
"""
scripts/run_phase7_rl.py — Phase 7: DQN sequential decision-support policy.

Fixes applied
-------------
H2  — SEQ_LEN=20 (matches Phase 3 production training and Phase 4).
H5  — Architecture loaded from model_meta.json (written by train_model.py)
      rather than fragile weight-shape inference.
H6  — Matched-episode evaluation: DQN (greedy) and rule-based evaluated
      from identical starting states at end of training.
Grid_env fix — info dict keys corrected: "fault_risk", "congestion_risk",
      "voltage_risk" (previously returned as "fault_mean" etc. in old env).
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.st_gnn   import STGAT_GRU
from src.train.utils     import make_sequences
from src.rl.grid_env     import GridResilienceEnv, ACTION_NAMES, N_ACTIONS
from src.rl.dqn_agent    import DQNAgent

DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
SEQ_LEN         = 20        # H2 fix: matches Phase 3 training
N_EPISODES      = 600
MAX_STEPS       = 12
WINDOW          = 20
N_EVAL_EPISODES = 50


# ---------------------------------------------------------------------------
# Model loading — uses model_meta.json (H5 fix)
# ---------------------------------------------------------------------------
def load_trained_stgat() -> STGAT_GRU:
    ckpt      = ROOT / "results/phase3/resiligraph_stgat/best_model.pt"
    meta_path = ROOT / "results/phase3/resiligraph_stgat/model_meta.json"

    if not ckpt.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt}. Run Phase 3 first.")

    if meta_path.exists():
        meta       = json.loads(meta_path.read_text())
        in_dim     = meta["in_dim"]
        hidden_dim = meta.get("hidden_dim", 64)
        heads      = meta.get("heads", 2)
    else:
        # Robust fallback: try common PyG weight key patterns
        state = torch.load(ckpt, map_location=DEVICE)
        in_dim, hidden_dim, heads = None, 64, 2
        for suffix in ("lin.weight", "lin_src.weight"):
            k1 = f"gat1.{suffix}"
            k2 = f"gat2.{suffix}"
            if k1 in state:
                in_dim     = state[k1].shape[1]
                hidden_dim = state[k2].shape[0]
                att_key    = "gat1.att_src"
                heads      = state[att_key].shape[1] if att_key in state else 2
                break
        if in_dim is None:
            raise RuntimeError(
                "Cannot load architecture. Save model_meta.json by running "
                "train_model.py (it now does this automatically)."
            )

    model = STGAT_GRU(in_dim=in_dim, hidden_dim=hidden_dim, heads=heads)
    state = torch.load(ckpt, map_location=DEVICE)
    model.load_state_dict(state)
    return model.to(DEVICE).eval()


def load_sequences():
    path = ROOT / "data/processed/phase2/train_graphs.pt"
    if not path.exists():
        raise FileNotFoundError(f"Missing: {path}. Run Phase 2 first.")
    graphs = torch.load(path, map_location="cpu", weights_only=False)
    seqs   = make_sequences(graphs, SEQ_LEN)
    if not seqs:
        raise RuntimeError("No sequences. Check data and SEQ_LEN.")
    return seqs


def rule_based_action(state: np.ndarray) -> int:
    """
    Static threshold policy for baseline comparison.
    State indices: [...emb (hidden_dim), fault(-4), cong(-3), volt(-2), renew(-1)]
    """
    fault = state[-4]
    cong  = state[-3]
    volt  = state[-2]
    if fault > 0.30:            return 4   # isolate_section
    if cong  > 0.35:            return 2   # reroute_power
    if volt  > 0.35:            return 3   # voltage_support
    if max(cong, volt) > 0.55:  return 1   # shed_load
    return 0                               # monitor


def running_avg(arr, w: int = WINDOW) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    return np.convolve(arr, np.ones(w) / w, mode="valid") if len(arr) >= w else arr


# ---------------------------------------------------------------------------
# H6 fix: matched-episode evaluation
# ---------------------------------------------------------------------------
def evaluate_matched(env, agent, n: int = N_EVAL_EPISODES) -> dict:
    """Both policies evaluated from identical starting states."""
    dqn_rewards, rule_rewards = [], []

    # Force greedy by pushing epsilon to 0
    saved_start = agent.epsilon_start
    saved_end   = agent.epsilon_end
    agent.epsilon_start = 0.0
    agent.epsilon_end   = 0.0

    for _ in range(n):
        # DQN greedy episode
        state    = env.reset()
        ep_snap  = list(env._episode)
        idx_snap = env._idx
        done     = False
        dr, ds   = 0.0, 0

        while not done:
            a           = agent.select_action(state)
            state, r, done, _ = env.step(a)
            dr += r; ds += 1

        dqn_rewards.append(dr / max(1, ds))

        # Rule-based on SAME episode
        env._episode = ep_snap
        env._idx     = idx_snap
        state        = env._get_state(env._episode[env._idx])
        done         = False
        rr, rs       = 0.0, 0

        while not done:
            a           = rule_based_action(state)
            state, r, done, _ = env.step(a)
            rr += r; rs += 1

        rule_rewards.append(rr / max(1, rs))

    # Restore epsilon
    agent.epsilon_start = saved_start
    agent.epsilon_end   = saved_end

    dqn_arr  = np.array(dqn_rewards)
    rule_arr = np.array(rule_rewards)
    return {
        "dqn_mean":          float(dqn_arr.mean()),
        "dqn_std":           float(dqn_arr.std()),
        "rule_mean":         float(rule_arr.mean()),
        "rule_std":          float(rule_arr.std()),
        "improvement":       float(dqn_arr.mean() - rule_arr.mean()),
        "improvement_pct":   float(
            100 * (dqn_arr.mean() - rule_arr.mean())
            / max(abs(rule_arr.mean()), 1e-8)
        ),
    }


# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("Phase 7 — DQN Sequential Decision-Support (seq_len=20)")
    print("=" * 70)

    model      = load_trained_stgat()
    train_seqs = load_sequences()
    env        = GridResilienceEnv(train_seqs, model, device=DEVICE, max_steps=MAX_STEPS)

    agent = DQNAgent(
        obs_dim=env.obs_dim, n_actions=N_ACTIONS,
        lr=1e-3, gamma=0.95, epsilon_decay=1200,
        batch_size=64, target_update=100, device=DEVICE,
    )

    dqn_rewards, rule_rewards, action_log = [], [], []

    for ep in range(N_EPISODES):
        # DQN training episode
        state      = env.reset()
        done       = False
        ep_reward  = 0.0
        last_info  = {}
        step_count = 0

        while not done:
            action               = agent.select_action(state)
            state, reward, done, info = env.step(action)
            agent.store(state, action, reward, state, done)
            agent.update()
            ep_reward  += reward
            last_info   = info
            step_count += 1

        dqn_rewards.append(ep_reward / max(1, step_count))

        # Rule-based on fresh episode (training comparison)
        state = env.reset()
        done  = False
        rr, rs = 0.0, 0
        while not done:
            a           = rule_based_action(state)
            state, r, done, _ = env.step(a)
            rr += r; rs += 1
        rule_rewards.append(rr / max(1, rs))

        # Log last 100 episodes (converged policy)
        if ep >= N_EPISODES - 100:
            action_log.append({
                "episode":         ep,
                "action":          action,
                "action_name":     ACTION_NAMES[action],
                "reward":          round(dqn_rewards[-1], 4),
                "fault_risk":      round(last_info.get("fault_risk",      0.0), 4),
                "congestion_risk": round(last_info.get("congestion_risk", 0.0), 4),
                "voltage_risk":    round(last_info.get("voltage_risk",    0.0), 4),
                "epsilon":         round(agent.epsilon, 4),
            })

        if (ep + 1) % 100 == 0:
            print(f"Episode {ep+1:4d}/{N_EPISODES} | "
                  f"DQN={np.mean(dqn_rewards[-50:]):.4f} | "
                  f"Rule={np.mean(rule_rewards[-50:]):.4f} | "
                  f"eps={agent.epsilon:.3f}")

    # Matched evaluation
    print("\nRunning matched-episode evaluation...")
    matched = evaluate_matched(env, agent)
    print(f"DQN={matched['dqn_mean']:.4f}±{matched['dqn_std']:.4f}  "
          f"Rule={matched['rule_mean']:.4f}±{matched['rule_std']:.4f}  "
          f"Improvement={matched['improvement']:+.4f} "
          f"({matched['improvement_pct']:+.1f}%)")

    # Save outputs
    out = ROOT / "results/phase7_rl"
    out.mkdir(parents=True, exist_ok=True)
    torch.save(agent.q_net.state_dict(), out / "dqn_model.pt")

    df_log = pd.DataFrame(action_log)
    df_log.to_csv(out / "policy_actions.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(running_avg(dqn_rewards),  label="DQN policy (training)")
    ax.plot(running_avg(rule_rewards), label="Rule-based (training)", linestyle="--")
    ax.set_title("DQN vs Rule-Based Resilience Policy")
    ax.set_xlabel("Episode")
    ax.set_ylabel(f"Mean reward per step (running avg w={WINDOW})")
    ax.grid(alpha=0.3); ax.legend()
    plt.tight_layout()
    plt.savefig(out / "reward_curve.png", dpi=150, bbox_inches="tight")
    plt.close()

    action_dist = Counter(df_log["action_name"].values) if not df_log.empty else Counter()

    summary = {
        "n_episodes":             N_EPISODES,
        # Training averages
        "dqn_final_avg_reward":  round(float(np.mean(dqn_rewards[-50:])), 4),
        "rule_final_avg_reward": round(float(np.mean(rule_rewards[-50:])), 4),
        "improvement":           round(float(
            np.mean(dqn_rewards[-50:]) - np.mean(rule_rewards[-50:])), 4),
        # Matched evaluation (use these for Table 4)
        "matched_dqn_mean":      round(matched["dqn_mean"],        4),
        "matched_dqn_std":       round(matched["dqn_std"],         4),
        "matched_rule_mean":     round(matched["rule_mean"],       4),
        "matched_rule_std":      round(matched["rule_std"],        4),
        "matched_improvement":   round(matched["improvement"],     4),
        "matched_improvement_pct": round(matched["improvement_pct"], 2),
        "action_distribution": {
            ACTION_NAMES[i]: int(action_dist.get(ACTION_NAMES[i], 0))
            for i in range(N_ACTIONS)
        },
    }

    with open(out / "rl_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\nSaved Phase 7 outputs to", out)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
