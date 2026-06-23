"""
collect_data.py

Collects real driving trajectories from HighwayEnv for FTL-IGM.

For each training scenario (highway, merge, intersection):
  - Runs N_EPISODES episodes
  - Records controlled vehicle trajectory: [T, 7]
  - Records initial state (full 5-vehicle obs): [35]
  - Saves concept label

For new scenario (roundabout):
  - Collects only N_FEWSHOT episodes (5) — the few-shot demonstrations

The "agent policy" is a simple heuristic (FASTER action) which gives
reasonable highway driving behaviour. For a research paper you'd use
the IDM planner, but this is sufficient for concept learning.

Output:
  Data/X.npy        [N, N_STEPS, 7]    trajectories
  Data/s0.npy       [N, 35]            initial states
  Data/y_str.npy    [N]                string labels
  Data/y_idx.npy    [N]                integer labels
  Data/fewshot_X.npy   [K, N_STEPS, 7]
  Data/fewshot_s0.npy  [K, 35]
  Data/meta.json
"""

import os, json
import numpy as np
import gymnasium as gym
import highway_env  # noqa: F401 — registers envs

from env_config import (
    N_STEPS, TRAINING_SCENARIOS, NEW_SCENARIO,
    SCENARIO_TO_IDX, ENV_IDS, make_env_config
)

# ── Actions ────────────────────────────────────────────────────────────────────
# DiscreteMetaAction: 0=LANE_LEFT, 1=IDLE, 2=LANE_RIGHT, 3=FASTER, 4=SLOWER
ACTION_FASTER = 3
ACTION_IDLE   = 1


def collect_episode(env, policy="faster"):
    """
    Run one episode, return (trajectory, initial_state).
    trajectory:    [T, 7]   controlled vehicle obs at each step
    initial_state: [35]     flattened 5-vehicle obs at t=0
    """
    obs, _ = env.reset()
    s0 = obs.flatten().astype(np.float32)   # [35]

    traj = []
    done = False
    step = 0

    while not done and step < N_STEPS:
        if policy == "faster":
            action = ACTION_FASTER
        else:
            action = env.action_space.sample()

        obs, reward, done, truncated, _ = env.step(action)
        traj.append(obs[0].astype(np.float32))   # controlled vehicle [7]
        done = done or truncated
        step += 1

    # Pad or truncate to exactly N_STEPS
    traj = np.stack(traj, axis=0)   # [actual_T, 7]
    if len(traj) < N_STEPS:
        pad = np.repeat(traj[-1:], N_STEPS - len(traj), axis=0)
        traj = np.concatenate([traj, pad], axis=0)
    else:
        traj = traj[:N_STEPS]

    return traj, s0


def collect_scenario(scenario, n_episodes, policy="faster", seed=42):
    env_id = ENV_IDS[scenario]
    config = make_env_config(scenario)
    env = gym.make(env_id, config=config)

    trajs, s0s = [], []
    for ep in range(n_episodes):
        env.reset(seed=seed + ep)
        traj, s0 = collect_episode(env, policy=policy)
        trajs.append(traj)
        s0s.append(s0)
        if (ep + 1) % 20 == 0:
            print(f"  {scenario}: {ep+1}/{n_episodes}")

    env.close()
    return np.stack(trajs), np.stack(s0s)


def main(
    data_dir   = "../Data",
    n_train    = 150,     # episodes per training scenario
    n_fewshot  = 5,       # few-shot demos for roundabout
    seed       = 42,
):
    os.makedirs(data_dir, exist_ok=True)
    np.random.seed(seed)

    all_X, all_s0, all_y_str, all_y_idx = [], [], [], []

    # ── Training scenarios ─────────────────────────────────────────────────
    for scenario in TRAINING_SCENARIOS:
        print(f"Collecting {n_train} episodes: {scenario}")
        X, s0 = collect_scenario(scenario, n_train, seed=seed)
        all_X.append(X)
        all_s0.append(s0)
        all_y_str.extend([scenario] * n_train)
        all_y_idx.extend([SCENARIO_TO_IDX[scenario]] * n_train)
        print(f"  done. traj shape={X.shape}")

    X_train    = np.concatenate(all_X,    axis=0).astype(np.float32)
    s0_train   = np.concatenate(all_s0,   axis=0).astype(np.float32)
    y_str      = np.array(all_y_str)
    y_idx      = np.array(all_y_idx, dtype=np.int64)

    np.save(os.path.join(data_dir, "X.npy"),     X_train)
    np.save(os.path.join(data_dir, "s0.npy"),    s0_train)
    np.save(os.path.join(data_dir, "y_str.npy"), y_str)
    np.save(os.path.join(data_dir, "y_idx.npy"), y_idx)

    # ── Few-shot: roundabout ───────────────────────────────────────────────
    print(f"\nCollecting {n_fewshot} few-shot demos: roundabout")
    X_few, s0_few = collect_scenario(NEW_SCENARIO, n_fewshot, seed=seed+999)
    np.save(os.path.join(data_dir, "fewshot_X.npy"),  X_few)
    np.save(os.path.join(data_dir, "fewshot_s0.npy"), s0_few)
    print(f"  done. shape={X_few.shape}")

    # ── Meta ──────────────────────────────────────────────────────────────
    meta = {
        "training_scenarios": TRAINING_SCENARIOS,
        "new_scenario":       NEW_SCENARIO,
        "scenario_to_idx":    SCENARIO_TO_IDX,
        "n_train_per_scenario": n_train,
        "n_fewshot":          n_fewshot,
        "n_steps":            N_STEPS,
        "obs_dim":            7,
        "s0_dim":             35,
        "total_train":        len(X_train),
    }
    with open(os.path.join(data_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nDataset saved → {data_dir}")
    print(f"  Train: {X_train.shape}  s0: {s0_train.shape}")
    print(f"  Few-shot: {X_few.shape}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",  default="../Data")
    p.add_argument("--n_train",   type=int, default=150)
    p.add_argument("--n_fewshot", type=int, default=5)
    args = p.parse_args()
    main(args.data_dir, args.n_train, args.n_fewshot)
