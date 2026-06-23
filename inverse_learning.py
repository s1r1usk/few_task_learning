"""
inverse_learning.py

FTL-IGM on HighwayEnv driving domain.

Core steps (all matching the paper):
  1. Invert concept c̃ from 5 roundabout demonstrations
  2. Generate trajectories with demo initial states
  3. Generate trajectories with novel initial states  (generalization test)
  4. Compose c̃ with training concepts
  5. Closed-loop rollout in the actual HighwayEnv environment
  6. t-SNE visualization of concept space (Figure 11 in paper)
"""

import os, json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.manifold import TSNE
import gymnasium as gym
import highway_env  # noqa

from model import ConditionalTrajModel
from env_config import (
    N_STEPS, TRAJ_DIM, S0_DIM,
    TRAINING_SCENARIOS, NEW_SCENARIO,
    SCENARIO_TO_IDX, ENV_IDS, make_env_config
)


# ─── Load frozen model ────────────────────────────────────────────────────────

def load_frozen_model(models_dir, device):
    with open(os.path.join(models_dir, "meta.json")) as f:
        meta = json.load(f)

    model = ConditionalTrajModel(
        n_concepts  = len(meta["training_scenarios"]),
        concept_dim = 32,
        latent_dim  = 32,
        hidden_dim  = 512,
    )
    ckpt = os.path.join(models_dir, "model_best.pth")
    if not os.path.exists(ckpt):
        ckpt = os.path.join(models_dir, "model.pth")
    model.load_state_dict(torch.load(ckpt, map_location=device))

    for p in model.parameters():
        p.requires_grad = False
    model.eval().to(device)

    norms = {
        "X_mean":  np.load(os.path.join(models_dir, "X_mean.npy")).astype(np.float32),
        "X_std":   np.load(os.path.join(models_dir, "X_std.npy")).astype(np.float32),
        "s0_mean": np.load(os.path.join(models_dir, "s0_mean.npy")).astype(np.float32),
        "s0_std":  np.load(os.path.join(models_dir, "s0_std.npy")).astype(np.float32),
    }
    return model, meta, norms


def norm_traj(X, norms):
    return (X - norms["X_mean"]) / norms["X_std"]

def norm_s0(s0, norms):
    return (s0 - norms["s0_mean"]) / norms["s0_std"]

def denorm_traj(X, norms):
    return X * norms["X_std"] + norms["X_mean"]


# ─── Concept inversion (core paper contribution) ──────────────────────────────

def invert_concept(demos, demo_s0s, model, norms, device,
                   concept_dim=32, steps=1000, lr=0.02):
    """
    Optimize c̃ so frozen G_θ(· | c̃, s0_k) ≈ τ̃_k for all k.

    Two-concept version (paper Section 5.3 shows this works better):
      c̃ = w1*c̃1 + w2*c̃2
    We learn c̃1, c̃2, w1, w2 jointly.
    """
    K = demos.shape[0]
    demos_n   = norm_traj(demos,    norms)
    demo_s0_n = norm_s0(demo_s0s,   norms)

    demos_t   = torch.from_numpy(demos_n).to(device)
    demo_s0_t = torch.from_numpy(demo_s0_n).to(device)

    # Two concept components + their weights (paper Eq. 2)
    c1 = torch.randn(concept_dim, device=device) * 0.1
    c2 = torch.randn(concept_dim, device=device) * 0.1
    w1 = torch.tensor(1.0, device=device)
    w2 = torch.tensor(1.0, device=device)

    c1.requires_grad_(True)
    c2.requires_grad_(True)
    w1.requires_grad_(True)
    w2.requires_grad_(True)

    optimizer = optim.Adam([c1, c2, w1, w2], lr=lr)
    mse = nn.MSELoss()
    losses = []

    for step in range(steps):
        optimizer.zero_grad()
        c_composed = w1 * c1 + w2 * c2   # composed concept

        loss = torch.tensor(0.0, device=device)
        for k in range(K):
            pred = model.decode_from_concept(c_composed, demo_s0_t[k])
            loss = loss + mse(pred[0], demos_t[k])
        loss = loss / K
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if (step + 1) % (steps // 5) == 0:
            print(f"  step {step+1:>4}/{steps}  loss={loss.item():.6f}")

    c_tilde = (w1 * c1 + w2 * c2).detach().cpu()
    c1_out  = c1.detach().cpu()
    c2_out  = c2.detach().cpu()
    return c_tilde, c1_out, c2_out, losses


# ─── Closed-loop evaluation ───────────────────────────────────────────────────

def closed_loop_rollout(model, c_tilde, s0_raw, norms, device,
                        scenario=NEW_SCENARIO, max_steps=N_STEPS):
    """
    Run the model in closed-loop inside the actual HighwayEnv environment.

    At each step:
      1. Get current observation from env
      2. Feed (c̃, current_s0) into frozen model → predicted next state
      3. Execute the action that moves closest to predicted next state
         (simplified: use FASTER if predicted vx > current vx, else IDLE)
      4. Repeat

    This mirrors the paper's closed-loop evaluation (Section 4.3).

    Returns: (trajectory, crashed, success)
    """
    env_id = ENV_IDS[scenario]
    config = make_env_config(scenario)
    env = gym.make(env_id, config=config)
    obs, _ = env.reset()

    traj_real = []
    crashed   = False
    success   = False

    for step in range(max_steps):
        # Current full obs as s0 context
        s0_current = obs.flatten().astype(np.float32)
        s0_n = norm_s0(s0_current, norms)
        s0_t = torch.from_numpy(s0_n).to(device)

        # Model predicts next state
        with torch.no_grad():
            pred = model.decode_from_concept(c_tilde.to(device), s0_t)
        pred_next = pred[0, 0].cpu().numpy()   # [7] — predicted next vehicle state

        # Simple action selection: compare predicted vx to current vx
        current_vx = obs[0, 2]   # controlled vehicle vx
        pred_vx    = pred_next[2]

        if pred_vx > current_vx + 0.05:
            action = 3   # FASTER
        elif pred_vx < current_vx - 0.05:
            action = 4   # SLOWER
        else:
            action = 1   # IDLE

        obs, reward, done, truncated, info = env.step(action)
        traj_real.append(obs[0].copy())

        if info.get("crashed", False):
            crashed = True
            break
        if done or truncated:
            success = True
            break

    env.close()
    return np.stack(traj_real), crashed, success


# ─── t-SNE visualization ──────────────────────────────────────────────────────

def plot_tsne(model, c_tilde, results_dir, device):
    """
    Visualize learned concept c̃ relative to training concept embeddings.
    Mirrors Figure 11 in the paper.
    """
    # Get all training concept embeddings
    training_concepts = []
    labels = []
    for scenario, idx in SCENARIO_TO_IDX.items():
        c = model.concept_embed(torch.tensor([idx])).squeeze(0).detach().cpu().numpy()
        training_concepts.append(c)
        labels.append(scenario)

    all_vecs = np.stack(training_concepts + [c_tilde.numpy()], axis=0)
    all_labels = labels + ["roundabout (learned)"]

    # t-SNE needs more points — perturb training concepts slightly to get spread
    perturbed = []
    perturb_labels = []
    for i, (vec, label) in enumerate(zip(training_concepts, labels)):
        for _ in range(20):
            perturbed.append(vec + np.random.normal(0, 0.1, size=vec.shape))
            perturb_labels.append(label)

    all_vecs_full   = np.stack(perturbed + [c_tilde.numpy()], axis=0)
    all_labels_full = perturb_labels + ["roundabout (learned)"]

    tsne = TSNE(n_components=2, perplexity=10, random_state=42, max_iter=1000)
    embedded = tsne.fit_transform(all_vecs_full.astype(np.float32))

    colors = {"highway": "steelblue", "merge": "tomato",
              "intersection": "seagreen", "roundabout (learned)": "gold"}
    markers = {"highway": "o", "merge": "s",
               "intersection": "^", "roundabout (learned)": "*"}

    fig, ax = plt.subplots(figsize=(7, 6))
    for label in set(all_labels_full):
        mask = [l == label for l in all_labels_full]
        pts  = embedded[mask]
        size = 200 if label == "roundabout (learned)" else 40
        ax.scatter(pts[:, 0], pts[:, 1],
                   label=label, color=colors[label],
                   marker=markers[label], s=size,
                   edgecolors="black" if label == "roundabout (learned)" else "none",
                   linewidths=1.0, alpha=0.85, zorder=5 if label == "roundabout (learned)" else 3)

    ax.set_title("t-SNE: Learned concept c̃ vs training concepts\n(mirrors Figure 11 in FTL-IGM paper)", fontsize=11)
    ax.legend(fontsize=9)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "tsne_concepts.png"), dpi=150)
    plt.close()
    print(f"  → tsne_concepts.png")


# ─── Trajectory visualization ─────────────────────────────────────────────────

def plot_traj_xy(ax, traj, title="", color="steelblue", alpha=1.0):
    """Plot x-y position from trajectory."""
    ax.plot(traj[:, 0], traj[:, 1], color=color, linewidth=1.5, alpha=alpha)
    ax.scatter(traj[0, 0],  traj[0, 1],  color="green", s=50, zorder=5)
    ax.scatter(traj[-1, 0], traj[-1, 1], color="red",   s=50, zorder=5)
    ax.set_title(title, fontsize=9)
    ax.set_aspect("equal")
    ax.axis("off")


def plot_loss_curve(losses, results_dir):
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(losses, color="steelblue", linewidth=1.2)
    ax.set_xlabel("Optimization step")
    ax.set_ylabel("MSE loss")
    ax.set_title("Concept inversion loss curve")
    ax.set_yscale("log")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "inversion_loss.png"), dpi=120)
    plt.close()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main(
    data_dir    = "../Data",
    models_dir  = "../Models",
    results_dir = "../Results",
    steps       = 1000,
    lr          = 0.02,
    n_closed_loop = 10,
):
    os.makedirs(results_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, meta, norms = load_frozen_model(models_dir, device)
    demos    = np.load(os.path.join(data_dir, "fewshot_X.npy")).astype(np.float32)
    demo_s0s = np.load(os.path.join(data_dir, "fewshot_s0.npy")).astype(np.float32)
    K = demos.shape[0]
    print(f"Loaded {K} roundabout demonstrations. Shape: {demos.shape}")

    # ── 1. Invert concept ─────────────────────────────────────────────────
    print("\n── Step 1: Inverting concept c̃ ──")
    c_tilde, c1, c2, losses = invert_concept(
        demos, demo_s0s, model, norms, device, steps=steps, lr=lr
    )
    np.save(os.path.join(results_dir, "c_tilde.npy"), c_tilde.numpy())
    plot_loss_curve(losses, results_dir)
    print(f"  Final loss: {losses[-1]:.6f}")

    # ── 2. Generate with demo s0s ─────────────────────────────────────────
    print("\n── Step 2: Decode with demonstrated initial states ──")
    fig, axes = plt.subplots(2, K, figsize=(3 * K, 6))
    fig.suptitle("Decoded trajectories vs roundabout demonstrations\n(x-y position of controlled vehicle)", fontsize=11)
    for k in range(K):
        s0_n = norm_s0(demo_s0s[k], norms)
        s0_t = torch.from_numpy(s0_n).to(device)
        with torch.no_grad():
            pred = model.decode_from_concept(c_tilde.to(device), s0_t)
        pred_traj = denorm_traj(pred.cpu().numpy()[0], norms)

        plot_traj_xy(axes[0, k], demos[k],    title=f"Demo {k+1}",      color="steelblue")
        plot_traj_xy(axes[1, k], pred_traj,   title=f"Generated {k+1}", color="tomato")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "step2_demo_s0.png"), dpi=130)
    plt.close()
    print(f"  → step2_demo_s0.png")

    # ── 3. Generate with novel s0s ────────────────────────────────────────
    print("\n── Step 3: Novel initial states (generalization test) ──")
    # Load some highway s0s as "novel" — very different distribution from roundabout
    X_train  = np.load(os.path.join(data_dir, "X.npy")).astype(np.float32)
    s0_train = np.load(os.path.join(data_dir, "s0.npy")).astype(np.float32)
    y_idx    = np.load(os.path.join(data_dir, "y_idx.npy"))
    highway_mask = y_idx == SCENARIO_TO_IDX["highway"]
    novel_s0s = s0_train[highway_mask][:4]   # highway s0s — never seen in roundabout demos

    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    fig.suptitle("Step 3 — Roundabout concept applied to novel (highway) initial states", fontsize=11)
    for i, s0 in enumerate(novel_s0s):
        s0_n = norm_s0(s0, norms)
        s0_t = torch.from_numpy(s0_n).to(device)
        with torch.no_grad():
            pred = model.decode_from_concept(c_tilde.to(device), s0_t)
        pred_traj = denorm_traj(pred.cpu().numpy()[0], norms)
        plot_traj_xy(axes[i], pred_traj, title=f"Novel s0 {i+1}", color="seagreen")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "step3_novel_s0.png"), dpi=130)
    plt.close()
    print(f"  → step3_novel_s0.png")

    # ── 4. Concept composition ────────────────────────────────────────────
    print("\n── Step 4: Compose roundabout c̃ with training concepts ──")
    fig, axes = plt.subplots(1, len(TRAINING_SCENARIOS), figsize=(4 * len(TRAINING_SCENARIOS), 3))
    fig.suptitle("Step 4 — Roundabout concept composed with training scenario concepts", fontsize=11)
    s0_compose = demo_s0s[0]
    s0_n = norm_s0(s0_compose, norms)
    s0_t = torch.from_numpy(s0_n).to(device)

    for ax, scenario in zip(axes, TRAINING_SCENARIOS):
        cidx   = SCENARIO_TO_IDX[scenario]
        c_known = model.concept_embed(torch.tensor([cidx])).squeeze(0).detach()
        with torch.no_grad():
            pred = model.decode_composed(c_tilde.to(device), c_known.to(device),
                                         w1=1.0, w2=0.7, s0=s0_t)
        pred_traj = denorm_traj(pred.cpu().numpy()[0], norms)
        plot_traj_xy(ax, pred_traj, title=f"roundabout + {scenario}", color="darkorchid")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "step4_composition.png"), dpi=130)
    plt.close()
    print(f"  → step4_composition.png")

    # ── 5. Closed-loop evaluation ─────────────────────────────────────────
    print(f"\n── Step 5: Closed-loop rollout in HighwayEnv ({n_closed_loop} episodes) ──")
    crashes   = 0
    successes = 0
    all_trajs = []
    for ep in range(n_closed_loop):
        traj_real, crashed, success = closed_loop_rollout(
            model, c_tilde, demo_s0s[0], norms, device
        )
        crashes   += int(crashed)
        successes += int(success)
        all_trajs.append(traj_real)
        print(f"  ep {ep+1:>2}: {'CRASH' if crashed else 'success' if success else 'timeout'}")

    crash_rate   = crashes   / n_closed_loop * 100
    success_rate = successes / n_closed_loop * 100
    print(f"\n  Crash rate:   {crash_rate:.0f}%")
    print(f"  Success rate: {success_rate:.0f}%")

    # Plot closed-loop trajectories
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    fig.suptitle(f"Step 5 — Closed-loop rollouts in roundabout env\n"
                 f"Crash rate: {crash_rate:.0f}%  Success rate: {success_rate:.0f}%", fontsize=11)
    for i, (ax, traj) in enumerate(zip(axes.flatten(), all_trajs)):
        plot_traj_xy(ax, traj, title=f"ep {i+1}", color="steelblue")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "step5_closed_loop.png"), dpi=130)
    plt.close()
    print(f"  → step5_closed_loop.png")

    # ── 6. t-SNE ─────────────────────────────────────────────────────────
    print("\n── Step 6: t-SNE concept visualization ──")
    plot_tsne(model, c_tilde, results_dir, device)

    # ── Summary ──────────────────────────────────────────────────────────
    results = {
        "crash_rate":   crash_rate,
        "success_rate": success_rate,
        "final_inversion_loss": losses[-1],
        "n_closed_loop_episodes": n_closed_loop,
    }
    with open(os.path.join(results_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nAll results saved → {results_dir}/")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",       default="../Data")
    p.add_argument("--models_dir",     default="../Models")
    p.add_argument("--results_dir",    default="../Results")
    p.add_argument("--steps",          type=int,   default=1000)
    p.add_argument("--lr",             type=float, default=0.02)
    p.add_argument("--n_closed_loop",  type=int,   default=10)
    args = p.parse_args()
    main(args.data_dir, args.models_dir, args.results_dir,
         args.steps, args.lr, args.n_closed_loop)
