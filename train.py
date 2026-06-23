"""
train.py

Pretraining phase: learn G_θ(τ | c, s0) on real HighwayEnv trajectories.

Training scenarios: highway, merge, intersection
The model never sees roundabout during training.
"""

import os, json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from model import ConditionalTrajModel


def load_data(data_dir):
    X     = np.load(os.path.join(data_dir, "X.npy")).astype(np.float32)
    s0    = np.load(os.path.join(data_dir, "s0.npy")).astype(np.float32)
    y_idx = np.load(os.path.join(data_dir, "y_idx.npy")).astype(np.int64)
    with open(os.path.join(data_dir, "meta.json")) as f:
        meta = json.load(f)
    return X, s0, y_idx, meta


def normalize(X, s0):
    X_mean  = X.mean(axis=(0, 1))
    X_std   = X.std(axis=(0, 1))  + 1e-8
    s0_mean = s0.mean(axis=0)
    s0_std  = s0.std(axis=0)  + 1e-8
    return (X - X_mean) / X_std, (s0 - s0_mean) / s0_std, X_mean, X_std, s0_mean, s0_std


def main(
    data_dir    = "../Data",
    models_dir  = "../Models",
    concept_dim = 32,
    latent_dim  = 32,
    hidden_dim  = 512,
    batch_size  = 64,
    epochs      = 80,
    lr          = 1e-3,
):
    os.makedirs(models_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    X, s0, y_idx, meta = load_data(data_dir)
    n_concepts = len(meta["training_scenarios"])

    X_n, s0_n, X_mean, X_std, s0_mean, s0_std = normalize(X, s0)

    # Save normalization stats
    np.save(os.path.join(models_dir, "X_mean.npy"),  X_mean)
    np.save(os.path.join(models_dir, "X_std.npy"),   X_std)
    np.save(os.path.join(models_dir, "s0_mean.npy"), s0_mean)
    np.save(os.path.join(models_dir, "s0_std.npy"),  s0_std)
    with open(os.path.join(models_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    dataset = TensorDataset(
        torch.from_numpy(X_n),
        torch.from_numpy(y_idx),
        torch.from_numpy(s0_n),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = ConditionalTrajModel(
        n_concepts  = n_concepts,
        concept_dim = concept_dim,
        latent_dim  = latent_dim,
        hidden_dim  = hidden_dim,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {total_params:,}")

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()

    best_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for traj_b, label_b, s0_b in loader:
            traj_b  = traj_b.to(device)
            label_b = label_b.to(device)
            s0_b    = s0_b.to(device)

            optimizer.zero_grad()
            recon, z, c = model(traj_b, label_b, s0_b)
            loss = criterion(recon, traj_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * traj_b.size(0)

        scheduler.step()
        avg = total_loss / len(loader.dataset)

        if avg < best_loss:
            best_loss = avg
            torch.save(model.state_dict(), os.path.join(models_dir, "model_best.pth"))

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:>3}/{epochs}  loss={avg:.6f}  best={best_loss:.6f}")

    torch.save(model.state_dict(), os.path.join(models_dir, "model.pth"))
    print(f"\nSaved → {models_dir}/model.pth  (best loss={best_loss:.6f})")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",   default="../Data")
    p.add_argument("--models_dir", default="../Models")
    p.add_argument("--epochs",     type=int,   default=80)
    p.add_argument("--lr",         type=float, default=1e-3)
    args = p.parse_args()
    main(data_dir=args.data_dir, models_dir=args.models_dir,
         epochs=args.epochs, lr=args.lr)
