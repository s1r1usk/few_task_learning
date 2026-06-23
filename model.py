"""
model.py

Conditional Trajectory Generator for driving domain.

G_θ(τ | c, s0) where:
  τ  = trajectory of controlled vehicle  [N_STEPS, 7]
  c  = concept vector (scenario embedding) [concept_dim]
  s0 = initial 5-vehicle observation      [35]

Architecture:
  Encoder:  [N_STEPS * 7] → latent_dim          (trajectory → z)
  Decoder:  [latent_dim + concept_dim + 35] → [N_STEPS * 7]

During training:  learn encoder + decoder + concept embedding table
During inversion: freeze everything, optimize c̃ only

Key difference from toy version:
  - s0 is now 35-dim (real multi-vehicle context from HighwayEnv)
  - trajectory is 7-dim per step (real kinematics)
  - concept_dim=32 (larger, more expressive)
  - deeper decoder with residual connections for better generalization
"""

import torch
import torch.nn as nn
from env_config import N_STEPS, TRAJ_DIM, S0_DIM


class ResBlock(nn.Module):
    """Simple residual block for better gradient flow during concept inversion."""
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(x + self.net(x))


class ConditionalTrajModel(nn.Module):
    def __init__(
        self,
        n_concepts  = 3,        # number of training scenarios
        concept_dim = 32,       # concept vector size
        latent_dim  = 32,       # trajectory style latent
        n_steps     = N_STEPS,  # 45
        traj_dim    = TRAJ_DIM, # 7
        s0_dim      = S0_DIM,   # 35
        hidden_dim  = 512,
    ):
        super().__init__()
        self.n_steps    = n_steps
        self.traj_dim   = traj_dim
        self.latent_dim = latent_dim
        self.concept_dim = concept_dim
        self.input_dim  = n_steps * traj_dim   # 315

        # ── Concept embedding table ──────────────────────────────────────
        self.concept_embed = nn.Embedding(n_concepts, concept_dim)

        # ── Encoder ─────────────────────────────────────────────────────
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
            ResBlock(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, latent_dim),
        )

        # ── Decoder ─────────────────────────────────────────────────────
        # Input: z + c + s0
        decoder_in = latent_dim + concept_dim + s0_dim
        self.decoder = nn.Sequential(
            nn.Linear(decoder_in, hidden_dim),
            nn.ReLU(),
            ResBlock(hidden_dim),
            ResBlock(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.input_dim),
        )

    def forward(self, traj, concept_idx, s0):
        """
        traj:        [B, N_STEPS, 7]
        concept_idx: [B]
        s0:          [B, 35]
        Returns: recon [B, N_STEPS, 7], z [B, latent_dim], c [B, concept_dim]
        """
        B = traj.size(0)
        x = traj.view(B, -1)
        z = self.encoder(x)
        c = self.concept_embed(concept_idx)
        recon = self._decode(z, c, s0)
        return recon, z, c

    def _decode(self, z, c, s0):
        inp = torch.cat([z, c, s0], dim=-1)
        out = self.decoder(inp)
        return out.view(-1, self.n_steps, self.traj_dim)

    def decode_from_concept(self, c_tilde, s0, z=None):
        """G_θ(· | c̃, s0) — used during inverse learning and generation."""
        if c_tilde.dim() == 1:
            c_tilde = c_tilde.unsqueeze(0)
        if s0.dim() == 1:
            s0 = s0.unsqueeze(0)
        B = max(c_tilde.size(0), s0.size(0))
        if c_tilde.size(0) == 1:
            c_tilde = c_tilde.expand(B, -1)
        if s0.size(0) == 1:
            s0 = s0.expand(B, -1)
        if z is None:
            z = torch.zeros(B, self.latent_dim, device=c_tilde.device)
        return self._decode(z, c_tilde, s0)

    def decode_composed(self, c1, c2, w1, w2, s0, z=None):
        """Concept composition: c_composed = w1*c1 + w2*c2."""
        c_composed = w1 * c1 + w2 * c2
        return self.decode_from_concept(c_composed, s0, z=z)
