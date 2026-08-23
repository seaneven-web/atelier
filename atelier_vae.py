#!/usr/bin/env python3
"""
The apprentice's own eye — a variational autoencoder trained on the artist's pieces, locally.

Adapted from the CelebA face-generation VAE in Czajka-Teaching/semester-project-benjaminlyons
(blyons1 / seven). What that project established, and what is carried over here:

  * plain VAE beat the GAN they also built — no adversarial game, just reconstruction + KL;
  * reconstruction must DOMINATE: BCE summed over pixels, KL summed, β≈1 — with mean-reduction
    the KL swamps the reconstruction and everything goes blurry;
  * BatchNorm + Dropout(0.25) + LeakyReLU(0.2) in every block;
  * deterministic at inference — decode μ, never a sample (sampling noise is blur);
  * consistently framed inputs (their Haar-cascade face crops, all 100×100) — the model only
    learns a crisp vocabulary if every training view is framed the same way. Here that means
    same-size patches taken at the artist's NATIVE brush scale;
  * hold out a validation split and keep the best-scoring checkpoint, not the last.

What it is used for here: not to generate the picture (a VAE trained on a handful of pieces
would only blur it), but as a *learned prior on the artist's own marks*. During style transfer
the styled image is asked to be something this VAE can reconstruct — "would the apprentice
recognise this as one of my brushstrokes?". VGG's deep layers know ImageNet objects, which is
what hallucinates faces into a prairie; this term knows only the artist's paint.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import atelier as A

PATCH = 64          # every training view is framed identically (their 100×100 face crops)
LATENT = 256        # theirs
DROP = 0.25         # theirs
LEAK = 0.2          # theirs
BETA = 1.0          # theirs (with SUM reduction — reconstruction dominates)


def _block(cin: int, cout: int, down: bool = True) -> nn.Sequential:
    conv = nn.Conv2d(cin, cout, 5, 2, 2) if down else nn.ConvTranspose2d(cin, cout, 4, 2, 1)
    return nn.Sequential(conv, nn.BatchNorm2d(cout), nn.Dropout(DROP), nn.LeakyReLU(LEAK, inplace=True))


class Encoder(nn.Module):
    def __init__(self, base: int = 64):
        super().__init__()
        c = [base, base * 2, base * 4, base * 8]                    # 64→32→16→8→4
        self.conv = nn.Sequential(_block(3, c[0]), _block(c[0], c[1]), _block(c[1], c[2]), _block(c[2], c[3]))
        self.flat = c[3] * 4 * 4
        self.fc_mu = nn.Linear(self.flat, LATENT)
        self.fc_logvar = nn.Linear(self.flat, LATENT)

    def forward(self, x):
        h = self.conv(x).flatten(1)
        return self.fc_mu(h), self.fc_logvar(h).clamp(-8, 6)


class Decoder(nn.Module):
    def __init__(self, base: int = 64):
        super().__init__()
        c = [base * 8, base * 4, base * 2, base]
        self.c0 = c[0]
        self.fc = nn.Sequential(nn.Linear(LATENT, c[0] * 4 * 4), nn.LeakyReLU(LEAK, inplace=True))
        self.conv = nn.Sequential(_block(c[0], c[1], False), _block(c[1], c[2], False), _block(c[2], c[3], False),
                                  nn.ConvTranspose2d(c[3], 3, 4, 2, 1), nn.Sigmoid())

    def forward(self, z):
        return self.conv(self.fc(z).view(-1, self.c0, 4, 4))


class ArtistVAE(nn.Module):
    def __init__(self, base: int = 64):
        super().__init__()
        self.encoder, self.decoder = Encoder(base), Decoder(base)

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = mu + torch.randn_like(mu) * (0.5 * logvar).exp() if self.training else mu   # μ at inference
        return self.decoder(z), mu, logvar


def vae_loss(out: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Their loss exactly: BCE summed over pixels + β·KL summed. Reconstruction dominates."""
    bce = F.binary_cross_entropy(out.clamp(1e-6, 1 - 1e-6), x, reduction="sum")
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return (bce + BETA * kld) / x.shape[0]


# --------------------------------------------------------------------------- patches (consistent framing)
def patch_bank(files: List[Path], device: torch.device, per_piece: int = 400, seed: int = 0,
               scale: float = 1.0) -> torch.Tensor:
    """N×3×PATCH×PATCH in [0,1], sampled at the artist's native brush scale — every view framed
    the same way, which is what made their face crops learnable."""
    rng = random.Random(seed)
    out = []
    for f in files:
        try:
            t = A.load_image(f, int(900 * scale))            # keep native-ish resolution
        except Exception:
            continue
        _, H, W = t.shape
        if min(H, W) < PATCH: continue
        for _ in range(per_piece):
            y, x = rng.randint(0, H - PATCH), rng.randint(0, W - PATCH)
            p = t[:, y:y + PATCH, x:x + PATCH]
            if rng.random() < 0.5: p = p.flip(-1)
            out.append(p)
    if not out:
        raise RuntimeError("no usable patches in this portfolio")
    return torch.stack(out).float().div(255)


# --------------------------------------------------------------------------- training
def model_path(files: List[Path]) -> Path:
    sig = hashlib.md5(("|".join(f"{f}:{f.stat().st_mtime_ns}" for f in files) + f"|p{PATCH}l{LATENT}").encode()).hexdigest()[:12]
    return A.MODEL_DIR / f"artistvae_{sig}.pt"


def train(files: List[Path], device: torch.device, minutes: float = 4.0, steps: int = 1200,
          batch: int = 48, base: int = 64, log=None, seed: int = 0) -> Tuple[ArtistVAE, dict]:
    """Train the artist's VAE on patches. Keeps the best-by-validation weights (theirs), not the last."""
    log = log or (lambda s: A.say(A.dim("    " + s)))
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    bank = patch_bank(files, device, seed=seed)
    n_val = max(64, int(len(bank) * 0.1))
    perm = torch.randperm(len(bank), generator=torch.Generator().manual_seed(seed))
    val_x = bank[perm[:n_val]][:256].to(device)
    train_idx = perm[n_val:].tolist()
    log(f"{len(bank)} patches ({PATCH}px) from {len(files)} piece(s) · {len(train_idx)} train / {n_val} held out")

    model = ArtistVAE(base).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    best, best_state, best_step, t0 = float("inf"), None, 0, time.time()
    for step in range(1, steps + 1):
        model.train()
        idx = [train_idx[random.randrange(len(train_idx))] for _ in range(batch)]
        x = bank[idx].to(device)
        out, mu, logvar = model(x)
        loss = vae_loss(out, x, mu, logvar)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if step % 100 == 0 or step == steps:
            model.eval()
            with torch.no_grad():
                o, m, lv = model(val_x)
                vl = float(vae_loss(o, val_x, m, lv))
                rec = float(F.l1_loss(o, val_x))
            if vl < best - 1e-4:
                best, best_step = vl, step
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            log(f"step {step}/{steps}  train {float(loss):.0f}  held-out {vl:.0f}  pixel err {rec:.3f}"
                f"{'  ✔ best' if best_step == step else ''}")
            if time.time() - t0 > minutes * 60:
                log(f"time budget reached at step {step}"); break
    if best_state is not None: model.load_state_dict(best_state)
    model.eval()
    meta = {"best_val": best, "best_step": best_step, "steps": step, "seconds": time.time() - t0,
            "patches": len(bank), "pieces": [str(f) for f in files], "base": base, "patch": PATCH, "latent": LATENT}
    return model, meta


def load_or_train(files: List[Path], device: torch.device, minutes: float = 4.0, log=None,
                  force: bool = False) -> Tuple[Optional[ArtistVAE], dict]:
    p = model_path(files)
    if p.exists() and not force:
        d = torch.load(p, map_location="cpu")
        model = ArtistVAE(d["meta"].get("base", 64))
        model.load_state_dict(d["state"]); model.to(device).eval()
        return model, d["meta"]
    model, meta = train(files, device, minutes=minutes, log=log)
    torch.save({"state": {k: v.cpu() for k, v in model.state_dict().items()}, "meta": meta}, p)
    p.with_suffix(".json").write_text(json.dumps(meta))     # sidecar: status without loading 50 MB
    return model, meta


def find_trained(files: List[Path], device: torch.device) -> Tuple[Optional[ArtistVAE], dict]:
    """The trained eye for this exact portfolio, if there is one. Never trains."""
    p = model_path(files)
    if not p.exists(): return None, {}
    d = torch.load(p, map_location="cpu")
    model = ArtistVAE(d["meta"].get("base", 64))
    model.load_state_dict(d["state"]); model.to(device).eval()
    for q in model.parameters(): q.requires_grad_(False)
    return model, d["meta"]


# --------------------------------------------------------------------------- use during painting
def prior_loss(model: ArtistVAE, x: torch.Tensor, tiles: int = 4, generator=None) -> torch.Tensor:
    """“Could the apprentice repaint this from its own vocabulary?” — reconstruction error of random
    PATCH-sized tiles of x under the artist's VAE. Differentiable; per-image."""
    B, _, H, W = x.shape
    if min(H, W) < PATCH: return torch.zeros(B, device=x.device)
    loss = torch.zeros(B, device=x.device)
    for _ in range(tiles):
        y0 = int(torch.randint(0, H - PATCH + 1, (1,), generator=generator))
        x0 = int(torch.randint(0, W - PATCH + 1, (1,), generator=generator))
        tile = x[:, :, y0:y0 + PATCH, x0:x0 + PATCH]
        mu, _ = model.encoder(tile)
        rec = model.decoder(mu)
        loss = loss + (rec - tile).abs().flatten(1).mean(1)
    return loss / tiles
