#!/usr/bin/env python3
"""
Atelier — learn one artist's style from a folder of JPGs, then paint new work
from plain-English prompts. Everything runs on this machine. No cloud, no
large language model: the "prompt engine" is a small, transparent optimizer
that turns words into moves through the learned style space.

Quick start
    python atelier.py                      # guided: trains if needed, then opens the studio
    python atelier.py train ~/DadsArt      # train on a folder of jpg/png (recursively)
    python atelier.py studio               # text studio
    python atelier.py paint "warm quiet harbour at dusk" --count 4

How it learns (the actor / critic pair)
    actor   = variational autoencoder: encoder squeezes a painting into a latent
              code, decoder paints it back. LeakyReLU everywhere. Cost = distance
              between the original and the repainting (pixels + critic features)
              plus a KL term that keeps the latent space smooth enough to sample.
    critic  = convolutional network that must tell the artist's originals from
              the actor's repaintings and dreams. Trained against the actor in a
              minimax game (BCE / non-saturating generator loss).
    ratchet = the two are never allowed to run away from each other: the critic
              sits out a step when it is winning too easily, the adversarial
              pressure on the actor is only applied once the critic is useful,
              and only checkpoints that improve the held-out score are kept.

Python 3.9+, torch, torchvision, numpy, pillow.  (tqdm optional)
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import random
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageOps

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:  # pragma: no cover
    sys.exit("PyTorch is missing. Run:  pip install -r requirements.txt")

# --------------------------------------------------------------------------- paths
HOME = Path(os.environ.get("ATELIER_HOME", Path(__file__).resolve().parent))
MODEL_DIR = HOME / "model"
GALLERY = HOME / "gallery"
PROGRESS = MODEL_DIR / "progress"
LATEST = MODEL_DIR / "latest.pt"
BEST = MODEL_DIR / "best.pt"
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".heic", ".heif"}
try:  # iPhone photos (HEIC/HEIF) — optional:  pip install pillow-heif
    import pillow_heif; pillow_heif.register_heif_opener(); HEIC_OK = True
except ImportError:
    HEIC_OK = False

# --------------------------------------------------------------------------- tiny terminal helpers
def _c(code: str, s: str) -> str:
    return f"\033[{code}m{s}\033[0m" if sys.stdout.isatty() else s

def bold(s): return _c("1", s)
def dim(s): return _c("2", s)
def green(s): return _c("32", s)
def yellow(s): return _c("33", s)
def cyan(s): return _c("36", s)
def red(s): return _c("31", s)

def say(msg: str = "") -> None:
    print(msg, flush=True)

def open_file(path: Path) -> None:
    """Open an image / folder with the OS viewer. Silent on failure."""
    try:
        if platform.system() == "Darwin":
            subprocess.Popen(["open", str(path)])
        elif platform.system() == "Windows":
            os.startfile(str(path))  # type: ignore[attr-defined]
        else:
            subprocess.Popen(["xdg-open", str(path)])
    except Exception:
        pass

def pick_device() -> torch.device:
    forced = os.environ.get("ATELIER_DEVICE")
    if forced:
        return torch.device(forced)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def fmt_secs(s: float) -> str:
    s = int(max(0, s))
    if s < 60: return f"{s}s"
    if s < 3600: return f"{s // 60}m{s % 60:02d}s"
    return f"{s // 3600}h{(s % 3600) // 60:02d}m"

# --------------------------------------------------------------------------- portfolio
STOP_TAGS = {
    "img", "image", "dsc", "dscn", "scan", "scanned", "final", "copy", "photo", "pic", "picture",
    "jpg", "jpeg", "png", "edit", "edited", "new", "old", "untitled", "art", "painting", "drawing",
    "work", "the", "and", "with", "for", "small", "large", "full", "crop", "cropped", "version",
    "low", "high", "res", "web", "print", "file", "done", "finished", "detail", "sketch",
}

def find_images(folder: Path) -> List[Path]:
    files = [p for p in sorted(folder.rglob("*")) if p.suffix.lower() in IMAGE_EXT and not p.name.startswith(".")]
    return files

def tags_for(path: Path, root: Path) -> List[str]:
    """Words hidden in the file name and sub-folder names become style tags."""
    parts = [path.stem] + [p.name for p in path.relative_to(root).parents if p.name]
    words = set()
    for part in parts:
        for w in re.split(r"[^a-zA-Z]+", part):
            w = w.lower()
            if len(w) >= 3 and w not in STOP_TAGS and not w.isdigit():
                words.add(w)
    return sorted(words)

def load_image(path: Path, short_side: int) -> torch.Tensor:
    """Load as uint8 CHW tensor, shorter side == short_side, aspect kept (capped at 2:1)."""
    img = Image.open(path)
    img.draft("RGB", (short_side * 2, short_side * 2))  # fast JPEG decode
    img = ImageOps.exif_transpose(img).convert("RGB")
    w, h = img.size
    scale = short_side / min(w, h)
    nw, nh = max(short_side, round(w * scale)), max(short_side, round(h * scale))
    img = img.resize((nw, nh), Image.LANCZOS)
    # cap extreme aspect ratios so crops still see most of the piece
    if nw > 2 * short_side:
        left = (nw - 2 * short_side) // 2; img = img.crop((left, 0, left + 2 * short_side, nh))
    if nh > 2 * short_side:
        top = (nh - 2 * short_side) // 2; img = img.crop((0, top, nw, top + 2 * short_side))
    arr = torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1).contiguous()
    return arr

class Portfolio:
    """All the artist's pieces in memory (uint8), with random-crop augmentation."""

    def __init__(self, root: Path, height: int, width: int, quiet: bool = False):
        self.root = Path(root).expanduser().resolve()
        self.h, self.w = height, width
        self.aspect = width / height
        self.store = int(max(height, width) * 1.25)
        self.files = find_images(self.root)
        heic = [f for f in self.files if f.suffix.lower() in (".heic", ".heif")]
        if heic and not HEIC_OK:
            say(yellow(f"  {len(heic)} HEIC/HEIF photo(s) found but the decoder is missing — run:  pip install pillow-heif   (skipping them)"))
            self.files = [f for f in self.files if f not in heic]
        if len(self.files) < 4:
            raise SystemExit(f"Need at least 4 images; found {len(self.files)} in {self.root}")
        self.images: List[torch.Tensor] = []
        kept: List[Path] = []
        t0 = time.time()
        for i, f in enumerate(self.files):
            try:
                self.images.append(load_image(f, self.store)); kept.append(f)
            except Exception as e:  # unreadable file: skip it
                say(yellow(f"  skipping {f.name}: {e}"))
            if not quiet and (i + 1) % 25 == 0:
                say(dim(f"  loaded {i + 1}/{len(self.files)} pieces…"))
        self.files = kept
        self.tags = [tags_for(f, self.root) for f in self.files]
        if not quiet:
            say(dim(f"  {len(self.images)} pieces ready in {fmt_secs(time.time() - t0)}"))

    def __len__(self): return len(self.images)

    def crop(self, img: torch.Tensor, train: bool) -> torch.Tensor:
        """Random canvas-shaped crop (train) or centred crop (eval), resized to the canvas, float in [0,1]."""
        _, h, w = img.shape
        ch_max = min(h, int(w / self.aspect))                 # tallest crop with the canvas aspect that fits
        ch = random.randint(int(ch_max * 0.72), ch_max) if train else ch_max
        cw = max(1, min(w, int(round(ch * self.aspect))))
        if train:
            top = random.randint(0, h - ch); left = random.randint(0, w - cw)
        else:
            top = (h - ch) // 2; left = (w - cw) // 2
        patch = img[:, top:top + ch, left:left + cw].float().div_(255.0).unsqueeze(0)
        patch = F.interpolate(patch, size=(self.h, self.w), mode="bilinear", align_corners=False, antialias=True)
        if train and random.random() < 0.5:
            patch = patch.flip(-1)
        return patch[0]

    def batch(self, idx: List[int], train: bool) -> torch.Tensor:
        return torch.stack([self.crop(self.images[i], train) for i in idx])

# --------------------------------------------------------------------------- image attributes (what the prompt engine steers)
ATTRS = ["brightness", "contrast", "saturation", "warmth", "detail", "red", "yellow", "green", "blue", "purple"]

def image_attributes(x: torch.Tensor) -> torch.Tensor:
    """x: B×3×H×W in [0,1] → B×10 cheap perceptual measurements."""
    r, g, b = x[:, 0], x[:, 1], x[:, 2]
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    mx, _ = x.max(1); mn, _ = x.min(1)
    sat = (mx - mn) / (mx + 1e-6)
    brightness = lum.mean((1, 2))
    contrast = lum.flatten(1).std(1)
    saturation = sat.mean((1, 2))
    warmth = (r - b).mean((1, 2))
    gx = (lum[:, :, 1:] - lum[:, :, :-1]).abs().mean((1, 2))
    gy = (lum[:, 1:, :] - lum[:, :-1, :]).abs().mean((1, 2))
    detail = gx + gy
    # hue in [0,1)
    d = (mx - mn) + 1e-6
    hue = torch.where(mx == r, ((g - b) / d) % 6, torch.where(mx == g, (b - r) / d + 2, (r - g) / d + 4)) / 6.0
    colored = (sat > 0.2) & (mx > 0.15)
    def frac(lo, hi):
        m = (hue >= lo) & (hue < hi) & colored
        return m.float().mean((1, 2))
    red_ = frac(0.95, 1.01) + frac(-0.01, 0.05)
    yellow_ = frac(0.05, 0.20)
    green_ = frac(0.20, 0.45)
    blue_ = frac(0.45, 0.75)
    purple_ = frac(0.75, 0.95)
    return torch.stack([brightness, contrast, saturation, warmth, detail, red_, yellow_, green_, blue_, purple_], 1)

# --------------------------------------------------------------------------- networks
def leaky(): return nn.LeakyReLU(0.2, inplace=True)

def plan(h: int, w: int, base: int) -> Tuple[List[int], Tuple[int, int]]:
    """How many stride-2 stages fit this canvas, and the bottleneck map they leave.
    Works for any canvas whose sides are divisible by 8 (e.g. 64, 96, 128, 160×96, 192×128, 256…)."""
    n = 0
    while n < 6 and h % 2 == 0 and w % 2 == 0 and min(h, w) // 2 >= 3:
        h //= 2; w //= 2; n += 1
    if n < 3:
        raise SystemExit("canvas sides must be divisible by 8 (try 64, 96, 128, 160x96, 192x128, 256)")
    return [min(base * 2 ** i, 256) for i in range(n)], (h, w)

class Encoder(nn.Module):
    def __init__(self, h: int, w: int, latent: int, base: int):
        super().__init__()
        ws, (bh, bw) = plan(h, w, base)
        layers, cin = [], 3
        for c in ws:
            layers += [nn.Conv2d(cin, c, 4, 2, 1), nn.BatchNorm2d(c), leaky()]
            cin = c
        self.conv = nn.Sequential(*layers)
        self.flat = ws[-1] * bh * bw
        self.mu = nn.Linear(self.flat, latent)
        self.logvar = nn.Linear(self.flat, latent)

    def forward(self, x):
        h = self.conv(x * 2 - 1).flatten(1)
        return self.mu(h), self.logvar(h).clamp(-8, 6)

class Decoder(nn.Module):
    def __init__(self, h: int, w: int, latent: int, base: int):
        super().__init__()
        ws, (self.bh, self.bw) = plan(h, w, base)
        ws = ws[::-1]
        self.w0 = ws[0]
        self.fc = nn.Sequential(nn.Linear(latent, ws[0] * self.bh * self.bw), leaky())
        layers = []
        for i in range(len(ws) - 1):   # upsample + conv (no checkerboard artifacts)
            layers += [nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(ws[i], ws[i + 1], 3, 1, 1),
                       nn.BatchNorm2d(ws[i + 1]), leaky()]
        layers += [nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(ws[-1], 3, 3, 1, 1), nn.Sigmoid()]
        self.deconv = nn.Sequential(*layers)

    def forward(self, z):
        return self.deconv(self.fc(z).view(-1, self.w0, self.bh, self.bw))

class Critic(nn.Module):
    """Spectral-normalised conv critic. Returns (logit, features) — features feed the actor's cost."""
    def __init__(self, h: int, w: int, base: int):
        super().__init__()
        ws, (bh, bw) = plan(h, w, base)
        blocks, cin = [], 3
        for c in ws:
            blocks.append(nn.Sequential(nn.utils.spectral_norm(nn.Conv2d(cin, c, 4, 2, 1)), leaky()))
            cin = c
        self.blocks = nn.ModuleList(blocks)
        self.head = nn.utils.spectral_norm(nn.Linear(ws[-1] * bh * bw, 1))

    def forward(self, x):
        h = x * 2 - 1
        feats = []
        for b in self.blocks:
            h = b(h); feats.append(h)
        return self.head(h.flatten(1)).squeeze(1), feats

@dataclass
class Config:
    height: int = 64          # canvas the model paints on (any size divisible by 8; need not be square)
    width: int = 64
    latent: int = 128
    base: int = 32
    batch: int = 16
    lr: float = 2e-4
    beta_kl: float = 0.3
    lam_pix: float = 10.0
    lam_feat: float = 1.0
    lam_adv: float = 0.5
    adv_start: int = 300      # steps before the critic's opinion counts
    adv_full: int = 1200      # steps until full adversarial weight
    kl_full: int = 600
    portfolio: str = ""
    name: str = ""

def cfg_from(d: dict) -> Config:
    """Config from a checkpoint dict (accepts the older square-only 'size' field)."""
    d = dict(d)
    if "size" in d:
        d["height"] = d["width"] = d.pop("size")
    return Config(**{k: v for k, v in d.items() if k in Config.__dataclass_fields__})

def build(cfg: Config):
    return (Encoder(cfg.height, cfg.width, cfg.latent, cfg.base), Decoder(cfg.height, cfg.width, cfg.latent, cfg.base),
            Critic(cfg.height, cfg.width, cfg.base))

def parse_canvas(size: Optional[int], canvas: Optional[str], default: int) -> Tuple[int, int]:
    """--size N → N×N square; --canvas WxH → rectangular. Returns (height, width)."""
    if canvas:
        m = re.fullmatch(r"\s*(\d+)\s*[xX×]\s*(\d+)\s*", canvas)
        if not m: raise SystemExit("--canvas wants WIDTHxHEIGHT, e.g. 160x96")
        w, h = int(m.group(1)), int(m.group(2))
    else:
        w = h = size or default
    if min(h, w) < 32 or h % 8 or w % 8:
        raise SystemExit("canvas sides must be ≥32 and divisible by 8 (e.g. 64, 96, 128, 160x96, 192x128, 256)")
    return h, w

def n_params(*ms): return sum(p.numel() for m in ms for p in m.parameters())

# --------------------------------------------------------------------------- image utilities
def to_pil(x: torch.Tensor) -> Image.Image:
    arr = (x.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
    return Image.fromarray(arr)

def grid(rows: List[torch.Tensor], pad: int = 2, bg=(24, 24, 24)) -> Image.Image:
    """rows: list of B×3×H×W tensors → one contact sheet."""
    n = max(r.shape[0] for r in rows); th, tw = rows[0].shape[-2], rows[0].shape[-1]
    W = n * (tw + pad) + pad; H = len(rows) * (th + pad) + pad
    sheet = Image.new("RGB", (W, H), bg)
    for ri, r in enumerate(rows):
        for ci in range(r.shape[0]):
            sheet.paste(to_pil(r[ci]), (pad + ci * (tw + pad), pad + ri * (th + pad)))
    return sheet

def upscale(img: Image.Image, long_side: int) -> Image.Image:
    """Resize so the longer side == long_side, aspect kept."""
    w, h = img.size
    if not long_side or max(w, h) == long_side: return img
    s = long_side / max(w, h)
    return img.resize((max(1, round(w * s)), max(1, round(h * s))), Image.LANCZOS)

# --------------------------------------------------------------------------- training (the ratcheting minimax loop)
def kl_div(mu, logvar):
    return (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).mean()

def ramp(step: int, start: int, full: int) -> float:
    if step <= start: return 0.0
    if step >= full: return 1.0
    return (step - start) / max(1, full - start)

def split_val(n: int) -> Tuple[List[int], List[int]]:
    idx = list(range(n)); random.Random(0).shuffle(idx)
    k = n // 10 if n >= 12 else 0
    return idx[k:], idx[:k]

def load_compatible(module: nn.Module, state: dict) -> Tuple[int, int]:
    """Copy every tensor whose name and shape match (lets a baseline trained on one canvas seed another)."""
    own = module.state_dict(); hit = 0
    for k, v in state.items():
        if k in own and own[k].shape == v.shape:
            own[k] = v; hit += 1
    module.load_state_dict(own)
    return hit, len(own)

class CifarSource:
    """CIFAR-10 (60k tiny natural images) resized to the canvas — a generic 'what images look like' baseline."""

    def __init__(self, height: int, width: int, root: Path):
        try:
            from torchvision.datasets import CIFAR10
        except ImportError:
            raise SystemExit("torchvision is needed for the CIFAR-10 baseline:  pip install torchvision")
        root.mkdir(parents=True, exist_ok=True)
        say(dim(f"  CIFAR-10 → {root}  (downloads ~170 MB from cs.toronto.edu on first use)"))
        tr = CIFAR10(str(root), train=True, download=True); te = CIFAR10(str(root), train=False, download=True)
        self.data = torch.from_numpy(tr.data).permute(0, 3, 1, 2).contiguous()        # N×3×32×32 uint8
        self.test = torch.from_numpy(te.data[:64]).permute(0, 3, 1, 2).contiguous()
        self.h, self.w = height, width
        self.aspect = width / height
        self.root = root; self.name = "cifar10"

    def __len__(self): return self.data.shape[0]

    def _to_canvas(self, x: torch.Tensor, train: bool) -> torch.Tensor:
        x = x.float().div(255.0)
        _, _, h, w = x.shape
        ch = min(h, int(round(w / self.aspect))); cw = min(w, int(round(h * self.aspect)))
        x = x[:, :, (h - ch) // 2:(h - ch) // 2 + ch, (w - cw) // 2:(w - cw) // 2 + cw]
        x = F.interpolate(x, size=(self.h, self.w), mode="bilinear", align_corners=False)
        if train:
            flip = torch.rand(x.shape[0]) < 0.5
            x[flip] = x[flip].flip(-1)
        return x

    def batch(self, idx: List[int], train: bool) -> torch.Tensor:
        return self._to_canvas(self.data[idx], train)

    def val(self) -> torch.Tensor:
        return self._to_canvas(self.test[:32], False)

def baseline_path(h: int, w: int) -> Path:
    return MODEL_DIR / f"baseline_cifar10_{w}x{h}.pt"

def find_baseline(h: int, w: int) -> Optional[Path]:
    exact = baseline_path(h, w)
    if exact.exists(): return exact
    others = sorted(MODEL_DIR.glob("baseline_cifar10_*x*.pt"), key=lambda p: p.stat().st_mtime)
    others = [p for p in others if not p.name.endswith("_latest.pt")]
    return others[-1] if others else None

def run_training(source, train_idx: List[int], val_x: torch.Tensor, real_mean: torch.Tensor, real_std: torch.Tensor,
                 cfg: Config, args, device: torch.device, latest: Path, best: Path, progress: Path, log_path: Path,
                 ckpt: Optional[dict] = None, init: Optional[dict] = None) -> dict:
    """The ratcheting minimax loop, shared by the CIFAR-10 baseline and the portfolio fine-tune."""
    progress.mkdir(parents=True, exist_ok=True)
    enc, dec, crit = build(cfg)
    enc.to(device); dec.to(device); crit.to(device)
    opt_g = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=cfg.lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(crit.parameters(), lr=cfg.lr, betas=(0.5, 0.999))
    step, best_score, best_step = 0, float("inf"), 0
    if ckpt:
        enc.load_state_dict(ckpt["enc"]); dec.load_state_dict(ckpt["dec"]); crit.load_state_dict(ckpt["crit"])
        opt_g.load_state_dict(ckpt["opt_g"]); opt_d.load_state_dict(ckpt["opt_d"])
        step, best_score, best_step = ckpt["step"], ckpt.get("best_score", float("inf")), ckpt.get("best_step", 0)
    elif init:
        hits = [load_compatible(m, init[k]) for m, k in ((enc, "enc"), (dec, "dec"), (crit, "crit"))]
        say(dim(f"  transferred baseline weights: actor {hits[0][0]+hits[1][0]}/{hits[0][1]+hits[1][1]} tensors · "
                f"critic {hits[2][0]}/{hits[2][1]} tensors"))
    say(dim(f"  actor {n_params(enc, dec)/1e6:.1f}M params · critic {n_params(crit)/1e6:.1f}M params"))
    say(dim(f"  budget: {args.steps} steps or {args.minutes} min or {args.patience} steps without progress — whichever comes first"))
    say(dim(f"  progress sheets → {progress}   (Ctrl-C stops early and keeps the best model)\n"))

    fixed_z = torch.randn(8, cfg.latent, device=device)
    eval_z = torch.randn(32, cfg.latent, device=device)
    new_log = not log_path.exists() or args.fresh or not ckpt
    log_f = open(log_path, "a", newline=""); log = csv.writer(log_f)
    if new_log: log.writerow(["step", "rec", "feat", "kl", "adv", "critic_acc", "val_score", "dream_gap", "sec_per_step"])

    acc_ema, t_start, t_last_ckpt, last_improve = 0.5, time.time(), time.time(), step
    start_step = step
    target = step + args.steps
    bce = F.binary_cross_entropy_with_logits
    hist = {"rec": [], "feat": [], "kl": [], "adv": [], "acc": []}
    stop_reason = "budget reached"

    def save(path: Path):
        torch.save({"cfg": cfg.__dict__, "enc": enc.state_dict(), "dec": dec.state_dict(), "crit": crit.state_dict(),
                    "opt_g": opt_g.state_dict(), "opt_d": opt_d.state_dict(), "step": step,
                    "best_score": best_score, "best_step": best_step}, path)

    if not ckpt:                     # a fresh run owns its files from the first step (no stale best.pt from an older canvas)
        save(latest); save(best)

    def evaluate() -> Tuple[float, float]:
        """Held-out score = how well originals are repainted + how far the dreams' palette/tone
        statistics sit from the real images'. Lower is better."""
        enc.eval(); dec.eval(); crit.eval()
        with torch.no_grad():
            mu, _ = enc(val_x); rec = dec(mu)
            _, fr = crit(val_x); _, ff = crit(rec)
            recon = F.l1_loss(rec, val_x).item() * cfg.lam_pix + F.l1_loss(ff[-1], fr[-1]).item() * cfg.lam_feat
            da = image_attributes(dec(eval_z)).cpu()
            gap = (((da.mean(0) - real_mean).abs() / real_std).mean() + 0.5 * ((da.std(0) - real_std).abs() / real_std).mean()).item()
        enc.train(); dec.train(); crit.train()
        return recon + 0.25 * gap, gap

    def sheet(tag: str):
        enc.eval(); dec.eval()
        with torch.no_grad():
            real = val_x[:8]; mu, _ = enc(real); rec = dec(mu); dream = dec(fixed_z)
        enc.train(); dec.train()
        img = grid([real.cpu(), rec.cpu(), dream.cpu()])
        img.save(progress / f"{tag}.png"); return img

    try:
        while step < target:
            t0 = time.time()
            x = source.batch(random.choices(train_idx, k=cfg.batch), train=True).to(device)
            w_adv = cfg.lam_adv * ramp(step, cfg.adv_start, cfg.adv_full)
            w_kl = cfg.beta_kl * ramp(step, 0, cfg.kl_full)

            # ----- actor forward
            mu, logvar = enc(x)
            z = mu + torch.randn_like(mu) * (0.5 * logvar).exp()
            x_rec = dec(z)
            x_dream = dec(torch.randn_like(mu))

            # ----- critic step (minimax, real vs repainting & dream)
            for p in crit.parameters(): p.requires_grad_(True)
            logit_real, feats_real = crit(x)
            logit_rec, _ = crit(x_rec.detach())
            logit_dream, _ = crit(x_dream.detach())
            loss_d = bce(logit_real, torch.full_like(logit_real, 0.9)) + \
                0.5 * (bce(logit_rec, torch.zeros_like(logit_rec)) + bce(logit_dream, torch.zeros_like(logit_dream)))
            with torch.no_grad():
                acc = 0.5 * ((logit_real > 0).float().mean() + 0.5 * ((logit_rec < 0).float().mean() + (logit_dream < 0).float().mean()))
            acc_ema = 0.95 * acc_ema + 0.05 * acc.item()
            critic_sits_out = acc_ema > 0.85 and step > cfg.adv_start   # ratchet: don't let the critic run away
            if not critic_sits_out:
                opt_d.zero_grad(set_to_none=True); loss_d.backward(); opt_d.step()

            # ----- actor step (reconstruction + critic features + KL + fooling the critic)
            for p in crit.parameters(): p.requires_grad_(False)
            logit_rec_g, feats_rec = crit(x_rec)
            logit_dream_g, _ = crit(x_dream)
            loss_rec = F.l1_loss(x_rec, x)
            loss_feat = F.l1_loss(feats_rec[-1], feats_real[-1].detach())
            loss_kl = kl_div(mu, logvar)
            loss_adv = bce(logit_rec_g, torch.ones_like(logit_rec_g)) + bce(logit_dream_g, torch.ones_like(logit_dream_g))
            actor_adv_on = acc_ema > 0.55 or step < cfg.adv_start   # ratchet: a lost critic teaches nothing
            loss_g = cfg.lam_pix * loss_rec + cfg.lam_feat * loss_feat * (w_adv > 0) + w_kl * loss_kl + \
                (w_adv * loss_adv if actor_adv_on else 0.0)
            opt_g.zero_grad(set_to_none=True); loss_g.backward(); opt_g.step()
            step += 1

            for k, v in zip(hist, [loss_rec.item(), loss_feat.item(), loss_kl.item(), loss_adv.item(), acc.item()]):
                hist[k].append(v)
            dt = time.time() - t0

            # ----- bookkeeping
            if step % 25 == 0 or step == 1:
                means = {k: float(np.mean(v[-25:])) for k, v in hist.items()}
                state = "critic sits out" if critic_sits_out else ("actor unpressured" if not actor_adv_on else "balanced")
                elapsed = time.time() - t_start
                eta = (target - step) * (elapsed / max(1, step - start_step))
                say(f"  step {step:>6}/{target}  rec {means['rec']:.4f}  feat {means['feat']:.3f}  kl {means['kl']:.3f}  "
                    f"adv {means['adv']:.2f}  critic {means['acc']*100:3.0f}%  {dim('[' + state + ']'):<28} {dt:.2f}s/step  eta {fmt_secs(min(eta, args.minutes*60 - elapsed))}")
            if step % 100 == 0:
                score, gap = evaluate()
                log.writerow([step, *[f"{np.mean(hist[k][-100:]):.5f}" for k in ("rec", "feat", "kl", "adv", "acc")], f"{score:.5f}", f"{gap:.4f}", f"{dt:.3f}"]); log_f.flush()
                if step < cfg.adv_full:
                    # warm-up: the critic is not yet in the game, so the score is not comparable with later ones.
                    save(best); last_improve = step
                elif score < best_score - 1e-4:
                    best_score, best_step, last_improve = score, step, step
                    save(best); say(green(f"  ✔ new best held-out score {score:.4f} (dream gap {gap:.2f}) at step {step} → {best.name}"))
                else:
                    say(dim(f"  held-out {score:.4f} (dream gap {gap:.2f}) — best {best_score:.4f} at step {best_step}"))
                if step % 300 == 0:
                    sheet(f"step_{step:06d}")
            if time.time() - t_last_ckpt > 60:
                save(latest); t_last_ckpt = time.time()
            if time.time() - t_start > args.minutes * 60:
                stop_reason = f"time budget ({args.minutes} min)"; break
            if step - last_improve > args.patience and step > cfg.adv_full:
                stop_reason = f"no held-out improvement for {args.patience} steps"; break
    except KeyboardInterrupt:
        stop_reason = "stopped by you"
    finally:
        log_f.close()

    save(latest)
    if not best.exists(): save(best)
    best_txt = f"Best held-out score {best_score:.4f} at step {best_step}." if best_step else \
        f"(still in warm-up at step {step} — held-out scoring starts at step {cfg.adv_full}; the latest weights are kept as best)"
    say(bold(f"\n■ training finished — {stop_reason}. {best_txt}"))
    sheet(f"final_step_{step:06d}")
    say(dim(f"  rows of the sheet: originals · repaintings · dreams  → {progress}"))
    if not args.no_open: open_file(progress / f"final_step_{step:06d}.png")
    return {"step": step, "best_score": best_score, "best_step": best_step, "stop_reason": stop_reason}

def pretrain(args) -> Path:
    """Stage 0: learn a generic image baseline on CIFAR-10 (so the portfolio fine-tune starts from
    'knows what images look like' instead of noise). Output: model/baseline_cifar10_WxH.pt"""
    torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)
    device = pick_device()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    h, w = parse_canvas(args.size, args.canvas, 128 if device.type != "cpu" else 64)
    best, latest = baseline_path(h, w), baseline_path(h, w).with_name(baseline_path(h, w).stem + "_latest.pt")
    ckpt = torch.load(latest, map_location="cpu") if latest.exists() and not args.fresh else None
    cfg = cfg_from(ckpt["cfg"]) if ckpt else Config(height=h, width=w, portfolio="", name="cifar10", batch=32)
    if args.batch: cfg.batch = args.batch
    say(bold(f"\n◆ Atelier · CIFAR-10 baseline ({cfg.width}×{cfg.height} canvas)" + (f" — resuming from step {ckpt['step']}" if ckpt else "")))
    say(dim(f"  device {device.type} · latent {cfg.latent} · batch {cfg.batch}"))
    src = CifarSource(cfg.height, cfg.width, HOME / "data")
    val_x = src.val().to(device)
    with torch.no_grad():
        ra = torch.cat([image_attributes(src.batch(list(range(i, i + 64)), train=False)) for i in range(0, 1024, 64)])
        real_mean, real_std = ra.mean(0), ra.std(0).clamp_min(1e-4)
    run_training(src, list(range(len(src))), val_x, real_mean, real_std, cfg, args, device,
                 latest, best, MODEL_DIR / "progress_cifar10", MODEL_DIR / "training_log_cifar10.csv", ckpt=ckpt)
    say(green(f"  baseline saved → {best}.  Now:  python atelier.py train <portfolio>   (it will start from this baseline)\n"))
    return best

def train(args) -> Path:
    """Stage 1 (or the only stage): learn the artist's style. Starts from the CIFAR-10 baseline when one
    exists (--init auto), builds one first with --init cifar10, or starts from noise with --init none."""
    torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)
    device = pick_device()
    portfolio_dir = Path(args.portfolio).expanduser().resolve()
    if not portfolio_dir.is_dir():
        raise SystemExit(f"Not a folder: {portfolio_dir}")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # resume?
    ckpt = None
    if LATEST.exists() and not args.fresh:
        ckpt = torch.load(LATEST, map_location="cpu")
        old = (ckpt["cfg"]["height"], ckpt["cfg"]["width"])
        if (args.size or args.canvas) and parse_canvas(args.size, args.canvas, old[1]) != old:
            say(yellow(f"  a model with a {old[1]}×{old[0]} canvas exists; you asked for a different one → starting fresh"))
            ckpt = None
    if ckpt:
        cfg = cfg_from(ckpt["cfg"])
        say(bold(f"↻ resuming from step {ckpt['step']} ({cfg.width}×{cfg.height} canvas, portfolio: {cfg.name})"))
    else:
        h, w = parse_canvas(args.size, args.canvas, 128 if device.type != "cpu" else 64)
        cfg = Config(height=h, width=w, portfolio=str(portfolio_dir), name=portfolio_dir.name)
    cfg.portfolio = str(portfolio_dir); cfg.name = portfolio_dir.name; remember_portfolio(portfolio_dir)
    if args.batch: cfg.batch = args.batch
    _, (bh, bw) = plan(cfg.height, cfg.width, cfg.base)
    if max(bh, bw) > 8:
        say(yellow(f"  note: a {cfg.width}×{cfg.height} canvas leaves a large {bw}×{bh} bottleneck (heavier model). "
                   f"Sides divisible by 32 (64, 96, 128, 160, 192, 256…) train leaner."))

    # baseline (transfer learning) — only when not resuming
    init, init_note = None, "from scratch"
    mode = getattr(args, "init", "auto")
    if not ckpt and mode != "none":
        bp = find_baseline(cfg.height, cfg.width)
        if bp is None and mode == "cifar10":
            say(bold("  no CIFAR-10 baseline yet → building one first"))
            pre = argparse.Namespace(size=cfg.width if cfg.width == cfg.height else None,
                                     canvas=None if cfg.width == cfg.height else f"{cfg.width}x{cfg.height}",
                                     batch=None, steps=getattr(args, "pretrain_steps", 4000), minutes=getattr(args, "pretrain_minutes", 30),
                                     patience=args.patience, seed=args.seed, fresh=False, no_open=True)
            bp = pretrain(pre)
        if bp is not None:
            init = torch.load(bp, map_location="cpu")
            init_note = f"from the CIFAR-10 baseline {bp.name}" + ("" if (init["cfg"]["height"], init["cfg"]["width"]) == (cfg.height, cfg.width)
                                                              else " (different canvas — conv layers transfer, heads restart)")

    say(bold(f"\n◆ Atelier · learning the style of “{cfg.name}” — {init_note}"))
    say(dim(f"  device {device.type} · canvas {cfg.width}×{cfg.height} · latent {cfg.latent} · batch {cfg.batch}"))
    say(dim("  loading the portfolio…"))
    pf = Portfolio(portfolio_dir, cfg.height, cfg.width)
    cfg.batch = max(4, min(cfg.batch, len(pf)))
    train_idx, val_idx = split_val(len(pf))
    if not val_idx:
        say(dim("  (fewer than 12 pieces → validating on the training pieces themselves)"))
        val_idx = train_idx[: min(8, len(train_idx))]
    val_x = pf.batch(val_idx[:32], train=False).to(device)
    with torch.no_grad():   # palette / tone statistics of the whole portfolio — dreams should match them
        ra = torch.cat([image_attributes(pf.batch(list(range(i, min(i + 32, len(pf)))), train=False)) for i in range(0, len(pf), 32)])
        real_mean, real_std = ra.mean(0), ra.std(0).clamp_min(1e-4)

    run_training(pf, train_idx, val_x, real_mean, real_std, cfg, args, device,
                 LATEST, BEST, PROGRESS, MODEL_DIR / "training_log.csv", ckpt=ckpt, init=init)

    say(dim("  indexing the style space (so words can steer it)…"))
    build_style_index(BEST, pf, device)
    say(green(f"  style index written into {BEST.name}. Open the studio:  python atelier.py studio\n"))
    return BEST


# --------------------------------------------------------------------------- style index: the bridge from words to latents
@torch.no_grad()
def build_style_index(ckpt_path: Path, pf: Portfolio, device: torch.device) -> dict:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = cfg_from(ckpt["cfg"])
    enc, _, _ = build(cfg); enc.load_state_dict(ckpt["enc"]); enc.to(device).eval()
    mus, attrs = [], []
    for i in range(0, len(pf), 32):
        x = pf.batch(list(range(i, min(i + 32, len(pf)))), train=False).to(device)
        mu, _ = enc(x); mus.append(mu.cpu()); attrs.append(image_attributes(x).cpu())
    Z = torch.cat(mus); A = torch.cat(attrs)
    z_mean, z_std = Z.mean(0), Z.std(0).clamp_min(1e-3)
    a_mean, a_std = A.mean(0), A.std(0).clamp_min(1e-4)
    n = len(pf); k = max(2, n // 3)
    dirs = []
    for j in range(len(ATTRS)):
        order = A[:, j].argsort()
        lo, hi = Z[order[:k]].mean(0), Z[order[-k:]].mean(0)
        dirs.append((hi - lo) / 2)                       # +1 unit ≈ from the middle to the vivid tercile
    tag_dirs: Dict[str, Tuple[torch.Tensor, int]] = {}
    counts: Dict[str, int] = {}
    for tl in pf.tags:
        for t in tl: counts[t] = counts.get(t, 0) + 1
    for t, c in counts.items():
        if c >= 2 and c < n:  # a tag on every piece says nothing
            idx = [i for i, tl in enumerate(pf.tags) if t in tl]
            tag_dirs[t] = (Z[idx].mean(0) - z_mean, c)
    style = {"z_mean": z_mean, "z_std": z_std, "a_mean": a_mean, "a_std": a_std, "dirs": torch.stack(dirs),
             "tags": {t: d for t, (d, c) in tag_dirs.items()}, "tag_counts": {t: c for t, (d, c) in tag_dirs.items()},
             "files": [str(f) for f in pf.files], "Z": Z}
    ckpt["style"] = style
    torch.save(ckpt, ckpt_path)
    return style

# --------------------------------------------------------------------------- the prompt engine
# word → (attribute, signed strength). Several words hit two attributes — that's intended.
_V: Dict[str, List[Tuple[str, float]]] = {}
def _add(attr: str, sign: float, words: str, w: float = 1.0):
    for word in words.split():
        _V.setdefault(word, []).append((attr, sign * w))

_add("brightness", +1, "bright light luminous sunny sunlit glowing radiant airy pale white day daylight noon morning snow highkey glow")
_add("brightness", -1, "dark dim moody night nocturnal shadow shadowy dusk evening twilight midnight gloomy black somber sombre lowkey stormy storm deep")
_add("contrast", +1, "contrast contrasty dramatic bold punchy stark crisp sharp striking graphic chiaroscuro strong")
_add("contrast", -1, "soft gentle hazy haze foggy fog mist misty dreamy faded washed smoky subtle flat quiet calm tender")
_add("saturation", +1, "vivid vibrant saturated colorful colourful rich intense neon lush juicy electric loud bold")
_add("saturation", -1, "muted pastel desaturated grey gray greyish grayish monochrome subdued faded dusty quiet sepia tonal earthy")
_add("warmth", +1, "warm orange golden gold amber sunset sunrise fire fiery autumn fall honey rust rusty copper peach cozy cosy tan brown sienna ochre ocher sepia red yellow earthy")
_add("warmth", -1, "cool cold icy ice blue winter frost frosty steel silver moonlight moonlit ocean sea water rain rainy teal cyan arctic chilly aqua indigo violet")
_add("detail", +1, "detailed intricate busy complex textured texture dense ornate crowded cluttered chaotic energetic lively pattern patterned fine delicate rough grainy noisy sketchy scribbled hatched crosshatch wild")
_add("detail", -1, "minimal minimalist simple sparse clean smooth calm quiet empty plain serene still tranquil flat open peaceful spacious bare blank sky washed")
_add("red", +1, "red crimson scarlet ruby cherry rose blood wine maroon burgundy coral pink")
_add("yellow", +1, "yellow golden gold lemon mustard sand sandy wheat straw amber orange ochre ocher sunflower honey")
_add("green", +1, "green emerald forest leaf leafy grass grassy moss mossy olive jade lime mint verdant jungle meadow garden trees tree foliage")
_add("blue", +1, "blue azure navy cobalt teal cyan aqua turquoise sky ocean sea water lake river sapphire denim")
_add("purple", +1, "purple violet lavender lilac magenta mauve plum orchid fuchsia pink amethyst grape")
INTENSIFIERS = {"very": 1.6, "really": 1.5, "extremely": 1.9, "super": 1.6, "deeply": 1.5, "intensely": 1.6,
                "strongly": 1.5, "much": 1.3, "heavily": 1.5, "more": 1.25, "high": 1.2, "extra": 1.4}
SOFTENERS = {"slightly": 0.5, "somewhat": 0.6, "bit": 0.5, "little": 0.5, "hint": 0.4, "touch": 0.4, "subtly": 0.5,
             "gently": 0.6, "faintly": 0.4, "lightly": 0.5, "mildly": 0.5}
NEGATIONS = {"not", "no", "without", "never", "non", "less", "low", "un", "hardly", "barely", "minus"}
FILLER = {"a", "an", "the", "of", "in", "at", "with", "and", "or", "to", "on", "for", "but", "is", "it", "its",
          "painting", "paintings", "art", "artwork", "piece", "picture", "image", "style", "please", "make", "create",
          "paint", "draw", "generate", "me", "i", "want", "would", "like", "something", "some", "kind", "sort", "feel",
          "feeling", "mood", "scene", "one", "two", "few", "new", "his", "her", "their", "my", "this", "that", "very",
          "looks", "looking", "look", "pieces", "works", "work", "by", "from", "into", "over", "under", "as", "so",
          "too", "lots", "lot", "of", "full", "more", "bit", "just", "maybe", "think", "about", "again", "another"}
PHRASES = {"black and white": "monochrome", "high contrast": "contrasty", "low contrast": "soft",
           "low key": "lowkey", "high key": "highkey", "cross hatch": "crosshatch", "cross hatched": "crosshatch"}

def _stems(w: str) -> List[str]:
    out = [w]
    for suf in ("ness", "ly", "ish", "ing", "ed", "er", "est", "es", "s", "y"):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            out.append(w[: -len(suf)])
    if w.endswith("ier"): out.append(w[:-3] + "y")
    if w.endswith("iest"): out.append(w[:-4] + "y")
    return out

@dataclass
class Recipe:
    text: str
    attrs: Dict[str, float] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    ignored: List[str] = field(default_factory=list)
    anchor: Optional[str] = None          # path of an image to start from

    def explain(self) -> str:
        bits = [f"{k} {v:+.1f}" for k, v in sorted(self.attrs.items(), key=lambda kv: -abs(kv[1])) if abs(v) > 0.05]
        s = "  ✎ optimized prompt → " + (" · ".join(bits) if bits else "(no steering words: free sample from the style)")
        if self.tags: s += "\n    tags from the portfolio → " + ", ".join(self.tags)
        if self.anchor: s += f"\n    starting from → {Path(self.anchor).name}"
        if self.ignored: s += dim("\n    ignored → " + ", ".join(self.ignored))
        return s

class PromptOptimizer:
    """Turns free text into a Recipe: steering weights for the style space (±2 std max) + tag pulls."""

    def __init__(self, tag_counts: Dict[str, int]):
        self.tags = dict(tag_counts)

    def optimize(self, text: str) -> Recipe:
        t = text.lower()
        for ph, rep in PHRASES.items(): t = t.replace(ph, rep)
        tokens = [w for w in re.split(r"[^a-z0-9]+", t) if w]
        r = Recipe(text=text)
        mult, neg_left = 1.0, 0
        for w in tokens:
            if w in NEGATIONS: neg_left = 3; continue
            if w in INTENSIFIERS: mult *= INTENSIFIERS[w]; continue
            if w in SOFTENERS: mult *= SOFTENERS[w]; continue
            hit = False
            # portfolio tags first (the artist's own vocabulary beats ours)
            for s in _stems(w):
                if s in self.tags:
                    if s not in r.tags: r.tags.append(s)
                    hit = True; break
            if not hit:
                for s in _stems(w):
                    if s in _V:
                        sign = -1.0 if neg_left > 0 else 1.0
                        for attr, strength in _V[s]:
                            r.attrs[attr] = r.attrs.get(attr, 0.0) + sign * strength * mult
                        hit = True; break
            if hit:
                mult, neg_left = 1.0, 0
            else:
                if neg_left > 0: neg_left -= 1
                if w not in FILLER and not w.isdigit() and len(w) > 1:
                    r.ignored.append(w)
        for k in list(r.attrs):
            r.attrs[k] = float(max(-2.0, min(2.0, r.attrs[k])))
        return r

# --------------------------------------------------------------------------- the painter (inference)
class Painter:
    def __init__(self, ckpt_path: Optional[Path] = None):
        path = ckpt_path or (BEST if BEST.exists() else LATEST)
        if not path.exists():
            raise FileNotFoundError("No trained model yet. Run:  python atelier.py train <portfolio folder>")
        self.device = pick_device()
        ck = torch.load(path, map_location="cpu")
        self.cfg = cfg_from(ck["cfg"])
        self.enc, self.dec, self.crit = build(self.cfg)
        self.enc.load_state_dict(ck["enc"]); self.dec.load_state_dict(ck["dec"]); self.crit.load_state_dict(ck["crit"])
        for m in (self.enc, self.dec, self.crit): m.to(self.device).eval()
        if "style" not in ck:
            pdir = Path(self.cfg.portfolio)
            if not pdir.is_dir():
                raise FileNotFoundError(f"Model has no style index and the portfolio folder moved ({pdir}). "
                                        f"Run:  python atelier.py index <portfolio folder>")
            say(dim("  building the style index (first run only)…"))
            ck["style"] = build_style_index(path, Portfolio(pdir, self.cfg.height, self.cfg.width, quiet=True), self.device)
        self.style = ck["style"]
        self.spread = float((self.style["Z"] - self.style["z_mean"]).norm(dim=1).mean())
        self.step = ck.get("step", 0)
        self.optimizer = PromptOptimizer(self.style["tag_counts"])
        self.temperature = 0.8
        self.last_recipe: Optional[Recipe] = None

    # ---- latents
    def _base(self, recipe: Recipe) -> torch.Tensor:
        st = self.style
        z0 = self.encode_file(Path(recipe.anchor)) if recipe.anchor else st["z_mean"].clone()
        steer = torch.zeros_like(z0)
        for i, a in enumerate(ATTRS):
            w = recipe.attrs.get(a, 0.0)
            if w: steer = steer + w * st["dirs"][i]
        if recipe.tags:
            steer = steer + torch.stack([st["tags"][t] for t in recipe.tags]).mean(0)
        cap = 1.25 * self.spread                      # stacked words must not leave the style manifold
        if steer.norm() > cap: steer = steer * (cap / steer.norm())
        return z0 + steer

    @torch.no_grad()
    def encode_file(self, path: Path) -> torch.Tensor:
        img = load_image(path, int(max(self.cfg.height, self.cfg.width) * 1.25))
        _, h, w = img.shape; aspect = self.cfg.width / self.cfg.height
        ch = min(h, int(w / aspect)); cw = max(1, min(w, int(round(ch * aspect))))
        top, left = (h - ch) // 2, (w - cw) // 2
        patch = img[:, top:top + ch, left:left + cw].float().div(255)[None]
        patch = F.interpolate(patch, size=(self.cfg.height, self.cfg.width), mode="bilinear", align_corners=False, antialias=True)
        mu, _ = self.enc(patch.to(self.device))
        return mu[0].cpu()

    @torch.no_grad()
    def paint(self, recipe: Recipe, count: int = 4, temperature: Optional[float] = None, seed: Optional[int] = None,
              candidates_per: int = 4) -> Tuple[List[torch.Tensor], List[dict]]:
        """Generate count images. The critic curates: we dream extra candidates and keep the most
        convincing ones that also land where the prompt asked."""
        temp = self.temperature if temperature is None else temperature
        if recipe.anchor: temp *= 0.5
        g = torch.Generator().manual_seed(seed) if seed is not None else None
        st = self.style
        K = max(count * candidates_per, 8)
        base = self._base(recipe)
        eps = torch.randn(K, self.cfg.latent, generator=g) * st["z_std"]
        if recipe.anchor:
            noise = temp * eps
        else:
            # borrow each candidate's deviation from a real piece: variety that stays inside the style
            pick = torch.randint(0, st["Z"].shape[0], (K,), generator=g)
            noise = temp * (0.8 * (st["Z"][pick] - st["z_mean"]) + 0.5 * eps)
        Z = base[None] + noise
        X = self.dec(Z.to(self.device)).cpu()
        logit, _ = self.crit(X.to(self.device)); real = torch.sigmoid(logit.cpu())
        A = (image_attributes(X) - st["a_mean"]) / st["a_std"]
        want = {a: w for a, w in recipe.attrs.items() if abs(w) > 0.05}
        if want:
            tgt = torch.tensor([want[a] for a in ATTRS if a in want])
            cols = [i for i, a in enumerate(ATTRS) if a in want]
            miss = ((A[:, cols] - tgt) ** 2).mean(1)
        else:
            miss = torch.zeros(K)
        score = real - 0.25 * miss
        ranked = score.argsort(descending=True)[: max(count, min(K, 2 * count))].tolist()
        order = [ranked[0]]                            # best first, then the most different of the rest
        while len(order) < count and len(order) < len(ranked):
            rest = [i for i in ranked if i not in order]
            d = torch.cdist(Z[rest], Z[order]).min(1).values
            order.append(rest[int(d.argmax())])
        picks = [X[i] for i in order]
        info = [{"critic": float(real[i]), "achieved": {a: float(A[i, j]) for j, a in enumerate(ATTRS) if a in want}}
                for i in order]
        self.last_recipe = recipe
        return picks, info

    @torch.no_grad()
    def dream(self, count: int = 4, temperature: Optional[float] = None, seed: Optional[int] = None):
        return self.paint(Recipe(text="(surprise)"), count, temperature, seed)

    @torch.no_grad()
    def vary(self, path: Path, count: int = 4, temperature: float = 0.5):
        r = Recipe(text=f"variations of {path.name}", anchor=str(path))
        return self.paint(r, count, temperature * 2)   # paint halves temp for anchors

# --------------------------------------------------------------------------- saving results
def slugify(s: str, n: int = 32) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")
    return (s[:n] or "untitled").rstrip("-")

def save_results(picks: List[torch.Tensor], info: List[dict], recipe: Recipe, out_size: int, want_open: bool) -> Tuple[List[Path], Path]:
    GALLERY.mkdir(parents=True, exist_ok=True)
    slug = slugify(recipe.text)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    n = 2
    while any(GALLERY.glob(f"{stamp}_{slug}_*")):        # same prompt twice in one second → keep both
        stamp = time.strftime("%Y%m%d-%H%M%S") + f"-{n}"; n += 1
    paths = []
    for i, x in enumerate(picks, 1):
        p = GALLERY / f"{stamp}_{slug}_{i}.png"
        upscale(to_pil(x), out_size).save(p); paths.append(p)
    # contact sheet with captions
    tile = min(out_size, 384); pad = 10
    cols = min(4, len(picks)); rows = math.ceil(len(picks) / cols)
    sheet = Image.new("RGB", (cols * (tile + pad) + pad, rows * (tile + pad + 18) + pad + 24), (22, 22, 22))
    dr = ImageDraw.Draw(sheet)
    dr.text((pad, 6), f"{recipe.text}"[:90], fill=(230, 230, 230))
    for i, x in enumerate(picks):
        r, c = divmod(i, cols)
        X0, Y0 = pad + c * (tile + pad), 24 + pad + r * (tile + pad + 18)
        sheet.paste(upscale(to_pil(x), tile), (X0, Y0))
        dr.text((X0, Y0 + tile + 3), f"#{i+1}  critic {info[i]['critic']*100:.0f}%", fill=(170, 170, 170))
    sheet_path = GALLERY / f"{stamp}_{slug}_sheet.png"
    sheet.save(sheet_path)
    if want_open: open_file(sheet_path)
    return paths, sheet_path

# --------------------------------------------------------------------------- prep: photos of drawings → just the paper
def _components(mask: np.ndarray):
    H, W = mask.shape; lab = np.zeros((H, W), np.int32); cur = 0; sizes = {}
    for y in range(H):
        for x in range(W):
            if mask[y, x] and lab[y, x] == 0:
                cur += 1; stack = [(y, x)]; lab[y, x] = cur; n = 0
                while stack:
                    cy, cx = stack.pop(); n += 1
                    for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                        if 0 <= ny < H and 0 <= nx < W and mask[ny, nx] and lab[ny, nx] == 0:
                            lab[ny, nx] = cur; stack.append((ny, nx))
                sizes[cur] = n
    return lab, sizes

def _close(mask: np.ndarray, r: int = 3) -> np.ndarray:
    m = mask.copy()
    for _ in range(r): m = m | np.roll(m, 1, 0) | np.roll(m, -1, 0) | np.roll(m, 1, 1) | np.roll(m, -1, 1)
    for _ in range(r): m = m & np.roll(m, 1, 0) & np.roll(m, -1, 0) & np.roll(m, 1, 1) & np.roll(m, -1, 1)
    return m

def crop_to_paper(img: Image.Image, margin: float = 0.015) -> Tuple[Image.Image, bool]:
    """Photographed drawing → the sheet of paper only (white, low-saturation region; split pieces merged)."""
    im = ImageOps.exif_transpose(img).convert("RGB")
    w0, h0 = im.size; s = 400 / max(w0, h0)
    small = im.resize((max(1, round(w0 * s)), max(1, round(h0 * s))), Image.BILINEAR)
    a = np.asarray(small).astype(np.float32) / 255
    mx, mn = a.max(2), a.min(2)
    white = _close((mn > 0.62) & ((mx - mn) < 0.16), 3)
    H, W = white.shape
    lab, sizes = _components(white)
    big = sorted([k for k, n in sizes.items() if n > 0.03 * H * W], key=lambda k: -sizes[k])
    if not big: return im, False
    boxes = []
    for k in big:
        ys, xs = np.where(lab == k)
        boxes.append([np.percentile(xs, 0.5), np.percentile(ys, 0.5), np.percentile(xs, 99.5), np.percentile(ys, 99.5)])
    main = boxes[0]
    for b in boxes[1:]:
        gap = max(b[0] - main[2], main[0] - b[2], b[1] - main[3], main[1] - b[3])
        if gap < 0.06 * max(W, H):
            main = [min(main[0], b[0]), min(main[1], b[1]), max(main[2], b[2]), max(main[3], b[3])]
    x0, y0, x1, y1 = main; mxm, mym = (x1 - x0) * margin, (y1 - y0) * margin
    return im.crop((int((x0 + mxm) / s), int((y0 + mym) / s), int((x1 - mxm) / s), int((y1 - mym) / s))), True

def prep(args) -> Path:
    src = Path(args.folder).expanduser().resolve(); out = Path(args.out).expanduser().resolve() if args.out else src.with_name(src.name + "_paper")
    out.mkdir(parents=True, exist_ok=True)
    files = find_images(src); n_ok = 0
    for f in files:
        try:
            im = Image.open(f)
            im, ok = crop_to_paper(im) if args.paper else (ImageOps.exif_transpose(im).convert("RGB"), True)
            im.thumbnail((args.max_side, args.max_side))
            im.save(out / (f.stem + ".jpg"), quality=92); n_ok += ok
            say(dim(f"  {f.name} → {f.stem}.jpg {'(paper found)' if ok else '(no paper found — kept whole)'}"))
        except Exception as e:
            say(yellow(f"  skipping {f.name}: {e}"))
    say(green(f"  {len(files)} files → {out}  ({n_ok} cropped to the paper). Use it as the portfolio:  --portfolio {out}"))
    return out

# --------------------------------------------------------------------------- which engine paints
PORTFOLIO_FILE = MODEL_DIR / "portfolio.txt"

def remember_portfolio(p: Path) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True); PORTFOLIO_FILE.write_text(str(Path(p).expanduser().resolve()))

def known_portfolio(arg: Optional[str] = None) -> Optional[Path]:
    """--portfolio → the last one used → the one the VAE was trained on."""
    if arg: return Path(arg).expanduser().resolve()
    if PORTFOLIO_FILE.exists():
        p = Path(PORTFOLIO_FILE.read_text().strip())
        if p.is_dir(): return p
    for ck in (BEST, LATEST):
        if ck.exists():
            p = Path(cfg_from(torch.load(ck, map_location="cpu")["cfg"]).portfolio)
            if p.is_dir(): return p
    return None

def make_engine(engine: str = "auto", portfolio: Optional[str] = None, res: int = 384, iters: int = 250):
    """neural = SD-Turbo content + VGG19 style transfer from the portfolio (no training, the default).
    vae = the small trained actor/critic (needs `train`)."""
    if engine in ("auto", "neural"):
        p = known_portfolio(portfolio)
        if p is None:
            if engine == "neural" or not (BEST.exists() or LATEST.exists()):
                raise FileNotFoundError("Which portfolio? Pass --portfolio <folder of the artist's pieces> (remembered afterwards).")
        else:
            import atelier_neural
            remember_portfolio(p)
            return atelier_neural.NeuralEngine(p, res=res, iters=iters)
    return Painter()

# --------------------------------------------------------------------------- the studio (text UI)
HELP = f"""
  {bold('Just type what you want to see.')}  Examples:
      a hippo eating cheese                a fishing boat at dawn, warm and quiet
      soft pastel morning, minimal         like the last one but colder
  Content words are drawn first (SD-Turbo), then repainted in the artist's style (neural style transfer);
  mood words (warm, dark, busy, soft…) pick which pieces define the style and grade the colour.
  The engine prints how it read your words (the “optimized prompt”) and paints {bold('N')} pieces,
  saves them to gallery/ and opens a contact sheet.

  {bold('Commands')}
    /more               more of the last prompt (fresh noise)
    /surprise           pure samples from the style, no steering
    /vary <file|last|N> repaint an image in the style (N = piece number from the last sheet)
    /like <file>        use that image as the content for every prompt until /like off
    /content <file>     same as /like
    /count N            pieces per prompt (default 4)
    /temp T             adventurousness 0.3–1.5 (default 0.8)
    /size PX            output size in pixels (default 512)
    /open on|off        auto-open the contact sheet
    /words              the steering vocabulary the engine understands
    /tags               words from the portfolio file names it can pull toward
    /gallery            open the gallery folder
    /train <folder>     (re)train on a portfolio — resumes if a model exists
    /status             model details
    /quit
"""

def studio(args) -> None:
    try:
        painter = make_engine(getattr(args, "engine", "auto"), getattr(args, "portfolio", None),
                              getattr(args, "res", 384), getattr(args, "iters", 250))
    except FileNotFoundError as e:
        say(yellow(f"  {e}")); return
    neural = type(painter).__name__ == "NeuralEngine"
    count, out_size, auto_open = args.count, args.size, not args.no_open
    anchor: Optional[str] = None
    last_paths: List[Path] = []
    say(bold(f"\n◆ Atelier studio — painting in the style of “{painter.cfg.name}”"))
    if neural:
        say(dim(f"  engine: SD-Turbo content → VGG19 style transfer · {len(painter.style['files'])} pieces in the style book · "
                f"{len(painter.style['tag_counts'])} portfolio tags · working res {painter.res}px · device {painter.device.type}"))
    else:
        say(dim(f"  engine: trained actor/critic · step {painter.step} · canvas {painter.cfg.width}×{painter.cfg.height} · "
                f"{len(painter.style['files'])} pieces studied · {len(painter.style['tag_counts'])} portfolio tags · device {painter.device.type}"))
    say(dim("  type a description, or /help\n"))

    def run(recipe: Recipe, temp: Optional[float] = None):
        nonlocal last_paths
        t0 = time.time()
        picks, info = painter.paint(recipe, count, temp)
        paths, sheet_path = save_results(picks, info, recipe, out_size, auto_open)
        last_paths = paths
        say(recipe.explain())
        for i, (p, inf) in enumerate(zip(paths, info), 1):
            ach = "  ".join(f"{a} {v:+.1f}" for a, v in inf["achieved"].items())
            say(dim(f"    #{i} {p.name}  critic {inf['critic']*100:.0f}%  {ach}"))
        say(green(f"  ✔ {len(paths)} pieces in {time.time()-t0:.1f}s → {sheet_path.name}\n"))

    while True:
        try:
            line = input(cyan("✎ ")).strip()
        except (EOFError, KeyboardInterrupt):
            say(); break
        if not line: continue
        if line.startswith("/"):
            parts = line.split(maxsplit=1); cmd = parts[0].lower(); arg = parts[1].strip() if len(parts) > 1 else ""
            if cmd in ("/quit", "/exit", "/q"): break
            elif cmd == "/help": say(HELP)
            elif cmd == "/more":
                if painter.last_recipe: run(painter.last_recipe)
                else: say(yellow("  nothing to repeat yet"))
            elif cmd == "/surprise": run(Recipe(text="surprise", anchor=anchor))
            elif cmd == "/vary":
                target = None
                if arg in ("", "last") and last_paths: target = last_paths[0]
                elif arg.isdigit() and last_paths and 1 <= int(arg) <= len(last_paths): target = last_paths[int(arg) - 1]
                elif arg: target = Path(arg.strip("'\"")).expanduser()
                if target is None or not Path(target).exists(): say(yellow("  give me an image file, a piece number, or 'last'")); continue
                run(Recipe(text=f"variations of {Path(target).name}", anchor=str(target)), temp=1.0)
            elif cmd in ("/like", "/content"):
                if arg.lower() in ("off", ""): anchor = None; say(dim("  anchor cleared"))
                else:
                    p = Path(arg.strip("'\"")).expanduser()
                    if p.exists(): anchor = str(p); say(dim(f"  every prompt now starts from {p.name}"))
                    else: say(yellow("  file not found"))
            elif cmd == "/count":
                if arg.isdigit() and 1 <= int(arg) <= 16: count = int(arg); say(dim(f"  {count} pieces per prompt"))
                else: say(yellow("  /count 1–16"))
            elif cmd == "/temp":
                try: painter.temperature = min(2.0, max(0.1, float(arg))); say(dim(f"  temperature {painter.temperature}"))
                except ValueError: say(yellow("  /temp 0.3–1.5"))
            elif cmd == "/size":
                if arg.isdigit() and 64 <= int(arg) <= 4096: out_size = int(arg); say(dim(f"  output {out_size}px"))
                else: say(yellow("  /size 64–4096"))
            elif cmd == "/open": auto_open = arg.lower() != "off"; say(dim(f"  auto-open {'on' if auto_open else 'off'}"))
            elif cmd == "/words":
                by = {}
                for w, hits in _V.items():
                    for a, s in hits: by.setdefault((a, s > 0), []).append(w)
                for a in ATTRS:
                    for sgn in (True, False):
                        ws = by.get((a, sgn))
                        if ws: say(f"  {a:<10} {'+' if sgn else '−'}  " + dim(" ".join(sorted(set(ws)))))
                say(dim("  modifiers: very / slightly / not / less / more … and portfolio tags (/tags)"))
            elif cmd == "/tags":
                tc = painter.style["tag_counts"]
                if tc: say("  " + dim(", ".join(f"{t}({c})" for t, c in sorted(tc.items(), key=lambda kv: -kv[1]))))
                else: say(dim("  no tags — name files like 'harbour-dusk-03.jpg' or use sub-folders to teach it words"))
            elif cmd == "/gallery": GALLERY.mkdir(exist_ok=True); open_file(GALLERY)
            elif cmd == "/status":
                say(f"  model: {BEST if BEST.exists() else LATEST}\n  portfolio: {painter.cfg.portfolio}\n  step {painter.step} · "
                    f"canvas {painter.cfg.width}×{painter.cfg.height} · latent {painter.cfg.latent} · temperature {painter.temperature} · count {count} · out {out_size}px")
            elif cmd == "/train":
                folder = arg or painter.cfg.portfolio
                ns = argparse.Namespace(portfolio=folder, size=None, canvas=None, batch=None, steps=3000, minutes=30,
                                        patience=1500, seed=0, fresh=False, no_open=not auto_open, init="auto")
                train(ns); painter = make_engine("vae") if not neural else painter
            else: say(yellow("  unknown command — /help"))
            continue
        recipe = painter.optimizer.optimize(line)
        recipe.anchor = anchor
        run(recipe)

# --------------------------------------------------------------------------- CLI
def main(argv=None):
    ap = argparse.ArgumentParser(prog="atelier", description="Learn an artist's style locally and paint from words.")
    sub = ap.add_subparsers(dest="cmd")

    t = sub.add_parser("train", help="train on a folder of jpg/png (resumes if a model exists)")
    t.add_argument("portfolio"); t.add_argument("--size", type=int, default=None,
                   help="square canvas in px, any multiple of 8 ≥32 (default 64 on CPU, 128 on GPU)")
    t.add_argument("--canvas", default=None, help="rectangular canvas WIDTHxHEIGHT, e.g. 160x96 or 96x128")
    t.add_argument("--steps", type=int, default=8000); t.add_argument("--minutes", type=float, default=45)
    t.add_argument("--patience", type=int, default=1500); t.add_argument("--batch", type=int, default=None)
    t.add_argument("--seed", type=int, default=0); t.add_argument("--fresh", action="store_true", help="ignore existing model")
    t.add_argument("--init", choices=["auto", "cifar10", "none"], default="auto",
                   help="auto: start from a CIFAR-10 baseline if one exists · cifar10: build one first if needed · none: from scratch")
    t.add_argument("--pretrain-steps", type=int, default=4000); t.add_argument("--pretrain-minutes", type=float, default=30)
    t.add_argument("--no-open", action="store_true")

    pt = sub.add_parser("pretrain", help="build the CIFAR-10 baseline (transfer-learning starting point for train)")
    pt.add_argument("--size", type=int, default=None); pt.add_argument("--canvas", default=None)
    pt.add_argument("--steps", type=int, default=4000); pt.add_argument("--minutes", type=float, default=30)
    pt.add_argument("--patience", type=int, default=2000); pt.add_argument("--batch", type=int, default=None)
    pt.add_argument("--seed", type=int, default=0); pt.add_argument("--fresh", action="store_true"); pt.add_argument("--no-open", action="store_true")

    def engine_flags(sp):
        sp.add_argument("--portfolio", default=None, help="folder of the artist's pieces (remembered after first use)")
        sp.add_argument("--engine", choices=["auto", "neural", "vae"], default="auto",
                        help="neural: SD-Turbo content + style transfer (default) · vae: the small trained actor/critic")
        sp.add_argument("--res", type=int, default=384, help="style-transfer working resolution (256 fast … 512 fine)")
        sp.add_argument("--iters", type=int, default=250, help="style-transfer iterations")

    s = sub.add_parser("studio", help="interactive text studio")
    s.add_argument("--count", type=int, default=4); s.add_argument("--size", type=int, default=512, help="output long side in px")
    s.add_argument("--no-open", action="store_true"); engine_flags(s)

    p = sub.add_parser("paint", help="one prompt, then exit (for scripts and agents)")
    p.add_argument("prompt"); p.add_argument("--count", type=int, default=4); p.add_argument("--size", type=int, default=512)
    p.add_argument("--temp", type=float, default=None); p.add_argument("--seed", type=int, default=None)
    p.add_argument("--like", "--content", dest="like", default=None, help="repaint this image (photo/sketch) as-is instead of drawing the words")
    p.add_argument("--sketch", default=None, help="a sketch/photo as the STARTING POINT: the words reinterpret it (SD-Turbo img2img), then the style is applied")
    p.add_argument("--freedom", type=float, default=0.55, help="with --sketch: 0.2 stays close to the sketch … 0.9 loosely inspired")
    p.add_argument("--structure", type=float, default=0.5, help="composition lock: 0 = style may rewrite the layout … 1 = the drawing's layout is kept")
    p.add_argument("--eye", type=float, default=1.0, help="weight of the artist's trained VAE prior (0 = off; needs `train-eye`)")
    p.add_argument("--no-open", action="store_true")
    p.add_argument("--json", action="store_true", help="print a JSON result instead of prose"); engine_flags(p)

    pp = sub.add_parser("prep", help="photos of drawings → clean JPEGs (crop to the sheet of paper with --paper)")
    pp.add_argument("folder"); pp.add_argument("--out", default=None); pp.add_argument("--paper", action="store_true", help="crop to the white sheet")
    pp.add_argument("--max-side", type=int, default=1600)

    te = sub.add_parser("train-eye", help="train the artist's own VAE on the portfolio (the apprentice's eye)")
    te.add_argument("portfolio"); te.add_argument("--minutes", type=float, default=4.0)

    i = sub.add_parser("index", help="rebuild the word→style index from the portfolio")
    i.add_argument("portfolio")

    args = ap.parse_args(argv)
    if args.cmd == "train":
        train(args)
    elif args.cmd == "pretrain":
        pretrain(args)
    elif args.cmd == "prep":
        prep(args)
    elif args.cmd == "train-eye":
        import atelier_vae as V
        folder = Path(args.portfolio).expanduser().resolve()
        files = [f for f in find_images(folder)]
        model, meta = V.load_or_train(files, pick_device(), minutes=args.minutes, force=True)
        say(green(f"  the apprentice's eye is trained: {meta['patches']} patches, best at step {meta['best_step']} "
                  f"({meta['seconds']:.0f}s) → {V.model_path(files).name}"))
    elif args.cmd == "studio":
        studio(args)
    elif args.cmd == "paint":
        try:
            painter = make_engine(args.engine, args.portfolio, args.res, args.iters)
        except FileNotFoundError as e:
            sys.exit(f"  {e}")
        recipe = painter.optimizer.optimize(args.prompt)
        if args.like: recipe.anchor = str(Path(args.like).expanduser())
        if type(painter).__name__ == "NeuralEngine":
            picks, info = painter.paint(recipe, args.count, args.temp, args.seed,
                                        sketch=Path(args.sketch).expanduser() if getattr(args, "sketch", None) else None,
                                        freedom=args.freedom, structure=args.structure, eye=args.eye)
        else:
            picks, info = painter.paint(recipe, args.count, args.temp, args.seed)
        paths, sheet = save_results(picks, info, recipe, args.size, not args.no_open and not args.json)
        if args.json:
            print(json.dumps({"prompt": args.prompt, "optimized": recipe.attrs, "tags": recipe.tags, "ignored": recipe.ignored,
                              "files": [str(x) for x in paths], "sheet": str(sheet), "info": info}, indent=2))
        else:
            say(recipe.explain()); say(green(f"  ✔ saved {len(paths)} pieces → {sheet}"))
    elif args.cmd == "index":
        ck = BEST if BEST.exists() else LATEST
        if not ck.exists(): sys.exit("no model yet — train first")
        cfg = cfg_from(torch.load(ck, map_location="cpu")["cfg"])
        build_style_index(ck, Portfolio(Path(args.portfolio), cfg.height, cfg.width), pick_device())
        say(green("  index rebuilt"))
    else:
        # guided mode: a portfolio folder is all it needs
        folder = known_portfolio()
        if folder is None:
            say(bold("\n◆ Atelier"))
            say("  Drop the folder with the artist's pieces here (or type its path), then press Enter.")
            try:
                folder = input(cyan("  portfolio folder: ")).strip().strip("'\"")
            except (EOFError, KeyboardInterrupt):
                say(); return
            if not folder: return
        studio(argparse.Namespace(count=4, size=512, no_open=False, engine="auto", portfolio=str(folder), res=384, iters=250))

if __name__ == "__main__":
    main()
