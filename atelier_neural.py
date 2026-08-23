#!/usr/bin/env python3
"""
Atelier · neural engine — pretrained models that are known to work, as the actor.

    content  Stable Diffusion Turbo (a UNet diffusion model) renders what the words describe.
    style    VGG19 neural style transfer (Gatys et al.) repaints it with the portfolio's palette and
             brushwork — style statistics (Gram matrices) averaged over the pieces the prompt selects.
    grade    the recipe's mood words (light, warmth, contrast…) applied as a final colour grade.
    critic   candidates ranked by how completely they took on the style (final style loss).

No training is required. A portfolio folder is indexed on first use (a "style book": tags, tone/palette
statistics, and per-piece style statistics) and cached in model/stylebook.pt.

    python atelier.py paint "a hippo eating cheese" --portfolio ~/DadsArt
    python atelier.py paint "harbour at dusk, quiet" --content photo.jpg      # style a photo/sketch instead of generating

Requires: torch, torchvision (VGG19 weights download once, ~550 MB). Text-to-image needs
`pip install diffusers transformers accelerate safetensors` (SD-Turbo downloads once, ~2.5 GB);
without it, pass --content to style an image you provide.
"""
from __future__ import annotations

import hashlib
import math
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import atelier as A

STYLE_RES = 512          # working resolution for style transfer
STYLE_CROPS = 4          # native-scale crops per piece — brush marks at their true size
BOOK_VERSION = 2         # bump to invalidate cached style books
CONTENT_RES = 512        # SD-Turbo native

# --------------------------------------------------------------------------- VGG19 features
_VGG = None
def vgg():
    global _VGG
    if _VGG is None:
        from torchvision.models import vgg19, VGG19_Weights
        m = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features.eval()
        for p in m.parameters(): p.requires_grad_(False)
        # average pooling gives smoother gradients for style transfer (as in the original paper)
        for i, l in enumerate(m):
            if isinstance(l, torch.nn.MaxPool2d): m[i] = torch.nn.AvgPool2d(2)
        _VGG = m
    return _VGG

STYLE_LAYERS = {1: "relu1_1", 6: "relu2_1", 11: "relu3_1", 20: "relu4_1", 29: "relu5_1"}   # indices in vgg19.features
# shallow layers carry palette and brush texture; deep layers carry *objects* — matching those is what
# hallucinates the artist's subjects into a new scene. So: weight the shallow ones, barely touch the deep.
STYLE_LAYER_W = {1: 1.0, 6: 0.8, 11: 0.5, 20: 0.15, 29: 0.05}
CONTENT_LAYERS = {8: 0.6, 22: 1.0}                                                       # relu2_2 (edges) + relu4_2 (layout)
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

def features(x: torch.Tensor, net, device) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
    """x in [0,1], B×3×H×W → ({style layer: fmap}, {content layer: fmap})."""
    h = (x - _MEAN.to(device)) / _STD.to(device)
    out, content = {}, {}
    last = max(max(STYLE_LAYERS), max(CONTENT_LAYERS))
    for i, layer in enumerate(net):
        h = layer(h)
        if i in STYLE_LAYERS: out[i] = h
        if i in CONTENT_LAYERS: content[i] = h
        if i >= last: break
    return out, content

def _style_views(f: Path, res: int) -> List[torch.Tensor]:
    """[whole piece fitted to res] + up to STYLE_CROPS res-sized crops at the piece's NATIVE scale.
    Consistent framing, true brush scale — the lesson from the face-crop VAE this engine borrows from."""
    big = A.load_image(f, min(1600, res * 3))
    _, H, W = big.shape
    side = min(H, W)
    whole = big[:, (H - side) // 2:(H - side) // 2 + side, (W - side) // 2:(W - side) // 2 + side].float().div(255)[None]
    views = [F.interpolate(whole, size=(res, res), mode="bilinear", align_corners=False, antialias=True)]
    if H >= res and W >= res:                       # deterministic grid of native-scale crops
        ys = [0, H - res] if H > res else [0]
        xs = [0, W - res] if W > res else [0]
        spots = [(y, x) for y in ys for x in xs][:STYLE_CROPS]
        if len(spots) < STYLE_CROPS: spots.append(((H - res) // 2, (W - res) // 2))
        for y, x in spots:
            views.append(big[:, y:y + res, x:x + res].float().div(255)[None])
    return views

def gram(f: torch.Tensor) -> torch.Tensor:
    b, c, h, w = f.shape
    f = f.reshape(b, c, h * w)
    return f @ f.transpose(1, 2) / (c * h * w)

# --------------------------------------------------------------------------- the style book (portfolio index, no training)
class StyleBook:
    """Everything the engine needs to know about a portfolio, computed once and cached."""

    def __init__(self, portfolio: Path, device: torch.device, quiet: bool = False):
        self.root = Path(portfolio).expanduser().resolve()
        self.device = device
        files = A.find_images(self.root)
        if len(files) < 1:
            raise SystemExit(f"no images in {self.root}")
        sig = hashlib.md5(("|".join(f"{f}:{f.stat().st_mtime_ns}" for f in files) + f"|v{BOOK_VERSION}").encode()).hexdigest()[:12]
        A.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        self.cache = A.MODEL_DIR / f"stylebook_{sig}.pt"
        if self.cache.exists():
            d = torch.load(self.cache, map_location="cpu")
            self.files, self.tags, self.attrs, self.grams = [Path(f) for f in d["files"]], d["tags"], d["attrs"], d["grams"]
        else:
            if not quiet: A.say(A.dim(f"  reading the portfolio into a style book ({len(files)} pieces)…"))
            self.files, self.tags, attrs, grams = [], [], [], []
            net = vgg().to(device)
            for i, f in enumerate(files):
                try:
                    views = _style_views(f, STYLE_RES)
                except Exception as e:
                    A.say(A.yellow(f"  skipping {f.name}: {e}")); continue
                # Gram averaged over the whole piece AND native-scale crops: the crops carry the brush
                # marks at their true size, which a squashed thumbnail throws away (that loss of scale
                # is what made the texture read as blur).
                acc: Dict[int, torch.Tensor] = {}
                with torch.no_grad():
                    for v in views:
                        fs, _ = features(v.to(device), net, device)
                        for k, val in fs.items():
                            g = gram(val)[0].cpu()
                            acc[k] = g if k not in acc else acc[k] + g
                grams.append({k: v / len(views) for k, v in acc.items()})
                attrs.append(A.image_attributes(views[0])[0])
                self.files.append(f); self.tags.append(A.tags_for(f, self.root))
                if not quiet and (i + 1) % 10 == 0: A.say(A.dim(f"  {i + 1}/{len(files)}"))
            self.attrs = torch.stack(attrs); self.grams = grams
            torch.save({"files": [str(f) for f in self.files], "tags": self.tags, "attrs": self.attrs, "grams": self.grams}, self.cache)
        self.a_mean, self.a_std = self.attrs.mean(0), self.attrs.std(0).clamp_min(1e-4) if len(self.files) > 1 else torch.ones(10) * 0.1
        counts: Dict[str, int] = {}
        for tl in self.tags:
            for t in tl: counts[t] = counts.get(t, 0) + 1
        self.tag_counts = {t: c for t, c in counts.items() if c >= 1}
        self.name = self.root.name

    def select(self, recipe: A.Recipe, k: int = 6) -> List[int]:
        """Which pieces define the style for this recipe: tagged ones first, then the closest in mood."""
        idx = list(range(len(self.files)))
        if recipe.anchor:
            hit = [i for i, f in enumerate(self.files) if str(f) == str(recipe.anchor) or f.name == Path(recipe.anchor).name]
            if hit: return hit
        if recipe.tags:
            tagged = [i for i in idx if any(t in self.tags[i] for t in recipe.tags)]
            if tagged: idx = tagged
        want = {a: w for a, w in recipe.attrs.items() if abs(w) > 0.05}
        if want and len(idx) > k:
            z = (self.attrs[idx] - self.a_mean) / self.a_std
            tgt = torch.tensor([want.get(a, 0.0) for a in A.ATTRS])
            mask = torch.tensor([1.0 if a in want else 0.0 for a in A.ATTRS])
            d = (((z - tgt) ** 2) * mask).sum(1)
            idx = [idx[i] for i in d.argsort()[:k].tolist()]
            return idx
        # no steering words: the artist's overall style = all pieces (a fixed sample of 30 for big portfolios)
        if len(idx) > 30:
            import random as _r; idx = sorted(_r.Random(0).sample(idx, 30))
        return idx

# --------------------------------------------------------------------------- content (text → image)
_PIPE = None
SD_IMPORT_ERROR: Optional[str] = None
def sd_available() -> bool:
    """Can this install draw from words? Records the real import error for the UI/log when it can't."""
    global SD_IMPORT_ERROR
    try:
        import diffusers  # noqa: F401
        from diffusers import AutoPipelineForText2Image  # noqa: F401
        import transformers  # noqa: F401
        SD_IMPORT_ERROR = None
        return True
    except Exception as e:  # ImportError, OSError (missing dylib), RuntimeError (version checks) …
        import traceback
        SD_IMPORT_ERROR = f"{type(e).__name__}: {e}"
        try: A.say(A.dim("  text-to-image unavailable: " + "".join(traceback.format_exception(type(e), e, e.__traceback__))[-1500:]))
        except Exception: pass
        return False

def content_from_text(prompt: str, seed: int, steps: int = 4, size: int = CONTENT_RES) -> Image.Image:
    """SD-Turbo on CPU (reliable on every machine; ~20–60 s per image). Cached pipeline."""
    global _PIPE
    if _PIPE is None:
        from diffusers import AutoPipelineForText2Image
        A.say(A.dim("  loading SD-Turbo (first time downloads ~2.5 GB)…"))
        _PIPE = AutoPipelineForText2Image.from_pretrained("stabilityai/sd-turbo", torch_dtype=torch.float32, variant="fp16")
        _PIPE.set_progress_bar_config(disable=True)
        try: _PIPE.safety_checker = None
        except Exception: pass
    g = torch.Generator().manual_seed(int(seed))
    return _PIPE(prompt=prompt, num_inference_steps=steps, guidance_scale=0.0, height=size, width=size, generator=g).images[0]

def medium_hint(book) -> str:
    """Guess the portfolio's medium from its tone statistics so the content model draws in kind —
    the style transfer then has much less to fight."""
    b, c, sat, _, det = [float(v) for v in book.a_mean[:5]]
    if b > 0.62 and sat < 0.35:
        return "simple coloured pencil and crayon drawing on white paper, clear outlines, children's illustration"
    if b < 0.35:
        return "dark moody painting, artwork"
    if sat > 0.5:
        return "vivid painting, bold colours, artwork"
    return "painting, artwork, clear composition"

_I2I = None
def content_from_sketch(sketch: Image.Image, prompt: str, seed: int, freedom: float = 0.55, steps: int = 4,
                        size: int = CONTENT_RES) -> Image.Image:
    """A sketch/photo as the starting point: SD-Turbo img2img. freedom 0.2 = stay close to the sketch,
    0.9 = only loosely inspired by it. (strength = freedom; steps scale so ≥1 denoising step always runs)"""
    global _PIPE, _I2I
    if _PIPE is None:
        content_from_text("warm-up", 0, steps=1, size=64)   # loads the shared components
    if _I2I is None:
        from diffusers import AutoPipelineForImage2Image
        _I2I = AutoPipelineForImage2Image.from_pipe(_PIPE)
        _I2I.set_progress_bar_config(disable=True)
    freedom = float(max(0.1, min(0.95, freedom)))
    steps = max(2, int(round(2 / freedom)))                  # strength·steps ≥ 1 (diffusers skips otherwise)
    img = ImageOpsFit(sketch, size)
    g = torch.Generator().manual_seed(int(seed))
    return _I2I(prompt=prompt, image=img, num_inference_steps=steps, strength=freedom, guidance_scale=0.0, generator=g).images[0]

def ImageOpsFit(img: Image.Image, size: int) -> Image.Image:
    """Centre-crop to square and resize (keeps the drawing's main subject)."""
    from PIL import ImageOps
    im = ImageOps.exif_transpose(img).convert("RGB")
    return ImageOps.fit(im, (size, size), Image.LANCZOS)

def content_prompt(text: str, recipe: A.Recipe, book=None) -> str:
    """What SD draws: the person's words, plus the portfolio's medium so the style takes well."""
    t = text.strip().rstrip(".")
    return f"{t}, {medium_hint(book) if book is not None else 'painting, artwork, clear composition'}"

# --------------------------------------------------------------------------- style transfer
def _gaussian(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable gaussian blur (MPS-safe: plain conv2d, no antialiased resize)."""
    r = max(1, int(round(3 * sigma))); n = 2 * r + 1
    k = torch.exp(-torch.arange(-r, r + 1, dtype=x.dtype, device=x.device) ** 2 / (2 * sigma * sigma)); k = k / k.sum()
    C = x.shape[1]
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="reflect"), k.view(1, 1, 1, n).expand(C, 1, 1, n), groups=C)
    return F.conv2d(F.pad(x, (0, 0, r, r), mode="reflect"), k.view(1, 1, n, 1).expand(C, 1, n, 1), groups=C)

def _luma(x: torch.Tensor) -> torch.Tensor:
    return (0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]).unsqueeze(1)

def structure_lock(styled: torch.Tensor, content: torch.Tensor, amount: float) -> torch.Tensor:
    """Keep the drawing's large-scale layout, keep the style's colour and fine detail.

    Composition lives in the low frequencies; brushwork and palette live above them. Style transfer
    is free to rewrite both, and when it rewrites the low frequencies you get the artist's *subjects*
    smeared over your scene. So: replace the styled image's low-frequency luminance with the drawing's
    (rescaled to the styled image's tonal range, so the grade survives) and leave everything else —
    all colour, all detail — exactly as the style transfer painted it."""
    if amount <= 0: return styled
    sigma = max(2.5, min(styled.shape[-2:]) / 40)
    ls, lc = _luma(styled), _luma(content)
    los, loc = _gaussian(ls, sigma), _gaussian(lc, sigma)
    loc = (loc - loc.mean((2, 3), keepdim=True)) / loc.std((2, 3), keepdim=True).clamp_min(1e-5)
    loc = loc * los.std((2, 3), keepdim=True) + los.mean((2, 3), keepdim=True)
    return (styled + amount * (loc - los)).clamp(0, 1)

def _box(x: torch.Tensor, r: int) -> torch.Tensor:
    k = 2 * r + 1
    return F.avg_pool2d(F.pad(x, (r, r, r, r), mode="reflect"), k, stride=1)

def guided_filter(p: torch.Tensor, guide: torch.Tensor, r: int = 4, eps: float = 1e-3) -> torch.Tensor:
    """He et al. guided filter: smooth p while keeping the edges of guide (the content drawing)."""
    g = guide.mean(1, keepdim=True)
    mean_g, mean_p = _box(g, r), _box(p, r)
    cov = _box(g * p, r) - mean_g * mean_p
    var = _box(g * g, r) - mean_g * mean_g
    a = cov / (var + eps); b = mean_p - a * mean_g
    return _box(a, r) * g + _box(b, r)

def _stylize(x0: torch.Tensor, G: Dict[int, torch.Tensor], Gn: Dict[int, torch.Tensor], c_target: Dict[int, torch.Tensor],
             c_scale: Dict[int, torch.Tensor], net, device, iters: int, sw: float, cw: float, tv_weight: float,
             lr: float, progress=None, offset: int = 0, total: int = 0, eye=None, eye_w: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
    x = x0.clone().requires_grad_(True)
    opt = torch.optim.Adam([x], lr=lr)
    last = torch.zeros(x0.shape[0])
    for it in range(iters):
        if it == int(iters * 0.6):
            for g in opt.param_groups: g["lr"] = lr * 0.4
        opt.zero_grad(set_to_none=True)
        fs, cs = features(x, net, device)
        ls = torch.zeros(x0.shape[0], device=device)
        for k, f in fs.items():
            ls = ls + STYLE_LAYER_W[k] * ((gram(f) - G[k]) ** 2).flatten(1).sum(1) / Gn[k]
        lc = torch.zeros(x0.shape[0], device=device)
        for k, f in cs.items():
            lc = lc + CONTENT_LAYERS[k] * ((f - c_target[k]) ** 2).flatten(1).mean(1) / c_scale[k]
        tv = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().flatten(1).mean(1) + (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().flatten(1).mean(1)
        loss = (sw * ls + cw * lc + tv_weight * tv).sum()
        if eye is not None and eye_w > 0:            # the artist's own VAE: is this paint in their vocabulary?
            import atelier_vae as V
            loss = loss + eye_w * V.prior_loss(eye, x).sum()
        loss.backward(); opt.step()
        with torch.no_grad(): x.clamp_(0, 1)
        last = ls.detach().cpu()
        if progress and (it % 25 == 0 or it == iters - 1): progress(offset + it + 1, total, float(last.mean()))
    return x.detach(), last

def transfer(contents: torch.Tensor, grams_target: Dict[int, torch.Tensor], device, iters: int = 250,
             style_weight: float = 30.0, content_weight: float = 1.0, tv_weight: float = 0.6, strength: float = 1.0,
             jitter: float = 0.0, progress=None, clean: float = 0.35, structure: float = 0.5,
             eye=None, eye_weight: float = 12.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """Gatys style transfer, batched and coarse-to-fine. contents: B×3×H×W in [0,1].
    Style loss per layer is *relative* (‖G−G*‖²/‖G*‖²) and weighted toward the shallow layers (palette,
    brush) so the artist's *subjects* don't get hallucinated into the scene; content is held at two depths
    (edges + layout). The large shapes are settled at 2/3 resolution first, then refined at full size —
    cleaner composition, far less texture noise. A light guided filter (edges from the content) finishes.
    Returns (images, final style loss per image)."""
    net = vgg().to(device)
    x_full = contents.to(device)
    B, _, H, W = x_full.shape
    G = {k: v.to(device)[None] for k, v in grams_target.items()}
    Gn = {k: (v ** 2).sum().clamp_min(1e-12) for k, v in G.items()}
    sw = style_weight * strength
    cw = content_weight * (1.0 + 0.8 * max(0.0, 1.0 - strength))     # gentle style → hold the drawing harder
    coarse = (max(96, int(H * 2 / 3) // 16 * 16), max(96, int(W * 2 / 3) // 16 * 16))
    stages = [(coarse, int(iters * 0.55)), ((H, W), iters - int(iters * 0.55))] if min(H, W) >= 320 else [((H, W), iters)]
    x, done, last = None, 0, torch.zeros(B)
    for si, ((h, w), n) in enumerate(stages):
        xc = F.interpolate(x_full.cpu(), size=(h, w), mode="bilinear", align_corners=False, antialias=True).to(device) if (h, w) != (H, W) else x_full   # antialias on CPU (not on MPS)
        with torch.no_grad():
            _, c_t = features(xc, net, device)
            c_scale = {k: (v ** 2).mean().clamp_min(1e-6) for k, v in c_t.items()}
        if x is None:
            x = (xc + jitter * torch.randn_like(xc)).clamp(0, 1)
        else:
            x = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)
        lr = 0.03 if si == 0 else 0.015
        x, last = _stylize(x, G, Gn, c_t, c_scale, net, device, n, sw, cw, tv_weight, lr, progress, done, iters,
                           eye=eye, eye_w=eye_weight)
        done += n
    with torch.no_grad():
        if clean > 0:
            x = (1 - clean) * x + clean * guided_filter(x, x_full, r=4, eps=2e-3).clamp(0, 1)
        x = structure_lock(x, x_full, structure)
    return x.cpu(), last

def grade(x: torch.Tensor, recipe: A.Recipe, book: StyleBook) -> torch.Tensor:
    """Mood words → colour grade: move brightness/contrast/saturation/warmth toward mean + w·std."""
    want = {a: w for a, w in recipe.attrs.items() if abs(w) > 0.05 and a in ("brightness", "contrast", "saturation", "warmth")}
    if not want: return x
    out = x.clone()
    for _ in range(2):
        a = A.image_attributes(out)
        for i in range(out.shape[0]):
            img = out[i]
            lum = 0.299 * img[0] + 0.587 * img[1] + 0.114 * img[2]
            tgt = {k: float(book.a_mean[A.ATTRS.index(k)] + 0.6 * w * book.a_std[A.ATTRS.index(k)]) for k, w in want.items()}
            if "saturation" in tgt:
                ks = float(max(0.5, min(1.9, tgt["saturation"] / (a[i, 2] + 1e-4)))); img = lum + (img - lum) * ks
            if "contrast" in tgt or "brightness" in tgt:
                m = float(a[i, 0]); kc = float(max(0.6, min(1.8, tgt.get("contrast", float(a[i, 1])) / (a[i, 1] + 1e-4))))
                db = tgt.get("brightness", m) - m; img = m + (img - m) * kc + db
            if "warmth" in tgt:
                dw = tgt["warmth"] - float(a[i, 3]); img = torch.stack([img[0] + dw / 2, img[1], img[2] - dw / 2])
            out[i] = img.clamp(0, 1)
    return out

# --------------------------------------------------------------------------- the engine
class NeuralEngine:
    """Same interface the studio and the Claude app use: paint(recipe, count, temperature, seed) → (images, info)."""

    def __init__(self, portfolio: Path, device: Optional[torch.device] = None, res: int = STYLE_RES, iters: int = 250):
        self.device = device or A.pick_device()
        self.book = StyleBook(portfolio, self.device)
        self.res, self.iters = res, iters
        self.optimizer = A.PromptOptimizer(self.book.tag_counts)
        self.temperature = 0.8
        self.last_recipe: Optional[A.Recipe] = None
        self.cfg = type("Cfg", (), {"name": self.book.name, "width": res, "height": res, "portfolio": str(self.book.root)})()
        self.style = {"files": [str(f) for f in self.book.files], "tag_counts": self.book.tag_counts, "tags": {t: None for t in self.book.tag_counts},
                      "a_mean": self.book.a_mean, "a_std": self.book.a_std}
        self.step = 0

    def _content_batch(self, recipe: A.Recipe, count: int, seed: Optional[int], content: Optional[Path], log,
                       sketch: Optional[Path] = None, freedom: float = 0.55) -> torch.Tensor:
        imgs = []
        if sketch is not None:
            if not sd_available():
                raise RuntimeError("A sketch as starting point needs SD-Turbo (pip install diffusers transformers accelerate safetensors); "
                                   "without it use --content to repaint the sketch as-is.")
            base = seed if seed is not None else int(time.time()) % 100000
            sk = Image.open(sketch)
            for i in range(count):
                t0 = time.time()
                im = content_from_sketch(sk, content_prompt(recipe.text, recipe, self.book), base + i * 7919, freedom)
                log(f"sketch → content {i + 1}/{count} in {time.time() - t0:.0f}s (freedom {freedom:.2f})")
                x = torch.from_numpy(np.asarray(im.convert("RGB")).copy()).permute(2, 0, 1).float().div(255)[None]
                imgs.append(F.interpolate(x, size=(self.res, self.res), mode="bilinear", align_corners=False, antialias=True))
            return torch.cat(imgs)
        if content is not None:
            t = A.load_image(Path(content), self.res); _, h, w = t.shape; s = min(h, w)
            x = t.float().div(255)[None, :, (h - s) // 2:(h - s) // 2 + s, (w - s) // 2:(w - s) // 2 + s]
            x = F.interpolate(x, size=(self.res, self.res), mode="bilinear", align_corners=False, antialias=True)
            return x.repeat(count, 1, 1, 1)
        if not sd_available():
            raise RuntimeError(f"Text-to-image is unavailable in this install ({SD_IMPORT_ERROR}). "
                               "From source: pip install diffusers transformers accelerate safetensors. "
                               "Meanwhile, start from a sketch or use --content <image> to style an image instead.")
        base = seed if seed is not None else int(time.time()) % 100000
        for i in range(count):
            t0 = time.time()
            im = content_from_text(content_prompt(recipe.text, recipe, self.book), base + i * 7919)
            log(f"content {i + 1}/{count} drawn in {time.time() - t0:.0f}s")
            x = torch.from_numpy(np.asarray(im.convert("RGB")).copy()).permute(2, 0, 1).float().div(255)[None]
            imgs.append(F.interpolate(x, size=(self.res, self.res), mode="bilinear", align_corners=False, antialias=True))
        return torch.cat(imgs)

    @torch.no_grad()
    def _rank(self, X: torch.Tensor, style_loss: torch.Tensor, recipe: A.Recipe):
        A_ = (A.image_attributes(X) - self.book.a_mean) / self.book.a_std
        want = {a: w for a, w in recipe.attrs.items() if abs(w) > 0.05}
        # style fit: relative to the batch (lower loss = took the style better)
        rel = (style_loss - style_loss.min()) / (style_loss.max() - style_loss.min() + 1e-9)
        fit = 1 - 0.6 * rel
        miss = torch.zeros(X.shape[0])
        if want:
            cols = [A.ATTRS.index(a) for a in want]; tgt = torch.tensor([want[a] for a in want])
            miss = ((A_[:, cols] - tgt) ** 2).mean(1)
        score = fit - 0.15 * miss
        order = score.argsort(descending=True).tolist()
        info = [{"critic": float(max(0.05, min(0.99, fit[i] * 0.9))), "style_loss": float(style_loss[i]),
                 "achieved": {a: float(A_[i, A.ATTRS.index(a)]) for a in want}} for i in order]
        return [X[i] for i in order], info

    def paint(self, recipe: A.Recipe, count: int = 4, temperature: Optional[float] = None, seed: Optional[int] = None,
              content: Optional[Path] = None, log=None, strength: Optional[float] = None,
              sketch: Optional[Path] = None, freedom: float = 0.55, iters: Optional[int] = None,
              structure: float = 0.5, eye: float = 1.0):
        log = log or (lambda s: A.say(A.dim(f"    {s}")))
        temp = self.temperature if temperature is None else temperature
        if content is None and recipe.anchor and Path(recipe.anchor).exists():
            content = Path(recipe.anchor)            # an anchored image is the content to repaint
        pieces = self.book.select(recipe)
        log(f"style from {len(pieces)} piece(s): " + ", ".join(self.book.files[i].name for i in pieces[:4]) + ("…" if len(pieces) > 4 else ""))
        G = {k: torch.stack([self.book.grams[i][k] for i in pieces]).mean(0) for k in STYLE_LAYERS}
        X0 = self._content_batch(recipe, count, seed, content, log, sketch=sketch, freedom=freedom)
        # temperature = how hard the style is pushed (0.3 gentle … 1.3 fully repainted)
        st = strength if strength is not None else max(0.25, min(2.0, temp))
        t0 = time.time()
        eye_model, eye_w = None, 0.0
        if eye > 0:
            try:
                import atelier_vae as V
                eye_model, meta = V.find_trained(self.book.files, self.device)
                if eye_model is not None:
                    eye_w = 12.0 * float(eye)
                    log(f"the apprentice's own eye is trained on this portfolio ({meta.get('patches', '?')} patches) — using it")
            except Exception as e:
                log(f"(trained eye unavailable: {e})")
        X, sl = transfer(X0, G, self.device, iters=iters or self.iters, strength=st, jitter=0.0,
                         clean=float(max(0.2, min(0.6, 0.7 - 0.4 * st))), structure=float(structure),
                         eye=eye_model, eye_weight=eye_w,
                         progress=lambda it, n, l: log(f"repainting in the style… {it}/{n}  style loss {l:.3g}"))
        log(f"style transfer done in {time.time() - t0:.0f}s")
        X = grade(X, recipe, self.book)
        picks, info = self._rank(X, sl, recipe)
        self.last_recipe = recipe
        return picks, info
