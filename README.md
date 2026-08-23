# Atelier

**An apprentice for your studio.** You draw the outline, it lays in the colour — in your palette, your hand — and you do the shading. It learns your style from your own pieces and runs on your own computer; your work never leaves the machine.

Give Atelier a few of an artist's pieces (JPG, PNG, phone HEIC — photographed drawings are cropped to the paper) and say what you'd like to see. It draws what you describe, then repaints it the way the artist paints. Start from a sketch if you like, turn the style up or down, ask for one piece or eight.

Website: https://seaneven-web.github.io/atelier/ · Downloads: https://github.com/seaneven-web/atelier/releases

## The app

```
python atelier_app.py        # or double-click the Mac / Windows / Linux build
```

Three benches: **the artist's work** → **ask for a piece** (prompt, optional sketch, sliders: style strength · sketch freedom · quality · detail · pieces · seed) → **the pieces** (open, save, variations). A **History** rail lists every past run (newest first, grouped by day); click one to see its pieces and settings again, *run again* with fresh noise, *load its settings* into the controls, or delete it. Each run is a small JSON record next to its images in the gallery folder, so the history survives restarts and is yours to keep. One-time setup downloads the two pretrained models (~3 GB) into the app's folder; after that it runs offline.

**Private by construction.** Everything — the artist's pieces, sketches, results, the per-artist style index and the downloaded models — lives in one folder:

| OS | folder |
|---|---|
| macOS | `~/Library/Application Support/Atelier` |
| Windows | `%LOCALAPPDATA%\Atelier` |
| Linux | `~/.local/share/atelier` |

No account, no upload, no server. The only network access is the one-time model download. Nothing trains a shared model on the artist's work: the "style book" is a small file describing this artist's palette and texture, used only to paint for them.

## How it paints

| stage | model | what it does |
|---|---|---|
| 1 · content | **SD-Turbo** (pretrained diffusion UNet, CPU-friendly) | draws what the words describe, in the artist's medium (the style book tells it coloured-pencil-on-paper vs. oil); or reinterprets your sketch (img2img, "sketch freedom") |
| 2 · style | **VGG19 neural style transfer** (Gatys — the "deep style" method) | repaints it with the portfolio's palette and brushwork; the pieces your words select (tags, mood words) define the style |
| 3 · grade + rank | — | mood words as a gentle colour grade; pieces ranked by how completely they took the style |

No training is required. A portfolio is indexed once into a cached *style book*. The experimental trained actor/critic (a VAE-GAN that learns palette/texture and paints abstract pieces; `train`, `--engine vae`) is still in the repo.

## Command line

```
python atelier.py paint "a hippo eating cheese" --portfolio ~/DadsArt [--count 4] [--temp 0.8] [--json]
python atelier.py paint "a fox reading" --sketch sketch.jpg --freedom 0.6      # sketch as starting point
python atelier.py paint "quiet, at dusk" --content photo.jpg                   # repaint an image as-is
python atelier.py prep ~/Photos --paper      # photographed drawings → clean JPEGs cropped to the sheet
python atelier.py studio                     # text studio (word engine)
python atelier_claude.py                     # Claude Opus directs the painter (needs an API key)
python atelier.py pretrain / train ~/DadsArt # optional: the small trained actor/critic
```

Knobs: `--temp` (style strength 0.4 gentle … 1.2 fully repainted), `--res 256|384|512`, `--iters 250`, `--size` (output long side). `ATELIER_HOME` relocates model/gallery; `ATELIER_DEVICE=cpu|mps|cuda` forces the device.

## Install from source

```
git clone https://github.com/seaneven-web/atelier && cd atelier
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt -r requirements-desktop.txt
.venv/bin/python atelier_app.py
```

Intel Macs get PyTorch 2.2.2 (the last x86_64 wheel) and matching diffusers pins automatically (see `requirements.txt`). HEIC needs `pillow-heif` (included).

## Building the desktop apps

`packaging/build_macos.sh`, `packaging/build_windows.ps1`, `packaging/build_linux.sh` (PyInstaller, onedir, windowed; headless smoke test). CI builds all four bundles on every push and publishes a GitHub Release on a `v*` tag. See `packaging/README.md` (unsigned-build caveats).

## Honest limits

Style transfer carries palette, texture and tone well; composition comes from the content model or your sketch. More pieces help; cleanly photographed pieces help most. A LoRA fine-tune of the content model would learn composition too — it needs a real GPU. About a minute per piece on a recent laptop (CPU for the content model, GPU when present for the style pass).

## The web preview

`atelier_web.html` is a self-contained browser page (also published as a Claude artifact) that previews the *mood/palette* half of the workflow in-browser with a small statistical painter; it cannot draw subjects. The app and the CLI are the real engine.
