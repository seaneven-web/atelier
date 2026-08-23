#!/usr/bin/env python3
"""
Atelier desktop — the app. A local server + a window. Nothing leaves the machine.

    python atelier_app.py                 # opens the window (or the browser if no webview backend)
    ATELIER_NO_WINDOW=1 python atelier_app.py   # headless: prints "ATELIER READY <url>" and serves (CI smoke test)
    ATELIER_BROWSER=1   …                 # force the system browser instead of a native window

Everything lives in ONE folder (the sandbox):
    macOS    ~/Library/Application Support/Atelier
    Windows  %LOCALAPPDATA%\\Atelier
    Linux    ~/.local/share/atelier
    (override: ATELIER_DATA_DIR)
  ├─ portfolios/<name>/   the artist's pieces (copied in; optionally cropped to the paper)
  ├─ sketches/            starting-point sketches
  ├─ gallery/             every result
  ├─ model/               style books (the per-artist style index) + optional trained actor/critic
  └─ models/              the pretrained weights (VGG19, SD-Turbo) — the only thing ever downloaded
The only network access is that one-time model download. After it, the app works with the network off.

Python 3.9+. Stdlib HTTP server (no web framework); the painter is atelier_neural (torch).
"""
from __future__ import annotations

import base64
import io
import json
import mimetypes
import os
import platform
import shutil
import signal
import socket
import sys
import threading
import time
import traceback
import uuid
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import unquote, urlparse

__version__ = "0.1.3"
APP_NAME = "Atelier"

# --------------------------------------------------------------------------- the sandbox folder (set BEFORE importing torch / atelier)
def default_data_dir() -> Path:
    env = os.environ.get("ATELIER_DATA_DIR")
    if env: return Path(env).expanduser()
    sysname = platform.system()
    if sysname == "Darwin": return Path.home() / "Library" / "Application Support" / APP_NAME
    if sysname == "Windows": return Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / APP_NAME
    return Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")) / APP_NAME.lower()

DATA_DIR = default_data_dir()
for sub in ("portfolios", "sketches", "gallery", "model", "models", "logs"):
    (DATA_DIR / sub).mkdir(parents=True, exist_ok=True)
os.environ.setdefault("ATELIER_HOME", str(DATA_DIR))                      # atelier.py: model/ + gallery/ here
os.environ.setdefault("HF_HOME", str(DATA_DIR / "models" / "huggingface"))  # SD-Turbo lives in the sandbox
os.environ.setdefault("TORCH_HOME", str(DATA_DIR / "models" / "torch"))      # VGG19 too
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

def _frozen() -> bool: return bool(getattr(sys, "frozen", False))
BASE = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
if str(BASE) not in sys.path: sys.path.insert(0, str(BASE))
WEB_DIR = BASE / "web"

LOG = DATA_DIR / "logs" / "atelier.log"
def log(msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    try:
        with open(LOG, "a", encoding="utf-8") as f: f.write(line + "\n")
    except Exception: pass
    if sys.stdout is not None:
        try: print(line, flush=True)
        except Exception: pass

# heavy imports are deferred so the window opens fast and the smoke test doesn't need the models
_A = None; _N = None
def A():
    global _A
    if _A is None:
        import atelier; _A = atelier
    return _A
def N():
    global _N
    if _N is None:
        import atelier_neural; _N = atelier_neural
    return _N

# --------------------------------------------------------------------------- state
class Jobs:
    """One painting/download job at a time, with a log the UI polls."""
    def __init__(self):
        self.lock = threading.Lock(); self.jobs: Dict[str, dict] = {}; self.busy = False
    def start(self, kind: str, fn, *args) -> str:
        with self.lock:
            if self.busy: raise RuntimeError("the studio is busy — wait for the current job to finish")
            self.busy = True
            jid = uuid.uuid4().hex[:10]
            self.jobs[jid] = {"id": jid, "kind": kind, "status": "running", "log": [], "result": None, "error": None, "t0": time.time()}
        def run():
            job = self.jobs[jid]
            try:
                job["result"] = fn(lambda m: job["log"].append(m), *args)
                job["status"] = "done"
            except Exception as e:
                job["error"] = f"{e}"; job["status"] = "error"; log("job error: " + traceback.format_exc())
            finally:
                job["dt"] = time.time() - job["t0"]
                with self.lock: self.busy = False
        threading.Thread(target=run, daemon=True).start()
        return jid
    def get(self, jid: str) -> Optional[dict]: return self.jobs.get(jid)

JOBS = Jobs()
ENGINES: Dict[str, Any] = {}       # portfolio name → NeuralEngine (style book cached on disk anyway)

def _has_file(root: Path, pred) -> bool:
    for dp, _, fns in os.walk(root, followlinks=True):
        if any(pred(fn) for fn in fns): return True
    return False

def models_status() -> dict:
    vgg = _has_file(DATA_DIR / "models" / "torch", lambda f: f.startswith("vgg19-") and f.endswith(".pth"))
    sd = DATA_DIR / "models" / "huggingface" / "hub" / "models--stabilityai--sd-turbo"
    sd_ok = sd.exists() and _has_file(sd, lambda f: f.endswith(".safetensors"))
    try:
        sd_lib = N().sd_available(); sd_err = N().SD_IMPORT_ERROR
    except Exception as e:
        sd_lib, sd_err = False, f"{type(e).__name__}: {e}"
    if not sd_lib and sd_err: log("text-to-image unavailable: " + sd_err)
    return {"vgg19": vgg, "sd_turbo": sd_ok, "sd_lib": sd_lib, "sd_error": sd_err, "ready": vgg and sd_ok and sd_lib,
            "folder": str(DATA_DIR / "models")}

def selfcheck() -> dict:
    """Import everything the painter needs — the frozen build runs this in CI so a bad bundle fails the build."""
    out = {"ok": True, "modules": {}}
    for name, fn in (("torch", lambda: __import__("torch").__version__), ("torchvision", lambda: __import__("torchvision").__version__),
                     ("pillow_heif", lambda: __import__("pillow_heif").__version__), ("atelier", lambda: bool(A())), ("atelier_neural", lambda: bool(N())),
                     ("transformers", lambda: __import__("transformers").__version__), ("diffusers", lambda: __import__("diffusers").__version__),
                     ("diffusers.AutoPipelineForText2Image", lambda: bool(__import__("diffusers", fromlist=["AutoPipelineForText2Image"]).AutoPipelineForText2Image)),
                     ("diffusers.AutoPipelineForImage2Image", lambda: bool(__import__("diffusers", fromlist=["AutoPipelineForImage2Image"]).AutoPipelineForImage2Image)),
                     ("transformers.CLIPTextModel", lambda: bool(__import__("transformers", fromlist=["CLIPTextModel"]).CLIPTextModel)),
                     ("transformers.CLIPTokenizer", lambda: bool(__import__("transformers", fromlist=["CLIPTokenizer"]).CLIPTokenizer))):
        try:
            out["modules"][name] = str(fn())
        except Exception as e:
            import traceback
            out["modules"][name] = "ERROR " + "".join(traceback.format_exception(type(e), e, e.__traceback__))[-2500:]
            out["ok"] = False
    return out

def portfolios() -> List[dict]:
    out = []
    for d in sorted((DATA_DIR / "portfolios").iterdir()):
        if d.is_dir():
            files = [f for f in d.iterdir() if f.suffix.lower() in A().IMAGE_EXT]
            out.append({"name": d.name, "count": len(files)})
    return out

def portfolio_dir(name: str) -> Path:
    safe = "".join(c for c in name if c.isalnum() or c in " -_").strip() or "portfolio"
    return DATA_DIR / "portfolios" / safe

def thumb_b64(path: Path, side: int = 160) -> str:
    t = A().load_image(path, side)
    im = A().to_pil(t.float() / 255); im.thumbnail((side * 2, side * 2))
    buf = io.BytesIO(); im.save(buf, "JPEG", quality=82)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

def engine_for(name: str, res: int, iters: int):
    key = f"{name}"
    eng = ENGINES.get(key)
    if eng is None:
        eng = N().NeuralEngine(portfolio_dir(name), res=res, iters=iters); ENGINES[key] = eng
    eng.res, eng.iters = res, iters
    return eng

# --------------------------------------------------------------------------- jobs
def job_add_portfolio(progress, name: str, files: List[dict], paper: bool) -> dict:
    d = portfolio_dir(name); d.mkdir(parents=True, exist_ok=True)
    from PIL import Image, ImageOps
    added = 0
    for f in files:
        raw = base64.b64decode(f["data"].split(",", 1)[-1])
        fname = Path(f["name"]).name
        try:
            im = Image.open(io.BytesIO(raw))
            if paper:
                im, ok = A().crop_to_paper(im)
                progress(f"{fname}: {'cropped to the paper' if ok else 'no paper found — kept whole'}")
            else:
                im = ImageOps.exif_transpose(im).convert("RGB")
            im.thumbnail((1600, 1600))
            im.save(d / (Path(fname).stem + ".jpg"), quality=92); added += 1
        except Exception as e:
            progress(f"{fname}: skipped ({e})")
    progress(f"{added} piece(s) in {d.name} — reading the style…")
    ENGINES.pop(name, None)
    eng = engine_for(name, 512, 250)
    progress(f"style book ready: {len(eng.book.files)} pieces, {len(eng.book.tag_counts)} words from file names")
    return portfolio_info(name)

def portfolio_info(name: str) -> dict:
    d = portfolio_dir(name)
    files = sorted(f for f in d.iterdir() if f.suffix.lower() in A().IMAGE_EXT)
    info = {"name": name, "count": len(files), "pieces": [{"name": f.name, "thumb": thumb_b64(f)} for f in files[:60]]}
    eng = ENGINES.get(name)
    if eng is not None:
        b = eng.book
        info["tags"] = b.tag_counts
        info["stats"] = {a: [float(b.a_mean[i]), float(b.a_std[i])] for i, a in enumerate(A().ATTRS[:5])}
        info["medium"] = N().medium_hint(b)
    return info

def job_download_models(progress) -> dict:
    import torch
    progress("VGG19 (style network, ~550 MB)…")
    t0 = time.time(); N().vgg(); progress(f"VGG19 ready ({time.time()-t0:.0f}s)")
    if not N().sd_available():
        progress("SD-Turbo library not installed — text-to-image unavailable (sketch/content repainting still works)")
    else:
        progress("SD-Turbo (content model, ~2.5 GB — one time)…")
        t0 = time.time()
        stop = threading.Event()
        def meter():
            root = DATA_DIR / "models" / "huggingface"
            while not stop.is_set():
                mb = sum(p.stat().st_size for p in root.rglob("*") if p.is_file()) / 1e6 if root.exists() else 0
                progress(f"  … {mb:,.0f} MB so far"); stop.wait(8)
        th = threading.Thread(target=meter, daemon=True); th.start()
        try:
            N().content_from_text("a test", 0, steps=1, size=64)
        finally:
            stop.set()
        progress(f"SD-Turbo ready ({time.time()-t0:.0f}s). From now on the app works offline.")
    os.environ["HF_HUB_OFFLINE"] = "1"
    return models_status()

def job_paint(progress, p: dict) -> dict:
    name = p["portfolio"]
    res, iters, count = int(p.get("res", 512)), int(p.get("iters", 250)), int(p.get("count", 4))
    eng = engine_for(name, res, iters)
    recipe = eng.optimizer.optimize(p.get("prompt") or "")
    if not (p.get("prompt") or "").strip() and not p.get("sketch"):
        raise RuntimeError("say what you'd like to see, or give a sketch")
    sketch = None
    if p.get("sketch"):
        sk = DATA_DIR / "sketches" / Path(p["sketch"]).name
        if sk.exists(): sketch = sk
    content = None
    if p.get("mode") == "repaint" and sketch is not None:
        content, sketch = sketch, None
    seed = int(p["seed"]) if p.get("seed") not in (None, "", "random") else None
    temp = float(p.get("strength", 0.8))
    picks, info = eng.paint(recipe, count, temp, seed, content=content, log=progress,
                            sketch=sketch, freedom=float(p.get("freedom", 0.55)), iters=iters)
    paths, sheet = A().save_results(picks, info, recipe, int(p.get("size", 768)), False)
    recipe_view = {"text": recipe.text, "attributes": recipe.attrs, "tags": recipe.tags, "ignored": recipe.ignored,
                   "content_prompt": N().content_prompt(recipe.text, recipe, eng.book) if not content else None,
                   "style_pieces": [eng.book.files[i].name for i in eng.book.select(recipe)]}
    stamp = sheet.name[: sheet.name.rfind("_sheet")]
    result = {"stamp": stamp, "time": time.time(), "title": (p.get("prompt") or "").strip() or ("sketch, repainted" if content else "from a sketch"),
              "portfolio": name, "settings": {k: v for k, v in p.items() if k != "files"},
              "files": [f"/gallery/{x.name}" for x in paths], "sheet": f"/gallery/{sheet.name}", "names": [x.name for x in paths],
              "info": info, "recipe": recipe_view,
              "sketch_url": f"/sketches/{Path(p['sketch']).name}" if p.get("sketch") else None}
    (DATA_DIR / "gallery" / f"{stamp}_run.json").write_text(json.dumps(result), encoding="utf-8")
    return result

def history() -> List[dict]:
    """Every past run, newest first: the saved records; older sheets without a record are listed bare."""
    out, seen = [], set()
    for f in sorted((DATA_DIR / "gallery").glob("*_run.json"), key=lambda f: f.stat().st_mtime, reverse=True):
        try:
            r = json.loads(f.read_text(encoding="utf-8"))
            out.append({"stamp": r["stamp"], "time": r.get("time", f.stat().st_mtime), "title": r.get("title", ""), "portfolio": r.get("portfolio"),
                        "thumb": (r.get("files") or [None])[0], "count": len(r.get("files") or []), "has_sketch": bool(r.get("sketch_url"))})
            seen.add(r["stamp"])
        except Exception:
            continue
    for f in sorted((DATA_DIR / "gallery").glob("*_sheet.png"), key=lambda f: f.stat().st_mtime, reverse=True):
        stamp = f.name[: f.name.rfind("_sheet")]
        if stamp in seen: continue
        files = sorted(x for x in (DATA_DIR / "gallery").glob(stamp + "_*.png") if not x.name.endswith("_sheet.png"))
        out.append({"stamp": stamp, "time": f.stat().st_mtime, "title": stamp.split("_", 1)[-1].replace("-", " "), "portfolio": None,
                    "thumb": f"/gallery/{files[0].name}" if files else f"/gallery/{f.name}", "count": len(files), "has_sketch": False, "bare": True})
    out.sort(key=lambda r: -r["time"])
    return out

def run_record(stamp: str) -> Optional[dict]:
    f = DATA_DIR / "gallery" / f"{Path(stamp).name}_run.json"
    if f.exists():
        return json.loads(f.read_text(encoding="utf-8"))
    files = sorted(x for x in (DATA_DIR / "gallery").glob(Path(stamp).name + "_*.png") if not x.name.endswith("_sheet.png"))
    if not files: return None
    return {"stamp": stamp, "title": stamp.split("_", 1)[-1].replace("-", " "), "files": [f"/gallery/{x.name}" for x in files],
            "names": [x.name for x in files], "info": [{"critic": 0, "achieved": {}} for _ in files], "recipe": None, "settings": None, "bare": True}

# --------------------------------------------------------------------------- HTTP
class Handler(BaseHTTPRequestHandler):
    server_version = f"Atelier/{__version__}"
    def log_message(self, fmt, *args): pass

    def _json(self, code: int, obj: Any) -> None:
        body = json.dumps(obj).encode()
        self.send_response(code); self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body))); self.send_header("Cache-Control", "no-store"); self.end_headers()
        self.wfile.write(body)

    def _file(self, path: Path, download: bool = False) -> None:
        if not path.is_file(): return self._json(404, {"error": "not found"})
        ctype = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        data = path.read_bytes()
        self.send_response(200); self.send_header("Content-Type", ctype); self.send_header("Content-Length", str(len(data)))
        if download: self.send_header("Content-Disposition", f'attachment; filename="{path.name}"')
        self.end_headers(); self.wfile.write(data)

    def do_GET(self):
        u = urlparse(self.path); p = unquote(u.path)
        try:
            if p in ("/", "/index.html"): return self._file(WEB_DIR / "index.html")
            if p.startswith("/web/"): return self._file(WEB_DIR / p[5:])
            if p.startswith("/gallery/"):
                return self._file(DATA_DIR / "gallery" / Path(p[9:]).name, download="download" in u.query)
            if p.startswith("/sketches/"): return self._file(DATA_DIR / "sketches" / Path(p[10:]).name)
            if p == "/api/status":
                return self._json(200, {"version": __version__, "data_dir": str(DATA_DIR), "models": models_status(),
                                        "portfolios": portfolios(), "busy": JOBS.busy, "frozen": _frozen(),
                                        "device": self._device()})
            if p.startswith("/api/portfolio/"):
                name = p.split("/", 3)[3]
                if not portfolio_dir(name).exists(): return self._json(404, {"error": "no such portfolio"})
                if name not in ENGINES:
                    try: engine_for(name, 384, 250)
                    except Exception as e: log(f"style book: {e}")
                return self._json(200, portfolio_info(name))
            if p.startswith("/api/job/"):
                j = JOBS.get(p.split("/", 3)[3]); return self._json(200 if j else 404, j or {"error": "no such job"})
            if p == "/api/last":
                sheets = sorted((DATA_DIR / "gallery").glob("*_sheet.png"), key=lambda f: f.stat().st_mtime, reverse=True)
                if not sheets: return self._json(200, {"files": [], "names": []})
                stamp = sheets[0].name[: sheets[0].name.rfind("_sheet")]
                files = sorted(f for f in (DATA_DIR / "gallery").glob(stamp + "_*.png") if not f.name.endswith("_sheet.png"))
                return self._json(200, {"files": [f"/gallery/{f.name}" for f in files], "names": [f.name for f in files], "title": stamp.split("_", 1)[-1].replace("-", " ")})
            if p == "/api/selfcheck": return self._json(200, selfcheck())
            if p == "/api/history": return self._json(200, history())
            if p.startswith("/api/run/"):
                r = run_record(p.split("/", 3)[3]); return self._json(200 if r else 404, r or {"error": "no such run"})
            if p == "/api/gallery":
                files = sorted((DATA_DIR / "gallery").glob("*_sheet.png"), key=lambda f: f.stat().st_mtime, reverse=True)[:40]
                return self._json(200, [{"sheet": f"/gallery/{f.name}", "name": f.name} for f in files])
            return self._json(404, {"error": "not found"})
        except Exception as e:
            log(traceback.format_exc()); return self._json(500, {"error": str(e)})

    def do_POST(self):
        u = urlparse(self.path); p = unquote(u.path)
        n = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
            if p == "/api/portfolio":
                jid = JOBS.start("portfolio", job_add_portfolio, body.get("name") or "portfolio", body.get("files") or [], bool(body.get("paper")))
                return self._json(200, {"job": jid})
            if p == "/api/sketch":
                raw = base64.b64decode(body["data"].split(",", 1)[-1])
                from PIL import Image, ImageOps
                im = ImageOps.exif_transpose(Image.open(io.BytesIO(raw))).convert("RGB"); im.thumbnail((1024, 1024))
                name = f"sketch_{int(time.time())}_{uuid.uuid4().hex[:4]}.jpg"
                im.save(DATA_DIR / "sketches" / name, quality=92)
                return self._json(200, {"sketch": name, "url": f"/sketches/{name}"})
            if p == "/api/models/download":
                return self._json(200, {"job": JOBS.start("models", job_download_models)})
            if p == "/api/paint":
                return self._json(200, {"job": JOBS.start("paint", job_paint, body)})
            if p == "/api/open":
                target = DATA_DIR / "gallery" / Path(body.get("name", "")).name
                if body.get("folder"): target = DATA_DIR / body["folder"]
                A().open_file(target); return self._json(200, {"ok": True})
            if p == "/api/run/delete":
                stamp = Path(body.get("stamp", "")).name
                n = 0
                for f in (DATA_DIR / "gallery").glob(stamp + "_*"):
                    f.unlink(); n += 1
                return self._json(200, {"deleted": n})
            if p == "/api/quit":
                threading.Thread(target=lambda: (time.sleep(0.3), os._exit(0)), daemon=True).start()
                return self._json(200, {"ok": True})
            return self._json(404, {"error": "not found"})
        except RuntimeError as e:
            return self._json(409, {"error": str(e)})
        except Exception as e:
            log(traceback.format_exc()); return self._json(500, {"error": str(e)})

    @staticmethod
    def _device() -> str:
        try:
            import torch
            return A().pick_device().type
        except Exception:
            return "unknown"

# --------------------------------------------------------------------------- run
def free_port(preferred: int = 8760) -> int:
    for port in (preferred, 0):
        s = socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(("127.0.0.1", port)); port = s.getsockname()[1]; s.close(); return port
        except OSError:
            s.close()
    return 0

def serve() -> str:
    port = free_port(int(os.environ.get("ATELIER_PORT", "8760")))
    httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler); httpd.daemon_threads = True
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{port}/"
    log(f"{APP_NAME} {__version__} serving {url}  data: {DATA_DIR}  frozen: {_frozen()}")
    return url

def main() -> None:
    if hasattr(sys, "frozen"):
        import multiprocessing; multiprocessing.freeze_support()
    url = serve()
    if os.environ.get("ATELIER_NO_WINDOW"):
        print(f"ATELIER READY {url}", flush=True)
        stop = threading.Event()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try: signal.signal(sig, lambda *_: stop.set())
            except Exception: pass
        while not stop.is_set(): stop.wait(0.5)
        return
    if not os.environ.get("ATELIER_BROWSER"):
        try:
            import webview
            webview.create_window(APP_NAME, url, width=1180, height=900, min_size=(900, 640))
            webview.start(); return
        except Exception as e:
            log(f"no native window ({e}); opening the browser")
    webbrowser.open(url)
    print(f"{APP_NAME} is running at {url}  (Ctrl-C to quit)", flush=True)
    try:
        while True: time.sleep(1)
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()
