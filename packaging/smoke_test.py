#!/usr/bin/env python3
"""Headless smoke test for an Atelier desktop build. Stdlib only, Python 3.9+.

Launches the app with ATELIER_NO_WINDOW=1 on a free port and a throw-away ATELIER_DATA_DIR,
waits for "ATELIER READY <url>" (or a 200 from /api/status), then checks: /, /web/app.js,
/api/status (json with models + data_dir inside the temp dir), /api/gallery. Exits 1 on
any failure; always kills the child.

    packaging/smoke_test.py dist/Atelier.app/Contents/MacOS/Atelier --frozen
    packaging/smoke_test.py "dist\\Atelier\\Atelier.exe" --frozen
    packaging/smoke_test.py ".venv/bin/python atelier_app.py"
"""
from __future__ import annotations
import argparse, json, os, shlex, shutil, socket, subprocess, sys, tempfile, threading, time, urllib.request

def free_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p

def get(url, timeout=5):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.status, r.read()

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("cmd"); ap.add_argument("--frozen", action="store_true")
    ap.add_argument("--timeout", type=float, default=120); ap.add_argument("--keep-data", action="store_true")
    ap.add_argument("--deep", action="store_true", help="also import torch/diffusers/transformers inside the app (/api/selfcheck)")
    a = ap.parse_args()
    port = free_port(); data = tempfile.mkdtemp(prefix="atelier-smoke-")
    env = dict(os.environ, ATELIER_NO_WINDOW="1", ATELIER_PORT=str(port), ATELIER_DATA_DIR=data, PYTHONUNBUFFERED="1")
    cmd = shlex.split(a.cmd) if not os.path.exists(a.cmd) else [a.cmd]
    print(f"smoke: launching {cmd} on port {port}, data {data}")
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    ready = {"url": None}; lines = []
    def pump():
        for line in proc.stdout:
            lines.append(line.rstrip()); print("  |", line.rstrip())
            if line.startswith("ATELIER READY"): ready["url"] = line.split()[2]
    threading.Thread(target=pump, daemon=True).start()
    url = f"http://127.0.0.1:{port}/"; t0 = time.time(); ok = False
    while time.time() - t0 < a.timeout:
        if proc.poll() is not None: break
        try:
            st, _ = get(url + "api/status"); ok = st == 200; break
        except Exception:
            time.sleep(0.5)
    results = []
    def check(name, fn):
        try: results.append((name, fn(), "")); 
        except Exception as e: results.append((name, False, str(e)))
    if not ok:
        results.append(("ready", False, f"no response within {a.timeout}s (exit={proc.poll()})"))
    else:
        check("GET /", lambda: get(url)[0] == 200 and b"<title>Atelier" in get(url)[1])
        check("GET /web/app.js", lambda: get(url + "web/app.js")[0] == 200)
        def status():
            st, body = get(url + "api/status"); j = json.loads(body)
            assert st == 200 and "models" in j and "portfolios" in j, j
            assert os.path.realpath(j["data_dir"]).startswith(os.path.realpath(data)), j["data_dir"]
            if a.frozen: assert j["frozen"] is True, "frozen flag not set"
            return True
        check("GET /api/status", status)
        check("GET /api/gallery", lambda: get(url + "api/gallery")[0] == 200)
        check("sandbox folders created", lambda: all(os.path.isdir(os.path.join(data, d)) for d in ("portfolios", "gallery", "models")))
        if a.deep:
            def deep():
                st, body = get(url + "api/selfcheck", timeout=600); j = json.loads(body)
                for k, v in j["modules"].items(): print(f"    {k:<40} {v[:90] if not v.startswith('ERROR') else 'ERROR'}")
                bad = {k: v for k, v in j["modules"].items() if v.startswith("ERROR")}
                for k, v in bad.items(): print(f"\n--- {k} ---\n{v}\n")
                assert j["ok"], f"{len(bad)} import(s) failed: {', '.join(bad)}"
                return True
            check("deep self-check (torch · diffusers · transformers)", deep)
    proc.terminate()
    try: proc.wait(10)
    except Exception: proc.kill()
    width = max(len(n) for n, _, _ in results) if results else 10
    fails = 0
    for n, r, msg in results:
        print(f"  {n:<{width}}  {'PASS' if r else 'FAIL'}  {msg}"); fails += (not r)
    if not a.keep_data: shutil.rmtree(data, ignore_errors=True)
    else: print("data dir kept:", data)
    print("smoke:", "OK" if not fails else f"{fails} failure(s)")
    sys.exit(1 if fails else 0)

if __name__ == "__main__":
    main()
