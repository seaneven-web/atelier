#!/usr/bin/env python3
"""
Atelier × Claude — a text studio where Claude Opus is the director and the small local
model (atelier.py) does the painting.

    python atelier_claude.py            # chat
    python atelier_claude.py --effort high --count 6

Claude never generates pixels. It studies the portfolio (stats + a contact sheet), turns what
you say into a recipe for the style space (that's the prompt optimisation — now done by Opus
instead of the word table), calls the local painter, LOOKS at the result, and iterates.

Needs:  pip install anthropic   and credentials — either  export ANTHROPIC_API_KEY=…  or  `ant auth login`.
Without credentials it tells you so and points you at the offline studio (python atelier.py studio).
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import anthropic
except ImportError:
    sys.exit("The Claude app needs the Anthropic SDK:  pip install anthropic")

import torch
from PIL import Image

import atelier as A

MODEL = "claude-opus-5"
MAX_TOOL_ROUNDS = 6          # tool calls Claude may chain per user message
KEEP_IMAGES = 3              # most recent images kept in context; older ones become text stubs

# --------------------------------------------------------------------------- image helpers
def image_block(img: Image.Image, max_side: int = 1024, quality: int = 85) -> dict:
    img = img.convert("RGB")
    w, h = img.size
    if max(w, h) > max_side:
        s = max_side / max(w, h); img = img.resize((round(w * s), round(h * s)), Image.LANCZOS)
    buf = io.BytesIO(); img.save(buf, "JPEG", quality=quality)
    return {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                         "data": base64.standard_b64encode(buf.getvalue()).decode()}}

def portfolio_sheet(files: List[str], h: int, w: int, n: int = 16) -> Optional[Image.Image]:
    """A contact sheet of up to n portfolio pieces so Claude can see the style, not just read numbers."""
    files = [f for f in files if Path(f).exists()]
    if not files: return None
    pick = random.Random(0).sample(files, min(n, len(files)))
    tiles = []
    for f in pick:
        try:
            t = A.load_image(Path(f), 160)
            tiles.append(A.to_pil(t.float() / 255))
        except Exception:
            continue
    if not tiles: return None
    tw, th = 160, 160
    cols = min(4, len(tiles)); rows = -(-len(tiles) // cols)
    sheet = Image.new("RGB", (cols * (tw + 6) + 6, rows * (th + 6) + 6), (22, 22, 22))
    for i, t in enumerate(tiles):
        t = t.copy(); t.thumbnail((tw, th))
        r, c = divmod(i, cols)
        sheet.paste(t, (6 + c * (tw + 6) + (tw - t.size[0]) // 2, 6 + r * (th + 6) + (th - t.size[1]) // 2))
    return sheet

# --------------------------------------------------------------------------- the studio Claude drives
class Studio:
    def __init__(self, count: int, out_size: int, auto_open: bool, engine: str = "auto", portfolio: Optional[str] = None):
        self.painter = A.make_engine(engine, portfolio)
        self.neural = type(self.painter).__name__ == "NeuralEngine"
        self.count, self.out_size, self.auto_open = count, out_size, auto_open
        self.last_paths: List[Path] = []
        self.usage = {"in": 0, "out": 0, "cache_read": 0, "cache_write": 0}

    # ---- what Claude knows (stable → cacheable)
    def system_prompt(self) -> str:
        st = self.painter.style; cfg = self.painter.cfg
        lines = []
        for i, a in enumerate(A.ATTRS):
            lines.append(f"  {a:<10} portfolio mean {st['a_mean'][i]:.3f} ± {st['a_std'][i]:.3f}")
        tags = sorted(st["tag_counts"].items(), key=lambda kv: -kv[1])
        tag_txt = ", ".join(f"{t} ({c})" for t, c in tags) if tags else "(none — file names carry no words)"
        files = [Path(f).name for f in st["files"]]
        shown = files[:80]
        vocab = sorted(set(A._V))
        how = (f"""WHAT THE PAINTER DOES (two stages). 1) A text-to-image model (SD-Turbo) draws what the words literally describe —
subjects, scenes, objects all work: "a hippo eating cheese" gives a hippo eating cheese. 2) Neural style transfer (VGG19)
repaints that in the artist's style, using the portfolio pieces the recipe selects. So: put the CONTENT in the title (that
is what gets drawn — write it as a clear, concrete picture description), and use tags/attributes/anchor to shape the STYLE.
The optional `content` argument repaints an existing image (a photo, sketch or earlier result) instead of drawing new content.
Temperature = how hard the style is pushed (0.4 gentle, 0.8 normal, 1.2 fully repainted). Each piece takes ~1 minute.""" if self.neural else
f"""WHAT THE PAINTER CAN DO. It paints in the artist's palette, tone, texture and compositional habits. It does NOT render
recognisable objects or scenes — if someone asks for "a boat", you can only aim at the moods/colours/compositions the
artist used for boats (via tags or anchors). Say so gently when it matters, then do the best steer.""")
        return f"""You are the studio director for Atelier: a painter that works in one artist's style, learned from {len(files)} pieces
("{cfg.name}"). You do not paint; you direct the painter through tools.

{how}

THE STYLE SPACE. A recipe is a set of steering weights in *standard-deviation units* of the portfolio (−2…+2; 1.0 is a clear
move, 2.0 is extreme), plus optional tags (pull toward pieces whose file names carry that word), optional anchor (start from a
specific piece), temperature (0.3 tight … 1.2 adventurous; default 0.8) and count.
Attributes:
{chr(10).join(lines)}
  (brightness = mean luminance; contrast = luminance spread; saturation; warmth = red−blue balance;
   detail = edge density/busyness; red/yellow/green/blue/purple = share of coloured pixels in that hue family)
Portfolio tags: {tag_txt}
Portfolio files you can anchor on (first {len(shown)} of {len(files)}): {", ".join(shown)}
The offline word-engine vocabulary (for reference, you are smarter than it): {" ".join(vocab[:160])} …

HOW TO WORK.
1. Translate what the person says into the best recipe — this is your main job (you are the prompt optimiser). Prefer 1–3
   attributes at moderate weights over many at extremes; use tags when the person's words match them; use an anchor when they
   point at a piece. Keep a short "title" for each recipe in their words.
2. Call paint. You will get the contact sheet back as an image plus, per piece: the critic's realism score and the achieved
   attribute values. LOOK at the sheet. Comment in one or two plain sentences (what worked, which piece you'd pick, one thing
   you'd change). Offer one concrete next move. Do not paint again unless asked or the result clearly missed.
3. If a result looks off-style, lower temperature or weights before anything else. If all pieces look alike, raise temperature.
4. Keep replies short and warm — this is a conversation with an artist, not a report. No markdown tables. Never invent
   numbers; quote the tool's.
5. You may look at any portfolio piece or gallery result with look_at to ground your judgement.
"""

    # ---- tools (schemas)
    def tools(self) -> List[dict]:
        attr_props = {a: {"type": "number", "minimum": -2, "maximum": 2} for a in A.ATTRS}
        return [
            {"name": "paint", "description": "Paint pieces with the local model from a recipe. Returns file paths, critic "
                                             "realism scores, achieved attribute values, and the contact sheet as an image.",
             "input_schema": {"type": "object", "properties": {
                 "title": {"type": "string", "description": "what to draw, as a clear picture description in the person's words (the content model draws exactly this; also used for file names)"},
                 "content": {"type": "string", "description": "optional: file name/path of an image to repaint in the style instead of drawing the title"},
                 "attributes": {"type": "object", "properties": attr_props, "additionalProperties": False,
                                "description": "steering weights in std units, −2…+2; omit what you don't want to move"},
                 "tags": {"type": "array", "items": {"type": "string"}, "description": "portfolio tags to pull toward"},
                 "anchor": {"type": "string", "description": "file name (or path) of a piece to start from"},
                 "temperature": {"type": "number", "minimum": 0.2, "maximum": 1.5},
                 "count": {"type": "integer", "minimum": 1, "maximum": 12},
                 "seed": {"type": "integer"}},
                 "required": ["title"]}},
            {"name": "variations", "description": "Paint variations of an existing image (a gallery result or a portfolio piece).",
             "input_schema": {"type": "object", "properties": {
                 "path": {"type": "string", "description": "file name or path; 'last:N' = piece N of the last sheet"},
                 "count": {"type": "integer", "minimum": 1, "maximum": 12},
                 "temperature": {"type": "number", "minimum": 0.2, "maximum": 1.5}},
                 "required": ["path"]}},
            {"name": "look_at", "description": "See an image: a portfolio piece (by file name) or a gallery result (by path).",
             "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}},
            {"name": "word_engine", "description": "How the offline word engine would read a phrase (attributes/tags/ignored). "
                                                   "Cheap sanity check when unsure what a word maps to.",
             "input_schema": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}},
            {"name": "open_in_viewer", "description": "Open a file in the person's image viewer.",
             "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}},
        ]

    # ---- tool execution → content blocks for the tool_result
    def resolve(self, name: str) -> Optional[Path]:
        if not name: return None
        if name.startswith("last:"):
            i = int(name.split(":")[1]) - 1
            return self.last_paths[i] if 0 <= i < len(self.last_paths) else None
        p = Path(name).expanduser()
        if p.exists(): return p
        for f in self.painter.style["files"]:
            if Path(f).name == name or Path(f).stem == name: return Path(f)
        g = sorted(A.GALLERY.glob(f"*{name}*")) if A.GALLERY.exists() else []
        return g[-1] if g else None

    def run_tool(self, name: str, inp: dict) -> List[dict]:
        if name == "paint":
            r = A.Recipe(text=inp.get("title") or "untitled")
            r.attrs = {k: float(max(-2, min(2, v))) for k, v in (inp.get("attributes") or {}).items() if k in A.ATTRS}
            known = self.painter.style["tags"]
            r.tags = [t for t in (inp.get("tags") or []) if t in known]
            unknown = [t for t in (inp.get("tags") or []) if t not in known]
            if inp.get("anchor"):
                p = self.resolve(inp["anchor"])
                if p is None: return [{"type": "text", "text": f"anchor not found: {inp['anchor']}"}]
                r.anchor = str(p)
            if inp.get("content"):
                p = self.resolve(inp["content"])
                if p is None: return [{"type": "text", "text": f"content image not found: {inp['content']}"}]
                r.anchor = str(p)
            return self._paint(r, int(inp.get("count") or self.count), inp.get("temperature"), inp.get("seed"),
                               note=(f"unknown tags ignored: {unknown}" if unknown else ""))
        if name == "variations":
            p = self.resolve(inp["path"])
            if p is None: return [{"type": "text", "text": f"not found: {inp['path']}"}]
            r = A.Recipe(text=f"variations of {p.stem[:24]}", anchor=str(p))
            return self._paint(r, int(inp.get("count") or self.count), (inp.get("temperature") or 0.5) * 2, None)
        if name == "look_at":
            p = self.resolve(inp["path"])
            if p is None: return [{"type": "text", "text": f"not found: {inp['path']}"}]
            return [{"type": "text", "text": f"{p.name}"}, image_block(Image.open(p))]
        if name == "word_engine":
            rec = self.painter.optimizer.optimize(inp["text"])
            return [{"type": "text", "text": json.dumps({"attributes": rec.attrs, "tags": rec.tags, "ignored": rec.ignored})}]
        if name == "open_in_viewer":
            p = self.resolve(inp["path"])
            if p is None: return [{"type": "text", "text": f"not found: {inp['path']}"}]
            A.open_file(p); return [{"type": "text", "text": f"opened {p.name}"}]
        return [{"type": "text", "text": f"unknown tool {name}"}]

    def _paint(self, recipe: A.Recipe, count: int, temp: Optional[float], seed: Optional[int], note: str = "") -> List[dict]:
        t0 = time.time()
        picks, info = self.painter.paint(recipe, count, temp, seed)
        paths, sheet = A.save_results(picks, info, recipe, self.out_size, self.auto_open)
        self.last_paths = paths
        A.say(A.dim(f"    ⟶ painted {len(paths)} pieces in {time.time()-t0:.1f}s → {sheet.name}"))
        report = {"sheet": str(sheet), "pieces": [
            {"n": i + 1, "file": p.name, "critic_realism": round(inf["critic"], 2),
             "achieved": {k: round(v, 2) for k, v in inf["achieved"].items()}} for i, (p, inf) in enumerate(zip(paths, info))],
            "recipe": {"attributes": recipe.attrs, "tags": recipe.tags, "anchor": Path(recipe.anchor).name if recipe.anchor else None}}
        if note: report["note"] = note
        return [{"type": "text", "text": json.dumps(report)}, image_block(Image.open(sheet))]

# --------------------------------------------------------------------------- conversation
def trim_images(messages: List[dict], keep: int) -> None:
    """Replace all but the newest `keep` images with text stubs (they'd cost tokens forever otherwise)."""
    seen = 0
    for m in reversed(messages[1:]):          # messages[0] carries the portfolio sheet — keep it for good
        if m["role"] != "user" or not isinstance(m["content"], list): continue
        for block in reversed(m["content"]):
            if block.get("type") == "tool_result" and isinstance(block.get("content"), list):
                for j in range(len(block["content"]) - 1, -1, -1):
                    if block["content"][j].get("type") == "image":
                        seen += 1
                        if seen > keep:
                            block["content"][j] = {"type": "text", "text": "[image shown earlier]"}
            elif block.get("type") == "image":
                seen += 1
                if seen > keep:
                    m["content"][m["content"].index(block)] = {"type": "text", "text": "[image shown earlier]"}

def chat(args) -> None:
    try:
        studio = Studio(args.count, args.size, not args.no_open, args.engine, args.portfolio)
    except FileNotFoundError as e:
        A.say(A.yellow(f"  {e}")); return
    try:
        client = anthropic.Anthropic()
    except anthropic.AnthropicError as e:
        A.say(A.yellow(f"  Claude needs credentials: export ANTHROPIC_API_KEY=…  (or `ant auth login`).  {e}"))
        A.say(A.dim("  The offline studio still works:  python atelier.py studio")); return

    system = [{"type": "text", "text": studio.system_prompt(), "cache_control": {"type": "ephemeral"}}]
    tools = studio.tools()
    messages: List[dict] = []
    cfg = studio.painter.cfg
    A.say(A.bold(f"\n◆ Atelier × Claude — {MODEL} directing the “{cfg.name}” painter"))
    A.say(A.dim(f"  {len(studio.painter.style['files'])} pieces studied · engine {'SD-Turbo → style transfer' if studio.neural else 'trained actor/critic'} · "
                f"effort {args.effort} · {args.count} pieces per prompt · say what you'd like to see; /help for commands\n"))

    # first turn carries a contact sheet of the portfolio so Claude has seen the work
    intro_sheet = portfolio_sheet(studio.painter.style["files"], cfg.height, cfg.width)
    intro: List[dict] = []
    if intro_sheet is not None:
        intro = [image_block(intro_sheet, 900, 80),
                 {"type": "text", "text": "(A contact sheet of pieces from the portfolio, so you have seen the style.)"}]

    while True:
        try:
            line = input(A.cyan("✎ ")).strip()
        except (EOFError, KeyboardInterrupt):
            A.say(); break
        if not line: continue
        if line.startswith("/"):
            cmd, _, arg = line.partition(" ")
            if cmd in ("/quit", "/exit", "/q"): break
            if cmd == "/help":
                A.say("  Just talk. Examples: “something like his harbour pieces but colder”, “more dramatic, fewer colours”,\n"
                      "  “variations of #2”, “what would you change?”.  Commands: /gallery /open on|off /count N /size PX /usage /reset /quit")
            elif cmd == "/gallery": A.GALLERY.mkdir(exist_ok=True); A.open_file(A.GALLERY)
            elif cmd == "/open": studio.auto_open = arg.strip().lower() != "off"; A.say(A.dim(f"  auto-open {'on' if studio.auto_open else 'off'}"))
            elif cmd == "/count" and arg.strip().isdigit(): studio.count = max(1, min(12, int(arg))); A.say(A.dim(f"  {studio.count} pieces per prompt"))
            elif cmd == "/size" and arg.strip().isdigit(): studio.out_size = int(arg); A.say(A.dim(f"  output {studio.out_size}px"))
            elif cmd == "/usage":
                u = studio.usage
                A.say(A.dim(f"  tokens in {u['in']} (cache read {u['cache_read']}, cache write {u['cache_write']}) · out {u['out']}"))
            elif cmd == "/reset": messages.clear(); A.say(A.dim("  conversation cleared"))
            else: A.say(A.yellow("  unknown command — /help"))
            continue

        content: List[dict] = (intro + [{"type": "text", "text": line}]) if (intro and not messages) else [{"type": "text", "text": line}]
        messages.append({"role": "user", "content": content})

        try:
            for _ in range(MAX_TOOL_ROUNDS + 1):
                with client.messages.stream(
                    model=MODEL, max_tokens=4096, system=system, tools=tools, messages=messages,
                    output_config={"effort": args.effort},
                ) as stream:
                    printed = False
                    for text in stream.text_stream:
                        if not printed: sys.stdout.write("  "); printed = True
                        sys.stdout.write(text); sys.stdout.flush()
                    response = stream.get_final_message()
                if printed: A.say()
                u = response.usage
                studio.usage["in"] += u.input_tokens; studio.usage["out"] += u.output_tokens
                studio.usage["cache_read"] += getattr(u, "cache_read_input_tokens", 0) or 0
                studio.usage["cache_write"] += getattr(u, "cache_creation_input_tokens", 0) or 0
                messages.append({"role": "assistant", "content": response.content})
                if response.stop_reason == "refusal":
                    A.say(A.yellow("  (Claude declined that one.)")); break
                if response.stop_reason != "tool_use":
                    break
                results = []
                for block in response.content:
                    if block.type == "tool_use":
                        A.say(A.dim(f"  ⚙ {block.name} {json.dumps(block.input)[:140]}"))
                        try:
                            out = studio.run_tool(block.name, dict(block.input))
                            results.append({"type": "tool_result", "tool_use_id": block.id, "content": out})
                        except Exception as e:  # tool failure → tell Claude, keep the loop alive
                            results.append({"type": "tool_result", "tool_use_id": block.id, "content": f"error: {e}", "is_error": True})
                messages.append({"role": "user", "content": results})
                trim_images(messages, KEEP_IMAGES)
            else:
                A.say(A.yellow("  (stopping after many tool calls — ask again to continue)"))
        except anthropic.AuthenticationError:
            A.say(A.yellow("  Invalid or missing API key. export ANTHROPIC_API_KEY=…  or  ant auth login")); break
        except anthropic.RateLimitError as e:
            A.say(A.yellow(f"  rate limited — try again in a moment ({e.message})")); messages.pop()
        except anthropic.APIStatusError as e:
            A.say(A.yellow(f"  API error {e.status_code}: {e.message}")); messages.pop()
        except anthropic.APIConnectionError:
            A.say(A.yellow("  no connection to the API — check the network")); messages.pop()
        except KeyboardInterrupt:
            A.say(A.dim("\n  (interrupted)"))
            if messages and messages[-1]["role"] == "user": messages.pop()
        A.say()

def main(argv=None):
    ap = argparse.ArgumentParser(description="Atelier × Claude — Opus directs the local painter")
    ap.add_argument("--count", type=int, default=4); ap.add_argument("--size", type=int, default=512, help="output long side px")
    ap.add_argument("--effort", choices=["low", "medium", "high", "xhigh", "max"], default="medium")
    ap.add_argument("--no-open", action="store_true")
    ap.add_argument("--portfolio", default=None, help="folder of the artist's pieces (remembered after first use)")
    ap.add_argument("--engine", choices=["auto", "neural", "vae"], default="auto")
    chat(ap.parse_args(argv))

if __name__ == "__main__":
    main()
