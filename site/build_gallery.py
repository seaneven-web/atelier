#!/usr/bin/env python3
"""Rebuild the carousel of pieces on the website.

    python3 site/build_gallery.py

Reads site/gallery.json — a list of {src, name, prompt, note} where `src` is a piece in
the app's gallery folder — writes web-sized JPEGs into site/assets/gallery/ and rewrites
the slide markup between the <!-- slides:start/end --> markers in site/index.html.
"""
import html
import json
import pathlib
import sys

from PIL import Image

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "assets" / "gallery"
WIDE = 1400          # long side of the shipped JPEG


def main() -> int:
    picks = json.loads((HERE / "gallery.json").read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    slides = []
    for n, p in enumerate(picks):
        src = pathlib.Path(p["src"]).expanduser()
        im = Image.open(src).convert("RGB")
        im.thumbnail((WIDE, WIDE), Image.LANCZOS)
        dst = OUT / f"{p['name']}.jpg"
        im.save(dst, "JPEG", quality=84, optimize=True, progressive=True)
        print(f"  {p['name']:<26} {im.size[0]}x{im.size[1]}  {dst.stat().st_size // 1024} KB")
        prompt, note = html.escape(p["prompt"]), html.escape(p["note"])
        slides.append(
            f'        <div class="slide" role="group" aria-roledescription="slide" aria-label="{n + 1} of {len(picks)}">\n'
            f'          <figure><div class="pic"><img src="assets/gallery/{p["name"]}.jpg"'
            f' alt="{prompt} — painted in the artist\'s style"'
            f'{"" if n == 0 else " loading=lazy"} width="{im.size[0]}" height="{im.size[1]}"></div>\n'
            f'          <figcaption><b>&ldquo;{prompt}&rdquo;</b> <span>{note}</span></figcaption></figure>\n'
            f'        </div>'
        )

    idx = HERE / "index.html"
    s = idx.read_text()
    a, b = "<!-- slides:start -->", "<!-- slides:end -->"
    i, j = s.index(a), s.index(b)
    idx.write_text(s[: i + len(a)] + "\n" + "\n".join(slides) + "\n" + s[j:])
    print(f"{len(picks)} slides -> site/index.html")
    return 0


if __name__ == "__main__":
    sys.exit(main())
