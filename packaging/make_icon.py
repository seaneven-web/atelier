#!/usr/bin/env python3
"""Render packaging/icon.icns (macOS, needs iconutil), icon.ico (any OS) and linux/icon-{256,512}.png.
A cobalt square with a soft brush stroke — drawn with Pillow, so the repo carries no binary icon source.
    python packaging/make_icon.py            # everything the current OS can make
    python packaging/make_icon.py --ico-only / --png
"""
from __future__ import annotations
import shutil, subprocess, sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter

HERE = Path(__file__).resolve().parent
def render(size=1024) -> Image.Image:
    im = Image.new("RGBA", (size, size), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    r = size * 0.22
    d.rounded_rectangle([0, 0, size, size], radius=r, fill=(47, 79, 203, 255))            # cobalt ground
    # a gesso-white sheet, slightly tilted, with a soft shadow
    sheet = Image.new("RGBA", (size, size), (0, 0, 0, 0)); sd = ImageDraw.Draw(sheet)
    sd.rounded_rectangle([size*0.2, size*0.16, size*0.8, size*0.84], radius=size*0.03, fill=(242, 243, 240, 255))
    sheet = sheet.rotate(-6, resample=Image.BICUBIC, center=(size/2, size/2))
    shadow = sheet.split()[3].filter(ImageFilter.GaussianBlur(size*0.03))
    im.paste((10, 20, 60, 120), (int(size*0.02), int(size*0.04)), shadow)
    im.alpha_composite(sheet)
    # a brush stroke in warm ochre across the sheet
    stroke = Image.new("RGBA", (size, size), (0, 0, 0, 0)); st = ImageDraw.Draw(stroke)
    pts = [(size*0.3, size*0.62), (size*0.42, size*0.42), (size*0.56, size*0.56), (size*0.7, size*0.36)]
    for i in range(len(pts)-1):
        st.line([pts[i], pts[i+1]], fill=(214, 140, 52, 235), width=int(size*0.085), joint="curve")
    for p in pts: st.ellipse([p[0]-size*0.042, p[1]-size*0.042, p[0]+size*0.042, p[1]+size*0.042], fill=(214, 140, 52, 235))
    im.alpha_composite(stroke.filter(ImageFilter.GaussianBlur(size*0.004)))
    return im

def main():
    args = set(sys.argv[1:]); base = render()
    if "--png" in args or not args:
        (HERE / "linux").mkdir(exist_ok=True)
        for s in (256, 512): base.resize((s, s), Image.LANCZOS).save(HERE / "linux" / f"icon-{s}.png")
        print("linux PNGs written")
    if "--ico-only" in args or not args or "--png" in args:
        base.save(HERE / "icon.ico", sizes=[(16,16),(32,32),(48,48),(64,64),(128,128),(256,256)]); print("icon.ico written")
    if ("--ico-only" not in args) and sys.platform == "darwin" and shutil.which("iconutil"):
        iconset = HERE / "icon.iconset"; iconset.mkdir(exist_ok=True)
        for s in (16, 32, 128, 256, 512):
            base.resize((s, s), Image.LANCZOS).save(iconset / f"icon_{s}x{s}.png")
            base.resize((s*2, s*2), Image.LANCZOS).save(iconset / f"icon_{s}x{s}@2x.png")
        subprocess.run(["iconutil", "-c", "icns", str(iconset), "-o", str(HERE / "icon.icns")], check=True)
        shutil.rmtree(iconset); print("icon.icns written")

if __name__ == "__main__":
    main()
