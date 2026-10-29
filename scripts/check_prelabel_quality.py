#!/usr/bin/env python3
"""把預標註「已通過」的樣本拼成核對圖，用眼睛驗證準確率。"""
import json
import sys
from pathlib import Path

sys.path.insert(0, "/home/nicklam-ai/repos/GBA-plate-recognition")

import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path("/home/nicklam-ai/repos/GBA-plate-recognition")
OUT = ROOT / "prelabel" / "hk"

rs = json.load(open(OUT / "labeled.json"))
ok = [r for r in rs if r["valid"]]

# 取前 12 張已通過的
picks = ok[:12]
tiles = []
for r in picks:
    f = OUT / "crops" / r["file"]
    img = cv2.imdecode(np.fromfile(str(f), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        continue
    p = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    w = 360
    p = p.resize((w, max(28, int(p.height * w / p.width))), Image.LANCZOS)
    tiles.append((p, r["text_clean"], r["paddle_mean_conf"],
                  r.get("double_line_merged", False)))

cols = 2
row_h = max(t[0].height for t in tiles) + 26
canvas = Image.new("RGB", (760, row_h * ((len(tiles) + cols - 1) // cols) + 10),
                   (24, 24, 32))
d = ImageDraw.Draw(canvas)
for i, (p, text, conf, dbl) in enumerate(tiles):
    x = 8 + (i % cols) * 376
    y = 6 + (i // cols) * row_h
    canvas.paste(p, (x, y))
    d.text((x, y + p.height + 4),
           f"#{i} PaddleOCR={text}  conf={conf:.2f}  雙行={dbl}",
           fill=(150, 220, 160))

canvas.save("/tmp/prelabel_check.png")
print(f"已通過 {len(ok)} 個，取 {len(tiles)} 張拼圖 → /tmp/prelabel_check.png")
for i, t in enumerate(tiles):
    print(f"  #{i} {t[1]}")
