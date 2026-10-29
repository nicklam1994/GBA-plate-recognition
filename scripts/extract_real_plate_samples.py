#!/usr/bin/env python3
"""
從 dataset_unified 的真實港澳標註裁出車牌小圖，供人工/視覺讀取真實號碼。

用途：澳門與粵Z 沒有像 plate.hk 那樣的號碼庫，
      只能從真實圖片反推生成器應該採用什麼格式。
"""
import glob
import os
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path("/home/nicklam-ai/repos/GBA-plate-recognition")
DS = ROOT / "dataset_unified"
OUT = ROOT / "real_plate_samples"

REGION = {1: "gangao", 2: "hk", 3: "mo"}
PAD = 0.25  # 多留邊距，方便看完整車牌


def collect(target_cls: int, limit: int):
    """收集中指定類別的真實車牌小圖"""
    crops = []
    for split in ("valid", "test", "train"):
        lbl_dir = DS / "labels" / split
        img_dir = DS / "images" / split
        if not lbl_dir.exists():
            continue
        files = sorted(lbl_dir.glob("*.txt"))
        random.shuffle(files)
        for lbl in files:
            img_path = None
            for ext in (".jpg", ".jpeg", ".png"):
                cand = img_dir / (lbl.stem + ext)
                if cand.exists():
                    img_path = cand
                    break
            if img_path is None:
                continue
            img = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8),
                               cv2.IMREAD_COLOR)
            if img is None:
                continue
            with open(lbl) as fh:
                for line in fh:
                    p = line.split()
                    if len(p) < 5:
                        continue
                    if int(float(p[0])) != target_cls:
                        continue
                    H, W = img.shape[:2]
                    xc, yc, w, h = [float(v) for v in p[1:5]]
                    bw, bh = w * W, h * H
                    x1 = max(0, int(xc * W - bw / 2 - bw * PAD))
                    x2 = min(W, int(xc * W + bw / 2 + bw * PAD))
                    y1 = max(0, int(yc * H - bh / 2 - bh * PAD))
                    y2 = min(H, int(yc * H + bh / 2 + bh * PAD))
                    if x2 - x1 < 20 or y2 - y1 < 10:
                        continue
                    crops.append(img[y1:y2, x1:x2].copy())
                    break
            if len(crops) >= limit:
                return crops
    return crops


def main():
    random.seed(11)
    OUT.mkdir(parents=True, exist_ok=True)

    plan = [(3, 24, "mo"), (1, 24, "gangao")]
    for cls, n, name in plan:
        crops = collect(cls, n)
        print(f"{name}: 收集 {len(crops)} 張")

        # 拼圖
        tiles = []
        for c in crops:
            im = Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB))
            r = 110 / im.height
            im = im.resize((max(40, int(im.width * r)), 110), Image.LANCZOS)
            tiles.append(im)

        if not tiles:
            continue
        cols = 4
        rows_n = (len(tiles) + cols - 1) // cols
        tw = max(t.width for t in tiles) + 10
        canvas = Image.new("RGB", (tw * cols, 130 * rows_n), (25, 25, 35))
        draw = ImageDraw.Draw(canvas)
        for i, t in enumerate(tiles):
            cx, cy = (i % cols) * tw + 5, (i // cols) * 130 + 8
            canvas.paste(t, (cx, cy))
            draw.text((cx, cy + 112), f"#{i}", fill=(255, 220, 90))
        canvas.save(f"/tmp/real_{name}.png")
        print(f"  saved /tmp/real_{name}.png {canvas.size}")


if __name__ == "__main__":
    main()
