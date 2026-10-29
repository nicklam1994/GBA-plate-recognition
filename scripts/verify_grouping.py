#!/usr/bin/env python3
"""用 three_region 的真實三地牌圖片驗證車輛級聚合。"""
import sys
from pathlib import Path

sys.path.insert(0, "/home/nicklam-ai/repos/GBA-plate-recognition")

import cv2
import numpy as np

from gba_plate.alpr import ALPR
from gba_plate.grouping import group_vehicles, check_combo, REGION_LABEL

ROOT = Path("/home/nicklam-ai/repos/GBA-plate-recognition")
IMGDIR = ROOT / "test_data" / "real" / "three_region"

alpr = ALPR(
    detect_model=str(ROOT / "weights" / "gba_4class_yolov8s_best.pt"),
    rec_model=str(ROOT / "weights" / "crnn_gba_v4.pth"),
    class_mode="region",
    device="cuda",
)

imgs = sorted([p for p in IMGDIR.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
print(f"three_region {len(imgs)} 張\n" + "=" * 74)

n_multi = 0
n_single = 0
for p in imgs:
    img = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        continue
    results = alpr.process(img)
    vehicles = group_vehicles(results)

    print(f"\n{p.name[:46]}   牌 {len(results)} 個 → 車輛 {len(vehicles)} 輛")
    for v in vehicles:
        if v.is_multi_plate:
            n_multi += 1
        else:
            n_single += 1
        print(f"  {v.describe()}")
        print(f"     組合={v.combo}  主牌={v.primary.text if v.primary else '-'}  "
              f"入出方向={v.entry_direction or '-'}")
        for note in check_combo(v):
            print(f"     ⚠ {note}")

print("\n" + "=" * 74)
print(f"多牌車輛 {n_multi} 輛 / 單牌車輛 {n_single} 輛")
print(f"聚合比：{len(imgs)} 張圖的 {sum(1 for _ in imgs)} 組預期均為 1 輛三牌車")
