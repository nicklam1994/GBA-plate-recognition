#!/usr/bin/env python3
"""
實測「一輛車多牌」：粵Z／港澳兩地牌車通常同時掛兩個牌照，
檢測能否同時抓到、空間關係如何、是否會被誤判成兩輛車。
"""
import sys
from pathlib import Path

sys.path.insert(0, "/home/nicklam-ai/repos/GBA-plate-recognition")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from gba_plate.alpr import ALPR  # noqa: E402

ROOT = Path("/home/nicklam-ai/repos/GBA-plate-recognition")
IMGDIR = ROOT / "test_data" / "real" / "three_region"

alpr = ALPR(
    detect_model=str(ROOT / "weights" / "gba_4class_yolov8s_best.pt"),
    rec_model=str(ROOT / "weights" / "crnn_gba_v4.pth"),
    class_mode="region",
    device="cuda",
)

imgs = sorted([p for p in IMGDIR.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
print(f"three_region 圖片 {len(imgs)} 張\n")

for p in imgs:
    img = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        continue
    H, W = img.shape[:2]
    res = alpr.process(img)
    print(f"--- {p.name[:44]}  ({W}x{H})  檢出 {len(res)} 個牌")
    for r in res:
        x1, y1, x2, y2 = r.detection.bbox
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        print(f"      {r.text or '(空)':<12} {getattr(r.region, 'value', r.region):<9} "
              f"conf={r.confidence:.2f}  中心=({cx:4.0f},{cy:4.0f})  "
              f"寬={x2-x1:3.0f} 高={y2-y1:3.0f}")

    # 兩兩距離分析（判斷能否聚類成同一輛車）
    if len(res) >= 2:
        print("      兩兩中心距離 / 平均牌寬:")
        widths = [(r.detection.bbox[2] - r.detection.bbox[0]) for r in res]
        avg_w = sum(widths) / len(widths)
        for i in range(len(res)):
            for j in range(i + 1, len(res)):
                a, b = res[i], res[j]
                ca = ((a.detection.bbox[0] + a.detection.bbox[2]) / 2, (a.detection.bbox[1] + a.detection.bbox[3]) / 2)
                cb = ((b.detection.bbox[0] + b.detection.bbox[2]) / 2, (b.detection.bbox[1] + b.detection.bbox[3]) / 2)
                dist = ((ca[0] - cb[0]) ** 2 + (ca[1] - cb[1]) ** 2) ** 0.5
                print(f"        [{i}]-[{j}] 距離 {dist:6.0f}px = {dist/avg_w:.2f}×平均牌寬")
    print()
