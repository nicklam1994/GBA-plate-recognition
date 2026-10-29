#!/usr/bin/env python3
"""
用 YOLO 裁出港澳真實車牌小圖，供 PaddleOCR 識別測試。

three_region 的 11 張圖已人工核對過真值（見下方 TRUTH），
用來判斷 PaddleOCR 對港澳車牌的真實能力。
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, "/home/nicklam-ai/repos/GBA-plate-recognition")

import cv2
import numpy as np

from gba_plate.alpr import ALPR

ROOT = Path("/home/nicklam-ai/repos/GBA-plate-recognition")
IMGDIR = ROOT / "test_data" / "real" / "three_region"
OUT = ROOT / "ocr_bench" / "gba_crops"
OUT.mkdir(parents=True, exist_ok=True)

# 人工核對的真值（按圖片由上而下）
TRUTH = {
    "ScreenShot_2026-10-29_101020_876.png": ["粵Z7366澳", "MI1388", "ZM6371"],
    "ScreenShot_2026-10-29_101011_362.png": ["TB8888", "MW8888", "粵Z288K港"],
    "ScreenShot_2026-10-29_101047_481.png": ["AJ1370", "MH3842", "粵Z8640澳"],
    "ScreenShot_2026-10-29_101103_961.png": ["SA33", "SAM33", "粵ZP653澳"],
    "b52c95214cd0409a99591526dbffc1f9.jpeg": ["粵Z3840澳", "MU8888", "ZM4200"],
}

alpr = ALPR(
    detect_model=str(ROOT / "weights" / "gba_4class_yolov8s_best.pt"),
    rec_model=str(ROOT / "weights" / "crnn_gba_v4.pth"),
    class_mode="region",
    device="cuda",
)

records = []
for p in sorted(IMGDIR.iterdir()):
    if p.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
        continue
    img = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        continue
    dets = alpr.detector.predict(img)
    # 由上而下排序
    dets = sorted(dets, key=lambda d: (d.bbox[1] + d.bbox[3]) / 2)
    truth = TRUTH.get(p.name, [])
    for i, d in enumerate(dets):
        x1, y1, x2, y2 = [int(v) for v in d.bbox]
        pad = 4
        H, W = img.shape[:2]
        crop = img[max(0, y1 - pad):min(H, y2 + pad),
                   max(0, x1 - pad):min(W, x2 + pad)]
        if crop.size == 0:
            continue
        # PaddleOCR 的文字檢測對小圖不敏感：放大到寬 320 再存
        ch, cw = crop.shape[:2]
        scale = max(1.0, 320.0 / max(1, cw))
        if scale > 1.0:
            crop = cv2.resize(crop, (int(cw * scale), int(ch * scale)),
                              interpolation=cv2.INTER_CUBIC)
        name = f"{p.stem[:28]}__{i}__{getattr(d.region, 'value', d.region)}.jpg"
        cv2.imwrite(str(OUT / name), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
        records.append({
            "file": name,
            "image": p.name,
            "index": i,
            "region": getattr(d.region, "value", str(d.region)),
            "det_conf": round(float(d.confidence), 3),
            "crop_w": int(crop.shape[1]),  # 已放大
            "crop_h": int(crop.shape[0]),
            "truth": truth[i] if i < len(truth) else None,
        })

json.dump(records, open(OUT / "index.json", "w"), ensure_ascii=False, indent=1)
print(f"裁出 {len(records)} 張車牌 → {OUT}")
for r in records:
    print(f"  {r['file'][:56]:<56} {r['region']:<8} 真值={r['truth']}")
