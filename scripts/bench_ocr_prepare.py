#!/usr/bin/env python3
"""
從 CCPD 裁出車牌小圖並輸出真值，用於 OCR 方案對比。

輸出：
    ocr_bench/crops/*.jpg   裁好的車牌小圖
    ocr_bench/truth.json    {filename: plate_text}
"""
import glob
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path("/home/nicklam-ai/repos/GBA-plate-recognition")
sys.path.insert(0, str(ROOT / "scripts"))

# 複用現有解碼邏輯
PROVINCES = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑",
             "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤",
             "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新"]
ALPHABETS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N',
             'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'O']
ADS = list('ABCDEFGHJKLMNPQRSTUVWXYZ0123456789O')


def parse_ccpd(fname: str):
    stem = fname.rsplit(".", 1)[0]
    parts = stem.split("-")
    if len(parts) < 5:
        return None
    try:
        x1, y1 = (int(v) for v in parts[2].split("_")[0].split("&"))
        x2, y2 = (int(v) for v in parts[2].split("_")[1].split("&"))
        codes = list(map(int, parts[4].split("_")))
        if len(codes) < 7:
            return None
        text = PROVINCES[codes[0]] + ALPHABETS[codes[1]] + "".join(ADS[c] for c in codes[2:7])
        return (x1, y1, x2, y2), text
    except Exception:
        return None


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    random.seed(7)

    out = ROOT / "ocr_bench"
    crops_dir = out / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    # 從多個子集取樣，覆蓋不同難度
    subsets = ["ccpd_base", "ccpd_blur", "ccpd_challenge", "ccpd_db", "ccpd_fn"]
    per = max(1, n // len(subsets))
    truth = {}
    saved = 0

    for sub in subsets:
        d = ROOT / "test_data/real/ccpd/CCPD2019" / sub
        if not d.exists():
            continue
        files = sorted(d.glob("*.jpg"))
        picked = random.sample(files, min(per, len(files)))
        for f in picked:
            parsed = parse_ccpd(f.name)
            if not parsed:
                continue
            (x1, y1, x2, y2), text = parsed
            img = cv2.imdecode(np.fromfile(str(f), dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue
            H, W = img.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W, x2), min(H, y2)
            if x2 - x1 < 20 or y2 - y1 < 8:
                continue
            crop = img[y1:y2, x1:x2]
            name = f"{sub}__{f.stem[:24]}.jpg"
            cv2.imwrite(str(crops_dir / name), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
            truth[name] = text
            saved += 1

    with open(out / "truth.json", "w", encoding="utf-8") as fh:
        json.dump(truth, fh, ensure_ascii=False, indent=1)

    print(f"裁出 {saved} 張車牌小圖 → {crops_dir}")
    print(f"真值: {out/'truth.json'}")
    for k, v in list(truth.items())[:5]:
        print(f"  {k[:50]:<52} {v}")


if __name__ == "__main__":
    main()
