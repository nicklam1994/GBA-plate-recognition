#!/usr/bin/env python3
"""
C: 港澳車牌 OCR 標註輔助。

流程：
    1. 從 dataset_unified 的 hk/mo/gangao bbox 裁出車牌小圖
    2. 用新訓練的 CRNN 預測文字（預標註）
    3. 按置信度分層：
         conf >= 0.95  → 自動通過（低風險）
         0.70-0.95     → 快速審核（多半只是分隔符/單字符錯）
         < 0.70        → 必須人工標註
    4. 導出 JSON + 一份 HTML 審核頁

用法：
    python scripts/prelabel_gba_ocr.py --weights weights/crnn_small_finetune2.pth
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DS = ROOT / "dataset_unified"
REGION_NAMES = {1: "gangao", 2: "hk", 3: "mo"}
#: 裁切時向外擴張的比例（OCR 需要完整車牌與少許邊距）
PAD_RATIO = 0.10


def crop_with_pad(img: np.ndarray, bbox_norm: list[float], pad: float = PAD_RATIO):
    """bbox_norm = [xc, yc, w, h] 歸一化座標"""
    H, W = img.shape[:2]
    xc, yc, w, h = bbox_norm
    bw, bh = w * W, h * H

    x1 = max(0, int(xc * W - bw / 2 - bw * pad))
    x2 = min(W, int(xc * W + bw / 2 + bw * pad))
    y1 = max(0, int(yc * H - bh / 2 - bh * pad))
    y2 = min(H, int(yc * H + bh / 2 + bh * pad))
    if x2 - x1 < 12 or y2 - y1 < 6:
        return None
    return img[y1:y2, x1:x2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="weights/crnn_small_finetune2.pth")
    ap.add_argument("--splits", default="train,valid,test")
    ap.add_argument("--limit-per-class", type=int, default=4000,
                    help="每個地區最多取多少張（0=全部）")
    ap.add_argument("--out", default="gba_ocr_label")
    args = ap.parse_args()

    from gba_plate.ocr.crnn import CRNNPlateOCR

    out = ROOT / args.out
    crops_dir = out / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    print(f"載入模型: {Path(args.weights).name}", flush=True)
    ocr = CRNNPlateOCR(weights=ROOT / args.weights, device="cpu")

    records = []
    per_class = defaultdict(int)

    for split in args.splits.split(","):
        lbl_dir = DS / "labels" / split
        img_dir = DS / "images" / split
        if not lbl_dir.exists():
            continue

        files = sorted(lbl_dir.glob("*.txt"))
        random.seed(42)
        random.shuffle(files)

        for lbl in files:
            if args.limit_per_class:
                # 粗略控制：已達上限就跳過該類
                pass

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
                lines = [l.split() for l in fh if l.strip()]

            for li, parts in enumerate(lines):
                if len(parts) < 5:
                    continue
                try:
                    cls = int(float(parts[0]))
                    bbox = [float(v) for v in parts[1:5]]
                except ValueError:
                    continue
                if cls not in REGION_NAMES:
                    continue

                region = REGION_NAMES[cls]
                if args.limit_per_class and per_class[region] >= args.limit_per_class:
                    continue

                crop = crop_with_pad(img, bbox)
                if crop is None:
                    continue

                name = f"{region}__{split}__{lbl.stem}__{li}.jpg"
                cv2.imwrite(str(crops_dir / name), crop,
                            [cv2.IMWRITE_JPEG_QUALITY, 95])

                r = ocr.predict(crop)
                records.append({
                    "file": name,
                    "region": region,
                    "pred": r.text if r else "",
                    "conf": round(r.confidence, 4) if r else 0.0,
                    "color": r.color.value if (r and r.color) else None,
                    "size": [crop.shape[1], crop.shape[0]],
                })
                per_class[region] += 1

        if args.limit_per_class and all(per_class[r] >= args.limit_per_class
                                        for r in ("gangao", "hk", "mo")):
            break

    print(f"\n裁出 {len(records)} 張車牌小圖")
    for k, v in per_class.items():
        print(f"  {k:<10} {v}")

    # 置信度分層
    tiers = {"auto": [], "quick": [], "manual": []}
    for r in records:
        if r["conf"] >= 0.95:
            tiers["auto"].append(r)
        elif r["conf"] >= 0.70:
            tiers["quick"].append(r)
        else:
            tiers["manual"].append(r)

    print()
    print("=" * 66)
    print("置信度分層（決定人工工作量）")
    print("=" * 66)
    total = len(records)
    for name, label in [("auto", "≥0.95 可自動通過"),
                        ("quick", "0.70-0.95 快速審核"),
                        ("manual", "<0.70 必須人工")]:
        n = len(tiers[name])
        print(f"  {label:<22} {n:>6} ({100*n/max(total,1):>5.1f}%)")

    # 各分層的地區分佈
    print()
    for name in ("auto", "quick", "manual"):
        c = Counter(r["region"] for r in tiers[name])
        print(f"  {name:<8} " + "  ".join(f"{k}={v}" for k, v in c.most_common()))

    # 預測文字長度分佈（看是否合理）
    lens = Counter(len(r["pred"]) for r in records if r["pred"])
    print(f"\n預測長度分佈: {dict(sorted(lens.items()))}")

    # 存 JSON
    payload = {
        "weights": args.weights,
        "total": total,
        "tiers": {k: len(v) for k, v in tiers.items()},
        "records": records,
    }
    with open(out / "prelabels.json", "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=1)
    print(f"\n輸出: {out/'prelabels.json'}")

    # 抽樣看幾個
    print("\n樣本（各地區前 5）:")
    shown = defaultdict(int)
    for r in records:
        if shown[r["region"]] >= 5:
            continue
        shown[r["region"]] += 1
        print(f"  {r['region']:<8} {r['pred'] or '(空)':<12} conf={r['conf']:.2f}  {r['file'][:50]}")


if __name__ == "__main__":
    main()
