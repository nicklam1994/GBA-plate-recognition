#!/usr/bin/env python3
"""
端到端測試：新 YOLOv8 檢測 + 新 CRNN 識別，跑用戶 607 張真實行車記錄儀圖。

無 ground truth，因此輸出：
    1. 檢測率（與舊模型 73.3% 對比）
    2. 識別結果按地區統計（合理分佈 = 真的在識別，而非亂猜）
    3. 抽樣輸出供人工判斷

用法：
    python scripts/e2e_test_real_hk.py --sample 120
"""
from __future__ import annotations

import argparse
import glob
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ROOT = Path("/home/nicklam-ai/repos/GBA-plate-recognition")
sys.path.insert(0, str(ROOT))

HK_DIR = ROOT / "test_data/real/hk"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det-weights",
                    default="runs/detect/gba_4class_yolov8s/weights/best.pt")
    ap.add_argument("--rec-weights", default="weights/crnn_gba_mix.pth")
    ap.add_argument("--sample", type=int, default=0, help="0=全部 607 張")
    ap.add_argument("--conf", type=float, default=0.3)
    args = ap.parse_args()

    from gba_plate.alpr import ALPR

    files = sorted(glob.glob(str(HK_DIR / "*")))
    if args.sample:
        random.seed(42)
        files = random.sample(files, min(args.sample, len(files)))

    print("=" * 74)
    print("端到端測試：真實行車記錄儀圖")
    print("=" * 74)
    print(f"檢測: {Path(args.det_weights).name}")
    print(f"識別: {Path(args.rec_weights).name}")
    print(f"樣本: {len(files)} 張\n")

    alpr = ALPR(
        detect_model=ROOT / args.det_weights,
        rec_model=ROOT / args.rec_weights,
        class_mode="region",
        conf_thres=args.conf,
    )

    hits = 0
    misses = 0
    total_plates = 0
    region_cnt = Counter()
    confs = []
    samples = []
    t0 = time.time()

    for i, f in enumerate(files, 1):
        img = cv2.imdecode(np.fromfile(f, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        try:
            results = alpr.process(img)
        except Exception as e:
            misses += 1
            continue

        if results:
            hits += 1
            total_plates += len(results)
            for r in results:
                region_cnt[r.region.value] += 1
                if r.ocr:
                    confs.append(r.confidence)
                    samples.append({
                        "file": Path(f).name,
                        "text": r.text,
                        "raw": r.raw_text,
                        "region": r.region.value,
                        "conf": round(r.confidence, 3),
                        "det_conf": round(r.detection.confidence, 3),
                        "valid": r.is_valid,
                        "note": r.validation_note,
                        "color": r.color.value if r.color else None,
                        "bbox": [round(v) for v in r.detection.bbox],
                    })
        else:
            misses += 1

        if i % 40 == 0:
            print(f"  {i}/{len(files)}  hit={hits} miss={misses} plates={total_plates}",
                  flush=True)

    elapsed = time.time() - t0
    n = hits + misses

    print()
    print("=" * 74)
    print("結果")
    print("=" * 74)
    print(f"圖片:        {n}")
    print(f"檢出圖片:    {hits} ({100*hits/max(n,1):.1f}%)   [舊模型 73.3%]")
    print(f"未檢出:      {misses} ({100*misses/max(n,1):.1f}%)")
    print(f"車牌總數:    {total_plates}")
    print(f"平均/圖:     {total_plates/max(n,1):.2f}")
    print(f"耗時:        {elapsed:.1f}s ({elapsed/max(n,1)*1000:.0f} ms/張)")

    if confs:
        a = np.array(confs)
        print(f"識別置信度:  中位 {np.median(a):.3f}  均值 {a.mean():.3f}")

    print()
    print("檢測到的地區分佈:")
    for k, v in region_cnt.most_common():
        print(f"  {k:<12} {v:>5} ({100*v/max(total_plates,1):.1f}%)")

    # 識別結果抽樣
    print()
    print("=" * 74)
    print("識別結果抽樣（30 筆）")
    print("=" * 74)
    print(f"{'車牌':<14} {'地區':<10} {'顏色':<5} {'conf':>5} {'det':>5} {'有效':<4}")
    print("-" * 74)
    random.seed(1)
    for s in random.sample(samples, min(30, len(samples))):
        print(f"{s['text'] or '(空)':<14} {s['region']:<10} "
              f"{s['color'] or '-':<5} {s['conf']:>5.2f} {s['det_conf']:>5.2f} "
              f"{'✓' if s['valid'] else '✗'}")

    out = Path("/tmp/e2e_real_hk.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({
            "det_weights": args.det_weights,
            "rec_weights": args.rec_weights,
            "images": n,
            "hit_rate": round(hits / max(n, 1), 4),
            "total_plates": total_plates,
            "region_dist": dict(region_cnt),
            "samples": samples,
        }, fh, ensure_ascii=False, indent=1)
    print(f"\n明細: {out}")


if __name__ == "__main__":
    main()
