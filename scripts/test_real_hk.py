#!/usr/bin/env python3
"""
在用户 607 張真實行車記錄儀 HK 圖上測試檢測模型（無標註，只看檢測率）。

這是關鍵驗收指標：對比舊 YOLOv5-face 模型（73.3%）與新四類模型。

用法：
    python scripts/test_real_hk.py --weights runs/detect/gba_4class_yolov8s/weights/best.pt
    python scripts/test_real_hk.py --weights weights/plate_detect.pt --legacy
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

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

HK_DIR = ROOT / "test_data" / "real" / "hk"
REGION_NAMES = {0: "mainland", 1: "gangao", 2: "hk", 3: "mo"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--legacy", action="store_true", help="用舊 YOLOv5-face 管線")
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--sample", type=int, default=0, help="0 = 全部 607 張")
    args = ap.parse_args()

    files = sorted(glob.glob(str(HK_DIR / "*")))
    if args.sample:
        random.seed(42)
        files = random.sample(files, min(args.sample, len(files)))

    print(f"測試圖: {len(files)} 張")
    print(f"權重: {args.weights}")
    print(f"imgsz={args.imgsz} conf={args.conf}")
    print()

    if args.legacy:
        from gba_plate.detectors.yolo_plate import YOLOPlateDetector
        det = YOLOPlateDetector(weights=args.weights, conf_thres=args.conf, img_size=640)

        def predict(img):
            res = det.predict(img)
            return [{"conf": d.confidence, "cls": None} for d in res]
    else:
        from ultralytics import YOLO
        model = YOLO(args.weights)

        def predict(img):
            r = model.predict(img, imgsz=args.imgsz, conf=args.conf, verbose=False)[0]
            out = []
            for b in r.boxes:
                out.append({
                    "conf": float(b.conf),
                    "cls": int(b.cls),
                    "xyxy": [float(v) for v in b.xyxy[0].tolist()],
                })
            return out

    hits, misses = 0, 0
    per_img = []
    confs = []
    cls_counter = Counter()
    t0 = time.time()
    details = []

    for i, f in enumerate(files, 1):
        img = cv2.imdecode(np.fromfile(f, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        try:
            res = predict(img)
        except Exception as e:
            print(f"  預測失敗 {Path(f).name}: {e}")
            misses += 1
            continue

        per_img.append(len(res))
        if res:
            hits += 1
            confs.extend(r["conf"] for r in res)
            for r in res:
                if r["cls"] is not None:
                    cls_counter[REGION_NAMES.get(r["cls"], r["cls"])] += 1
            details.append({"file": Path(f).name, "n": len(res)})
        else:
            misses += 1

        if i % 100 == 0:
            print(f"  {i}/{len(files)} ... hit={hits} miss={misses}", flush=True)

    elapsed = time.time() - t0
    n = hits + misses

    print()
    print("=" * 66)
    print(f"樣本: {n}")
    print(f"檢測到: {hits} ({100*hits/max(1,n):.1f}%)")
    print(f"未檢出: {misses} ({100*misses/max(1,n):.1f}%)")
    print(f"平均框數/圖: {np.mean(per_img):.2f}")
    print(f"耗時: {elapsed:.1f}s ({elapsed/max(1,n)*1000:.0f} ms/張)")
    if confs:
        a = np.array(confs)
        print(f"置信度: 中位={np.median(a):.3f} P10={np.percentile(a,10):.3f} P90={np.percentile(a,90):.3f}")
    if cls_counter:
        print("檢測到的類別分佈:")
        for k, v in cls_counter.most_common():
            print(f"    {k:<12} {v:>5}")
    print("框數分佈:", {k: per_img.count(k) for k in sorted(set(per_img))})
    print("=" * 66)

    out = Path("/tmp") / f"real_hk_{Path(args.weights).stem}.json"
    with open(out, "w") as fh:
        json.dump({
            "weights": args.weights,
            "n": n, "hits": hits, "misses": misses,
            "hit_rate": round(hits / max(1, n), 4),
            "ms_per_img": round(elapsed / max(1, n) * 1000, 1),
            "class_dist": dict(cls_counter),
            "details": details,
        }, fh, ensure_ascii=False, indent=1)
    print(f"明細: {out}")


if __name__ == "__main__":
    main()
