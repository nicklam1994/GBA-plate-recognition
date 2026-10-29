#!/usr/bin/env python3
"""
用現有 CRNN（weights/plate_rec_color.pth）跑 OCR 基準測試。
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path("/home/nicklam-ai/repos/GBA-plate-recognition")
sys.path.insert(0, str(ROOT))

from gba_plate.ocr.crnn import CRNNPlateOCR  # noqa: E402


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(ROOT / "weights/plate_rec_color.pth"))
    args = ap.parse_args()

    bench = ROOT / "ocr_bench"
    truth = json.load(open(bench / "truth.json", encoding="utf-8"))

    print(f"載入 CRNN 模型: {Path(args.weights).name}", flush=True)
    ocr = CRNNPlateOCR(weights=args.weights, device="cpu")

    exact = 0
    char_total = 0
    char_correct = 0
    fails = []
    t0 = time.time()

    for name, gt in truth.items():
        img_path = bench / "crops" / name
        r = ocr.predict_file(img_path) if hasattr(ocr, "predict_file") else None
        if r is None:
            import cv2
            import numpy as np
            img = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_COLOR)
            r = ocr.predict(img)

        pred = (r.text if r else "") or ""
        if pred == gt:
            exact += 1
        else:
            fails.append((gt, pred, round(r.confidence, 2) if r else 0))

        # 字符級準確率（按位置對齊）
        char_total += max(len(gt), len(pred))
        for i in range(min(len(gt), len(pred))):
            if gt[i] == pred[i]:
                char_correct += 1

    elapsed = time.time() - t0
    n = len(truth)

    print()
    print("=" * 62)
    print("現有 CRNN（plate_rec_color.pth）")
    print("=" * 62)
    print(f"樣本:       {n}")
    print(f"整串正確:   {exact} ({100*exact/max(n,1):.1f}%)")
    print(f"字符級正確: {char_correct}/{char_total} ({100*char_correct/max(char_total,1):.1f}%)")
    print(f"耗時:       {elapsed:.1f}s ({elapsed/max(n,1)*1000:.0f} ms/張)")
    print()
    print(f"錯誤樣本（前 20）:")
    for gt, pred, conf in fails[:20]:
        print(f"  真值 {gt:<10} → 預測 {pred:<10} (conf {conf})")


if __name__ == "__main__":
    main()
