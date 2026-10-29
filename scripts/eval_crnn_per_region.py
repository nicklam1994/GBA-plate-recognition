#!/usr/bin/env python3
"""
分地區評估 CRNN 模型：CCPD 真實內地牌 vs 合成粵港澳。

混合驗證集只有一個總體數字，看不出港澳是否真的學會了。
本腳本按圖檔名前綴分組：
    ccpd__*     → 真實內地（皖為主）
    gangao__*   → 合成粵Z
    hk__*       → 合成香港
    mo__*       → 合成澳門
    guangdong__*→ 合成粵牌

用法：
    python scripts/eval_crnn_per_region.py --weights weights/crnn_gba_mix.pth
"""
from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path("/home/nicklam-ai/repos/GBA-plate-recognition")
sys.path.insert(0, str(ROOT))

CHARSET = (
    "京沪津渝冀晋蒙辽吉黑苏浙皖闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新"
    "学警港澳挂使领民航危"
    "0123456789"
    "ABCDEFGHJKLMNPQRSTUVWXYZ"
    "险品"
)
IDX2CHAR = {i + 1: c for i, c in enumerate(CHARSET)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="weights/crnn_gba_mix.pth")
    ap.add_argument("--data", default="crnn_dataset_mix")
    ap.add_argument("--split", default="val.txt")
    ap.add_argument("--per-group", type=int, default=400, help="每組最多測多少")
    args = ap.parse_args()

    from gba_plate.ocr.crnn import CRNNPlateOCR

    print(f"載入 {Path(args.weights).name}", flush=True)
    ocr = CRNNPlateOCR(weights=ROOT / args.weights, device="cpu")

    groups: dict[str, list[tuple[Path, str]]] = defaultdict(list)
    with open(ROOT / args.data / args.split, encoding="utf-8") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 2:
                continue
            path, idxs = parts
            label = "".join(IDX2CHAR.get(int(v), "") for v in idxs.split())
            name = Path(path).name
            group = name.split("__")[0] if "__" in name else "other"
            groups[group].append((Path(path), label))

    random.seed(7)
    print()
    print("=" * 70)
    print(f"{'地區':<14} {'樣本':>6} {'整串正確':>10} {'字符級':>8}")
    print("=" * 70)

    overall_exact = 0
    overall_total = 0

    for group in sorted(groups):
        items = groups[group]
        if len(items) > args.per_group:
            items = random.sample(items, args.per_group)

        exact = 0
        c_ok = 0
        c_all = 0
        fails = []

        for path, gt in items:
            img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8),
                               cv2.IMREAD_COLOR)
            if img is None:
                continue
            r = ocr.predict(img)
            pred = r.text if r else ""
            if pred == gt:
                exact += 1
            elif len(fails) < 4:
                fails.append((gt, pred))

            c_all += max(len(gt), len(pred))
            for i in range(min(len(gt), len(pred))):
                if gt[i] == pred[i]:
                    c_ok += 1

        n = len(items)
        overall_exact += exact
        overall_total += n
        print(f"{group:<14} {n:>6} {exact:>5} ({100*exact/max(n,1):>5.1f}%) "
              f"{100*c_ok/max(c_all,1):>6.1f}%")
        for gt, pred in fails:
            print(f"               {gt:<10} → {pred:<10}")

    print("=" * 70)
    print(f"{'總計':<14} {overall_total:>6} {overall_exact:>5} "
          f"({100*overall_exact/max(overall_total,1):>5.1f}%)")


if __name__ == "__main__":
    main()
