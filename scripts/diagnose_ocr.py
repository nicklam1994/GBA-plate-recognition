#!/usr/bin/env python3
"""
診斷 CRNN 準確率瓶頸：是模型容量不足，還是輸入分辨率不足？

現有 INPUT_SIZE = (168, 48)。7 位車牌每位只有 24px 寬，
省份漢字筆畫多，24×48 可能不夠。

做法：同一批圖用不同放大倍率送入模型，看準確率變化。
注意：模型是用 168×48 訓練的，放大輸入不一定更好，但能看出信息量瓶頸。
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path("/home/nicklam-ai/repos/GBA-plate-recognition")
sys.path.insert(0, str(ROOT))

from gba_plate.ocr.crnn import CRNNPlateOCR, ctc_greedy_decode  # noqa: E402


def eval_at_scale(ocr, bench, truth, scale_label, pre_resize_width):
    """用指定預處理寬度評估（高度按比例）"""
    exact = 0
    char_ok = 0
    char_all = 0

    for name, gt in truth.items():
        img = cv2.imdecode(np.fromfile(str(bench / "crops" / name), dtype=np.uint8),
                           cv2.IMREAD_COLOR)
        if img is None:
            continue

        # 先放大到更高分辨率，再讓模型 resize 回 168×48
        # 這樣保留更多細節給 resize 算法
        if pre_resize_width:
            h, w = img.shape[:2]
            ratio = pre_resize_width / w
            if ratio > 1:
                img = cv2.resize(img, (int(w * ratio), int(h * ratio)),
                                 interpolation=cv2.INTER_CUBIC)

        r = ocr.predict(img)
        pred = r.text if r else ""
        if pred == gt:
            exact += 1
        char_all += max(len(gt), len(pred))
        for i in range(min(len(gt), len(pred))):
            if gt[i] == pred[i]:
                char_ok += 1

    n = len(truth)
    print(f"  {scale_label:<28} 整串 {exact:>2}/{n} ({100*exact/n:>5.1f}%)   "
          f"字符 {100*char_ok/max(char_all,1):>5.1f}%")
    return exact


def main():
    bench = ROOT / "ocr_bench"
    truth = json.load(open(bench / "truth.json", encoding="utf-8"))

    print("載入模型...", flush=True)
    ocr = CRNNPlateOCR(weights=ROOT / "weights/plate_rec_color.pth", device="cpu")

    print()
    print("=" * 70)
    print("輸入分辨率對準確率的影響（同一模型，不同預放大倍率）")
    print("=" * 70)
    eval_at_scale(ocr, bench, truth, "原始（直接 resize 168x48）", 0)
    eval_at_scale(ocr, bench, truth, "預放大到 336 寬", 336)
    eval_at_scale(ocr, bench, truth, "預放大到 504 寬", 504)
    eval_at_scale(ocr, bench, truth, "預放大到 672 寬", 672)

    print()
    print("=" * 70)
    print("按車牌長度分組的準確率（原始輸入）")
    print("=" * 70)
    by_len = {}
    for name, gt in truth.items():
        img = cv2.imdecode(np.fromfile(str(bench / "crops" / name), dtype=np.uint8),
                           cv2.IMREAD_COLOR)
        if img is None:
            continue
        r = ocr.predict(img)
        pred = r.text if r else ""
        L = len(gt)
        ok, tot = by_len.get(L, (0, 0))
        by_len[L] = (ok + (1 if pred == gt else 0), tot + 1)

    for L in sorted(by_len):
        ok, tot = by_len[L]
        print(f"  {L} 位: {ok}/{tot} ({100*ok/tot:.0f}%)")

    print()
    print("=" * 70)
    print("首位（省份）識別準確率")
    print("=" * 70)
    prov_ok = 0
    prov_all = 0
    confusions = {}
    for name, gt in truth.items():
        img = cv2.imdecode(np.fromfile(str(bench / "crops" / name), dtype=np.uint8),
                           cv2.IMREAD_COLOR)
        if img is None:
            continue
        r = ocr.predict(img)
        pred = r.text if r else ""
        prov_all += 1
        if pred and pred[0] == gt[0]:
            prov_ok += 1
        elif pred:
            key = f"{gt[0]}→{pred[0]}"
            confusions[key] = confusions.get(key, 0) + 1

    print(f"  省份正確: {prov_ok}/{prov_all} ({100*prov_ok/prov_all:.1f}%)")
    print(f"  主要混淆:")
    for k, v in sorted(confusions.items(), key=lambda x: -x[1])[:12]:
        print(f"    {k}  x{v}")


if __name__ == "__main__":
    main()
