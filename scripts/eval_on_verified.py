#!/usr/bin/env python3
"""
用人工核對的真值評估模型表現。

比較兩個 OCR：
    PaddleOCR  目前預標註用的（無需訓練，開箱即用）
    CRNN       專案自己訓的 gba_plate/ocr/crnn.py

指標：精確匹配率 EM（車牌場景真正關心的數字）與字符錯誤率 CER。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from gba_plate.evaluate import evaluate_pairs  # noqa: E402


def load_pairs(data_dir: Path) -> list[dict]:
    labeled = json.loads((data_dir / "labeled.json").read_text(encoding="utf-8"))
    verified = json.loads((data_dir / "verified.json").read_text(encoding="utf-8"))
    out = []
    for r in labeled:
        v = verified.get(r["file"])
        if not v:
            continue
        gt = (v.get("text") or "").strip()
        if not gt or v.get("action") in ("unusable", "not_plate"):
            continue
        out.append({
            "file": r["file"],
            "gt": gt,
            "pred_paddle": (r.get("text_clean") or "").strip(),
            "raw": r.get("paddle_raw") or [],
            "region": v.get("region") or r.get("region_det", ""),
            "crop": data_dir / "crops" / r["file"],
        })
    return out


def show(title: str, rep) -> None:
    d = rep.as_dict()
    print(f"\n{'='*66}")
    print(f"{title}")
    print(f"{'='*66}")
    print(f"  樣本 {d['total']}   EM {d['em']}/{d['total']} = {d['em_rate']:.1%}   CER {d['cer']:.3f}")
    print(f"\n  依地區:")
    for k, s in sorted(d["by_region"].items(), key=lambda x: -x[1]["n"]):
        print(f"    {k:<10} {s['n']:>3} 筆   EM {s['em_rate']:>6.1%}   CER {s['cer']:.3f}")
    if d["top_errors"]:
        print(f"\n  最常見字符錯誤: {d['top_errors'][:8]}")
    if d["failures"]:
        print(f"\n  錯誤樣本（前 12）:")
        for f in d["failures"][:12]:
            print(f"    真值 {f['gt']:<12} 預測 {f['pred'] or '(空)':<12} {f['region']:<9} 錯 {f['dist']} 字")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="prelabel/hk")
    ap.add_argument("--crnn", default="weights/crnn_gba_v2.pth")
    ap.add_argument("--skip-crnn", action="store_true")
    args = ap.parse_args()

    rows = load_pairs(ROOT / args.data)
    print(f"有效真值 {len(rows)} 筆")

    # --- PaddleOCR（用預標註結果，不必重跑推論） ---
    rep_p = evaluate_pairs([(r["gt"], r["pred_paddle"], r["region"]) for r in rows])
    show("PaddleOCR（預標註時的輸出）", rep_p)

    if args.skip_crnn:
        return

    # --- CRNN ---
    wpath = ROOT / args.crnn
    if not wpath.exists():
        print(f"\n[略過 CRNN] 找不到權重 {wpath}")
        return

    from gba_plate.ocr.crnn import CRNNOcr

    ocr = CRNNOcr(weights=wpath, device="0")
    pairs = []
    for r in rows:
        img = cv2.imdecode(np.fromfile(str(r["crop"]), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        try:
            text = ocr.recognize(img)
        except Exception:
            text = ""
        pairs.append((r["gt"], text, r["region"]))
    rep_c = evaluate_pairs(pairs)
    show(f"CRNN（{wpath.name}）", rep_c)


if __name__ == "__main__":
    main()
