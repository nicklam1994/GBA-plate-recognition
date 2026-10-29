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
from datetime import datetime
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


def _flush(history: list[dict], args) -> None:
    """把本次評估追加到歷史檔，方便比較不同模型／不同輪次。"""
    if not args.record or not history:
        return
    path = ROOT / "crnn_verified" / "eval_history.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        for h in history:
            fh.write(json.dumps(h, ensure_ascii=False) + "\n")
    print(f"\n已記錄 {len(history)} 條評估結果 → {path.relative_to(ROOT)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="prelabel/hk")
    ap.add_argument("--crnn", default="weights/crnn_gba_v2.pth")
    ap.add_argument("--skip-crnn", action="store_true")
    ap.add_argument("--split", choices=["all", "train", "test"], default="all",
                    help="只評某個切分。test 為凍結測試集，是唯一可跨版本比較的分數")
    ap.add_argument("--frozen", default="crnn_verified/test_frozen.json")
    ap.add_argument("--record", action="store_true",
                    help="把結果追加到 eval_history.jsonl")
    args = ap.parse_args()

    rows = load_pairs(ROOT / args.data)
    print(f"有效真值 {len(rows)} 筆")

    # 按凍結名單篩選切分
    if args.split != "all":
        fp = ROOT / args.frozen
        if not fp.exists():
            raise SystemExit(f"缺少凍結名單 {fp}，先跑 export_verified_dataset.py")
        frozen = json.loads(fp.read_text(encoding="utf-8"))
        test_files = set(frozen["files"])
        if args.split == "test":
            rows = [r for r in rows if r["file"] in test_files]
        else:
            rows = [r for r in rows if r["file"] not in test_files]
        print(f"切分 {args.split}：{len(rows)} 筆"
              f"（凍結集 {frozen['sha256_16']}）")

    history: list[dict] = []

    def record(model: str, rep) -> None:
        d = rep.as_dict()
        history.append({
            "ts": datetime.now().isoformat(timespec="seconds"),
            "split": args.split,
            "frozen_sha": (json.loads((ROOT / args.frozen).read_text(encoding="utf-8"))["sha256_16"]
                           if (ROOT / args.frozen).exists() else None),
            "model": model,
            "n": d["total"],
            "em": d["em"],
            "em_rate": d["em_rate"],
            "cer": d["cer"],
            "by_region": {k: v["em_rate"] for k, v in d["by_region"].items()},
        })

    # --- PaddleOCR（用預標註結果，不必重跑推論） ---
    rep_p = evaluate_pairs([(r["gt"], r["pred_paddle"], r["region"]) for r in rows])
    show("PaddleOCR（預標註時的輸出）", rep_p)
    record("paddleocr", rep_p)

    if args.skip_crnn:
        _flush(history, args)
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
    record(f"crnn:{wpath.name}", rep_c)
    _flush(history, args)


if __name__ == "__main__":
    main()
