#!/usr/bin/env python3
"""
預標註流水線 步驟 1：檢測 + 裁圖。

對指定目錄的真實圖跑 YOLO 四類檢測，裁出車牌並放大到寬 320，
輸出 crops 與 index.json，供 PaddleOCR 識別。

用法：
    ./venv/bin/python scripts/prelabel_stage1_crop.py --dir test_data/real/hk --out prelabel/hk
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from gba_plate.alpr import ALPR

ROOT = Path(__file__).resolve().parent.parent

# --- 雙行拼接（與 gba_plate.preprocess.double_line 同邏輯） ---
UPPER_END_RATIO = 0.52
LOWER_START_RATIO = 0.48


def looks_double_line(img) -> bool:
    """
    判斷是否雙行牌。

    只靠寬高比會誤判：裁圖含上下留白時，單行長牌的比例也會掉到 2.4 以下
    （實測 135 張裡誤判 122 張為雙行，反而把單行牌拼壞）。
    改為寬高比粗篩 + 水平投影找行間空白帶 —— 雙行牌在 35%~65% 高度處
    必然有一段近空白的間隙，單行牌的中段則密佈筆劃。
    """
    h, w = img.shape[:2]
    if h < 16 or w < 16:
        return False
    ratio = w / float(h)
    if ratio >= 2.6:
        return False           # 明顯單行
    if ratio < 1.3:
        return True            # 接近方形／豎長，必然雙行

    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, bw = cv2.threshold(gray, 0, 255,
                              cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        row_ink = bw.sum(axis=1) / 255.0
        band = row_ink[int(h * 0.35):int(h * 0.65)]
        if band.size == 0:
            return False
        mean_ink = row_ink.mean()
        if mean_ink <= 0:
            return False
        # 中段有明顯低谷 → 兩行之間的分隔帶
        return bool(float(band.min()) < 0.22 * float(mean_ink))
    except Exception:
        return False


def split_merge_double(img):
    """上排（字母）與下排（數字）水平拼接成單行。"""
    h = img.shape[0]
    upper = img[0:int(UPPER_END_RATIO * h), :]
    lower = img[int(LOWER_START_RATIO * h):h, :]
    if upper.size == 0 or lower.size == 0:
        return img
    upper = cv2.resize(upper, (lower.shape[1], lower.shape[0]))
    return np.hstack((upper, lower))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="test_data/real/hk")
    ap.add_argument("--out", default="prelabel/hk")
    ap.add_argument("--target-w", type=int, default=320,
                    help="裁圖放大後的目標寬度（PaddleOCR 對小圖不敏感）")
    ap.add_argument("--pad", type=int, default=5, help="裁切外擴像素")
    ap.add_argument("--limit", type=int, default=0, help="只處理前 N 張（0=全部）")
    args = ap.parse_args()

    src = (ROOT / args.dir) if not Path(args.dir).is_absolute() else Path(args.dir)
    out = (ROOT / args.out) if not Path(args.out).is_absolute() else Path(args.out)
    crops_dir = out / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    alpr = ALPR(
        detect_model=str(ROOT / "weights" / "gba_4class_yolov8s_best.pt"),
        rec_model=str(ROOT / "weights" / "crnn_gba_v4.pth"),
        class_mode="region",
        device="cuda",
    )

    imgs = sorted(p for p in src.rglob("*")
                  if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if args.limit:
        imgs = imgs[: args.limit]

    print(f"來源 {src} 共 {len(imgs)} 張 → 輸出 {out}")

    records = []
    n_det = 0
    for i, p in enumerate(imgs, 1):
        img = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        H, W = img.shape[:2]
        try:
            dets = alpr.detector.predict(img)
        except Exception:
            continue
        # 由上而下排序，保證索引穩定
        dets = sorted(dets, key=lambda d: (d.bbox[1] + d.bbox[3]) / 2)
        n_det += len(dets)

        for j, d in enumerate(dets):
            x1, y1, x2, y2 = [int(v) for v in d.bbox]
            pd = args.pad
            crop = img[max(0, y1 - pd):min(H, y2 + pd),
                       max(0, x1 - pd):min(W, x2 + pd)]
            if crop.size == 0:
                continue
            # 雙行方牌先拼成單行，否則 OCR 只會讀到其中一行
            merged = looks_double_line(crop)
            if merged:
                crop = split_merge_double(crop)

            ch, cw = crop.shape[:2]
            if cw < args.target_w:
                s = args.target_w / float(cw)
                crop = cv2.resize(crop, (int(cw * s), int(ch * s)),
                                  interpolation=cv2.INTER_CUBIC)
            region = getattr(d.region, "value", str(d.region))
            name = f"{p.stem[:40]}__{j}__{region}.jpg"
            cv2.imwrite(str(crops_dir / name), crop,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            records.append({
                "file": name,
                "src": str(p.relative_to(ROOT)),
                "idx": j,
                "region_det": region,
                "det_conf": round(float(d.confidence), 3),
                "bbox": [x1, y1, x2, y2],
                "crop_wh": [int(crop.shape[1]), int(crop.shape[0])],
                "double_line_merged": bool(merged),
            })
        if i % 100 == 0:
            print(f"  {i}/{len(imgs)}  已裁 {len(records)} 個牌", flush=True)

    json.dump(records, open(out / "index.json", "w"), ensure_ascii=False, indent=1)
    print(f"\n完成：{len(imgs)} 張圖 → {len(records)} 個車牌裁圖（{n_det} 個檢測框）")
    from collections import Counter
    print("地區分佈:", dict(Counter(r["region_det"] for r in records)))


if __name__ == "__main__":
    main()
