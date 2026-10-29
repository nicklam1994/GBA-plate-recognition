#!/usr/bin/env python3
"""
從 CCPD 全量裁出車牌小圖 + 文字標註，用於訓練更強的 CRNN。

CCPD 的優勢：檔名自帶 bbox + 車牌號碼，零人工標註成本。
355k 張圖可以產出 300k+ 訓練樣本。

輸出結構（we0091234 的 CRNN 訓練格式）：
    crnn_dataset/
        images/<車牌號>_<序號>.jpg
        train.txt     "<路徑> <c0> <c1> ... <cN>"   ← 字符索引序列
        val.txt
        chars.txt     字符表

用法：
    python scripts/build_crnn_dataset.py --limit 100000 --workers 8
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

ROOT = Path("/home/nicklam-ai/repos/GBA-plate-recognition")
CCPD = ROOT / "test_data/real/ccpd/CCPD2019"
OUT = ROOT / "crnn_dataset"

#: CCPD 用綠牌標記的特殊字符（新能源車牌第 3 位）
PROVINCES = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑",
             "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤",
             "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新"]
ALPHABETS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N',
             'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'O']
ADS = list('ABCDEFGHJKLMNPQRSTUVWXYZ0123456789O')

#: 與 gba_plate/ocr/vocab.py 的 BASE_VOCAB_V78 保持一致
#: 注意：BASE_VOCAB_V78 的 index 0 是 '#'（CTC blank），所以真實字符從 1 開始
CHARSET = list(
    "京沪津渝冀晋蒙辽吉黑苏浙皖闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新"
    "学警港澳挂使领民航危"
    "0123456789"
    "ABCDEFGHJKLMNPQRSTUVWXYZ"
    "险品"
)
#: ★ 索引 +1，對齊 BASE_VOCAB_V78（0 保留給 CTC blank）
CHAR2IDX = {c: i + 1 for i, c in enumerate(CHARSET)}


def parse_ccpd(fname: str):
    stem = fname.rsplit(".", 1)[0]
    parts = stem.split("-")
    if len(parts) < 5:
        return None
    try:
        xy = parts[2].split("_")
        x1, y1 = (int(v) for v in xy[0].split("&"))
        x2, y2 = (int(v) for v in xy[1].split("&"))
        codes = list(map(int, parts[4].split("_")))
        if len(codes) < 7:
            return None
        text = PROVINCES[codes[0]] + ALPHABETS[codes[1]] + "".join(ADS[c] for c in codes[2:7])
        return (x1, y1, x2, y2), text
    except Exception:
        return None


def work(args):
    """裁一張圖，回傳 (檔名, 文字, 是否成功)"""
    fpath, out_dir, idx = args
    f = Path(fpath)
    parsed = parse_ccpd(f.name)
    if parsed is None:
        return None
    (x1, y1, x2, y2), text = parsed

    if not all(c in CHAR2IDX for c in text):
        return None

    img = cv2.imdecode(np.fromfile(str(f), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return None
    H, W = img.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W, x2), min(H, y2)
    w, h = x2 - x1, y2 - y1
    if w < 24 or h < 8:
        return None

    crop = img[y1:y2, x1:x2]
    name = f"{text}_{idx}.jpg"
    cv2.imwrite(str(out_dir / name), crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return (name, text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=100000, help="最多裁多少張")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--val-ratio", type=float, default=0.02)
    args = ap.parse_args()

    img_dir = OUT / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    subsets = ["ccpd_base", "ccpd_challenge", "ccpd_blur", "ccpd_db",
               "ccpd_fn", "ccpd_rotate", "ccpd_tilt", "ccpd_weather"]
    all_files = []
    for sub in subsets:
        d = CCPD / sub
        if d.exists():
            fs = sorted(d.glob("*.jpg"))
            all_files.extend(fs)
            print(f"  {sub:<16} {len(fs):>7}")

    print(f"\n總計 {len(all_files)} 張可裁")
    random.seed(42)
    random.shuffle(all_files)
    picked = all_files[: args.limit]
    print(f"本次裁 {len(picked)} 張，workers={args.workers}\n")

    tasks = [(str(f), img_dir, i) for i, f in enumerate(picked)]

    results = []
    with Pool(args.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(work, tasks, chunksize=200), 1):
            if r is not None:
                results.append(r)
            if i % 5000 == 0:
                print(f"  已處理 {i}/{len(tasks)}  成功 {len(results)}", flush=True)

    print(f"\n成功裁出 {len(results)} 張")

    # 字符索引序列
    random.shuffle(results)
    n_val = max(1, int(len(results) * args.val_ratio))
    val_set, train_set = results[:n_val], results[n_val:]

    def write_list(entries, path: Path):
        with open(path, "w", encoding="utf-8") as fh:
            for name, text in entries:
                idxs = " ".join(str(CHAR2IDX[c]) for c in text)
                fh.write(f"{img_dir / name}\t{idxs}\n")

    write_list(train_set, OUT / "train.txt")
    write_list(val_set, OUT / "val.txt")

    with open(OUT / "chars.txt", "w", encoding="utf-8") as fh:
        fh.write("\n".join(CHARSET))

    # 統計
    cnt = Counter(text[0] for _, text in results)
    print(f"\ntrain: {len(train_set)}  val: {len(val_set)}")
    print(f"輸出: {OUT}")
    print(f"\n省份分佈（前 10）:")
    for k, v in cnt.most_common(10):
        print(f"  {k}  {v:>6}")

    lens = Counter(len(t) for _, t in results)
    print(f"\n車牌長度分佈: {dict(sorted(lens.items()))}")


if __name__ == "__main__":
    main()
