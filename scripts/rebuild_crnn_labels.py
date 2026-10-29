#!/usr/bin/env python3
"""
重建 crnn_dataset 的 train.txt/val.txt（修正索引偏移）。

原因：build_crnn_dataset.py 首次執行時 CHAR2IDX 從 0 開始，
但 BASE_VOCAB_V78 的 index 0 是 CTC blank，真實字符需 +1。

本腳本直接從檔名（`車牌號_序號.jpg`）重建，順便重新劃分 train/val。
"""
import random
import sys
from pathlib import Path

ROOT = Path("/home/nicklam-ai/repos/GBA-plate-recognition")
DS = ROOT / "crnn_dataset"

CHARSET = (
    "京沪津渝冀晋蒙辽吉黑苏浙皖闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新"
    "学警港澳挂使领民航危"
    "0123456789"
    "ABCDEFGHJKLMNPQRSTUVWXYZ"
    "险品"
)
CHAR2IDX = {c: i + 1 for i, c in enumerate(CHARSET)}  # 0 = blank


def main():
    val_ratio = float(sys.argv[1]) if len(sys.argv) > 1 else 0.02
    img_dir = DS / "images"
    files = sorted(img_dir.glob("*.jpg"))
    print(f"找到 {len(files)} 張圖")

    entries = []
    skipped = 0
    for f in files:
        plate = f.stem.rsplit("_", 1)[0]
        if not plate or any(c not in CHAR2IDX for c in plate):
            skipped += 1
            continue
        entries.append((f, plate))

    print(f"有效 {len(entries)}，跳過 {skipped}")

    random.seed(42)
    random.shuffle(entries)
    n_val = max(1, int(len(entries) * val_ratio))
    val_set, train_set = entries[:n_val], entries[n_val:]

    def write(rows, path):
        with open(path, "w", encoding="utf-8") as fh:
            for f, plate in rows:
                idxs = " ".join(str(CHAR2IDX[c]) for c in plate)
                fh.write(f"{f}\t{idxs}\n")
        print(f"  {path.name}: {len(rows)} 行")

    write(train_set, DS / "train.txt")
    write(val_set, DS / "val.txt")

    # 驗證：抽查第一行
    with open(DS / "train.txt", encoding="utf-8") as fh:
        first = fh.readline().strip()
    path, idxs = first.split("\t")
    decoded = "".join(CHARSET[int(v) - 1] for v in idxs.split())
    actual = Path(path).stem.rsplit("_", 1)[0]
    print(f"\n抽查: {actual} → 索引 {idxs} → 解碼 {decoded}  {'✅' if decoded == actual else '❌'}")


if __name__ == "__main__":
    main()
