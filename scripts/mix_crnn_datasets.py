#!/usr/bin/env python3
"""
合併 CCPD 真實數據與粵港澳合成數據，產生混合訓練集。

目的：
    CCPD 提供真實內地牌（皖 92.8%）→ 內地識別強
    合成數據提供粵港澳字符分佈   → 補齊粵 Z / 港 / 澳

比例控制：
    真實:合成 ≈ 3:1。合成太多會拉低內地牌準確率（domain gap），
    太少則港澳學不到。

輸出：
    crnn_dataset_mix/train.txt  （絕對路徑指回原圖，不複製檔案）
    crnn_dataset_mix/val.txt
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

ROOT = Path("/home/nicklam-ai/repos/GBA-plate-recognition")


def read_pairs(path: Path) -> list[tuple[str, str]]:
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                continue
            out.append((parts[0], parts[1]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real-dir", default="crnn_dataset")
    ap.add_argument("--synth-dir", default="crnn_dataset_gba")
    ap.add_argument("--out-dir", default="crnn_dataset_mix")
    ap.add_argument("--real-sample", type=int, default=98000,
                    help="從真實數據抽多少（0=全部）")
    ap.add_argument("--repeat-synth", type=int, default=1,
                    help="合成數據重複幾次（提升其採樣概率）")
    args = ap.parse_args()

    real = read_pairs(ROOT / args.real_dir / "train.txt")
    synth = read_pairs(ROOT / args.synth_dir / "train.txt")
    real_val = read_pairs(ROOT / args.real_dir / "val.txt")
    synth_val = read_pairs(ROOT / args.synth_dir / "val.txt")

    print(f"真實 train: {len(real)}   val: {len(real_val)}")
    print(f"合成 train: {len(synth)}   val: {len(synth_val)}")

    random.seed(42)
    if args.real_sample and len(real) > args.real_sample:
        real = random.sample(real, args.real_sample)
        print(f"真實數據抽樣至 {len(real)}")

    synth_expanded = synth * max(1, args.repeat_synth)

    train = real + synth_expanded
    val = real_val + synth_val
    random.shuffle(train)
    random.shuffle(val)

    out = ROOT / args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    def write(rows, path):
        with open(path, "w", encoding="utf-8") as fh:
            for img, idxs in rows:
                fh.write(f"{img}\t{idxs}\n")
        print(f"  {path.name}: {len(rows)}")

    write(train, out / "train.txt")
    write(val, out / "val.txt")

    n_real = len(real)
    n_synth = len(synth_expanded)
    total = n_real + n_synth
    print()
    print(f"混合結果: {total}")
    print(f"  真實: {n_real} ({100*n_real/total:.1f}%)")
    print(f"  合成: {n_synth} ({100*n_synth/total:.1f}%)")
    print(f"  val:  {len(val)}（真實 {len(real_val)} + 合成 {len(synth_val)}）")


if __name__ == "__main__":
    main()
