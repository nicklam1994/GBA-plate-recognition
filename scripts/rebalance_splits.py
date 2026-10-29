#!/usr/bin/env python3
"""
修正 dataset_unified 的 valid/test 分割：把 mainland 樣本從 train 移一部分過去。

問題：CCPD 採樣只寫入 train，導致 valid/test 幾乎沒有 mainland（6/4 框），
      評估時無法反映內地車牌的真實表現。

目標比例（valid/test 各類至少 ~15%）：
    valid: mainland ~500 框
    test:  mainland ~250 框
"""
from __future__ import annotations

import random
import shutil
from collections import Counter
from pathlib import Path

ROOT = Path("/home/nicklam-ai/repos/GBA-plate-recognition")
DS = ROOT / "dataset_unified"

MOVE_VALID = 500
MOVE_TEST = 250
SEED = 42


def label_classes(lbl: Path) -> set[int]:
    out = set()
    try:
        with open(lbl) as f:
            for line in f:
                p = line.split()
                if p:
                    out.add(int(float(p[0])))
    except Exception:
        pass
    return out


def move_samples(files: list[Path], src_split: str, dst_split: str) -> int:
    n = 0
    for lbl in files:
        stem = lbl.stem
        img = None
        for ext in (".jpg", ".jpeg", ".png", ".JPG"):
            cand = DS / "images" / src_split / (stem + ext)
            if cand.exists():
                img = cand
                break
        if img is None:
            continue
        (DS / "images" / dst_split).mkdir(parents=True, exist_ok=True)
        (DS / "labels" / dst_split).mkdir(parents=True, exist_ok=True)
        shutil.move(str(img), str(DS / "images" / dst_split / img.name))
        shutil.move(str(lbl), str(DS / "labels" / dst_split / lbl.name))
        n += 1
    return n


def main():
    random.seed(SEED)

    train_labels = sorted((DS / "labels" / "train").glob("*.txt"))
    mainland = [l for l in train_labels if 0 in label_classes(l)]

    print(f"train 總標註: {len(train_labels)}")
    print(f"含 mainland 的標註: {len(mainland)}")

    need = MOVE_VALID + MOVE_TEST
    if len(mainland) < need:
        print(f"⚠️ mainland 只有 {len(mainland)}，不足 {need}")
        need = len(mainland)

    picked = random.sample(mainland, need)
    to_valid = picked[:MOVE_VALID]
    to_test = picked[MOVE_VALID:]

    nv = move_samples(to_valid, "train", "valid")
    nt = move_samples(to_test, "train", "test")
    print(f"移動到 valid: {nv} 圖")
    print(f"移動到 test:  {nt} 圖")

    names = {0: "mainland", 1: "gangao", 2: "hk", 3: "mo"}
    print()
    for split in ("train", "valid", "test"):
        cnt = Counter()
        n_img = len(list((DS / "images" / split).glob("*")))
        empty = 0
        for f in (DS / "labels" / split).glob("*.txt"):
            cls = label_classes(f)
            if not cls:
                empty += 1
            for c in cls:
                cnt[c] += 1
        total = sum(cnt.values())
        print(f"{split:<7} {n_img:>6} 圖  {total:>6} 框  (空 {empty})")
        for k in sorted(cnt):
            print(f"          {names[k]:<10} {cnt[k]:>6}  ({100*cnt[k]/max(1,total):.1f}%)")
        print()


if __name__ == "__main__":
    main()
