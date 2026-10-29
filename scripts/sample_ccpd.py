#!/usr/bin/env python3
"""
從 CCPD2019 採樣轉成 YOLO bbox，類別 0 = mainland，寫入 dataset_unified。

CCPD 標註編碼在檔名裡，格式：
    025-95_113-154&383_386&473-386&473_177&454_154&383_363&402-0_0_22_27_27_33_16-37-15.jpg
    [面積]-[傾斜]-[bbox:x1&y1_x2&y2]-[4角點]-[車牌索引]-[亮度]-[模糊度]

用法：
    python scripts/sample_ccpd.py --total 13000
"""
from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
CCPD = ROOT / "test_data" / "real" / "ccpd" / "CCPD2019"
OUT = ROOT / "dataset_unified"

# 各子集採樣配額（按場景多樣性分配）
SUBSET_QUOTA = {
    "ccpd_base": 6000,       # 標準停車場場景
    "ccpd_challenge": 2000,  # 挑戰性（遮擋/多牌）
    "ccpd_rotate": 1000,     # 旋轉
    "ccpd_tilt": 1000,       # 傾斜
    "ccpd_blur": 1000,       # 模糊
    "ccpd_fn": 1000,         # 過遠/過近
    "ccpd_weather": 500,     # 雨雪
    "ccpd_db": 500,          # 過暗/過亮
}


def parse_bbox(fname: str) -> tuple[int, int, int, int] | None:
    """從 CCPD 檔名解析 bbox"""
    stem = fname.rsplit(".", 1)[0]
    parts = stem.split("-")
    if len(parts) < 3:
        return None
    try:
        coords = parts[2].split("_")
        if len(coords) != 2:
            return None
        x1, y1 = (int(v) for v in coords[0].split("&"))
        x2, y2 = (int(v) for v in coords[1].split("&"))
        if x2 <= x1 or y2 <= y1:
            return None
        return x1, y1, x2, y2
    except (ValueError, IndexError):
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--total", type=int, default=0,
                    help="總採樣數（0 = 用 SUBSET_QUOTA 預設配額）")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)

    # 計算實際配額
    quota = dict(SUBSET_QUOTA)
    if args.total > 0:
        base_sum = sum(quota.values())
        quota = {k: max(1, round(v / base_sum * args.total)) for k, v in quota.items()}

    out_img = OUT / "images" / "train"
    out_lbl = OUT / "labels" / "train"
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    total_written = 0
    total_skip = 0

    for subset, n in quota.items():
        d = CCPD / subset
        if not d.exists():
            print(f"⏭  {subset} 不存在，跳過")
            continue

        files = sorted(d.glob("*.jpg"))
        if len(files) < n:
            print(f"⚠️  {subset} 只有 {len(files)} 張，取全部")
            n = len(files)

        picked = random.sample(files, n)
        written = 0
        for f in picked:
            box = parse_bbox(f.name)
            if box is None:
                total_skip += 1
                continue
            try:
                with Image.open(f) as im:
                    W, H = im.size
            except Exception:
                total_skip += 1
                continue

            x1, y1, x2, y2 = box
            # 裁剪到圖內
            x1 = max(0, min(x1, W - 1))
            x2 = max(1, min(x2, W))
            y1 = max(0, min(y1, H - 1))
            y2 = max(1, min(y2, H))
            bw, bh = x2 - x1, y2 - y1
            if bw < 8 or bh < 4:
                total_skip += 1
                continue

            xc = (x1 + x2) / 2 / W
            yc = (y1 + y2) / 2 / H
            nw = bw / W
            nh = bh / H
            if not (0 <= xc <= 1 and 0 <= yc <= 1 and 0 < nw <= 1 and 0 < nh <= 1):
                total_skip += 1
                continue

            stem = f"ccpd__{subset}__{f.stem}"
            try:
                shutil.copy2(f, out_img / (stem + ".jpg"))
            except Exception:
                total_skip += 1
                continue
            with open(out_lbl / (stem + ".txt"), "w") as fh:
                fh.write(f"0 {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}\n")
            written += 1

        print(f"  {subset:<18} 採樣 {n:>6} → 寫出 {written:>6}")
        total_written += written

    print()
    print("=" * 60)
    print(f"CCPD 總寫出: {total_written}")
    print(f"跳過（解析失敗/異常）: {total_skip}")
    print(f"輸出: {out_img}")


if __name__ == "__main__":
    main()
