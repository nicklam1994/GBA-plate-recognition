#!/usr/bin/env python3
"""
驗證 gba_plate 與舊管線功能等價。

用法：
    python scripts/verify_refactor.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CASES = [
    ("imgs/hongkang1.jpg", "粵Z 兩地牌"),
    ("imgs/single_blue.jpg", "內地藍牌"),
    ("imgs/single_green.jpg", "內地新能源"),
    ("imgs/double_yellow.jpg", "內地雙層黃牌"),
    ("imgs/police.jpg", "內地警車牌"),
    ("imgs/xue.jpg", "內地教練牌"),
]


def main() -> int:
    from gba_plate import ALPR

    print("載入 gba_plate.ALPR（會載入兩個模型，約需數秒）...")
    alpr = ALPR(
        detect_model=str(ROOT / "weights/plate_detect.pt"),
        rec_model=str(ROOT / "weights/plate_rec_color.pth"),
    )
    print("載入完成\n")

    print("=" * 96)
    print(f"{'圖片':<28} {'地區':<10} {'識別文字':<14} {'置信':>6} {'顏色':<5} {'校驗':<6} {'備註'}")
    print("=" * 96)

    total = 0
    ok_count = 0

    for rel, desc in CASES:
        path = ROOT / rel
        if not path.exists():
            print(f"{rel:<28} 檔案不存在")
            continue

        img = cv2.imdecode(
            __import__("numpy").fromfile(str(path), dtype=__import__("numpy").uint8),
            cv2.IMREAD_COLOR,
        )
        results = alpr.process(img)

        if not results:
            print(f"{Path(rel).name:<28} {'-':<10} {'未檢測到':<14}")
            continue

        for r in results:
            total += 1
            if r.is_valid:
                ok_count += 1
            color = r.color.value if r.color else "-"
            mark = "OK" if r.is_valid else "FAIL"
            note = r.validation_note[:22] if r.validation_note else ""
            text = r.text or "(空)"
            print(
                f"{Path(rel).name:<28} {r.region.value:<10} {text:<14} "
                f"{r.confidence:>6.2f} {color:<5} {mark:<6} {note}"
            )

    print("=" * 96)
    print(f"共 {total} 個車牌，格式校驗通過 {ok_count} 個 ({ok_count/max(total,1)*100:.0f}%)")

    # 地區統計
    from collections import Counter

    region_counts: Counter[str] = Counter()
    for rel, _ in CASES:
        path = ROOT / rel
        if not path.exists():
            continue
        img = cv2.imdecode(
            __import__("numpy").fromfile(str(path), dtype=__import__("numpy").uint8),
            cv2.IMREAD_COLOR,
        )
        for r in alpr.process(img):
            region_counts[r.region.value] += 1

    print(f"地區分佈: {dict(region_counts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
