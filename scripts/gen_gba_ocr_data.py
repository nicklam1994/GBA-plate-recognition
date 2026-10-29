#!/usr/bin/env python3
"""
用項目自帶的合成生成器產生粵港澳車牌 OCR 訓練數據。

為什麼需要合成：
    CCPD 10 萬張裡「粵」只有 410 張（0.4%），港澳為 0。
    純真實數據無法覆蓋粵港澳字符，合成是唯一低成本方式。

標籤處理：
    香港 "AB 1234" → "AB1234"（去空格）
    澳門 "MG 12 34" → "MG1234"（去空格與連字符）
    識別後由 postprocess/rules.py 的格式規則還原分隔符，
    這樣可以維持既有 78 類字符集，不必加 '-' 重訓。

增強策略：
    合成圖過於乾淨，直接訓練會與真實照片產生 domain gap。
    每張隨機施加：噪點 / 運動模糊 / 亮度偏移 / 透視變形 / JPEG 壓縮。

用法：
    python scripts/gen_gba_ocr_data.py --gangao 12000 --hk 8000 --mo 6000
"""
from __future__ import annotations

import argparse
import random
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CHARSET = (
    "京沪津渝冀晋蒙辽吉黑苏浙皖闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新"
    "学警港澳挂使领民航危"
    "0123456789"
    "ABCDEFGHJKLMNPQRSTUVWXYZ"
    "险品"
)
CHAR2IDX = {c: i + 1 for i, c in enumerate(CHARSET)}


def clean_label(text: str) -> str:
    """去掉空格、换行、連字符，只留可識別字符"""
    return "".join(c for c in text if c in CHAR2IDX)


#: 真實香港車牌號緩存（由 fetch_hk_plate_db.py 抓取）
_HK_PLATES: list[str] | None = None


def load_hk_plates() -> list[str]:
    """載入 plate.hk 抓來的真實香港車牌號（找不到就回退隨機生成）"""
    global _HK_PLATES
    if _HK_PLATES is not None:
        return _HK_PLATES
    p = ROOT / "data" / "hk_plates.json"
    if p.exists():
        try:
            import json
            d = json.load(open(p, encoding="utf-8"))
            _HK_PLATES = d.get("plates", [])
        except Exception:
            _HK_PLATES = []
    else:
        _HK_PLATES = []
    return _HK_PLATES


def render_one(args):
    """生成單張圖，回傳 (檔名, 標籤) 或 None"""
    kind, idx, out_dir = args
    from synthetic_data.plate_specs import (
        MAINLAND_BLUE,
        MAINLAND_GREEN,
        GANGAO_SUFFIX,
        generate_gangao_plate,
        generate_hk_plate,
        generate_mo_plate,
    )
    from synthetic_data.plate_renderer import render_plate

    try:
        if kind == "gangao":
            text, spec = generate_gangao_plate()
        elif kind == "hk":
            # 優先使用 plate.hk 真實車牌號：模型學到的是真實分佈（長度、字母組合規律）
            db = load_hk_plates()
            if db:
                import re as _re
                import synthetic_data.plate_specs as _ps
                plate = random.choice(db)
                m = _re.match(r"^([A-Z]{1,2})(\d{1,4})$", plate)
                if m:
                    text = f"{m.group(1)} {m.group(2)}"
                else:
                    letters = "".join(c for c in plate if c.isalpha())
                    digits = "".join(c for c in plate if c.isdigit())
                    text = f"{letters} {digits}"
                spec = random.choice([_ps.HK_FRONT, _ps.HK_REAR])
            else:
                text, spec = generate_hk_plate()
        elif kind == "mo":
            text, spec = generate_mo_plate()
        else:
            # guangdong：直接構造粵牌（避免隨機省份的低命中率）
            import string as _s
            alpha = random.choice("ABCDEFGHJKLMNPQRSTUVWXYZ")
            if random.random() < 0.25:
                # 新能源 8 位
                special = random.choice(["D", "F"])
                rest = "".join(random.choices(
                    "ABCDEFGHJKLMNPQRSTUVWXYZ0123456789", k=5))
                text = f"粤{alpha}{special}{rest}"
                spec = MAINLAND_GREEN
            else:
                rest = "".join(random.choices(
                    "ABCDEFGHJKLMNPQRSTUVWXYZ0123456789", k=5))
                text = f"粤{alpha}{rest}"
                spec = MAINLAND_BLUE

        label = clean_label(text)
        # 香港真實車牌最短是 A1（2 字符），但 2-3 位樣本對訓練幫助有限且易歧義，
        # 下限取 3；上限 9 覆蓋粵Z雙行拼接後的最長情況
        if not (3 <= len(label) <= 9):
            return None

        # 隨機增強（模擬真實拍攝）
        img = render_plate(
            text,
            spec,
            noise_level=random.uniform(0, 0.35),
            blur_level=random.uniform(0, 2.0),
            brightness_offset=random.uniform(-35, 35),
            perspective=(random.uniform(-6, 6), random.uniform(-4, 4)),
        )

        # 額外的運動模糊（行車記錄儀最常見的退化成因）
        if random.random() < 0.3:
            from synthetic_data.augmentations import add_motion_blur
            img = add_motion_blur(img, size=random.choice([3, 5, 7]),
                                  angle=random.uniform(0, 180))

        if random.random() < 0.25:
            quality = random.randint(45, 85)
            ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
            if ok:
                img = cv2.imdecode(enc, cv2.IMREAD_COLOR)

        name = f"{kind}__{label}__{idx}.jpg"
        cv2.imwrite(str(out_dir / name), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return (name, label)
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gangao", type=int, default=12000, help="粵Z 兩地牌")
    ap.add_argument("--hk", type=int, default=8000, help="香港車牌")
    ap.add_argument("--mo", type=int, default=6000, help="澳門車牌")
    ap.add_argument("--guangdong", type=int, default=8000, help="粵牌（普通內地）")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="crnn_dataset_gba")
    args = ap.parse_args()

    out = ROOT / args.out
    img_dir = out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    plan = []
    for kind, n in [("gangao", args.gangao), ("hk", args.hk),
                    ("mo", args.mo), ("guangdong", args.guangdong)]:
        plan.extend([(kind, i, img_dir) for i in range(n)])

    print("=" * 66)
    print("生成粵港澳 OCR 訓練數據")
    print("=" * 66)
    print(f"粵Z: {args.gangao}  香港: {args.hk}  澳門: {args.mo}  粵牌: {args.guangdong}")
    print(f"總計 {len(plan)} 張，workers={args.workers}")
    print()

    random.seed(42)
    results = []
    with Pool(args.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(render_one, plan, chunksize=100), 1):
            if r is not None:
                results.append(r)
            if i % 3000 == 0:
                print(f"  {i}/{len(plan)}  成功 {len(results)}", flush=True)

    print(f"\n成功生成 {len(results)} 張")

    random.shuffle(results)
    n_val = max(1, int(len(results) * 0.03))
    val_set, train_set = results[:n_val], results[n_val:]

    def write(rows, path):
        with open(path, "w", encoding="utf-8") as fh:
            for name, label in rows:
                idxs = " ".join(str(CHAR2IDX[c]) for c in label)
                fh.write(f"{img_dir / name}\t{idxs}\n")
        print(f"  {path.name}: {len(rows)}")

    write(train_set, out / "train.txt")
    write(val_set, out / "val.txt")

    # 抽樣驗證
    from collections import Counter
    kind_cnt = Counter(n.split("__")[0] for n, _ in results)
    print(f"\n各類型:")
    for k, v in kind_cnt.most_common():
        print(f"  {k:<12} {v}")
    print(f"\n樣本示例:")
    for name, label in results[:6]:
        print(f"  {name:<38} → {label}")


if __name__ == "__main__":
    main()
