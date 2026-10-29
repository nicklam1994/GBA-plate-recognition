#!/usr/bin/env python3
"""
排序策略 A/B 測試。

同一批真值、同一份 PaddleOCR 輸出，只改變文字塊排序方式，
比較哪種排序的 EM 最高。

    A. 原始順序        PaddleOCR 自己給的順序
    B. y_min 差值      舊策略：|y_min 差| <= 中位高度 * 0.5 視為同一行
    C. 垂直重疊        新策略：垂直投影重疊 >= 35% 才視為同一行

用法：./venv/bin/python scripts/bench_sort_strategies.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PROVINCE_CHARS = (
    "京津冀晋蒙辽吉黑沪苏浙皖闽赣鲁豫鄂湘粤桂琼渝川贵云藏陕甘青宁新"
    "京津滬晉蒙遼吉黑滬蘇浙皖閩贛魯豫鄂湘粵桂瓊渝川貴雲藏陝甘青寧新"
    "使领領警学學挂掛港澳"
)
_CAR_TEXT = {
    "TOYOTA", "HONDA", "NISSAN", "BENZ", "BMW", "AUDI", "MAZDA", "SUBARU",
    "MITSUBISHI", "SUZUKI", "LEXUS", "HYUNDAI", "KIA", "VOLVO", "FORD",
    "FERRARI", "MASERATI", "PORSCHE", "ALPHARD", "VELLFIRE", "CROWN",
    "COMFORT", "TAXI", "POLICE", "HONG", "KONG", "MODEL", "PLATE",
}


def strip_noise(s: str) -> str:
    s = s.replace("粵", "粤")
    s = re.sub(r"[\s\-·・．.．]", "", s)
    return s.upper()


def pick_plate(texts: list[str]) -> str:
    parts = []
    for t in texts:
        c = strip_noise(t)
        if not c or any(ch.islower() for ch in t) or len(c) > 8:
            continue
        if c in _CAR_TEXT or any(b in c for b in _CAR_TEXT if len(b) > 3):
            continue
        if not re.fullmatch(rf"[A-Z0-9{PROVINCE_CHARS}]+", c):
            continue
        parts.append(c)
    return "".join(parts)


def order_raw(texts, polys):
    return list(range(len(texts)))


def order_ymin(texts, polys):
    """舊策略：只看 y_min 差值。"""
    boxes = []
    for i, pp in enumerate(polys):
        a = np.array(pp, dtype=np.float32).reshape(-1, 2)
        boxes.append((i, float(a[:, 0].min()), float(a[:, 1].min()), float(a[:, 1].max())))
    heights = [b[3] - b[2] for b in boxes]
    tol = max(4.0, (float(np.median(heights)) if heights else 10.0) * 0.5)

    lines: list[list] = []
    for b in sorted(boxes, key=lambda x: x[2]):
        for line in lines:
            if abs(b[2] - line[0][2]) <= tol:
                line.append(b)
                break
        else:
            lines.append([b])

    out = []
    for line in lines:
        out.extend(x[0] for x in sorted(line, key=lambda x: x[1]))
    return out


def order_overlap(texts, polys):
    """新策略：要求垂直投影真的重疊。"""
    boxes = []
    for i, pp in enumerate(polys):
        a = np.array(pp, dtype=np.float32).reshape(-1, 2)
        boxes.append((i, float(a[:, 0].min()), float(a[:, 1].min()), float(a[:, 1].max())))

    def same_line(a, b):
        top, bot = max(a[2], b[2]), min(a[3], b[3])
        ov = bot - top
        if ov <= 0:
            return False
        return ov >= 0.35 * max(min(a[3] - a[2], b[3] - b[2]), 1.0)

    lines: list[list] = []
    for b in sorted(boxes, key=lambda x: x[2]):
        for line in lines:
            if same_line(b, line[0]):
                line.append(b)
                break
        else:
            lines.append([b])

    out = []
    for line in lines:
        out.extend(x[0] for x in sorted(line, key=lambda x: x[1]))
    return out


def edit_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a or not b:
        return max(len(a), len(b))
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def main() -> None:
    data = ROOT / "prelabel/hk"
    labeled = json.loads((data / "labeled.json").read_text(encoding="utf-8"))
    verified = json.loads((data / "verified.json").read_text(encoding="utf-8"))

    rows = []
    for r in labeled:
        v = verified.get(r["file"])
        if not v:
            continue
        gt = (v.get("text") or "").strip()
        if not gt or v.get("action") in ("unusable", "not_plate"):
            continue
        crop = data / "crops" / r["file"]
        if crop.exists():
            rows.append((crop, gt))

    print(f"樣本 {len(rows)} 筆，載入 PaddleOCR…")
    from paddleocr import PaddleOCR
    ocr = PaddleOCR(lang="ch", use_textline_orientation=False)

    strategies = {"A 原始順序": order_raw, "B y_min 差值": order_ymin, "C 垂直重疊": order_overlap}
    score = {k: {"em": 0, "dist": 0, "n": 0} for k in strategies}

    for crop, gt in rows:
        img = cv2.imdecode(np.fromfile(str(crop), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        if img.shape[1] > 340:
            s = 320.0 / img.shape[1]
            img = cv2.resize(img, (320, max(1, int(img.shape[0] * s))), interpolation=cv2.INTER_AREA)
        r = ocr.predict(img)[0]
        d = r["res"] if isinstance(r, dict) and "res" in r else r
        texts = d.get("rec_texts") or []
        polys = d.get("rec_polys") or d.get("dt_polys") or []

        for name, fn in strategies.items():
            if len(polys) == len(texts) and len(texts) > 1:
                idx = fn(texts, polys)
                ts = [texts[i] for i in idx]
            else:
                ts = texts
            pred = pick_plate(ts)
            dist = edit_distance(gt, pred)
            score[name]["n"] += 1
            score[name]["em"] += int(dist == 0)
            score[name]["dist"] += dist

    print(f"\n{'策略':<16} {'EM':>8} {'EM%':>8} {'平均錯字':>10}")
    print("-" * 46)
    for name, s in score.items():
        n = max(s["n"], 1)
        print(f"{name:<16} {s['em']:>4}/{n:<4} {s['em']/n:>7.1%} {s['dist']/n:>9.2f}")


if __name__ == "__main__":
    main()
