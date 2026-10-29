#!/usr/bin/env python3
"""
從 plate.hk 抓取完整真實香港車牌號碼庫（11 萬+）。

與 fetch_hk_plates.py 的差異：
    原腳本只抓 500 個並直接渲染測試圖；
    本腳本抓完整號碼庫並輸出 JSON，供兩個用途：
      1. 合成數據用真實存在的車牌號（提升真實度）
      2. OCR 後處理校驗（預測結果是否為真實號碼）

API: https://plate.hk/api/results?dataset=tvrm_physical&page=N&page_size=200

用法：
    python scripts/fetch_hk_plate_db.py --target 110000
"""
from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

ROOT = Path("/home/nicklam-ai/repos/GBA-plate-recognition")
OUT = ROOT / "data" / "hk_plates.json"
API = "https://plate.hk/api/results"

#: 香港車牌禁用字母（I/O/Q 易與 1/0 混淆）
FORBIDDEN = {"I", "O", "Q"}
#: 1-2 字母 + 1-4 數字
HK_PATTERN = re.compile(r"^([A-Z]{1,2})\s*(\d{1,4})$")


def normalize(text: str | None) -> str | None:
    if not text:
        return None
    t = str(text).strip().upper()
    if not t or t in {"SALE", "UNSOLD", "N/A"}:
        return None
    m = HK_PATTERN.match(t)
    if not m:
        return None
    letters, digits = m.group(1), m.group(2)
    if any(c in FORBIDDEN for c in letters):
        return None
    if digits.startswith("0"):
        return None
    if len(letters) + len(digits) > 6:
        return None
    return f"{letters}{digits}"


def fetch_page(page: int, page_size: int, retries: int = 3) -> list[dict]:
    for attempt in range(retries):
        try:
            r = requests.get(
                API,
                params={"dataset": "tvrm_physical", "sort": "date_desc",
                        "page": page, "page_size": page_size},
                timeout=20,
            )
            if r.status_code == 200:
                return r.json().get("rows", [])
            time.sleep(1 + attempt)
        except Exception:
            time.sleep(1 + attempt)
    return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=110000)
    ap.add_argument("--page-size", type=int, default=200)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    # 取總數
    try:
        meta = requests.get(API, params={"dataset": "tvrm_physical", "page": 1,
                                         "page_size": 1}, timeout=20).json()
    except Exception as e:
        raise SystemExit(f"API 無回應: {e}")

    total = meta.get("total", 0)
    n_pages = (total + args.page_size - 1) // args.page_size

    print("=" * 66)
    print("抓取 plate.hk 真實香港車牌號碼庫")
    print("=" * 66)
    print(f"API 總筆數: {total:,}")
    print(f"頁數:       {n_pages}  (page_size={args.page_size}, workers={args.workers})")
    print()

    seen: set[str] = set()
    raw = 0
    t0 = time.time()

    def take(rows):
        nonlocal raw
        for row in rows:
            for key in ("single_line", "double_line"):
                v = row.get(key)
                if not v:
                    continue
                raw += 1
                n = normalize(v)
                if n:
                    seen.add(n)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(fetch_page, p, args.page_size): p
                for p in range(1, n_pages + 1)}
        done = 0
        for fut in as_completed(futs):
            done += 1
            take(fut.result())
            if done % 60 == 0:
                print(f"  頁 {done}/{n_pages}  唯一 {len(seen):,}", flush=True)

    elapsed = time.time() - t0
    plates = sorted(seen)
    if len(plates) > args.target:
        plates = plates[: args.target]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump({"count": len(plates), "plates": plates}, fh,
                  ensure_ascii=False, indent=1)

    print()
    print(f"耗時 {elapsed:.0f}s")
    print(f"原始 {raw:,} → 正規化唯一 {len(plates):,}")
    print(f"輸出 {OUT}  ({OUT.stat().st_size/1024:.0f} KB)")
    print(f"\n樣本: {plates[:15]}")
    lens = Counter(len(p) for p in plates)
    print(f"長度分佈: {dict(sorted(lens.items()))}")

    # 字母前綴分佈（檢查覆蓋度）
    pre = Counter(p[:2] for p in plates if p[:2].isalpha())
    print(f"\n雙字母前綴數: {len(pre)}")
    print(f"最常見: {pre.most_common(8)}")


if __name__ == "__main__":
    main()
