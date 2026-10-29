#!/usr/bin/env python3
"""
從 plate.hk 抓取完整真實香港車牌號碼庫（全 dataset）。

關鍵修正：原腳本只抓 dataset=tvrm_physical（112,719 條），
漏掉 pvrm（個性化）與 tvrm_eauction（電子拍賣），
且正則假設「1-2 字母 + 數字」，把純數字、純字母車牌全部丟棄。
實際 plate.hk 三集合合計 215,051 條，含：
    1314 / 188881        純數字（早期車牌）
    ASTRA / AWW / MLH    純字母
    TR0UBLE / P1G B0SS   個性化
    AA1 / ZZ9888         雙字母 + 數字（主流）

用法：
    python scripts/fetch_hk_plate_db.py --dataset all --target 220000
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

FORBIDDEN = {"I", "O", "Q"}


def classify(text: str) -> str:
    """判定車牌格式類別"""
    t = text.replace(" ", "").replace("-", "").upper()
    if not t:
        return "空"
    if re.fullmatch(r"\d+", t):
        return "numeric"
    if re.fullmatch(r"[A-Z]", t):
        return "single_letter"
    if re.fullmatch(r"[A-Z]{2,}", t):
        return "letters_only"
    if re.fullmatch(r"[A-Z]\d+", t):
        return "one_letter_digits"
    if re.fullmatch(r"[A-Z]{2}\d+", t):
        return "two_letter_digits"
    if re.fullmatch(r"[A-Z0-9]+", t):
        return "alnum_mixed"
    return "other"


def normalize(text: str | None) -> tuple[str, str] | None:
    """
    正規化車牌文字，回傳 (清理後文字, 類別)。

    保留純數字、純字母、個性化車牌 —— 這些都是真實存在的香港車牌。
    """
    if not text:
        return None
    t = str(text).strip().upper()
    if not t or t in {"SALE", "UNSOLD", "N/A", "NONE"}:
        return None
    # 有些欄位是 list 的字串表示，例如 "['P1G', 'B0SS']"
    if t.startswith("[") and t.endswith("]"):
        try:
            parts = json.loads(t.replace("'", '"'))
            t = "".join(str(p) for p in parts)
        except Exception:
            t = re.sub(r"[^A-Z0-9]", "", t)
    t = re.sub(r"[\s\-·]", "", t)
    if not t or len(t) > 8:
        return None
    if not re.fullmatch(r"[A-Z0-9]+", t):
        return None
    # 排除萬能牌等非車牌文字
    if t in {"SALE", "UNSOLD", "NONE", "NA"}:
        return None
    # 長度合理性：香港車牌 1-8 字符（個性化最多 8）
    if len(t) < 1:
        return None
    return t, classify(t)


def fetch_page(page: int, page_size: int, dataset: str, retries: int = 4) -> list[dict]:
    for attempt in range(retries):
        try:
            r = requests.get(
                API,
                params={"dataset": dataset, "sort": "date_desc",
                        "page": page, "page_size": page_size},
                timeout=25,
            )
            if r.status_code == 200:
                return r.json().get("rows", [])
            time.sleep(1 + attempt)
        except Exception:
            time.sleep(1 + attempt)
    return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="all")
    ap.add_argument("--target", type=int, default=220000)
    ap.add_argument("--page-size", type=int, default=200)
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args()

    meta = requests.get(API, params={"dataset": args.dataset, "page": 1,
                                     "page_size": 1}, timeout=20).json()
    total = meta.get("total", 0)
    n_pages = (total + args.page_size - 1) // args.page_size

    print("=" * 68)
    print(f"抓取 plate.hk 完整車牌庫  dataset={args.dataset}")
    print("=" * 68)
    print(f"API 總筆數: {total:,}   頁數: {n_pages}")
    print()

    buckets: dict[str, set[str]] = {}
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
                res = normalize(v)
                if res:
                    text, cat = res
                    buckets.setdefault(cat, set()).add(text)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(fetch_page, p, args.page_size, args.dataset): p
                for p in range(1, n_pages + 1)}
        done = 0
        for fut in as_completed(futs):
            done += 1
            take(fut.result())
            if done % 100 == 0:
                tot = sum(len(s) for s in buckets.values())
                print(f"  頁 {done}/{n_pages}  唯一 {tot:,}", flush=True)

    all_plates: set[str] = set()
    for s in buckets.values():
        all_plates |= s
    plates = sorted(all_plates)
    if len(plates) > args.target:
        plates = plates[: args.target]

    # 分類統計（在去重後的完整集合上重算）
    cat_counter = Counter(classify(p) for p in plates)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "count": len(plates),
                "source": f"plate.hk dataset={args.dataset}",
                "by_format": {k: sorted(v) for k, v in buckets.items()},
                "plates": plates,
            },
            fh, ensure_ascii=False, indent=1,
        )

    print()
    print(f"耗時 {time.time()-t0:.0f}s   原始 {raw:,} → 唯一 {len(plates):,}")
    print(f"輸出 {OUT} ({OUT.stat().st_size/1024/1024:.1f} MB)")
    print()
    print("格式分佈:")
    for k, v in cat_counter.most_common():
        print(f"  {k:<20} {v:>8,} ({100*v/len(plates):5.1f}%)")
    print()
    for cat in ("numeric", "single_letter", "one_letter_digits", "letters_only"):
        ex = sorted(buckets.get(cat, set()))[:10]
        if ex:
            print(f"  {cat:<20} 樣本: {ex}")


if __name__ == "__main__":
    main()
