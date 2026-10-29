#!/usr/bin/env python3
"""
把人工核對過的真值匯出成 CRNN 訓練／評估集。

資料來源：
    prelabel/hk/labeled.json    OCR 預標註（含裁圖檔名、地區、置信度）
    prelabel/hk/verified.json   人工核對結果（text / action / region）

輸出：
    <out>/images/*.jpg          裁圖副本（檔名帶地區前綴，避免跨區重名）
    <out>/all.txt               全部真值，格式 path\\t<char_idx> <char_idx> ...
    <out>/train.txt / test.txt  分層切分（依地區分層，避免某一區全落在測試）
    <out>/report.json           統計：各地區筆數、字符覆蓋、被排除原因

標籤編碼與 crnn_dataset 一致：index = chars.txt 中的位置 + 1（0 保留給 CTC blank）。

用法：
    ./venv/bin/python scripts/export_verified_dataset.py
    ./venv/bin/python scripts/export_verified_dataset.py --out crnn_verified --test-ratio 0.25
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

#: 不算有效真值的動作
EXCLUDE_ACTIONS = {"unusable", "not_plate"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="prelabel/hk")
    ap.add_argument("--out", default="crnn_verified")
    ap.add_argument("--chars", default="crnn_dataset/chars.txt")
    ap.add_argument("--test-ratio", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    data = ROOT / args.data
    out = ROOT / args.out
    img_dir = out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    chars = (ROOT / args.chars).read_text(encoding="utf-8").split("\n")
    char_to_idx = {c: i + 1 for i, c in enumerate(chars) if c}
    print(f"字符表 {len(char_to_idx)} 個（index 從 1 起，0 為 CTC blank）")

    labeled = json.loads((data / "labeled.json").read_text(encoding="utf-8"))
    verified = json.loads((data / "verified.json").read_text(encoding="utf-8"))
    meta = {r["file"]: r for r in labeled}
    region_of = {f: (d.get("region") or meta.get(f, {}).get("region_det", ""))
                 for f, d in verified.items()}

    rows: list[dict] = []
    skipped = Counter()
    outside = Counter()

    for f, v in verified.items():
        action = v.get("action") or ""
        text = (v.get("text") or "").strip().upper()

        if action in EXCLUDE_ACTIONS:
            skipped[action] += 1
            continue
        if not text:
            skipped["空值"] += 1
            continue

        src = data / "crops" / f
        if not src.exists():
            skipped["裁圖缺失"] += 1
            continue

        # 檢查字符是否都在字符表裡 —— 缺一個字就整筆編不出標籤
        missing = [c for c in text if c not in char_to_idx]
        if missing:
            outside["".join(sorted(set(missing)))] += 1
            skipped["含字符表外字符"] += 1
            continue

        region = region_of.get(f, "")
        rows.append({
            "file": f, "text": text, "region": region,
            "chars_idx": [char_to_idx[c] for c in text],
            "src": src,
        })

    print(f"\n可用真值 {len(rows)} 筆")
    if skipped:
        print(f"排除：{dict(skipped)}")
    if outside:
        print(f"字符表外字符（需補進 chars.txt 才能用）：{dict(outside)}")

    # 依地區分層切分，避免小地區（粵Z 只有 9 筆）全落在測試集
    by_region: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_region[r["region"] or "unknown"].append(r)

    rng = random.Random(args.seed)
    train, test = [], []
    for region, group in sorted(by_region.items()):
        group = group[:]
        rng.shuffle(group)
        n_test = max(1, round(len(group) * args.test_ratio)) if len(group) > 1 else 0
        test.extend(group[:n_test])
        train.extend(group[n_test:])
        print(f"  {region:<10} {len(group):>3} 筆 → 訓練 {len(group)-n_test:>3} / 測試 {n_test:>2}")

    # 複製裁圖並寫標籤
    for r in rows:
        dst = img_dir / f"{r['region'] or 'unk'}__{r['file']}"
        shutil.copy2(r["src"], dst)
        r["dst"] = dst

    def write_txt(path: Path, subset: list[dict]) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            for r in subset:
                fh.write(f"{r['dst']}\t{' '.join(map(str, r['chars_idx']))}\n")

    write_txt(out / "all.txt", rows)
    write_txt(out / "train.txt", train)
    write_txt(out / "test.txt", test)

    report = {
        "total": len(rows),
        "train": len(train),
        "test": len(test),
        "by_region": dict(Counter(r["region"] for r in rows)),
        "skipped": dict(skipped),
        "chars_outside_vocab": dict(outside),
        "avg_len": round(sum(len(r["text"]) for r in rows) / max(1, len(rows)), 2),
        "len_dist": dict(Counter(len(r["text"]) for r in rows)),
    }
    (out / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"\n輸出 {out}")
    print(f"  images/  {len(rows)} 張")
    print(f"  all.txt / train.txt / test.txt")
    print(f"  地區分佈 {report['by_region']}")
    print(f"  平均長度 {report['avg_len']}  長度分佈 {report['len_dist']}")


if __name__ == "__main__":
    main()
