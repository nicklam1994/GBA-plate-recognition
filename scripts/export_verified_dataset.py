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
from datetime import datetime
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
    ap.add_argument("--reset-frozen", action="store_true",
                    help="重新劃定測試集（會破壞歷史評估的可比性，慎用）")
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

    # ---- 測試集凍結 ----
    # 同一批標註既當訓練材料又當測試基準，最大的風險是數據泄漏：
    # 模型見過的樣本拿去測，準確率必然虛高，而且換一版模型就不可比。
    # 做法：第一次劃定後把測試集的 file 列表寫進 test_frozen.json，
    # 之後每次重新導出都沿用同一份名單，新標註的數據只進訓練集。
    frozen_path = out / "test_frozen.json"
    frozen = None
    if frozen_path.exists() and not args.reset_frozen:
        frozen = json.loads(frozen_path.read_text(encoding="utf-8"))

    if frozen:
        test_files = set(frozen["files"])
        available = {r["file"] for r in rows}
        missing = test_files - available
        test = [r for r in rows if r["file"] in test_files]
        train = [r for r in rows if r["file"] not in test_files]
        print(f"沿用凍結測試集（{frozen['frozen_at'][:10]} 劃定，"
              f"{len(frozen['files'])} 筆）")
        if missing:
            print(f"  ⚠ {len(missing)} 筆凍結樣本已不在資料中（可能被標為不可用）")
        # 凍結後新增的數據全部歸入訓練集
        print(f"  測試 {len(test)} 筆 / 訓練 {len(train)} 筆")
    else:
        # 首次：依地區分層切分，避免小地區（粵Z 只有 6 筆）全落在測試集
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

        import hashlib
        files = sorted(r["file"] for r in test)
        digest = hashlib.sha256("\n".join(files).encode()).hexdigest()[:16]
        frozen_path.write_text(json.dumps({
            "files": files,
            "count": len(files),
            "sha256_16": digest,
            "seed": args.seed,
            "test_ratio": args.test_ratio,
            "frozen_at": datetime.now().isoformat(timespec="seconds"),
        }, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\n已凍結測試集 → {frozen_path.name}（sha256 {digest}）")
        print("  之後重新導出會沿用這份名單，新數據只進訓練集")

    # 複製裁圖並寫標籤
    for r in rows:
        dst = img_dir / f"{r['region'] or 'unk'}__{r['file']}"
        shutil.copy2(r["src"], dst)
        r["dst"] = dst

    def write_txt(path: Path, subset: list[dict]) -> None:
        # 按檔名排序輸出，讓同樣的輸入產生位元組級相同的檔案 ——
        # 沿用凍結測試集時，集合內容雖然不變，但 rows 的原始順序會變，
        # 不排序就每次產生不同的 test.txt，難以用 diff 確認「測試集沒被動過」。
        with open(path, "w", encoding="utf-8") as fh:
            for r in sorted(subset, key=lambda x: x["file"]):
                fh.write(f"{r['dst']}\t{' '.join(map(str, r['chars_idx']))}\n")

    write_txt(out / "all.txt", rows)
    write_txt(out / "train.txt", train)
    write_txt(out / "test.txt", test)

    frozen_info = (json.loads(frozen_path.read_text(encoding="utf-8"))
                   if frozen_path.exists() else {})
    report = {
        "total": len(rows),
        "train": len(train),
        "test": len(test),
        "test_frozen": {
            "sha256_16": frozen_info.get("sha256_16"),
            "frozen_at": frozen_info.get("frozen_at"),
        },
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
