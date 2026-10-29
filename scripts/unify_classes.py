#!/usr/bin/env python3
"""
把各來源數據集的類別標籤統一映射到 GBA 4 類別。

目標類別：
    0 = mainland  內地（藍牌/綠牌/黃牌/警/學等）
    1 = gangao    粵Z 兩地牌（港澳入出內地車輛）
    2 = hk        香港車牌（前牌白底 / 後牌黃底）
    3 = mo        澳門車牌（黑底白字 / 黃底，含 MEV/MTP/兩行牌）

同時處理：
    - polygon 標註（>5 列）→ 轉成外接矩形 bbox
    - 空標註檔（背景圖）→ 保留，作為負樣本
    - 未知類別 → 記錄並跳過

用法：
    python scripts/unify_classes.py --dry-run     # 只統計不寫檔
    python scripts/unify_classes.py               # 產出 unified/
"""
from __future__ import annotations

import argparse
import shutil
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "test_data" / "real"
OUT = ROOT / "dataset_unified"

#: 類別名 → 目標 id。None 表示丟棄。
CLASS_MAP: dict[str, int | None] = {
    # ---- 內地 ----
    "mainland": 0,
    "china": 0,
    "cn": 0,
    "china_carplate": 0,
    "cn-license-plate": 0,    # trial-gdyuh/car-plates-zihyp 的內地牌類
    "License_Plate_cn": 0,   # hk_big_10k 待驗證後調整

    # ---- 粵Z 兩地牌 ----
    "gangao": 1,
    "hkcn-license-plate": 1,  # 港車北上 / 粵Z 兩地牌
    "gdm": 1,                 # GuangDong-Macau（已目視驗證：粵Z·AK75澳）
    "GDM": 1,

    # ---- 香港 ----
    "hk": 2,
    "hk-license-plate": 2,
    "hk-front-license-plate": 2,
    "hk-back-license-plate": 2,
    "hk_carplate": 2,
    "hk-car-license-plate": 2,
    "hk-car-license-plate": 2,
    "hk_front": 2,
    "hk_back": 2,

    # ---- 澳門 ----
    "mo": 3,
    "MO": 3,
    "mo-license-plate": 3,
    "macau_carplate": 3,
    "mev": 3,      # Macau Electric Vehicle（已驗證：MW-56-57 橙紅底）
    "MEV": 3,
    "mtp": 3,      # Macau 特殊牌
    "MTP": 3,
    "hkr": 3,      # ★ 已目視驗證：HKR 實為澳門車尾牌（MS-11-04 / ZM 5922）
    "HKR": 3,
    "hkf": 3,      # ★ 已目視驗證：HKF 實為澳門車頭牌（MX-44-44 / ZM 1444）
    "HKF": 3,
    "Licence-Plate": 3,      # macau_clear / macau_shh5r 專案（澳門）
    "licence-plate": 3,

    # ---- 丟棄 ----
    "OTH": None,   # 已驗證：多為低質量/無法辨識樣本
    "oth": None,
    "hk-car-license-plate-plate": None,
}

#: 已知只有單一類別名的數據集 → 直接指定目標（優先於 CLASS_MAP）
#: 因為同一個 "License-Plate" 在 hk_lpr 是香港、在 macau_clear 是澳門
DATASET_CLASS_OVERRIDE: dict[str, dict[str, int | None] | None] = {
    "hk_lpr": {"License-Plate": 2},           # 已目視驗證：LL 4675 / SVB 典型港牌
    "hk_big_10k": {"License_Plate": 2},       # 待驗證，暫按 HK 處理
    "macau_clear": {"Licence-Plate": 3},
    "macau_shh5r": {"Licence-Plate": 3},
    "macau_parkinglot": {"MO": 3},
    "macau_shareimages": {"MO": 3, "CN": 0},
    "hk_mius": {"hk-car-license-plate": 2},
    "hk_itp": {"hk-car-license-plate": 2},
    "hk_alan_carplate": {
        "china_carplate": 0,
        "hk-car-license-plate": 2,
        "hk_carplate": 2,
        "macau_carplate": 3,
    },
    "macau_merged": None,      # 用 data.yaml 的類別名走 CLASS_MAP
    "hk_default": None,
    "hk_trial_zihyp": None,
    "hk_trial_0824": None,
}

REGION_NAMES = {0: "mainland", 1: "gangao", 2: "hk", 3: "mo", None: "(丟棄)"}


def read_yaml_names(yaml_path: Path) -> list[str]:
    """從 data.yaml 讀取類別名（不依賴 pyyaml 的完整功能）"""
    names: list[str] = []
    try:
        import yaml
        with open(yaml_path) as f:
            d = yaml.safe_load(f)
        ns = d.get("names", [])
        if isinstance(ns, dict):
            names = [ns[k] for k in sorted(ns)]
        elif isinstance(ns, list):
            names = ns
    except Exception:
        # 後備：手動解析
        with open(yaml_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith("- "):
                    names.append(line[2:].strip())
    return names


def polygon_to_bbox(vals: list[float]) -> tuple[float, float, float, float] | None:
    """把 polygon 的 x1 y1 x2 y2 ... 轉成 xc yc w h（歸一化）"""
    if len(vals) < 6 or len(vals) % 2 != 0:
        return None
    xs = vals[0::2]
    ys = vals[1::2]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    w, h = x2 - x1, y2 - y1
    if w <= 0 or h <= 0:
        return None
    return (x1 + w / 2, y1 + h / 2, w, h)


def process_dataset(ds_dir: Path, dry_run: bool, stats: dict, log: list[str]):
    name = ds_dir.name

    # 找 data.yaml
    yamls = list(ds_dir.glob("*.yaml")) + list(ds_dir.glob("*.yml"))
    names = read_yaml_names(yamls[0]) if yamls else []

    override = DATASET_CLASS_OVERRIDE.get(name)

    for split in ("train", "valid", "test"):
        img_dir = ds_dir / split / "images"
        lbl_dir = ds_dir / split / "labels"
        if not lbl_dir.exists():
            continue

        out_img = OUT / "images" / split
        out_lbl = OUT / "labels" / split
        if not dry_run:
            out_img.mkdir(parents=True, exist_ok=True)
            out_lbl.mkdir(parents=True, exist_ok=True)

        for lbl in sorted(lbl_dir.glob("*.txt")):
            stem = lbl.stem
            # 找對應圖片
            img_src = None
            for ext in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"):
                cand = img_dir / (stem + ext)
                if cand.exists():
                    img_src = cand
                    break
            if img_src is None:
                stats["missing_image"] += 1
                continue

            new_lines = []
            with open(lbl) as f:
                for line in f:
                    parts = line.split()
                    if not parts:
                        continue
                    try:
                        cls_id = int(float(parts[0]))
                    except ValueError:
                        stats["bad_class"] += 1
                        continue
                    nums = [float(v) for v in parts[1:]]

                    # 決定類別名
                    if override is not None:
                        cls_name = names[cls_id] if cls_id < len(names) else None
                        target = override.get(cls_name) if cls_name else None
                    else:
                        cls_name = names[cls_id] if cls_id < len(names) else None
                        target = CLASS_MAP.get(cls_name) if cls_name is not None else None

                    stats["src_class"][cls_name] += 1

                    if target is None:
                        stats["dropped"] += 1
                        continue

                    # bbox 還是 polygon
                    if len(nums) == 4:
                        coords = nums
                        stats["bbox"] += 1
                    elif len(nums) > 4:
                        b = polygon_to_bbox(nums)
                        if b is None:
                            stats["bad_polygon"] += 1
                            continue
                        coords = list(b)
                        stats["polygon_converted"] += 1
                    else:
                        stats["bad_format"] += 1
                        continue

                    # 檢查歸一化範圍
                    if not all(-0.01 <= v <= 1.01 for v in coords):
                        stats["out_of_range"] += 1
                        continue

                    new_lines.append(
                        f"{target} " + " ".join(f"{v:.6f}" for v in coords)
                    )
                    stats["kept"] += 1
                    stats["dst_class"][target] += 1

            if not new_lines:
                stats["empty_label"] += 1

            if not dry_run:
                # 統一改名避免跨數據集衝突
                out_stem = f"{name}__{split}__{stem}"
                shutil.copy2(img_src, out_img / (out_stem + img_src.suffix))
                with open(out_lbl / (out_stem + ".txt"), "w") as f:
                    f.write("\n".join(new_lines))
                    if new_lines:
                        f.write("\n")
                stats["files_written"] += 1

    log.append(f"  {name}: names={names} override={'yes' if override else 'no'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--src", default=str(SRC))
    args = ap.parse_args()

    src = Path(args.src)
    dirs = sorted([d for d in src.iterdir() if d.is_dir() and d.name not in ("ccpd", "hk", "mo", "three_region")])

    stats = {
        "kept": 0, "dropped": 0, "bbox": 0, "polygon_converted": 0,
        "bad_polygon": 0, "bad_format": 0, "bad_class": 0,
        "out_of_range": 0, "missing_image": 0, "empty_label": 0,
        "files_written": 0,
        "src_class": Counter(), "dst_class": Counter(),
    }
    log: list[str] = []

    print(f"來源: {src}")
    print(f"數據集目錄: {[d.name for d in dirs]}")
    print()

    for d in dirs:
        print(f"處理 {d.name} ...", flush=True)
        process_dataset(d, args.dry_run, stats, log)

    print()
    for line in log:
        print(line)

    print()
    print("=" * 70)
    print("類別映射結果")
    print("=" * 70)
    print("來源類別 →")
    for k, v in stats["src_class"].most_common():
        # 顯示時優先看 override 有沒有覆蓋（同名類別在不同數據集含義不同）
        tgt = CLASS_MAP.get(k, "?")
        for ov in DATASET_CLASS_OVERRIDE.values():
            if ov and k in ov:
                tgt = ov[k]
                break
        print(f"  {str(k):<32} {v:>6}  → {REGION_NAMES.get(tgt, tgt)}")
    print()
    print("目標類別分佈:")
    for k in (0, 1, 2, 3):
        v = stats["dst_class"].get(k, 0)
        print(f"  {REGION_NAMES[k]:<12} {v:>7}")
    print()
    print(f"保留框:      {stats['kept']}")
    print(f"丟棄框:      {stats['dropped']}  (OTH 等低質量)")
    print(f"bbox 直通:   {stats['bbox']}")
    print(f"polygon 轉換: {stats['polygon_converted']}")
    print(f"空標註檔:    {stats['empty_label']}  (背景圖，保留為負樣本)")
    print(f"異常: 多邊形={stats['bad_polygon']} 格式={stats['bad_format']} "
          f"類別={stats['bad_class']} 越界={stats['out_of_range']} 無圖={stats['missing_image']}")
    print(f"寫出檔案:    {stats['files_written']} {'(dry-run，未寫)' if args.dry_run else ''}")
    print()
    if not args.dry_run:
        print(f"輸出目錄: {OUT}")


if __name__ == "__main__":
    main()
