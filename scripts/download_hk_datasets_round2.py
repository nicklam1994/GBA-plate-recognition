#!/usr/bin/env python3
"""
下載新發現的香港/澳門車牌數據集（第二輪搜尋）

第一輪已下載 6 個數據集（6,311 圖）但香港只有 223 框，
本輪找到類別定義更清晰的數據集，香港框數可達 8,000+。

用法：
    export ROBOFLOW_API_KEY=wpRMtxKqnZYhlj6bCo0A
    python scripts/download_hk_datasets_round2.py --priority 1
"""
import argparse
import os
import sys
from pathlib import Path

# 第二輪找到的數據集（按價值排序）
DATASETS = [
    {
        "workspace": "default-mn0jt",
        "project": "hk-license-plate",
        "local_name": "hk_default",
        "desc": "HK License Plate (4,205 圖 / 7,679 框)",
        "classes": "hk-back 3413, hk-front 2343, hkcn 1321, mo 602",
        "priority": 1,
    },
    {
        "workspace": "trial-gdyuh",
        "project": "car-plates-zihyp",
        "local_name": "hk_trial_zihyp",
        "desc": "Trial car-plates (951 圖 / 2,531 框)",
        "classes": "hk 1550, hkcn 696, hk-front 227, mo 58",
        "priority": 1,
    },
    {
        "workspace": "trial-gdyuh",
        "project": "0824car-plates",
        "local_name": "hk_trial_0824",
        "desc": "0824 car-plates (597 圖 / 1,070 框)",
        "classes": "hk 468, hk-front 329, hkcn 228, mo 45",
        "priority": 1,
    },
    {
        "workspace": "macaocarplaterecognitiontraining",
        "project": "macau_license_plate_shareimages",
        "local_name": "macau_shareimages",
        "desc": "Macau ShareImages (803 圖 / 874 框)",
        "classes": "MO 867, CN 7",
        "priority": 2,
    },
    {
        "workspace": "mius-workspace",
        "project": "hk-car-license-plate-detection-gmcf1",
        "local_name": "hk_mius",
        "desc": "HK Car Plate Detection (150 圖 / 178 框)",
        "classes": "hk-car-license-plate",
        "priority": 2,
    },
    {
        "workspace": "itp-project",
        "project": "hk-car-license-plate-detection",
        "local_name": "hk_itp",
        "desc": "HK Car Plate Detection (107 圖 / 112 框)",
        "classes": "hk-car-license-plate",
        "priority": 2,
    },
    {
        "workspace": "hong-kong",
        "project": "license-plate-recognition-rxg4e-7lvtq",
        "local_name": "hk_big_10k",
        "desc": "HK License Plate Recognition (10,125 圖 / 10,637 框) — 單類，需驗證",
        "classes": "License_Plate",
        "priority": 3,
    },
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="test_data/real")
    parser.add_argument("--priority", type=int, default=2,
                        help="下載 priority <= N（1=最高價值，3=需驗證）")
    args = parser.parse_args()

    api_key = os.environ.get("ROBOFLOW_API_KEY", "wpRMtxKqnZYhlj6bCo0A").strip()
    if not api_key:
        print("❌ 未設定 ROBOFLOW_API_KEY")
        sys.exit(1)

    try:
        from roboflow import Roboflow
    except ImportError:
        print("❌ pip install roboflow")
        sys.exit(1)

    rf = Roboflow(api_key=api_key)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    results = []
    for ds in DATASETS:
        if ds["priority"] > args.priority:
            print(f"⏭  跳過 {ds['local_name']}（priority {ds['priority']} > {args.priority}）")
            continue

        print()
        print("=" * 74)
        print(f"📦 {ds['desc']}")
        print(f"   {ds['workspace']}/{ds['project']}")
        print(f"   類別: {ds['classes']}")
        print("=" * 74)

        try:
            proj = rf.workspace(ds["workspace"]).project(ds["project"])
            versions = proj.versions()
            if not versions:
                print("   ⚠️ 無版本，跳過")
                results.append((ds["local_name"], "no versions", 0))
                continue

            vnum = versions[-1].version
            print(f"   v{vnum} 下載中...", flush=True)

            ds_obj = proj.version(vnum).download(
                "yolov8",
                location=str(out / ds["local_name"]),
                overwrite=True,
            )
            loc = Path(ds_obj.location)
            n_img = len(list(loc.rglob("*.jpg"))) + len(list(loc.rglob("*.png")))
            n_lbl = len(list(loc.rglob("labels/*.txt")))
            print(f"   ✅ {n_img} 圖 / {n_lbl} 標註檔")
            results.append((ds["local_name"], "ok", n_img))
        except Exception as e:
            print(f"   ❌ 失敗: {e}")
            results.append((ds["local_name"], f"error: {str(e)[:60]}", 0))

    print()
    print("=" * 74)
    print("📊 匯總")
    print("=" * 74)
    total = 0
    for name, status, cnt in results:
        mark = "✅" if status == "ok" else "❌"
        print(f"{mark} {name:<20} {cnt:>6} 圖  {status}")
        if status == "ok":
            total += cnt
    print("-" * 74)
    print(f"新增總計: {total} 圖")


if __name__ == "__main__":
    main()
