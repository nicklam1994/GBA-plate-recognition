#!/usr/bin/env python3
"""
下載 Roboflow 上公開的港澳車牌數據集

前置：需要 Roboflow 免費 API key
    1. 註冊 https://app.roboflow.com
    2. 取得 key: https://app.roboflow.com/settings/api
    3. export ROBOFLOW_API_KEY=rf_xxxxx

用法：
    export ROBOFLOW_API_KEY=rf_xxxxx
    python scripts/download_hk_mo_datasets.py --output test_data/real

數據集（均為 CC BY 4.0）：
    - V2 Merged Macau License Plate   FYP, 2,495 張, 類別 GDM/HKF/HKR/MEV/MO
    - alan carplate                   HK License plates, 134 張, 三地牌同車
    - Macau_License_Plate_ParkingLot  macaocarplaterecognitiontraining, 239 張
    - HK License Plate Recognition    hong-kong-license-plate-detect, 33 張
"""
import argparse
import os
import sys
from pathlib import Path


# 目標數據集：workspace / project / 說明
DATASETS = [
    {
        "workspace": "fyp-dmovy",
        "project": "v2-merged-macau-license-plate",
        "local_name": "macau_merged",
        "desc": "V2 Merged Macau License Plate (2,495 張, GDM/HKF/HKR/MEV/MO)",
        "priority": 1,
    },
    {
        "workspace": "hk-license-plates",
        "project": "alan-carplate",
        "local_name": "hk_alan_carplate",
        "desc": "alan carplate (134 張, 三地牌同車場景)",
        "priority": 1,
    },
    {
        "workspace": "macaocarplaterecognitiontraining",
        "project": "macau_license_plate_parkinglot",
        "local_name": "macau_parkinglot",
        "desc": "Macau_License_Plate_ParkingLot (239 張)",
        "priority": 2,
    },
    {
        "workspace": "hong-kong-license-plate-detect",
        "project": "hk-license-plate-recognition",
        "local_name": "hk_recognition",
        "desc": "HK License Plate Recognition (33 張)",
        "priority": 3,
    },
]


def check_api_key() -> str:
    """檢查 API key"""
    key = os.environ.get("ROBOFLOW_API_KEY", "").strip()
    if not key:
        print("❌ 未設定 ROBOFLOW_API_KEY")
        print("")
        print("請先取得免費 API key：")
        print("  1. 註冊 https://app.roboflow.com")
        print("  2. 複製 key: https://app.roboflow.com/settings/api")
        print("  3. export ROBOFLOW_API_KEY=rf_xxxxx")
        sys.exit(1)
    if not key.startswith("rf_"):
        print(f"⚠️  API key 格式可疑（應以 rf_ 開頭）: {key[:8]}...")
    return key


def download_all(output_dir: Path, only_priority: int = None):
    """下載所有港澳數據集"""
    try:
        from roboflow import Roboflow
    except ImportError:
        print("❌ 未安裝 roboflow SDK")
        print("   pip install roboflow")
        sys.exit(1)

    api_key = check_api_key()
    rf = Roboflow(api_key=api_key)

    output_dir.mkdir(parents=True, exist_ok=True)

    results = []

    for ds in DATASETS:
        if only_priority and ds["priority"] > only_priority:
            continue

        print("")
        print("=" * 70)
        print(f"📦 {ds['desc']}")
        print(f"   workspace: {ds['workspace']}")
        print(f"   project:   {ds['project']}")
        print("=" * 70)

        try:
            project = rf.workspace(ds["workspace"]).project(ds["project"])

            # 列出可用版本
            versions = project.versions()
            if not versions:
                print("   ⚠️  沒有可用版本，跳過")
                results.append((ds["local_name"], "no versions", 0))
                continue

            # 取最新版本
            latest = versions[-1]
            version_num = latest.version
            print(f"   版本: v{version_num}")

            target = output_dir / ds["local_name"]
            dataset = project.version(version_num).download(
                "yolov8",
                location=str(target),
                overwrite=True,
            )

            # 統計圖片數
            img_count = len(list(Path(dataset.location).rglob("*.jpg"))) + \
                        len(list(Path(dataset.location).rglob("*.png")))

            print(f"   ✅ 下載完成: {dataset.location}")
            print(f"   圖片數: {img_count}")
            results.append((ds["local_name"], "ok", img_count))

        except Exception as e:
            print(f"   ❌ 失敗: {e}")
            results.append((ds["local_name"], f"error: {e}", 0))

    # 匯總
    print("")
    print("=" * 70)
    print("📊 下載結果匯總")
    print("=" * 70)
    total = 0
    for name, status, count in results:
        mark = "✅" if status == "ok" else "❌"
        print(f"{mark} {name:25s} {count:6d} 張  {status}")
        if status == "ok":
            total += count
    print("-" * 70)
    print(f"   總計新增: {total} 張")
    print("")


def main():
    parser = argparse.ArgumentParser(
        description="下載 Roboflow 公開港澳車牌數據集",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--output",
        default="test_data/real",
        help="輸出目錄（默認 test_data/real）",
    )
    parser.add_argument(
        "--priority",
        type=int,
        default=None,
        help="只下載 priority <= N 的數據集（1=最高價值）",
    )
    args = parser.parse_args()

    download_all(Path(args.output), args.priority)


if __name__ == "__main__":
    main()
