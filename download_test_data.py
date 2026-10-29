#!/usr/bin/env python3
"""
下載真實車牌測試圖片
"""
import os
import requests
from pathlib import Path


def download_roboflow_hk_dataset():
    """下載 Roboflow 香港車牌數據集"""
    print("=" * 60)
    print("下載 Roboflow 香港車牌數據集")
    print("=" * 60)
    
    # Roboflow 公開數據集（需要 API key）
    # 這裡提供下載鏈接，手動下載
    url = "https://universe.roboflow.com/hong-kong-license-plate-detect/hk-license-plate-recognition"
    
    print(f"\n請訪問以下鏈接手動下載：")
    print(f"{url}")
    print(f"\n下載後解壓到: test_data/real/hk_roboflow/")
    

def download_ccpd_sample():
    """下載 CCPD 數據集樣本"""
    print("\n" + "=" * 60)
    print("CCPD 數據集（中國城市停車）")
    print("=" * 60)
    
    print(f"\n數據集規模: 25萬+ 張真實車牌圖片")
    print(f"下載地址: https://opendatalab.com/OpenDataLab/CCPD")
    print(f"文件大小: 13.11 GB")
    print(f"\n請訪問上述鏈接下載完整數據集")
    print(f"下載後解壓到: test_data/real/ccpd/")


def download_cblprd_sample():
    """下載 CBLPRD-330k 數據集"""
    print("\n" + "=" * 60)
    print("CBLPRD-330k 數據集")
    print("=" * 60)
    
    print(f"\n數據集規模: 33萬張中國車牌圖片")
    print(f"GitHub: https://github.com/SunlifeV/CBLPRD-330k")
    print(f"\n請訪問上述鏈接下載數據集")
    print(f"下載後解壓到: test_data/real/cblprd/")


def create_test_structure():
    """創建測試數據目錄結構"""
    print("\n" + "=" * 60)
    print("創建測試數據目錄結構")
    print("=" * 60)
    
    base_dir = Path("test_data/real")
    
    dirs = [
        base_dir / "hk_roboflow",      # 香港車牌
        base_dir / "ccpd",             # 內地車牌（CCPD）
        base_dir / "cblprd",           # 中國車牌（CBLPRD）
        base_dir / "gangao",           # 粵Z兩地牌
        base_dir / "mo",               # 澳門車牌
    ]
    
    for dir_path in dirs:
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"✅ 創建目錄: {dir_path}")
    
    print(f"\n✅ 目錄結構創建完成")
    print(f"\n請按以下步驟下載數據：")
    print(f"1. Roboflow 香港車牌 → test_data/real/hk_roboflow/")
    print(f"2. CCPD 數據集 → test_data/real/ccpd/")
    print(f"3. CBLPRD-330k → test_data/real/cblprd/")
    print(f"4. 粵Z兩地牌 → test_data/real/gangao/ (需手動收集)")
    print(f"5. 澳門車牌 → test_data/real/mo/ (需手動收集)")


def print_summary():
    """打印數據集摘要"""
    print("\n" + "=" * 70)
    print("📊 真實車牌測試圖片資源摘要")
    print("=" * 70)
    
    summary = """
┌─────────────────────────────────────────────────────────────────┐
│ 數據集名稱          │ 規模      │ 地區    │ 狀態      │ 優先級 │
├─────────────────────────────────────────────────────────────────┤
│ HZM (港珠澳大橋)    │ 1,376 張  │ 三地    │ 需聯繫作者 │ ⭐⭐⭐⭐⭐ │
│ CCPD                │ 25萬+ 張  │ 內地    │ ✅ 可下載  │ ⭐⭐⭐⭐  │
│ CBLPRD-330k         │ 33萬張    │ 中國    │ ✅ 可下載  │ ⭐⭐⭐⭐  │
│ Roboflow HK         │ 33 張     │ 香港    │ ✅ 可下載  │ ⭐⭐⭐   │
│ 粵Z兩地牌 (手動)    │ 需收集    │ 粵Z     │ ❌ 需收集  │ ⭐⭐⭐⭐⭐ │
│ 澳門車牌 (手動)     │ 需收集    │ 澳門    │ ❌ 需收集  │ ⭐⭐⭐⭐  │
└─────────────────────────────────────────────────────────────────┘

建議優先級：
1. 先下載 CCPD + CBLPRD（內地車牌基準測試）
2. 下載 Roboflow HK（香港車牌測試）
3. 聯繫 HZM 作者獲取三地數據集
4. 手動收集粵Z和澳門車牌圖片
"""
    print(summary)


if __name__ == "__main__":
    create_test_structure()
    download_roboflow_hk_dataset()
    download_ccpd_sample()
    download_cblprd_sample()
    print_summary()
    
    print("\n" + "=" * 70)
    print("✅ 完成！請按上述指引下載真實車牌測試圖片")
    print("=" * 70)
