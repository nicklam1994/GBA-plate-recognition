#!/usr/bin/env python3
"""
從 plate.hk API 獲取真實香港車牌號碼，生成測試數據
"""
import json
import os
import sys
import requests
from pathlib import Path

# 添加項目根目錄
sys.path.insert(0, str(Path(__file__).parent.parent))

from synthetic_data.plate_specs import HK_FRONT, HK_REAR
from synthetic_data.plate_renderer import render_plate


def fetch_hk_plates_from_api(num_plates: int = 500) -> list:
    """從 plate.hk API 獲取真實香港車牌號碼"""
    plates = []
    page = 1
    page_size = 100
    
    print(f"從 plate.hk API 獲取香港車牌號碼...")
    
    while len(plates) < num_plates:
        url = f"https://plate.hk/api/results?dataset=tvrm_physical&sort=date_desc&page={page}&page_size={page_size}"
        
        try:
            response = requests.get(url, timeout=10)
            data = response.json()
            
            if 'rows' not in data or not data['rows']:
                break
            
            for row in data['rows']:
                single = row.get('single_line', '')
                double = row.get('double_line', '')
                
                # 過濾有效車牌（排除 SALE, UNSOLD 等）
                if single and single not in ['SALE', 'UNSOLD', ''] and len(single) <= 8:
                    plates.append({
                        'text': single,
                        'type': 'hk',
                        'source': 'plate.hk'
                    })
                
                if double and len(double) <= 8:
                    plates.append({
                        'text': double,
                        'type': 'hk',
                        'source': 'plate.hk'
                    })
                
                if len(plates) >= num_plates:
                    break
            
            page += 1
            print(f"  已獲取 {len(plates)} 個車牌 (page {page-1})")
            
        except Exception as e:
            print(f"  錯誤: {e}")
            break
    
    return plates[:num_plates]


def generate_hk_test_images(plates: list, output_dir: str):
    """生成香港車牌測試圖片"""
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n生成 {len(plates)} 張香港車牌圖片...")
    
    labels = []
    
    for i, plate_info in enumerate(plates):
        text = plate_info['text']
        
        # 隨機選擇前牌（白底）或後牌（黃底）
        spec = HK_FRONT if i % 2 == 0 else HK_REAR
        
        # 渲染車牌（無增強，保持清晰）
        img = render_plate(text, spec, noise_level=0, blur_level=0)
        
        # 保存圖片
        filename = f"hk_real_{i:04d}.jpg"
        filepath = os.path.join(output_dir, filename)
        
        import cv2
        cv2.imwrite(filepath, img)
        
        # 記錄標籤
        labels.append({
            'filename': filename,
            'plate_text': text,
            'region': 'hk',
            'spec_name': spec.name,
            'source': 'plate.hk'
        })
    
    # 保存標籤文件
    label_path = os.path.join(output_dir, 'labels.txt')
    with open(label_path, 'w', encoding='utf-8') as f:
        for label in labels:
            line = f"{label['filename']} {label['plate_text']} {label['region']} {label['spec_name']}\n"
            f.write(line)
    
    print(f"✅ 生成完成！")
    print(f"   圖片目錄: {output_dir}")
    print(f"   標籤文件: {label_path}")
    print(f"   總計圖片: {len(labels)}")
    
    # 顯示樣本
    print(f"\n樣本車牌:")
    for label in labels[:10]:
        print(f"   {label['plate_text']:10s} -> {label['filename']}")


def main():
    output_dir = Path(__file__).parent / "real" / "hk_plate_hk"
    
    # 獲取真實車牌號碼
    plates = fetch_hk_plates_from_api(num_plates=500)
    
    if not plates:
        print("❌ 無法獲取車牌號碼")
        return
    
    # 生成測試圖片
    generate_hk_test_images(plates, str(output_dir))


if __name__ == '__main__':
    main()
