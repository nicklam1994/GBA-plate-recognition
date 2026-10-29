#!/usr/bin/env python3
"""
生成 10,000 張香港車牌合成數據（使用 plate.hk 真實號碼）
"""
import json
import os
import sys
import requests
from pathlib import Path

# 添加項目根目錄
sys.path.insert(0, str(Path(__file__).parent.parent))

from synthetic_data.plate_specs import HK_FRONT, HK_REAR, HK_FRONT_DOUBLE, HK_REAR_DOUBLE
from synthetic_data.plate_renderer import render_plate
from synthetic_data.augmentations import augment_plate, simulate_capture_condition


def fetch_all_hk_plates(num_plates: int = 10000) -> list:
    """從 plate.hk API 獲取大量真實香港車牌號碼"""
    plates = []
    page = 1
    page_size = 200  # 最大 page_size
    
    print(f"從 plate.hk API 獲取 {num_plates} 個香港車牌號碼...")
    
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
                    plates.append(single)
                
                if double and len(double) <= 8:
                    plates.append(double)
                
                if len(plates) >= num_plates:
                    break
            
            page += 1
            if page % 10 == 0:
                print(f"  已獲取 {len(plates)} 個車牌 (page {page-1})")
            
        except Exception as e:
            print(f"  錯誤: {e}")
            break
    
    return plates[:num_plates]


def generate_hk_synthetic_dataset(plates: list, output_dir: str, augment: bool = True):
    """生成香港車牌合成數據集"""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'labels_yolo'), exist_ok=True)
    
    print(f"\n生成 {len(plates)} 張香港車牌合成圖片...")
    
    labels = []
    
    for i, plate_text in enumerate(plates):
        # 隨機選擇格式：70% 單行，30% 雙行
        is_double = (i % 10) < 3
        
        if is_double:
            # 雙行格式：將車牌分成兩行
            # 例如 "AB 1234" → "AB\n1234"
            parts = plate_text.split()
            if len(parts) >= 2:
                top_line = parts[0]
                bottom_line = ' '.join(parts[1:])
                display_text = f"{top_line}\n{bottom_line}"
            else:
                # 如果無法分割，使用單行
                display_text = plate_text
                is_double = False
            
            # 隨機選擇前牌或後牌
            if i % 2 == 0:
                spec = HK_FRONT_DOUBLE if is_double else HK_FRONT
            else:
                spec = HK_REAR_DOUBLE if is_double else HK_REAR
        else:
            # 單行格式
            display_text = plate_text
            spec = HK_FRONT if i % 2 == 0 else HK_REAR
        
        # 渲染車牌
        img = render_plate(display_text, spec, noise_level=0, blur_level=0)
        
        # 數據增強（70% 概率）
        if augment and i % 10 < 7:
            img = augment_plate(img, level='medium')
            img = simulate_capture_condition(img)
        
        # 保存圖片
        filename = f"hk_synthetic_{i:05d}.jpg"
        filepath = os.path.join(output_dir, 'images', filename)
        
        import cv2
        cv2.imwrite(filepath, img)
        
        # 生成 YOLO 標籤（整個車牌區域）
        h, w = img.shape[:2]
        label_file = os.path.join(output_dir, 'labels_yolo', filename.replace('.jpg', '.txt'))
        with open(label_file, 'w') as f:
            # YOLO 格式: class_id x_center y_center width height (歸一化)
            # class_id: 2 = hk
            f.write(f"2 0.5 0.5 1.0 1.0\n")
        
        # 記錄標籤
        labels.append({
            'filename': filename,
            'plate_text': plate_text,
            'display_text': display_text,
            'is_double': is_double,
            'region': 'hk',
            'spec_name': spec.name,
            'source': 'plate.hk'
        })
        
        if (i + 1) % 1000 == 0:
            print(f"  已生成 {i + 1}/{len(plates)} 張")
    
    # 保存標籤文件
    label_path = os.path.join(output_dir, 'labels.txt')
    with open(label_path, 'w', encoding='utf-8') as f:
        for label in labels:
            line = f"{label['filename']} {label['plate_text']} {label['region']} {label['spec_name']}\n"
            f.write(line)
    
    # 保存數據集配置
    config = {
        'num_samples': len(labels),
        'region': 'hk',
        'source': 'plate.hk',
        'augment': augment,
    }
    config_path = os.path.join(output_dir, 'dataset_config.json')
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ 生成完成！")
    print(f"   圖片目錄: {output_dir}/images")
    print(f"   YOLO 標籤: {output_dir}/labels_yolo")
    print(f"   標籤文件: {label_path}")
    print(f"   總計圖片: {len(labels)}")


def main():
    output_dir = Path(__file__).parent.parent / 'dataset_hk_10k'
    
    # 獲取真實車牌號碼
    plates = fetch_all_hk_plates(num_plates=10000)
    
    if not plates:
        print("❌ 無法獲取車牌號碼")
        return
    
    print(f"✅ 獲取到 {len(plates)} 個真實香港車牌號碼")
    
    # 生成合成數據集
    generate_hk_synthetic_dataset(plates, str(output_dir), augment=True)


if __name__ == '__main__':
    main()
