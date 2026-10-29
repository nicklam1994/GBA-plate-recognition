#!/usr/bin/env python3
"""
生成四地車牌合成數據（修正版）
- 內地車牌：永遠單行
- 粵Z牌：永遠單行
- 香港車牌：70% 單行，30% 雙行
- 澳門車牌：70% 單行，30% 雙行
"""
import json
import os
import sys
import random
import string
import requests
from pathlib import Path

# 添加項目根目錄
sys.path.insert(0, str(Path(__file__).parent.parent))

from synthetic_data.plate_specs import (
    MAINLAND_BLUE, MAINLAND_YELLOW, MAINLAND_GREEN, MAINLAND_WHITE,
    GANGAO_SINGLE, HK_FRONT, HK_REAR, HK_FRONT_DOUBLE, HK_REAR_DOUBLE,
    MO_PLATE
)
from synthetic_data.plate_renderer import render_plate
from synthetic_data.augmentations import augment_plate, simulate_capture_condition


def fetch_hk_plates(num_plates: int = 5000) -> list:
    """從 plate.hk API 獲取真實香港車牌號碼，並根據真實規格過濾"""
    plates = []
    page = 1
    page_size = 200
    
    print(f"從 plate.hk API 獲取 {num_plates} 個香港車牌號碼...")
    
    # 香港車牌禁用字母
    forbidden_letters = {'I', 'O', 'Q'}
    
    while len(plates) < num_plates:
        url = f"https://plate.hk/api/results?dataset=tvrm_physical&sort=date_desc&page={page}&page_size={page_size}"
        
        try:
            response = requests.get(url, timeout=10)
            data = response.json()
            
            if 'rows' not in data or not data['rows']:
                break
            
            for row in data['rows']:
                single = row.get('single_line', '')
                
                # 過濾有效車牌
                if single and single not in ['SALE', 'UNSOLD', ''] and len(single) <= 8:
                    # 檢查是否包含禁用字母
                    if not any(letter in forbidden_letters for letter in single.upper()):
                        # 檢查數字部分是否以 0 開頭
                        parts = single.split()
                        if len(parts) >= 2:
                            numbers = parts[-1]
                            if not numbers.startswith('0'):
                                plates.append({'text': single, 'region': 'hk'})
                        else:
                            # 純數字車牌（如 18, 168）
                            if not single.startswith('0'):
                                plates.append({'text': single, 'region': 'hk'})
                
                if len(plates) >= num_plates:
                    break
            
            page += 1
            if page % 10 == 0:
                print(f"  已獲取 {len(plates)} 個車牌 (page {page-1})")
            
        except Exception as e:
            print(f"  錯誤: {e}")
            break
    
    return plates[:num_plates]


def generate_mainland_plates(num_plates: int = 3000) -> list:
    """生成內地車牌（永遠單行）"""
    from synthetic_data.plate_specs import generate_mainland_plate
    plates = []
    for _ in range(num_plates):
        text, spec = generate_mainland_plate()
        plates.append({'text': text, 'region': 'mainland', 'spec': spec})
    return plates


def generate_mo_plates(num_plates: int = 2000) -> list:
    """生成澳門車牌（70% 單行，30% 雙行）"""
    plates = []
    for i in range(num_plates):
        # 澳門車牌格式：M + 字母 + 數字 + 數字
        letter1 = random.choice(string.ascii_uppercase)
        letter2 = random.choice(string.ascii_uppercase)
        num1 = random.randint(10, 99)
        num2 = random.randint(10, 99)
        
        is_double = (i % 10) < 3  # 30% 雙行
        
        if is_double:
            # 雙行格式：MA / 12-30（無連字符在字母後）
            text = f"M{letter1} {num1}-{num2}"
        else:
            # 單行格式：MA-12-30（有連字符）
            text = f"M{letter1}-{num1}-{num2}"
        
        plates.append({'text': text, 'region': 'mo', 'spec': MO_PLATE, 'is_double': is_double})
    return plates


def generate_gangao_plates(num_plates: int = 1000) -> list:
    """生成粵Z牌（永遠單行）"""
    from synthetic_data.plate_specs import generate_gangao_plate
    plates = []
    for _ in range(num_plates):
        text, spec = generate_gangao_plate()
        plates.append({'text': text, 'region': 'gangao', 'spec': spec})
    return plates


def render_and_save(plate_info: dict, output_dir: str, idx: int, augment: bool = True):
    """渲染並保存車牌圖片"""
    text = plate_info['text']
    region = plate_info['region']
    
    # 根據地區選擇規格
    if region == 'mainland':
        spec = plate_info['spec']
        display_text = text
        is_double = False
    
    elif region == 'gangao':
        spec = plate_info['spec']
        display_text = text
        is_double = False
    
    elif region == 'hk':
        # 香港：70% 單行，30% 雙行
        is_double = (idx % 10) < 3
        
        if is_double:
            # 雙行格式：字母 / 數字
            parts = text.split()
            if len(parts) >= 2:
                display_text = f"{parts[0]}\n{' '.join(parts[1:])}"
            else:
                display_text = text
                is_double = False
            
            spec = HK_FRONT_DOUBLE if idx % 2 == 0 else HK_REAR_DOUBLE
        else:
            display_text = text
            spec = HK_FRONT if idx % 2 == 0 else HK_REAR
    
    elif region == 'mo':
        # 澳門：70% 單行，30% 雙行
        is_double = plate_info.get('is_double', False)
        
        if is_double:
            # 雙行格式：MA / 12-34
            parts = text.replace('-', ' ').split()
            if len(parts) >= 2:
                display_text = f"{parts[0]}\n{'-'.join(parts[1:])}"
            else:
                display_text = text
                is_double = False
            
            # 澳門沒有專門的雙行規格，使用方形規格
            from synthetic_data.plate_specs import PlateSpec
            mo_double = PlateSpec(
                name="mo_double",
                width=220, height=260,
                bg_color=(15, 15, 15),
                text_color=(245, 245, 245),
                font_size=70,
                char_spacing=6,
                margin_x=20, margin_y=20,
                border_width=4,
                border_color=(245, 245, 245),
                is_double=True,
                top_height=120,
                bottom_height=120,
            )
            spec = mo_double
        else:
            display_text = text
            spec = MO_PLATE
    
    else:
        return None
    
    # 渲染車牌
    img = render_plate(display_text, spec, noise_level=0, blur_level=0)
    
    # 數據增強（70% 概率）
    if augment and idx % 10 < 7:
        img = augment_plate(img, level='medium')
        img = simulate_capture_condition(img)
    
    # 保存圖片
    filename = f"{region}_{idx:05d}.jpg"
    filepath = os.path.join(output_dir, 'images', filename)
    
    import cv2
    cv2.imwrite(filepath, img)
    
    # 生成 YOLO 標籤
    label_file = os.path.join(output_dir, 'labels_yolo', filename.replace('.jpg', '.txt'))
    
    # 類別 ID：0=mainland, 1=gangao, 2=hk, 3=mo
    class_id = {'mainland': 0, 'gangao': 1, 'hk': 2, 'mo': 3}[region]
    
    with open(label_file, 'w') as f:
        f.write(f"{class_id} 0.5 0.5 1.0 1.0\n")
    
    return {
        'filename': filename,
        'plate_text': text,
        'display_text': display_text,
        'is_double': is_double,
        'region': region,
        'spec_name': spec.name,
    }


def main():
    output_dir = Path(__file__).parent.parent / 'dataset_4region_15k'
    
    os.makedirs(output_dir / 'images', exist_ok=True)
    os.makedirs(output_dir / 'labels_yolo', exist_ok=True)
    
    print("=" * 60)
    print("生成四地車牌合成數據（修正版）")
    print("=" * 60)
    
    # 生成各地區車牌
    print("\n1. 生成內地車牌（3,000 張，永遠單行）...")
    mainland_plates = generate_mainland_plates(3000)
    
    print("\n2. 獲取香港車牌號碼（5,000 張，70% 單行 / 30% 雙行）...")
    hk_plates = fetch_hk_plates(5000)
    
    print("\n3. 生成澳門車牌（2,000 張，70% 單行 / 30% 雙行）...")
    mo_plates = generate_mo_plates(2000)
    
    print("\n4. 生成粵Z牌（1,000 張，永遠單行）...")
    gangao_plates = generate_gangao_plates(1000)
    
    # 合併所有車牌
    all_plates = []
    all_plates.extend([{'text': p['text'], 'region': 'mainland', 'spec': p['spec']} for p in mainland_plates])
    all_plates.extend([{'text': p['text'], 'region': 'hk'} for p in hk_plates])
    all_plates.extend(mo_plates)
    all_plates.extend([{'text': p['text'], 'region': 'gangao', 'spec': p['spec']} for p in gangao_plates])
    
    print(f"\n總計 {len(all_plates)} 張車牌")
    print(f"  - 內地: {len(mainland_plates)} 張（單行）")
    print(f"  - 香港: {len(hk_plates)} 張（70% 單行 / 30% 雙行）")
    print(f"  - 澳門: {len(mo_plates)} 張（70% 單行 / 30% 雙行）")
    print(f"  - 粵Z: {len(gangao_plates)} 張（單行）")
    
    # 渲染並保存
    print(f"\n開始渲染車牌...")
    labels = []
    
    for i, plate_info in enumerate(all_plates):
        label = render_and_save(plate_info, str(output_dir), i, augment=True)
        if label:
            labels.append(label)
        
        if (i + 1) % 1000 == 0:
            print(f"  已渲染 {i + 1}/{len(all_plates)} 張")
    
    # 保存標籤文件
    label_path = output_dir / 'labels.txt'
    with open(label_path, 'w', encoding='utf-8') as f:
        for label in labels:
            line = f"{label['filename']} {label['plate_text']} {label['region']} {label['spec_name']}\n"
            f.write(line)
    
    # 保存數據集配置
    config = {
        'num_samples': len(labels),
        'regions': {
            'mainland': len(mainland_plates),
            'hk': len(hk_plates),
            'mo': len(mo_plates),
            'gangao': len(gangao_plates),
        },
        'format_rules': {
            'mainland': '永遠單行',
            'gangao': '永遠單行',
            'hk': '70% 單行 / 30% 雙行',
            'mo': '70% 單行 / 30% 雙行',
        }
    }
    config_path = output_dir / 'dataset_config.json'
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'=' * 60}")
    print(f"✅ 生成完成！")
    print(f"   圖片目錄: {output_dir}/images")
    print(f"   YOLO 標籤: {output_dir}/labels_yolo")
    print(f"   標籤文件: {label_path}")
    print(f"   總計圖片: {len(labels)}")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
