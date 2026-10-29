#!/usr/bin/env python3
"""
將 CCPD 數據集轉換為 YOLO 格式
CCPD 文件名包含標註信息，無需額外的標註文件
"""
import os
import re
import cv2
from pathlib import Path
from tqdm import tqdm


# CCPD 省份編碼
PROVINCES = [
    "皖", "沪", "津", "渝", "冀", "晉", "蒙", "遼", "吉", "黑",
    "蘇", "浙", "京", "閩", "贛", "魯", "豫", "鄂", "湘", "粵",
    "桂", "瓊", "川", "貴", "雲", "藏", "陝", "甘", "青", "寧", "新"
]

# 字母和數字編碼
ALPHABETS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'O']
ADS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']


def parse_ccpd_filename(filename: str):
    """
    解析 CCPD 文件名
    格式: 0018-0_7-339&483_411&505-411&505_342&505_339&483_408&483-0_0_16_9_27_27_27-82-8.jpg
    
    返回:
    - bbox: [x1, y1, x2, y2]
    - vertices: 4個頂點坐標
    - plate_number: 車牌號碼
    """
    # 移除擴展名
    name = filename.replace('.jpg', '')
    
    # 分割各部分
    parts = name.split('-')
    if len(parts) < 7:
        return None
    
    try:
        # 解析邊框坐標 (第3部分)
        bbox_str = parts[2]
        bbox_parts = bbox_str.split('_')
        x1, y1 = map(int, bbox_parts[0].split('&'))
        x2, y2 = map(int, bbox_parts[1].split('&'))
        bbox = [x1, y1, x2, y2]
        
        # 解析頂點坐標 (第4部分)
        vertices_str = parts[3]
        vertices_parts = vertices_str.split('_')
        vertices = []
        for vp in vertices_parts:
            vx, vy = map(int, vp.split('&'))
            vertices.append([vx, vy])
        
        # 解析車牌號碼 (第5部分)
        plate_str = parts[4]
        plate_codes = list(map(int, plate_str.split('_')))
        
        # 解碼車牌號碼
        if len(plate_codes) >= 7:
            province = PROVINCES[plate_codes[0]]
            alphabet = ALPHABETS[plate_codes[1]]
            ads = ''.join([ADS[code] for code in plate_codes[2:7]])
            plate_number = f"{province}{alphabet}{ads}"
        else:
            plate_number = ""
        
        return {
            'bbox': bbox,
            'vertices': vertices,
            'plate_number': plate_number
        }
    
    except Exception as e:
        return None


def convert_ccpd_to_yolo(ccpd_dir: str, output_dir: str, max_samples: int = None):
    """
    將 CCPD 數據集轉換為 YOLO 格式
    
    Args:
        ccpd_dir: CCPD 圖片目錄
        output_dir: 輸出目錄
        max_samples: 最大樣本數（None 表示全部）
    """
    ccpd_path = Path(ccpd_dir)
    output_path = Path(output_dir)
    
    # 創建輸出目錄
    images_dir = output_path / 'images'
    labels_dir = output_path / 'labels_yolo'
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    # 獲取所有圖片
    image_files = list(ccpd_path.glob('*.jpg'))
    if max_samples:
        image_files = image_files[:max_samples]
    
    print(f"找到 {len(image_files)} 張 CCPD 圖片")
    
    # 轉換
    converted = 0
    failed = 0
    
    for img_file in tqdm(image_files, desc="轉換中"):
        # 解析文件名
        annotation = parse_ccpd_filename(img_file.name)
        
        if annotation is None:
            failed += 1
            continue
        
        bbox = annotation['bbox']
        plate_number = annotation['plate_number']
        
        # 讀取圖片獲取尺寸
        img = cv2.imread(str(img_file))
        if img is None:
            failed += 1
            continue
        
        h, w = img.shape[:2]
        
        # 轉換為 YOLO 格式 (歸一化)
        x_center = (bbox[0] + bbox[2]) / 2 / w
        y_center = (bbox[1] + bbox[3]) / 2 / h
        bbox_w = (bbox[2] - bbox[0]) / w
        bbox_h = (bbox[3] - bbox[1]) / h
        
        # 複製圖片
        new_img_name = f"ccpd_{img_file.name}"
        new_img_path = images_dir / new_img_name
        cv2.imwrite(str(new_img_path), img)
        
        # 寫入標籤文件 (class_id = 0 表示內地車牌)
        label_file = labels_dir / f"{new_img_name.replace('.jpg', '.txt')}"
        with open(label_file, 'w') as f:
            f.write(f"0 {x_center:.6f} {y_center:.6f} {bbox_w:.6f} {bbox_h:.6f}\n")
        
        converted += 1
    
    print(f"\n✅ 轉換完成！")
    print(f"   成功: {converted}")
    print(f"   失敗: {failed}")
    print(f"   輸出目錄: {output_dir}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='轉換 CCPD 數據集為 YOLO 格式')
    parser.add_argument('--ccpd_dir', type=str, 
                       default='/home/nicklam-ai/repos/chinese-plate-recognition/test_data/real/ccpd/CCPD2019/ccpd_fn',
                       help='CCPD 圖片目錄')
    parser.add_argument('--output_dir', type=str,
                       default='/home/nicklam-ai/repos/chinese-plate-recognition/dataset_ccpd_yolo',
                       help='輸出目錄')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='最大樣本數（None 表示全部）')
    
    args = parser.parse_args()
    
    convert_ccpd_to_yolo(args.ccpd_dir, args.output_dir, args.max_samples)
