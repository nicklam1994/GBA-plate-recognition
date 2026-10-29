#!/usr/bin/env python3
"""
劃分數據集為訓練集/驗證集/測試集
"""
import os
import random
import shutil
from pathlib import Path
from collections import defaultdict


def split_dataset(
    dataset_dir: str,
    output_dir: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
):
    """
    劃分數據集
    
    Args:
        dataset_dir: 原始數據集目錄
        output_dir: 輸出目錄
        train_ratio: 訓練集比例
        val_ratio: 驗證集比例
        test_ratio: 測試集比例
        seed: 隨機種子
    """
    random.seed(seed)
    
    # 創建輸出目錄
    for split in ['train', 'val', 'test']:
        os.makedirs(f"{output_dir}/{split}/images", exist_ok=True)
        os.makedirs(f"{output_dir}/{split}/labels", exist_ok=True)
    
    # 獲取所有圖片
    images_dir = Path(dataset_dir) / 'images'
    labels_dir = Path(dataset_dir) / 'labels_yolo'
    
    all_images = list(images_dir.glob('*.jpg'))
    print(f"總計圖片: {len(all_images)}")
    
    # 按地區分組
    region_images = defaultdict(list)
    for img_path in all_images:
        name = img_path.stem
        if name.startswith('mainland'):
            region_images['mainland'].append(img_path)
        elif name.startswith('gangao'):
            region_images['gangao'].append(img_path)
        elif name.startswith('hk'):
            region_images['hk'].append(img_path)
        elif name.startswith('mo'):
            region_images['mo'].append(img_path)
    
    print(f"\n地區分佈:")
    for region, images in region_images.items():
        print(f"  {region}: {len(images)} 張")
    
    # 分層抽樣
    train_files = []
    val_files = []
    test_files = []
    
    for region, images in region_images.items():
        random.shuffle(images)
        n = len(images)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        
        train_files.extend(images[:n_train])
        val_files.extend(images[n_train:n_train + n_val])
        test_files.extend(images[n_train + n_val:])
    
    print(f"\n劃分結果:")
    print(f"  訓練集: {len(train_files)} 張")
    print(f"  驗證集: {len(val_files)} 張")
    print(f"  測試集: {len(test_files)} 張")
    
    # 複製文件
    def copy_files(files, split_name):
        for img_path in files:
            # 複製圖片
            dst_img = Path(output_dir) / split_name / 'images' / img_path.name
            shutil.copy2(img_path, dst_img)
            
            # 複製標籤
            label_name = img_path.stem + '.txt'
            src_label = labels_dir / label_name
            dst_label = Path(output_dir) / split_name / 'labels' / label_name
            if src_label.exists():
                shutil.copy2(src_label, dst_label)
    
    print(f"\n複製文件...")
    copy_files(train_files, 'train')
    copy_files(val_files, 'val')
    copy_files(test_files, 'test')
    
    print(f"\n✅ 數據集劃分完成！")
    print(f"輸出目錄: {output_dir}")


if __name__ == '__main__':
    split_dataset(
        dataset_dir='./dataset_100k',
        output_dir='./dataset_split',
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
    )
