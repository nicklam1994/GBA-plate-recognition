#!/usr/bin/env python3
"""
整合所有車牌數據集
- dataset_100k: 100,000 張合成數據（四地車牌）
- dataset_hk_10k: 10,000 張香港車牌合成數據
- dataset_ccpd_yolo: CCPD 真實數據集
"""
import os
import shutil
from pathlib import Path
from tqdm import tqdm


def merge_datasets(output_dir: str):
    """合併所有數據集"""
    output_path = Path(output_dir)
    images_dir = output_path / 'images'
    labels_dir = output_path / 'labels_yolo'
    
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    datasets = [
        {
            'name': 'dataset_100k',
            'path': '/home/nicklam-ai/repos/chinese-plate-recognition/dataset_100k',
            'class_id': None  # 使用原始標籤
        },
        {
            'name': 'dataset_hk_10k',
            'path': '/home/nicklam-ai/repos/chinese-plate-recognition/dataset_hk_10k',
            'class_id': 2  # 香港車牌
        },
        {
            'name': 'dataset_ccpd_yolo',
            'path': '/home/nicklam-ai/repos/chinese-plate-recognition/dataset_ccpd_yolo',
            'class_id': 0  # 內地車牌
        }
    ]
    
    total_copied = 0
    
    for dataset in datasets:
        print(f"\n整合 {dataset['name']}...")
        
        ds_path = Path(dataset['path'])
        ds_images = ds_path / 'images'
        ds_labels = ds_path / 'labels_yolo'
        
        if not ds_images.exists():
            print(f"  ⚠️ 目錄不存在: {ds_images}")
            continue
        
        # 獲取所有圖片
        image_files = list(ds_images.glob('*.jpg'))
        print(f"  找到 {len(image_files)} 張圖片")
        
        # 複製圖片和標籤
        copied = 0
        for img_file in tqdm(image_files, desc=f"複製 {dataset['name']}"):
            # 生成新的文件名（添加數據集前綴）
            new_img_name = f"{dataset['name']}_{img_file.name}"
            new_img_path = images_dir / new_img_name
            
            # 複製圖片
            shutil.copy2(img_file, new_img_path)
            
            # 複製或轉換標籤
            label_file = ds_labels / f"{img_file.stem}.txt"
            if label_file.exists():
                new_label_path = labels_dir / f"{new_img_name.replace('.jpg', '.txt')}"
                
                if dataset['class_id'] is None:
                    # 直接複製原始標籤
                    shutil.copy2(label_file, new_label_path)
                else:
                    # 轉換為指定類別
                    with open(label_file, 'r') as f:
                        lines = f.readlines()
                    
                    with open(new_label_path, 'w') as f:
                        for line in lines:
                            parts = line.strip().split()
                            if len(parts) >= 5:
                                # 替換 class_id
                                parts[0] = str(dataset['class_id'])
                                f.write(' '.join(parts) + '\n')
                
                copied += 1
        
        print(f"  ✅ 複製 {copied} 張")
        total_copied += copied
    
    # 統計
    print(f"\n{'='*60}")
    print(f"✅ 整合完成！")
    print(f"   總計圖片: {total_copied}")
    print(f"   輸出目錄: {output_dir}")
    print(f"{'='*60}")
    
    # 創建 data.yaml
    data_yaml = output_path / 'data.yaml'
    with open(data_yaml, 'w') as f:
        f.write("""# 整合車牌數據集
path: {}
train: images
val: images
test: images

# 類別定義
nc: 4  # 四地車牌
names:
  0: mainland   # 內地車牌
  1: gangao     # 粵Z兩地牌
  2: hk         # 香港車牌
  3: mo         # 澳門車牌

# 數據來源:
# - dataset_100k: 100,000 張合成數據（四地車牌）
# - dataset_hk_10k: 10,000 張香港車牌合成數據
# - dataset_ccpd_yolo: CCPD 真實數據集
""".format(output_dir))
    
    print(f"   配置文件: {data_yaml}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='整合車牌數據集')
    parser.add_argument('--output_dir', type=str,
                       default='/home/nicklam-ai/repos/chinese-plate-recognition/dataset_merged',
                       help='輸出目錄')
    
    args = parser.parse_args()
    
    merge_datasets(args.output_dir)
