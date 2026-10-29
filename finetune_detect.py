#!/usr/bin/env python3
"""
微調 we0091234 檢測模型（YOLOv5-based）

用法：
    # 使用合成數據集微調
    python finetune_detect.py --data ./dataset_100k/data.yaml --epochs 50 --batch 16
    
    # 從預訓練權重開始
    python finetune_detect.py --weights weights/plate_detect.pt --data ./dataset_100k/data.yaml
"""
import argparse
import sys
from pathlib import Path

# 添加項目根目錄到 path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import yaml
from models.experimental import attempt_load
from utils.general import check_img_size
from utils.torch_utils import select_device


def prepare_dataset(data_yaml: str):
    """準備數據集配置"""
    with open(data_yaml, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"數據集配置:")
    print(f"  路徑: {config['path']}")
    print(f"  類別數: {config['nc']}")
    print(f"  類別名: {config['names']}")
    
    # 檢查數據集是否存在
    dataset_path = Path(config['path'])
    images_dir = dataset_path / config['train']
    labels_dir = dataset_path / 'labels_yolo'
    
    if not images_dir.exists():
        raise FileNotFoundError(f"圖片目錄不存在: {images_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"標籤目錄不存在: {labels_dir}")
    
    num_images = len(list(images_dir.glob('*.jpg')))
    num_labels = len(list(labels_dir.glob('*.txt')))
    
    print(f"  圖片數量: {num_images}")
    print(f"  標籤數量: {num_labels}")
    
    if num_images == 0:
        raise ValueError("數據集為空，請先生成合成數據")
    
    return config


def finetune_model(
    weights: str,
    data_yaml: str,
    epochs: int = 50,
    batch_size: int = 16,
    img_size: int = 640,
    device: str = '0',
    project: str = 'runs/finetune',
    name: str = 'plate_detect_4region',
):
    """微調檢測模型"""
    
    # 準備數據集
    config = prepare_dataset(data_yaml)
    
    # 選擇設備
    device = select_device(device)
    print(f"\n使用設備: {device}")
    
    # 加載預訓練模型
    print(f"\n加載預訓練權重: {weights}")
    model = attempt_load(weights, map_location=device)
    
    # 凍結 backbone（可選）
    # for name, param in model.named_parameters():
    #     if 'detect' not in name:
    #         param.requires_grad = False
    
    # 修改檢測頭（適配 4 類別）
    # we0091234 的檢測模型原本只檢測車牌（1 類），需要改為 4 類
    print(f"\n修改檢測頭: 1 類 -> {config['nc']} 類")
    
    # 獲取檢測層
    detect_layer = model.model[-1]
    
    # 檢查是否需要修改
    if hasattr(detect_layer, 'nc'):
        old_nc = detect_layer.nc
        if old_nc != config['nc']:
            print(f"  修改 nc: {old_nc} -> {config['nc']}")
            detect_layer.nc = config['nc']
            
            # 重新初始化分類頭
            # 這裡需要根據實際模型結構調整
            print("  注意：需要重新初始化分類頭權重")
    
    # 使用 Ultralytics YOLO 訓練（如果可用）
    try:
        from ultralytics import YOLO
        print("\n使用 Ultralytics YOLO 訓練...")
        
        # 加載 YOLOv5 模型
        model = YOLO('yolov5n.pt')  # 或 'yolov5s.pt'
        
        # 訓練
        results = model.train(
            data=data_yaml,
            epochs=epochs,
            batch=batch_size,
            imgsz=img_size,
            device=device,
            project=project,
            name=name,
            pretrained=weights,
        )
        
        print(f"\n訓練完成！")
        print(f"最佳模型: {project}/{name}/weights/best.pt")
        
    except ImportError:
        print("\nUltralytics YOLO 未安裝，使用原生訓練腳本...")
        print("請運行: python train.py --data {data_yaml} --epochs {epochs} --batch {batch_size}")
        print("注意：需要手動修改 train.py 以支持 4 類別檢測")


def main():
    parser = argparse.ArgumentParser(description="微調車牌檢測模型")
    parser.add_argument('--weights', type=str, default='weights/plate_detect.pt',
                        help='預訓練權重路徑')
    parser.add_argument('--data', type=str, default='./dataset_100k/data.yaml',
                        help='數據集配置文件')
    parser.add_argument('--epochs', type=int, default=50,
                        help='訓練輪數')
    parser.add_argument('--batch', type=int, default=16,
                        help='批次大小')
    parser.add_argument('--img-size', type=int, default=640,
                        help='圖片尺寸')
    parser.add_argument('--device', type=str, default='0',
                        help='設備 (0, 0,1,2,3 or cpu)')
    parser.add_argument('--project', type=str, default='runs/finetune',
                        help='輸出項目目錄')
    parser.add_argument('--name', type=str, default='plate_detect_4region',
                        help='實驗名稱')
    
    args = parser.parse_args()
    
    finetune_model(
        weights=args.weights,
        data_yaml=args.data,
        epochs=args.epochs,
        batch_size=args.batch,
        img_size=args.img_size,
        device=args.device,
        project=args.project,
        name=args.name,
    )


if __name__ == '__main__':
    main()
