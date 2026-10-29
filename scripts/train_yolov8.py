#!/usr/bin/env python3
"""
訓練 YOLOv8 車牌檢測模型
使用整合後的 141,967 張圖片數據集
"""
from ultralytics import YOLO
import os
from pathlib import Path


def train_model():
    """訓練 YOLOv8 車牌檢測模型"""
    
    print("=" * 60)
    print("訓練 YOLOv8 車牌檢測模型")
    print("=" * 60)
    
    # 數據集配置
    data_yaml = "/home/nicklam-ai/repos/chinese-plate-recognition/dataset_merged/data.yaml"
    
    # 檢查數據集是否存在
    if not os.path.exists(data_yaml):
        print(f"❌ 數據集配置文件不存在: {data_yaml}")
        return
    
    print(f"\n數據集配置: {data_yaml}")
    
    # 加載預訓練模型（YOLOv8n - nano 版本，速度快）
    print("\n加載預訓練模型: YOLOv8n")
    model = YOLO('yolov8n.pt')
    
    # 訓練參數
    print("\n開始訓練...")
    print("訓練參數:")
    print("  - epochs: 100")
    print("  - batch: 16")
    print("  - imgsz: 640")
    print("  - device: cpu")
    print("  - patience: 20")
    
    # 開始訓練
    results = model.train(
        data=data_yaml,
        epochs=100,           # 訓練輪數
        batch=16,             # 批次大小
        imgsz=640,            # 圖片尺寸
        device='cpu',         # 使用 CPU（如果有 GPU 可改為 'cuda'）
        patience=20,          # 早停耐心值
        save=True,            # 保存最佳模型
        save_period=10,       # 每 10 輪保存一次
        project='runs/detect',
        name='plate_yolov8',
        exist_ok=True,
        pretrained=True,
        verbose=True,
        # 數據增強
        hsv_h=0.015,          # 色相增強
        hsv_s=0.7,            # 飽和度增強
        hsv_v=0.4,            # 亮度增強
        degrees=10,           # 旋轉角度
        translate=0.1,        # 平移
        scale=0.5,            # 縮放
        fliplr=0.5,           # 水平翻轉
        mosaic=1.0,           # Mosaic 增強
        mixup=0.1,            # MixUp 增強
    )
    
    print("\n" + "=" * 60)
    print("✅ 訓練完成！")
    print("=" * 60)
    
    # 最佳模型路徑
    best_model = Path('runs/detect/plate_yolov8/weights/best.pt')
    if best_model.exists():
        print(f"\n最佳模型: {best_model}")
        print(f"模型大小: {best_model.stat().st_size / 1024 / 1024:.2f} MB")
    else:
        print("\n⚠️ 未找到最佳模型文件")
    
    return results


if __name__ == '__main__':
    train_model()
