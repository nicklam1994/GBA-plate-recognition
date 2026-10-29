#!/usr/bin/env python3
"""
批量生成四地車牌合成數據

輸出格式：
- images/: 車牌圖片
- labels.txt: 標註文件（每行: 文件名 車牌號 地區 規格）
- labelme/: YOLO 格式標註（可選）

用法：
    python generate.py --num 1000 --output ./dataset --regions mainland gangao hk mo
"""
import os
import sys
import json
import argparse
import random
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import numpy as np

# 添加父目錄到 path
sys.path.insert(0, str(Path(__file__).parent.parent))

from synthetic_data.plate_specs import generate_plate
from synthetic_data.plate_renderer import render_plate
from synthetic_data.augmentations import augment_plate, simulate_capture_condition, add_occlusion


def generate_single_plate(
    idx: int,
    output_dir: str,
    region: str = "random",
    augment: bool = True,
    img_size: tuple = None,
) -> dict:
    """
    生成單張車牌

    Returns:
        dict with keys: filename, plate_text, region, spec_name, bbox
    """
    # 生成車牌文字和規格
    text, spec, plate_region = generate_plate(region)

    # 渲染基礎車牌
    img = render_plate(text, spec)

    # 數據增強
    if augment:
        # 基礎增強（輕度為主，避免過度模糊）
        level = random.choices(
            ['light', 'medium', 'heavy'],
            weights=[50, 35, 15],
            k=1
        )[0]
        img = augment_plate(img, level=level)

        # 模擬拍攝條件（50% 概率）
        if random.random() < 0.5:
            img = simulate_capture_condition(img)

        # 隨機遮擋（15% 概率）
        if random.random() < 0.15:
            img = add_occlusion(img)

    # 縮放到目標尺寸（如果指定）
    if img_size:
        img = cv2.resize(img, img_size, interpolation=cv2.INTER_LINEAR)

    # 保存圖片
    filename = f"{plate_region}_{idx:06d}_{spec.name}.jpg"
    filepath = os.path.join(output_dir, "images", filename)
    cv2.imwrite(filepath, img)

    # 計算邊框（整個車牌區域）
    h, w = img.shape[:2]
    bbox = [0, 0, w, h]  # [x1, y1, x2, y2]

    return {
        "filename": filename,
        "plate_text": text.replace('\n', ''),  # 雙層車牌去掉換行
        "region": plate_region,
        "spec_name": spec.name,
        "bbox": bbox,
        "width": w,
        "height": h,
    }


def generate_batch(
    num_samples: int,
    output_dir: str,
    region_weights: dict = None,
    augment: bool = True,
    img_size: tuple = None,
    num_workers: int = 4,
):
    """
    批量生成車牌數據

    Args:
        num_samples: 生成數量
        output_dir: 輸出目錄
        region_weights: 各地區權重 {"mainland": 50, "gangao": 15, "hk": 20, "mo": 15}
        augment: 是否增強
        img_size: 目標尺寸 (w, h)
        num_workers: 並行進程數
    """
    # 創建輸出目錄
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)

    # 默認權重
    if region_weights is None:
        region_weights = {"mainland": 50, "gangao": 15, "hk": 20, "mo": 15}

    # 生成地區列表
    regions = []
    for region, weight in region_weights.items():
        regions.extend([region] * weight)

    # 標籤列表
    labels = []

    print(f"開始生成 {num_samples} 張車牌圖片...")
    print(f"地區分佈: {region_weights}")
    print(f"輸出目錄: {output_dir}")
    print(f"增強: {'開' if augment else '關'}")
    print()

    start_time = time.time()

    # 並行生成
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for i in range(num_samples):
            region = random.choice(regions)
            future = executor.submit(
                generate_single_plate,
                i, output_dir, region, augment, img_size
            )
            futures.append(future)

        # 收集結果
        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()
            labels.append(result)

            if i % 100 == 0 or i == num_samples:
                elapsed = time.time() - start_time
                speed = i / elapsed
                eta = (num_samples - i) / speed if speed > 0 else 0
                print(f"\r進度: {i}/{num_samples} ({i/num_samples*100:.1f}%) "
                      f"速度: {speed:.1f} img/s ETA: {eta:.0f}s", end="")

    print()

    # 保存標籤文件
    label_path = os.path.join(output_dir, "labels.txt")
    with open(label_path, 'w', encoding='utf-8') as f:
        for label in labels:
            line = f"{label['filename']} {label['plate_text']} {label['region']} {label['spec_name']}\n"
            f.write(line)

    # 保存 YOLO 格式標籤（用於檢測模型訓練）
    yolo_dir = os.path.join(output_dir, "labels_yolo")
    os.makedirs(yolo_dir, exist_ok=True)

    # 地區 -> 類別 ID 映射
    region_to_id = {"mainland": 0, "gangao": 1, "hk": 2, "mo": 3}

    for label in labels:
        yolo_file = os.path.join(yolo_dir, label['filename'].replace('.jpg', '.txt'))
        class_id = region_to_id.get(label['region'], 0)

        # YOLO 格式: class_id x_center y_center width height (歸一化)
        x1, y1, x2, y2 = label['bbox']
        x_center = (x1 + x2) / 2 / label['width']
        y_center = (y1 + y2) / 2 / label['height']
        width = (x2 - x1) / label['width']
        height = (y2 - y1) / label['height']

        with open(yolo_file, 'w') as f:
            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

    # 保存數據集配置
    config = {
        "num_samples": num_samples,
        "region_weights": region_weights,
        "augment": augment,
        "img_size": img_size,
        "region_to_id": region_to_id,
        "id_to_region": {v: k for k, v in region_to_id.items()},
        "total_generated": len(labels),
    }
    config_path = os.path.join(output_dir, "dataset_config.json")
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    # 統計信息
    region_counts = {}
    spec_counts = {}
    for label in labels:
        region_counts[label['region']] = region_counts.get(label['region'], 0) + 1
        spec_counts[label['spec_name']] = spec_counts.get(label['spec_name'], 0) + 1

    print(f"\n生成完成！")
    print(f"總計圖片: {len(labels)}")
    print(f"標籤文件: {label_path}")
    print(f"YOLO 標籤: {yolo_dir}/")
    print(f"數據集配置: {config_path}")
    print(f"\n地區分佈:")
    for region, count in sorted(region_counts.items()):
        print(f"  {region:10s}: {count:5d} ({count/len(labels)*100:.1f}%)")
    print(f"\n規格分佈:")
    for spec, count in sorted(spec_counts.items()):
        print(f"  {spec:25s}: {count:5d} ({count/len(labels)*100:.1f}%)")

    return labels


def main():
    parser = argparse.ArgumentParser(description="四地車牌合成數據生成器")
    parser.add_argument("--num", type=int, default=100, help="生成數量")
    parser.add_argument("--output", type=str, default="./dataset", help="輸出目錄")
    parser.add_argument("--mainland", type=int, default=50, help="內地車牌權重")
    parser.add_argument("--gangao", type=int, default=15, help="粵Z車牌權重")
    parser.add_argument("--hk", type=int, default=20, help="香港車牌權重")
    parser.add_argument("--mo", type=int, default=15, help="澳門車牌權重")
    parser.add_argument("--no-augment", action="store_true", help="關閉數據增強")
    parser.add_argument("--img-width", type=int, default=None, help="目標寬度")
    parser.add_argument("--img-height", type=int, default=None, help="目標高度")
    parser.add_argument("--workers", type=int, default=4, help="並行進程數")

    args = parser.parse_args()

    region_weights = {
        "mainland": args.mainland,
        "gangao": args.gangao,
        "hk": args.hk,
        "mo": args.mo,
    }

    img_size = None
    if args.img_width and args.img_height:
        img_size = (args.img_width, args.img_height)

    generate_batch(
        num_samples=args.num,
        output_dir=args.output,
        region_weights=region_weights,
        augment=not args.no_augment,
        img_size=img_size,
        num_workers=args.workers,
    )


if __name__ == "__main__":
    main()
