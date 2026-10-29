#!/usr/bin/env python3
"""
訓練 GBA 四地區車牌 OBB 模型（旋轉框 + 角度回歸）。

與 train_4class.py 的差別：
    detect 版輸出軸對齊 bbox，warp_plate() 拿到的四角是矩形角，透視變換退化成裁切；
    OBB 版輸出帶角度的旋轉矩形，四角是真實的，warp 才有意義。

數據：dataset_obb（合成，四角由變換矩陣精確計算）
骨幹：yolov8s-obb.pt（DOTA 預訓練，已具備旋轉目標的先天）

增強策略沿用 detect 版的顏色保護思路（車牌底色是類別判別核心特徵）：
  hsv_s / hsv_v 壓低，mixup 關閉，mosaic 在最後 10 epoch 關閉。
另外 degrees 也壓低 —— 合成數據本身已含 ±32° 傾斜，
訓練期再大幅旋轉會讓角度分佈偏離真實場景。

用法：
    ./venv/bin/python scripts/train_obb.py --epochs 100 --batch 32
"""
from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
DATA_YAML = ROOT / "dataset_obb" / "data.yaml"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="yolov8s-obb.pt")
    ap.add_argument("--data", default=str(DATA_YAML))
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--name", default="gba_4class_obb")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--amp", action="store_true",
                    help="啟用 AMP（torch 2.14+CUDA13 的 check_amp 會失敗，默認關閉）")
    args = ap.parse_args()

    if not Path(args.data).exists():
        raise SystemExit(f"找不到數據集設定：{args.data}\n先跑 scripts/gen_obb_dataset.py")

    model = YOLO(args.weights)
    model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        workers=args.workers,
        project=str(ROOT / "runs" / "obb"),
        name=args.name,
        exist_ok=True,
        resume=args.resume,
        amp=args.amp,
        # --- 顏色保護（同 detect 版）---
        hsv_h=0.010,        # 色相微調，避免底色被改（藍/綠/黃/白/黑是類別核心）
        hsv_s=0.30,         # 默認 0.7，壓低
        hsv_v=0.30,         # 默認 0.4，壓低
        mixup=0.0,          # 混合兩張圖會污染底色與類別
        # --- 幾何 ---
        degrees=5.0,        # 合成數據已含 ±32°，訓練期只小幅再旋轉
        translate=0.10,
        scale=0.40,
        shear=0.0,
        perspective=0.0,    # OBB 標籤是旋轉矩形，透視變形會讓標籤失真
        fliplr=0.0,         # 車牌左右翻轉會變成不存在的號碼
        flipud=0.0,
        mosaic=0.80,
        close_mosaic=10,    # 最後 10 epoch 關閉 mosaic，讓模型適應真實構圖
        # --- 優化 ---
        optimizer="SGD",
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        cos_lr=True,
        patience=30,
        # --- 輸出 ---
        val=True,
        plots=True,
        save=True,
        verbose=True,
    )

    best = ROOT / "runs" / "obb" / args.name / "weights" / "best.pt"
    if best.exists():
        dst = ROOT / "weights" / "gba_4class_obb_best.pt"
        import shutil
        shutil.copy2(best, dst)
        print(f"\n最佳權重已複製到 {dst.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
