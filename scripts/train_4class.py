#!/usr/bin/env python3
"""
訓練 GBA 四地區車牌檢測模型（YOLOv8）

模型選型理由（YOLOv8s vs YOLOv10s）：
  1. 車牌是密集小目標，YOLOv10 的 NMS-free 一對一標籤分配在密集場景召回不如
     YOLOv8 的一對多分配（停車場一行 5-10 個車牌時尤其明顯）
  2. YOLOv8 的 ONNX / TensorRT 導出路徑經過最廣泛驗證，部署風險最低
  3. 4090 上 v8s 推理 ~2-3ms/張，速度完全夠用；v10 的低延遲優勢只在邊緣設備顯現
  4. v8 有完整的蒸餾 / 剪枝工具鏈，後續要壓到邊緣時有退路

數據增強策略（關鍵）：
  車牌底色是四地區類別判別的核心特徵（藍=內地、白/黃=香港、黑=澳門），
  一般車牌論文的增強配方會破壞顏色資訊，因此：
    - hsv_s / hsv_v 降低到 0.3（默認 0.7 / 0.4）
    - mixup 關閉（會混合兩張圖的顏色，直接污染類別標籤）
    - degrees 降到 5（車牌在實景中傾斜幅度有限）
    - mosaic 降到 0.8 並在最後 10 epoch 關閉

用法：
    python scripts/train_4class.py --model yolov8s --epochs 100 --batch 32
"""
from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
DATA_YAML = ROOT / "configs" / "gba_plate_4class.yaml"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolov8s", help="yolov8n/s/m")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0", help="0=GPU, cpu")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--amp", action="store_true",
                    help="啟用 AMP（torch 2.14+CUDA13 的 check_amp 會失敗，默認關閉）")
    args = ap.parse_args()

    if not DATA_YAML.exists():
        raise SystemExit(f"data.yaml 不存在: {DATA_YAML}")

    weights = f"{args.model}.pt"
    print("=" * 68)
    print(f"訓練 {args.model} — GBA 四地區車牌檢測")
    print("=" * 68)
    print(f"數據集: {DATA_YAML}")
    print(f"epochs={args.epochs} batch={args.batch} imgsz={args.imgsz} device={args.device}")
    print()

    model = YOLO(weights)

    results = model.train(
        data=str(DATA_YAML),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        workers=args.workers,
        patience=25,
        save=True,
        save_period=10,
        project=str(ROOT / "runs" / "detect"),
        name=f"gba_4class_{args.model}",
        exist_ok=True,
        pretrained=True,
        resume=args.resume,
        amp=args.amp,
        optimizer="auto",
        cos_lr=True,
        # ---- 顏色相關增強壓低（車牌底色 = 類別特徵）----
        # 實測 train_batch 發現 hsv_h=0.01 已足夠讓藍牌偏綠，
        # 與內地新能源綠牌混淆，因此再壓到 0.005（±1.8°）
        hsv_h=0.005,
        hsv_s=0.20,
        hsv_v=0.30,
        # ---- 幾何增強 ----
        degrees=5.0,
        translate=0.10,
        scale=0.45,
        shear=2.0,
        perspective=0.0005,
        fliplr=0.5,
        flipud=0.0,
        # ---- 組合增強 ----
        mosaic=0.8,
        close_mosaic=10,
        mixup=0.0,
        copy_paste=0.0,
        erasing=0.15,
        # ---- 其他 ----
        val=True,
        plots=True,
        verbose=True,
    )

    best = ROOT / "runs" / "detect" / f"gba_4class_{args.model}" / "weights" / "best.pt"
    print()
    print("=" * 68)
    if best.exists():
        print(f"✅ 最佳模型: {best}")
        print(f"   大小: {best.stat().st_size / 1024 / 1024:.2f} MB")
    else:
        print("⚠️ 未找到 best.pt")
    print("=" * 68)
    return results


if __name__ == "__main__":
    main()
