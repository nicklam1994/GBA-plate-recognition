#!/usr/bin/env python3
"""
評估 GBA 四地區檢測模型：整體 mAP + 每類 AP/召回率 + 分地區統計。

用法：
    python scripts/evaluate_4class.py --weights runs/detect/gba_4class_yolov8s/weights/best.pt
    python scripts/evaluate_4class.py --weights best.pt --split test --imgsz 960
"""
from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent

REGION_NAMES = {0: "mainland 內地", 1: "gangao 粵Z", 2: "hk 香港", 3: "mo 澳門"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(ROOT / "runs/detect/gba_4class_yolov8s/weights/best.pt"))
    ap.add_argument("--data", default=str(ROOT / "configs/gba_plate_4class.yaml"))
    ap.add_argument("--split", default="test", choices=["val", "test", "train"])
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--conf", type=float, default=0.001, help="評估用低閾值（mAP 計算需要）")
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    w = Path(args.weights)
    if not w.exists():
        raise SystemExit(f"權重不存在: {w}")

    print("=" * 72)
    print(f"評估 {w.name}")
    print(f"split={args.split} imgsz={args.imgsz} conf={args.conf}")
    print("=" * 72)

    model = YOLO(str(w))
    metrics = model.val(
        data=args.data,
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        conf=args.conf,
        iou=0.7,
        device=args.device,
        plots=True,
        verbose=True,
    )

    print()
    print("=" * 72)
    print("整體指標")
    print("=" * 72)
    print(f"  mAP@50:     {metrics.box.map50:.4f}")
    print(f"  mAP@50-95:  {metrics.box.map:.4f}")
    print(f"  Precision:  {metrics.box.mp:.4f}")
    print(f"  Recall:     {metrics.box.mr:.4f}")

    print()
    print("=" * 72)
    print("分地區指標")
    print("=" * 72)
    print(f"{'類別':<18} {'AP@50':>8} {'AP@50-95':>10} {'P':>8} {'R':>8}")
    print("-" * 72)

    try:
        ap50 = metrics.box.ap50
        ap_all = metrics.box.ap
        p = metrics.box.p
        r = metrics.box.r
        for i in range(len(ap50)):
            name = REGION_NAMES.get(i, f"class{i}")
            print(f"{name:<18} {ap50[i]:>8.4f} {ap_all[i]:>10.4f} {p[i]:>8.4f} {r[i]:>8.4f}")
    except Exception as e:
        print(f"  逐類指標讀取失敗: {e}")
        print(f"  可用屬性: {[a for a in dir(metrics.box) if not a.startswith('_')]}")

    print()
    print("=" * 72)
    # 找混淆矩陣與 PR 曲線
    save_dir = Path(metrics.save_dir) if hasattr(metrics, "save_dir") else None
    if save_dir and save_dir.exists():
        print(f"圖表輸出: {save_dir}")
        for f in sorted(save_dir.glob("*.png")):
            print(f"  {f.name}")


if __name__ == "__main__":
    main()
