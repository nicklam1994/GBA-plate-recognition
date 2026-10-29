#!/usr/bin/env python3
"""
OBB vs 標準 detect 推論速度對照。

問題：改用 YOLOv8-OBB 做四角檢測，會不會拖慢識別速度？
方法：同一台機器、同一張圖、同樣 imgsz，分別計時三種模型，
      並拆出 preprocess / inference / postprocess 各階段耗時。
      速度瓶頸若在 inference，換 OBB 幾乎沒差；若在 postprocess，
      OBB 的 rotated NMS 才是要注意的地方。
"""
import time
import numpy as np
import cv2
import torch
from ultralytics import YOLO

WARMUP = 20
RUNS = 100
IMG_SIZE = 640

# 拿一張真實行車記錄儀圖，貼近實際場景
IMG = "test_data/real/hk/IMG-20261015-WA0213.jpg"
img = cv2.imdecode(np.fromfile(IMG, dtype=np.uint8), cv2.IMREAD_COLOR)
print(f"測試圖 {IMG}  shape={img.shape[:2]}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"imgsz={IMG_SIZE}  warmup={WARMUP}  runs={RUNS}\n")


def bench(name: str, weights: str) -> dict:
    try:
        model = YOLO(weights)
    except Exception as e:
        print(f"[skip] {name}: {e}")
        return {}

    # 預熱
    for _ in range(WARMUP):
        model.predict(img, imgsz=IMG_SIZE, device="0", verbose=False)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    pre = inf = post = 0.0
    for _ in range(RUNS):
        r = model.predict(img, imgsz=IMG_SIZE, device="0", verbose=False)[0]
        s = r.speed  # {'preprocess': ms, 'inference': ms, 'postprocess': ms}
        pre += s["preprocess"]
        inf += s["inference"]
        post += s["postprocess"]
    torch.cuda.synchronize()
    wall = (time.perf_counter() - t0) / RUNS * 1000

    n_det = len(r.boxes) if getattr(r, "boxes", None) is not None else 0
    out = {
        "name": name,
        "wall_ms": wall,
        "pre": pre / RUNS,
        "inf": inf / RUNS,
        "post": post / RUNS,
        "fps": 1000.0 / wall,
        "dets": n_det,
    }
    print(f"{name:<26} 總計 {out['wall_ms']:>7.2f} ms  "
          f"{out['fps']:>6.1f} FPS   "
          f"[前處理 {out['pre']:>5.2f} | 推論 {out['inf']:>5.2f} | 後處理 {out['post']:>5.2f}]  "
          f"檢出 {n_det}")
    return out


results = []
results.append(bench("YOLOv8s detect (現有4類)", "weights/gba_4class_yolov8s_best.pt"))
results.append(bench("YOLOv8s detect (COCO通用)", "yolov8s.pt"))
results.append(bench("YOLOv8s-OBB (DOTA)", "yolov8s-obb.pt"))
results.append(bench("YOLOv8n-OBB (DOTA)", "yolov8n-obb.pt"))

base = next((r for r in results if "現有4類" in r.get("name", "")), None)
print("\n" + "=" * 78)
if base:
    for r in results:
        if not r or r is base:
            continue
        d = r["wall_ms"] - base["wall_ms"]
        print(f"{r['name']:<26} 相對現有模型 {d:+.2f} ms  "
              f"({100 * d / base['wall_ms']:+.1f}%)")
print("\n註：speed 為 ultralytics 內建分段計時；wall 含 Python 呼叫開銷。")
