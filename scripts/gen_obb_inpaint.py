#!/usr/bin/env python3
"""
原位替牌：把真實圖裡原有的車牌 inpaint 掉，在同一位置貼上新車牌。

與 gen_obb_dataset.py 的差別：
    舊版把合成牌隨機貼在背景圖上，位置不講道理（貼在路面、浮在保險桿上）。
    模型會學到「車牌可以出現在任何地方」的錯誤先驗。

    這版改為「原位替換」：detect 找到原本的車牌 → 遮罩後 inpaint 成車身
    → 在同一位置貼新牌。車牌位置天然合理，與車身的遮擋關係也保留。

為什麼不用 ComfyUI 生成車牌號碼：
    Diffusion 對精確文字的渲染不可靠，車牌需要逐字準確，生成必出亂碼。
    所以車牌仍由 render_plate 產生（號碼、字體、底色完全可控），
    ComfyUI 的角色只能是「清掉舊牌」，這裡先用 cv2.inpaint 代替，
    效果不如 ComfyUI 自然但零依賴。

輸出：YOLO-OBB 格式，與 gen_obb_dataset.py 一致。

用法：
    ./venv/bin/python scripts/gen_obb_inpaint.py --n 16000 --out dataset_obb2
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.gen_obb_dataset import (  # noqa: E402
    CLASS_NAMES, to_obb_label, make_plate,
)


def detect_plates(img: np.ndarray, model, conf: float = 0.30):
    """用四類 detect 模型找圖中所有車牌，回傳 [(x1,y1,x2,y2,cls)]。"""
    r = model.predict(img, imgsz=640, conf=conf, verbose=False)[0]
    out = []
    if r.boxes is not None and len(r.boxes):
        for (x1, y1, x2, y2), cls in zip(r.boxes.xyxy.cpu().numpy(),
                                        r.boxes.cls.cpu().numpy().astype(int)):
            out.append((float(x1), float(y1), float(x2), float(y2), int(cls)))
    return out


def inpaint_plate(img: np.ndarray, box, pad_ratio: float = 0.12) -> np.ndarray:
    """
    把指定區域 inpaint 成周圍顏色。

    遮罩向外擴張一點，確保車牌的黑邊框與螺絲也被蓋掉，
    否則貼上新牌後邊緣會露出一圈舊痕跡。
    """
    x1, y1, x2, y2 = [int(v) for v in box]
    w, h = x2 - x1, y2 - y1
    pad_x, pad_y = int(w * pad_ratio), int(h * pad_ratio * 1.6)
    H, W = img.shape[:2]
    mx1, my1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
    mx2, my2 = min(W, x2 + pad_x), min(H, y2 + pad_y)

    mask = np.zeros((H, W), dtype=np.uint8)
    mask[my1:my2, mx1:mx2] = 255
    out = cv2.inpaint(img, mask, 5, cv2.INPAINT_TELEA)
    return out


def paste_plate_at(img: np.ndarray, plate: np.ndarray, box, rng: random.Random):
    """
    在原車牌位置貼新牌。

    新牌寬度參考原框（略作浮動），傾斜角以原框為中心小幅隨機 —— 
    原位替換的意義就在於位置與尺度都來自真實場景。
    """
    H, W = img.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = max(8.0, x2 - x1), max(6.0, y2 - y1)

    # 新牌寬度在原框寬度的 0.85~1.15 倍之間浮動
    target_w = int(np.clip(bw * rng.uniform(0.85, 1.15), 30, 400))
    ph, pw = plate.shape[:2]
    scale = target_w / pw
    plate_r = cv2.resize(plate, (target_w, max(2, int(ph * scale))),
                         interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
    ph2, pw2 = plate_r.shape[:2]

    angle = rng.gauss(0, 6)
    angle = max(-20.0, min(20.0, angle))

    diag = int(np.ceil(np.hypot(pw2, ph2))) + 6
    canvas = np.zeros((diag, diag, 3), dtype=np.uint8)
    ox, oy = (diag - pw2) // 2, (diag - ph2) // 2
    canvas[oy:oy + ph2, ox:ox + pw2] = plate_r
    M = cv2.getRotationMatrix2D((diag / 2, diag / 2), angle, 1.0)
    rot = cv2.warpAffine(canvas, M, (diag, diag), flags=cv2.INTER_LINEAR)

    # 中心對齊原框中心
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    x0, y0 = int(cx - diag / 2), int(cy - diag / 2)
    sx0, sy0 = max(0, x0), max(0, y0)
    sx1, sy1 = min(W, x0 + diag), min(H, y0 + diag)
    if sx1 <= sx0 or sy1 <= sy0:
        return img, None

    sub = rot[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0]
    mask = (sub.max(axis=2) > 12).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    roi = img[sy0:sy1, sx0:sx1]
    roi[mask > 0] = sub[mask > 0]

    rect = np.array([[ox, oy], [ox + pw2, oy], [ox + pw2, oy + ph2], [ox, oy + ph2]],
                    dtype=np.float32)
    ones = np.ones((4, 1), dtype=np.float32)
    pts = np.hstack([rect, ones]) @ M.T
    pts += np.array([x0, y0], dtype=np.float32)
    return img, pts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=16000)
    ap.add_argument("--out", default="dataset_obb2")
    ap.add_argument("--src", default="test_data/real/hk", help="真實圖來源")
    ap.add_argument("--val-ratio", type=float, default=0.12)
    ap.add_argument("--seed", type=int, default=2027)
    ap.add_argument("--conf", type=float, default=0.30)
    ap.add_argument("--max-plates", type=int, default=2, help="每張圖最多替換幾塊牌")
    args = ap.parse_args()

    out = ROOT / args.out
    srcs = sorted(ROOT.glob(f"{args.src}/*.jpg")) + sorted(ROOT.glob(f"{args.src}/*.png"))
    if not srcs:
        raise SystemExit(f"找不到真實圖：{args.src}")

    from ultralytics import YOLO
    model = YOLO(str(ROOT / "weights/gba_4class_yolov8s_best.pt"))
    print(f"真實圖 {len(srcs)} 張，開始偵測車牌…")

    # 預先偵測全部真實圖的車牌（同一張圖會被重複抽樣多次，快取省時間）
    cache_path = out / "src_detections.json"
    detections: dict[str, list] = {}
    if cache_path.exists():
        detections = json.loads(cache_path.read_text(encoding="utf-8"))
        print(f"沿用偵測快取（{len(detections)} 張）")
    if len(detections) < len(srcs) * 0.9:
        detections = {}
        for i, p in enumerate(srcs, 1):
            img = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue
            detections[p.name] = [list(d) for d in detect_plates(img, model, args.conf)]
            if i % 100 == 0:
                print(f"  偵測 {i}/{len(srcs)}", flush=True)
        out.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(detections, ensure_ascii=False), encoding="utf-8")
        print(f"偵測完成，快取 → {cache_path.name}")

    usable = [k for k, v in detections.items() if v]
    print(f"含車牌的圖 {len(usable)} / {len(detections)}")

    rng = random.Random(args.seed)
    n_val = int(args.n * args.val_ratio)
    splits = {"train": args.n - n_val, "val": n_val}
    for sp in splits:
        (out / "images" / sp).mkdir(parents=True, exist_ok=True)
        (out / "labels" / sp).mkdir(parents=True, exist_ok=True)

    made = {"train": 0, "val": 0}
    angles: list[float] = []
    idx = 0

    for split, count in splits.items():
        for k in range(count):
            name = rng.choice(usable)
            p = ROOT / args.src / name
            img0 = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)
            if img0 is None:
                continue
            H, W = img0.shape[:2]
            boxes = [tuple(b) for b in detections[name]]

            # 先 inpaint 掉所有要被替換的舊牌，避免貼新牌時露出舊牌殘影
            img = img0.copy()
            replace_n = min(len(boxes), rng.randint(1, args.max_plates))
            chosen = rng.sample(boxes, replace_n)
            for b in chosen:
                img = inpaint_plate(img, b[:4])

            labels: list[str] = []
            # 未被替換的原牌：保留為 angle=0 的真實標註
            for b in boxes:
                if b in chosen:
                    continue
                x1, y1, x2, y2, cls = b
                if x2 - x1 < 12 or y2 - y1 < 6:
                    continue
                row = (f"{cls} {(x1+x2)/2/W:.6f} {(y1+y2)/2/H:.6f} "
                       f"{(x2-x1)/W:.6f} {(y2-y1)/H:.6f} 0.000000")
                labels.append(row)

            # 在被替換的位置貼新牌
            for b in chosen:
                plate, cls = make_plate(rng)
                img, pts = paste_plate_at(img, plate, b[:4], rng)
                if pts is None:
                    continue
                lab = to_obb_label(pts, cls, W, H)
                if lab:
                    labels.append(lab)
                    angles.append(float(lab.split()[-1]))

            if not labels:
                continue
            fname = f"inp_{split}_{idx:06d}"
            idx += 1
            cv2.imwrite(str(out / "images" / split / f"{fname}.jpg"), img,
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
            (out / "labels" / split / f"{fname}.txt").write_text(
                "\n".join(labels) + "\n", encoding="utf-8")
            made[split] += 1
            if (k + 1) % 2000 == 0:
                print(f"  {split} {k+1}/{count}", flush=True)

    yaml = (f"# GBA 四地區車牌 OBB（原位替牌版）\n"
            f"path: {out}\ntrain: images/train\nval: images/val\n\n"
            f"nc: 4\nnames:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(CLASS_NAMES)))
    (out / "data.yaml").write_text(yaml, encoding="utf-8")

    stats = {
        "train": made["train"], "val": made["val"],
        "angle_deg_abs_mean": round(float(np.degrees(np.mean(np.abs(angles)))), 2) if angles else 0,
    }
    (out / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1),
                                    encoding="utf-8")
    print(f"\n完成：train {made['train']} / val {made['val']}")
    print(f"替換牌傾斜 |mean| {stats['angle_deg_abs_mean']}°")
    print(f"輸出 {out}")


if __name__ == "__main__":
    main()
