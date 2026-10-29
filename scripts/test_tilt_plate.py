#!/usr/bin/env python3
"""
斜拍車牌四角矯正對照實驗。

針對一輛掛雙牌（MY1153 + ZM2089）的斜拍照片，比較三種裁切方式對 OCR 的影響：
    A. bbox 直切        現況（軸對齊矩形，斜拍時會把車牌切歪）
    B. 傳統 CV 四角矯正  在框內找車牌真實四邊形，perspective warp 拉正
    C. 四角矯正 + 放大   在 B 基礎上放大到固定寬度供 OCR

輸出視覺化圖，讓四角是否真的抓對一眼可見。
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SRC = "test_data/real/macau_merged/train/images/MY1153_ZM2089_jpg.rf.8644a4184464cd1f5c0297399bffb74c.jpg"
OUT = Path("/home/nicklam-ai/.hermes/profiles/architect/cache/scratch")


def order_corners(pts: np.ndarray) -> np.ndarray:
    """把四點排成 tl, tr, br, bl。"""
    pts = pts.reshape(-1, 2).astype(np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    return np.array([
        pts[np.argmin(s)], pts[np.argmin(d)],
        pts[np.argmax(s)], pts[np.argmax(d)],
    ], dtype=np.float32)


def find_quad(gray: np.ndarray):
    """
    在裁圖內找車牌四邊形。

    做法：Otsu 二值化 → 找最大輪廓 → 多邊形逼近。
    取不到四邊形時回退到 minAreaRect（旋轉矩形）。
    """
    h, w = gray.shape[:2]
    # 上採樣讓小圖的輪廓更完整
    scale = max(1.0, 320.0 / max(w, 1))
    g = cv2.resize(gray, (int(w * scale), int(h * scale)),
                   interpolation=cv2.INTER_CUBIC) if scale > 1 else gray

    blur = cv2.GaussianBlur(g, (5, 5), 0)
    _, bw = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # 車牌區域通常是「一塊亮的矩形」，先做閉運算連通
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k, iterations=2)

    cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)

    if len(approx) == 4:
        quad = approx.reshape(4, 2).astype(np.float32)
    else:
        rect = cv2.minAreaRect(c)
        quad = cv2.boxPoints(rect).astype(np.float32)

    quad /= scale
    return order_corners(quad)


def warp_quad(img: np.ndarray, quad: np.ndarray):
    """依四角做透視變換，輸出拉正後的車牌。"""
    tl, tr, br, bl = quad
    w = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    h = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    if w < 8 or h < 4:
        return None
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]],
                   dtype=np.float32)
    M = cv2.getPerspectiveTransform(quad, dst)
    return cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_CUBIC)


def angle_of(quad: np.ndarray) -> float:
    """上邊與水平的夾角（度），用來判斷斜拍程度。"""
    tl, tr = quad[0], quad[1]
    return float(np.degrees(np.arctan2(tr[1] - tl[1], tr[0] - tl[0])))


def main():
    p = ROOT / SRC
    img = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        print(f"讀不到圖：{p}")
        return
    H, W = img.shape[:2]
    print(f"原圖 {W}x{H}")

    from gba_plate.detectors.yolov8_plate import YOLOv8PlateDetector

    det = YOLOv8PlateDetector(
        weights=ROOT / "weights/gba_4class_yolov8s_best.pt",
        conf_thres=0.25, img_size=640, device="0",
    )
    dets = det.predict(img)
    print(f"檢測到 {len(dets)} 個車牌\n")

    vis = img.copy()
    crops = {}
    for i, d in enumerate(dets):
        x1, y1, x2, y2 = [int(v) for v in d.bbox]
        print(f"[{i}] region={d.region.value} conf={d.confidence:.3f} "
              f"bbox=({x1},{y1},{x2},{y2}) 寬{x2-x1} 高{y2-y1}")

        pad = 6
        cx1, cy1 = max(0, x1 - pad), max(0, y1 - pad)
        cx2, cy2 = min(W, x2 + pad), min(H, y2 + pad)
        crop_a = img[cy1:cy2, cx1:cx2]
        crops[f"{i}_A_bbox"] = crop_a

        gray = cv2.cvtColor(crop_a, cv2.COLOR_BGR2GRAY)
        quad_local = find_quad(gray)
        if quad_local is not None:
            quad_abs = quad_local + np.array([cx1, cy1], dtype=np.float32)
            ang = angle_of(quad_local)
            print(f"     四角矯正：斜角 {ang:+.1f}°")
            crop_b = warp_quad(img, quad_abs)
            if crop_b is not None:
                crops[f"{i}_B_quad"] = crop_b
                hh, ww = crop_b.shape[:2]
                s = 320.0 / max(ww, 1)
                crops[f"{i}_C_quad_up"] = cv2.resize(
                    crop_b, (320, max(1, int(hh * s))), interpolation=cv2.INTER_CUBIC)
            # 視覺化四角
            for k in range(4):
                cv2.line(vis, tuple(quad_abs[k].astype(int)),
                         tuple(quad_abs[(k + 1) % 4].astype(int)), (0, 220, 255), 2)
        # bbox 視覺化
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(vis, f"{i}:{d.region.value}", (x1, max(14, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    OUT.mkdir(parents=True, exist_ok=True)
    for name, c in crops.items():
        if c is None or c.size == 0:
            continue
        cv2.imwrite(str(OUT / f"tilt_{name}.jpg"), c, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"  已存 tilt_{name}.jpg  {c.shape[1]}x{c.shape[0]}")

    cv2.imwrite(str(OUT / "tilt_full.jpg"), vis, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"\n視覺化：{OUT/'tilt_full.jpg'}")


if __name__ == "__main__":
    main()
