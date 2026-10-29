"""
透視變換：把車牌四邊形校正成正矩形。

與原 detect_plate.py 的 four_point_transform 邏輯一致，
但增加了角點排序（原版假設檢測器輸出的角點已是 tl,tr,br,bl 順序）。
"""

from __future__ import annotations

import cv2
import numpy as np


def order_corners(pts: np.ndarray) -> np.ndarray:
    """
    把 4 個角點排成 tl, tr, br, bl 順序。

    用「和最小 = 左上、差最小 = 右上」的幾何技巧，對任意凸四邊形都成立。

    Args:
        pts: (4, 2) 任意順序的四個角點

    Returns:
        (4, 2) 排序為 tl, tr, br, bl
    """
    pts = np.asarray(pts, dtype=np.float32)
    if pts.shape != (4, 2):
        raise ValueError(f"角點形狀必須是 (4,2)，收到 {pts.shape}")

    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]   # tl: x+y 最小
    rect[2] = pts[np.argmax(s)]   # br: x+y 最大
    d = np.diff(pts, axis=1).ravel()
    rect[1] = pts[np.argmin(d)]   # tr: y-x 最小（差最小）
    rect[3] = pts[np.argmax(d)]   # bl: y-x 最大
    return rect


def warp_plate(
    image: np.ndarray,
    corners: np.ndarray,
    target_size: tuple[int, int] | None = None,
) -> np.ndarray:
    """
    以 4 角點做透視變換，裁出正向的車牌小圖。

    Args:
        image: 原圖 BGR
        corners: (4, 2) 車牌角點（順序不限）
        target_size: (width, height)，None 表示由角點實際尺寸決定

    Returns:
        校正後的車牌小圖 BGR
    """
    rect = order_corners(corners)

    if target_size is None:
        (tl, tr, br, bl) = rect
        width = max(
            np.linalg.norm(br - bl),
            np.linalg.norm(tr - tl),
        )
        height = max(
            np.linalg.norm(tr - br),
            np.linalg.norm(tl - bl),
        )
        w, h = max(int(width), 1), max(int(height), 1)
    else:
        w, h = target_size

    dst = np.array(
        [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, matrix, (w, h))


def crop_by_bbox(image: np.ndarray, bbox: tuple[float, float, float, float]) -> np.ndarray:
    """按軸對齊框裁切（無角點時的退化方案）"""
    h, w = image.shape[:2]
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(int(x1), w - 1))
    y1 = max(0, min(int(y1), h - 1))
    x2 = max(x1 + 1, min(int(x2), w))
    y2 = max(y1 + 1, min(int(y2), h))
    return image[y1:y2, x1:x2]
