"""
雙行車牌拼接。

跨項目驗證過的關鍵比例（zhahoi-ncnn 與 pcb9382 完全一致）：
    上層：取前 5/12 (41.67%) 高度
    下層：從 1/3 (33.33%) 高度開始
兩者刻意重疊 8.33%，避免字符跨越分割線時被切斷。

拼接後交給單行 CRNN 識別（不是分兩次識別再拼字符串）。
"""

from __future__ import annotations

import cv2
import numpy as np

#: 上層結束位置（佔總高度比例）
#: 上層結束位置（佔總高度比例）。兩行各半，略留重疊容錯。
UPPER_END_RATIO = 0.52
#: 下層起始位置（佔總高度比例）
LOWER_START_RATIO = 0.48
#: 拼接填充色（與原項目 letterbox 填充值一致）
PAD_VALUE = 114


def split_merge_double(plate_img: np.ndarray) -> np.ndarray:
    """
    把雙行車牌（上下兩行）拼成單行寬圖。

    Args:
        plate_img: 已透視校正的車牌小圖 BGR

    Returns:
        寬度約為原本高度的 2 倍多，高度不變的單行圖
    """
    h, w = plate_img.shape[:2]
    if h < 8 or w < 8:
        return plate_img

    upper_end = int(UPPER_END_RATIO * h)
    lower_start = int(LOWER_START_RATIO * h)

    img_upper = plate_img[0:upper_end, :]
    img_lower = plate_img[lower_start:h, :]

    if img_upper.size == 0 or img_lower.size == 0:
        return plate_img

    # 把上層縮放到與下層同高
    img_upper = cv2.resize(img_upper, (img_lower.shape[1], img_lower.shape[0]))

    # 水平拼接（原項目用 hstack；這裡用帶填充的版本，便於對齊視覺基準線）
    return np.hstack((img_upper, img_lower))


def looks_double_line(plate_img: np.ndarray) -> bool:
    """
    啟發式判斷是否為雙行車牌。

    僅靠寬高比不夠：香港大方牌 330×170 = 1.94，舊閾值 1.6 判不出，
    導致雙行牌被當單行送進 OCR（字符被壓成兩半）。
    改為寬高比粗篩 + 水平投影找行間空白帶。

    典型寬高比：香港長牌 4.73 / 內地 3.14 → 單行；
                香港方牌 1.94 / 澳門方形 1.9 → 雙行。
    """
    h, w = plate_img.shape[:2]
    if h < 16 or w < 16:
        return False
    ratio = w / float(h)

    if ratio >= 2.4:
        return False          # 明顯是長條單行牌
    if ratio < 1.3:
        return True           # 接近方形或豎長，必然是雙行

    # 中間地帶（1.3 ~ 2.4）：用水平投影找兩行之間的空白帶
    try:
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        _, bw = cv2.threshold(gray, 0, 255,
                              cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        row_ink = bw.sum(axis=1) / 255.0
        band = row_ink[int(h * 0.35):int(h * 0.65)]
        if band.size == 0:
            return False
        mean_ink = row_ink.mean()
        if mean_ink <= 0:
            return False
        # 中間有明顯低谷 → 兩行之間的分隔帶
        return float(band.min()) < 0.25 * mean_ink
    except Exception:
        return False
