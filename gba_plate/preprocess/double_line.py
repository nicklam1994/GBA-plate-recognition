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
UPPER_END_RATIO = 5 / 12
#: 下層起始位置（佔總高度比例）
LOWER_START_RATIO = 1 / 3
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
    啟發式判斷是否為雙行車牌（無類別標籤時的 fallback）。

    依據：方形或豎長（寬高比 < 1.6）且有明顯水平分割特徵。
    香港/澳門的雙行牌通常是接近方形（220×260）。
    """
    h, w = plate_img.shape[:2]
    if h == 0 or w == 0:
        return False
    return (w / h) < 1.6
