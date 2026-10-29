"""前處理測試（透視變換、雙行拼接）"""

import cv2
import numpy as np
import pytest

from gba_plate.preprocess.perspective import crop_by_bbox, order_corners, warp_plate
from gba_plate.preprocess.double_line import (
    LOWER_START_RATIO,
    UPPER_END_RATIO,
    looks_double_line,
    split_merge_double,
)


class TestOrderCorners:
    def test_already_ordered(self):
        pts = np.array([[0, 0], [100, 0], [100, 50], [0, 50]], dtype=np.float32)
        out = order_corners(pts)
        np.testing.assert_allclose(out[0], [0, 0])      # tl
        np.testing.assert_allclose(out[1], [100, 0])    # tr
        np.testing.assert_allclose(out[2], [100, 50])   # br
        np.testing.assert_allclose(out[3], [0, 50])     # bl

    def test_shuffled_input(self):
        """任意順序輸入都要排出 tl,tr,br,bl"""
        pts = np.array([[100, 50], [0, 0], [0, 50], [100, 0]], dtype=np.float32)
        out = order_corners(pts)
        np.testing.assert_allclose(out[0], [0, 0])
        np.testing.assert_allclose(out[1], [100, 0])
        np.testing.assert_allclose(out[2], [100, 50])
        np.testing.assert_allclose(out[3], [0, 50])

    def test_wrong_shape_raises(self):
        with pytest.raises(ValueError):
            order_corners(np.array([[0, 0], [1, 1]], dtype=np.float32))


class TestWarpPlate:
    def test_frontal_rect_keeps_shape(self):
        img = np.zeros((200, 400, 3), dtype=np.uint8)
        pts = np.array([[10, 10], [390, 10], [390, 130], [10, 130]], dtype=np.float32)
        out = warp_plate(img, pts)
        h, w = out.shape[:2]
        assert abs(w - 381) <= 2
        assert abs(h - 121) <= 2

    def test_target_size_respected(self):
        img = np.zeros((200, 400, 3), dtype=np.uint8)
        pts = np.array([[10, 10], [390, 10], [390, 130], [10, 130]], dtype=np.float32)
        out = warp_plate(img, pts, target_size=(168, 48))
        assert out.shape[:2] == (48, 168)

    def test_trapezoid_corrected(self):
        """梯形輸入（透視變形）應被校正為矩形"""
        img = np.zeros((200, 400, 3), dtype=np.uint8)
        # 上邊比下邊窄 → 典型的遠近透視
        pts = np.array([[60, 10], [340, 10], [390, 130], [10, 130]], dtype=np.float32)
        out = warp_plate(img, pts, target_size=(300, 100))
        assert out.shape[:2] == (100, 300)


class TestCropByBbox:
    def test_basic(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        out = crop_by_bbox(img, (10, 20, 60, 80))
        assert out.shape[:2] == (60, 50)

    def test_clamped_to_image(self):
        """超出邊界的框要被截斷，不能報錯"""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        out = crop_by_bbox(img, (-50, -50, 500, 500))
        assert out.shape[:2] == (100, 100)

    def test_degenerate_bbox(self):
        """零尺寸框不能返回空陣列"""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        out = crop_by_bbox(img, (50, 50, 50, 50))
        assert out.size > 0


class TestDoubleLine:
    def test_ratios_match_cross_project_consensus(self):
        """5/12 + 1/3 是 zhahoi-ncnn 與 pcb9382 一致的比例"""
        assert abs(UPPER_END_RATIO - 5 / 12) < 1e-9
        assert abs(LOWER_START_RATIO - 1 / 3) < 1e-9
        # 兩者刻意重疊，避免字符被切斷
        assert UPPER_END_RATIO > LOWER_START_RATIO

    def test_merge_doubles_width(self):
        """拼接後寬度約為原本 2 倍（上層 + 下層）"""
        img = np.zeros((260, 220, 3), dtype=np.uint8)
        out = split_merge_double(img)
        assert out.shape[1] == pytest.approx(440, abs=2)
        assert out.shape[0] == pytest.approx(173, abs=2)

    def test_merge_preserves_height_of_lower(self):
        img = np.zeros((120, 100, 3), dtype=np.uint8)
        out = split_merge_double(img)
        expected_h = 120 - int(1 / 3 * 120)   # 下層高度
        assert out.shape[0] == expected_h

    def test_tiny_image_passthrough(self):
        """過小的圖直接返回，不崩"""
        img = np.zeros((4, 4, 3), dtype=np.uint8)
        out = split_merge_double(img)
        assert out.shape == img.shape

    @pytest.mark.parametrize(
        "shape,expected",
        [
            ((260, 220), True),    # 方形雙行（港/澳雙行牌）
            ((140, 440), False),   # 長條單行
            ((100, 380), False),
        ],
    )
    def test_looks_double_line(self, shape, expected):
        img = np.zeros((*shape, 3), dtype=np.uint8)
        assert looks_double_line(img) is expected
