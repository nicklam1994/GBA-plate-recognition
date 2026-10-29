"""YOLOv8 四地區檢測器測試（不依賴真實權重）"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from gba_plate.base import Region
from gba_plate.detectors.yolov8_plate import DOUBLE_PLATE_AR_MAX


class TestRegionMapping:
    def test_from_class_id(self):
        assert Region.from_class_id(0) == Region.MAINLAND
        assert Region.from_class_id(1) == Region.GANGAO
        assert Region.from_class_id(2) == Region.HK
        assert Region.from_class_id(3) == Region.MO

    def test_unknown_class(self):
        assert Region.from_class_id(99) == Region.UNKNOWN
        assert Region.from_class_id(-1) == Region.UNKNOWN


class TestDetectorLogic:
    """用 mock 的 ultralytics 模型測試 bbox → Detection 轉換"""

    @pytest.fixture
    def fake_model(self):
        """構造一個回傳固定 3 個框的假模型"""
        import torch

        boxes = MagicMock()
        boxes.__len__ = lambda self: 3
        boxes.xyxy = torch.tensor([
            [10.0, 20.0, 110.0, 50.0],    # 寬100 高30 → 長寬比 3.33 → 單行
            [200.0, 100.0, 260.0, 148.0],  # 寬60 高48 → 1.25 → 單行（>1.2）
            [300.0, 200.0, 340.0, 320.0],  # 寬40 高120 → 0.33 → 雙行
        ])
        boxes.conf = torch.tensor([0.95, 0.88, 0.72])
        boxes.cls = torch.tensor([0.0, 2.0, 3.0])

        result = MagicMock()
        result.boxes = boxes

        model = MagicMock()
        model.predict.return_value = [result]
        model.names = {0: "mainland", 1: "gangao", 2: "hk", 3: "mo"}
        return model

    @pytest.fixture
    def detector(self, fake_model, tmp_path):
        # 建立假的權重文件讓 __init__ 的 exists() 通過
        w = tmp_path / "fake.pt"
        w.write_bytes(b"")
        with patch("ultralytics.YOLO", return_value=fake_model):
            from gba_plate.detectors.yolov8_plate import YOLOv8PlateDetector

            return YOLOv8PlateDetector(weights=w, device="cpu")

    def test_returns_all_boxes(self, detector):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        dets = detector.predict(img)
        assert len(dets) == 3

    def test_bbox_preserved(self, detector):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        dets = detector.predict(img)
        assert dets[0].bbox == (10.0, 20.0, 110.0, 50.0)

    def test_corners_are_rectangle(self, detector):
        """bbox 生成的 corners 必須是矩形四角，順序 tl,tr,br,bl"""
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        dets = detector.predict(img)
        c = dets[0].corners
        assert c.shape == (4, 2)
        assert list(c[0]) == [10.0, 20.0]      # tl
        assert list(c[1]) == [110.0, 20.0]     # tr
        assert list(c[2]) == [110.0, 50.0]     # br
        assert list(c[3]) == [10.0, 50.0]      # bl

    def test_region_from_class(self, detector):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        dets = detector.predict(img)
        assert dets[0].region == Region.MAINLAND   # cls=0
        assert dets[1].region == Region.HK         # cls=2
        assert dets[2].region == Region.MO         # cls=3

    def test_double_line_detection(self, detector):
        """長寬比 < 1.2 判為雙行"""
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        dets = detector.predict(img)
        assert dets[0].is_double is False   # ar=3.33
        assert dets[1].is_double is False   # ar=1.25
        assert dets[2].is_double is True    # ar=0.33

    def test_confidence_preserved(self, detector):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        dets = detector.predict(img)
        assert abs(dets[0].confidence - 0.95) < 1e-6

    def test_empty_image(self, detector):
        assert detector.predict(np.zeros((0, 0, 3), dtype=np.uint8)) == []

    def test_empty_result(self, tmp_path):
        fake = MagicMock()
        fake.predict.return_value = [MagicMock(boxes=None)]
        fake.names = {}
        w = tmp_path / "f.pt"
        w.write_bytes(b"")
        with patch("ultralytics.YOLO", return_value=fake):
            from gba_plate.detectors.yolov8_plate import YOLOv8PlateDetector

            d = YOLOv8PlateDetector(weights=w, device="cpu")
        assert d.predict(np.zeros((100, 100, 3), dtype=np.uint8)) == []


class TestConstants:
    def test_double_ar_threshold(self):
        assert 1.0 < DOUBLE_PLATE_AR_MAX <= 1.5


class TestAlprSelection:
    def test_region_mode_picks_yolov8(self, tmp_path):
        """class_mode='region' 時 ALPR 應選 YOLOv8PlateDetector"""
        w = tmp_path / "yolov8.pt"
        w.write_bytes(b"")
        with patch("ultralytics.YOLO", return_value=MagicMock()), \
             patch("gba_plate.detectors.yolov8_plate.YOLOv8PlateDetector") as mock_cls, \
             patch("gba_plate.ocr.crnn.CRNNPlateOCR"):
            from gba_plate.alpr import ALPR

            ALPR(detect_model=w, class_mode="region")
            assert mock_cls.called

    def test_yolov8_filename_picks_yolov8(self, tmp_path):
        """權重檔名含 yolov8 時自動選 YOLOv8 管線"""
        w = tmp_path / "my_yolov8_best.pt"
        w.write_bytes(b"")
        with patch("ultralytics.YOLO", return_value=MagicMock()), \
             patch("gba_plate.detectors.yolov8_plate.YOLOv8PlateDetector") as mock_cls, \
             patch("gba_plate.ocr.crnn.CRNNPlateOCR"):
            from gba_plate.alpr import ALPR

            ALPR(detect_model=w)
            assert mock_cls.called
