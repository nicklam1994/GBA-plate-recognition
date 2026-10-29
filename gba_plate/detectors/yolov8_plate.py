"""
YOLOv8 四地區車牌檢測器。

與 yolo_plate.py（YOLOv5-face，輸出 4 角點）的差異：
    - YOLOv8 只輸出軸對齊 bbox，沒有角點 → 用 bbox 生成矩形四角
      （warp_plate 收到矩形四角時等價於直接裁切，不會出錯，只是沒有透視校正效果）
    - class_id 直接就是地區：0=mainland 1=gangao 2=hk 3=mo
    - 因此 region 是一等公民，不需要事後推斷（舊權重只能推 UNKNOWN）

class_mode 固定為 "region"。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from gba_plate.base import BaseDetector, Detection, Region

#: 寬高比低於此值視為雙行車牌
DOUBLE_PLATE_AR_MAX = 1.2


class YOLOv8PlateDetector(BaseDetector):
    """基於 YOLOv8 的四地區車牌檢測器。"""

    class_mode = "region"

    def __init__(
        self,
        weights: str | Path = "runs/detect/gba_4class_yolov8s/weights/best.pt",
        device: str | torch.device | None = None,
        img_size: int = 640,
        conf_thres: float = 0.25,
        iou_thres: float = 0.5,
        max_det: int = 50,
        half: bool = False,
    ) -> None:
        self.weights = Path(weights)
        if not self.weights.exists():
            raise FileNotFoundError(f"檢測權重不存在: {self.weights}")

        if device is None:
            self.device = "0" if torch.cuda.is_available() else "cpu"
        else:
            self.device = str(device)

        self.img_size = img_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.max_det = max_det
        self.half = half and torch.cuda.is_available()

        from ultralytics import YOLO

        self.model = YOLO(str(self.weights))

    @property
    def names(self) -> dict[int, str]:
        return self.model.names if isinstance(self.model.names, dict) else dict(enumerate(self.model.names))

    def predict(self, image: np.ndarray) -> list[Detection]:
        """
        Args:
            image: BGR uint8 (H, W, 3)

        Returns:
            Detection 列表，region 已按 class_id 填好
        """
        if image is None or image.size == 0:
            return []

        res = self.model.predict(
            source=image,
            imgsz=self.img_size,
            conf=self.conf_thres,
            iou=self.iou_thres,
            max_det=self.max_det,
            device=self.device,
            verbose=False,
        )[0]

        detections: list[Detection] = []
        boxes = res.boxes
        if boxes is None or len(boxes) == 0:
            return detections

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        clss = boxes.cls.cpu().numpy().astype(int)

        for (x1, y1, x2, y2), conf, cls_id in zip(xyxy, confs, clss):
            bbox = (float(x1), float(y1), float(x2), float(y2))
            w = max(x2 - x1, 1e-6)
            h = max(y2 - y1, 1e-6)

            # 矩形四角，順序 tl, tr, br, bl（與 warp_plate 的期望一致）
            corners = np.array(
                [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                dtype=np.float32,
            )

            detections.append(
                Detection(
                    bbox=bbox,  # type: ignore[arg-type]
                    corners=corners,
                    confidence=float(conf),
                    is_double=bool(w / h < DOUBLE_PLATE_AR_MAX),
                    region=Region.from_class_id(int(cls_id)),
                )
            )

        return detections


def is_available(weights: str | Path) -> bool:
    """檢查權重與 ultralytics 是否可用（供 ALPR 自動選擇）"""
    try:
        import ultralytics  # noqa: F401
    except ImportError:
        return False
    return Path(weights).exists()
