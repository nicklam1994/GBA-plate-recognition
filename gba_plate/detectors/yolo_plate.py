"""
YOLO 車牌檢測器（封裝 we0091234 的 YOLOv5 權重）。

支援兩種類別語義（class_mode）：
    "single_double"  class_id: 0=單層 1=雙層       ← 現有權重 plate_detect.pt
    "region"         class_id: 0=mainland 1=gangao 2=hk 3=mo  ← 未來重訓的權重

這樣同一個類可以吃舊權重（立即可用）和新權重（重訓後替換），無需改調用方。
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

from gba_plate.base import BaseDetector, Detection, Region


class YOLOPlateDetector(BaseDetector):
    """基於 YOLOv5 的車牌檢測器，輸出邊界框 + 4 角點。"""

    def __init__(
        self,
        weights: str | Path = "weights/plate_detect.pt",
        device: str | torch.device | None = None,
        img_size: int = 640,
        conf_thres: float = 0.3,
        iou_thres: float = 0.5,
        class_mode: str = "single_double",
    ) -> None:
        """
        Args:
            weights: 檢測權重 .pt
            device: torch 設備
            img_size: 推理輸入邊長（權重以 640 訓練）
            conf_thres: 置信度閾值
            iou_thres: NMS IoU 閾值
            class_mode: "single_double" 或 "region"
        """
        self.weights = Path(weights)
        if not self.weights.exists():
            raise FileNotFoundError(f"檢測權重不存在: {self.weights}")

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.img_size = img_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

        if class_mode not in ("single_double", "region"):
            raise ValueError(f"class_mode 必須是 single_double 或 region，收到 {class_mode}")
        self.class_mode = class_mode

        self.model = self._load_model()
        self._stride = int(self.model.stride.max())

    # ------------------------------------------------------------------ 內部

    def _load_model(self):
        from models.experimental import attempt_load

        model = attempt_load(str(self.weights), map_location=self.device)
        model.eval()
        return model

    def _preprocess(self, image: np.ndarray) -> tuple[torch.Tensor, float, tuple[float, float]]:
        """
        letterbox 到 32 倍數尺寸。

        Returns:
            (tensor, ratio, (pad_w, pad_h))
        """
        from utils.datasets import letterbox

        h0, w0 = image.shape[:2]
        size = max(self.img_size, self._stride * 2)

        img, ratio, pad = letterbox(image, new_shape=size, auto=False, scaleup=False)
        # letterbox 回傳 ratio 為 (r, r) tuple
        ratio = float(ratio[0]) if isinstance(ratio, (tuple, list)) else float(ratio)

        img = img[:, :, ::-1].transpose(2, 0, 1)          # BGR→RGB, HWC→CHW
        img = np.ascontiguousarray(img)                    # 避免負 stride
        tensor = torch.from_numpy(img).to(self.device).float() / 255.0
        return tensor.unsqueeze(0), ratio, (float(pad[0]), float(pad[1]))

    @staticmethod
    def _scale_back(
        coords: torch.Tensor, ratio: float, pad: tuple[float, float], orig_hw: tuple[int, int]
    ) -> torch.Tensor:
        """把 letterbox 座標映射回原圖"""
        pad_w, pad_h = pad
        h0, w0 = orig_hw
        coords = coords.clone()
        # 偶數索引是 x，奇數是 y
        coords[:, 0::2] = (coords[:, 0::2] - pad_w) / ratio
        coords[:, 1::2] = (coords[:, 1::2] - pad_h) / ratio
        coords[:, 0::2].clamp_(0, w0)
        coords[:, 1::2].clamp_(0, h0)
        return coords

    def _interpret(
        self, class_id: int, corners: np.ndarray, aspect_ratio: float
    ) -> tuple[bool, Region]:
        """把 class_id 解釋成 (is_double, region)"""
        if self.class_mode == "region":
            region = Region.from_class_id(class_id)
            # 4 類別模型：雙行靠寬高比判斷（方形 → 雙行）
            return aspect_ratio < 1.6, region
        # 舊權重：0=單層 1=雙層
        return bool(class_id), Region.UNKNOWN

    # ------------------------------------------------------------------ 對外

    def predict(self, image: np.ndarray) -> list[Detection]:
        if image is None or image.size == 0:
            return []

        from utils.general import non_max_suppression_face

        tensor, ratio, pad = self._preprocess(image)
        h0, w0 = image.shape[:2]

        with torch.no_grad():
            pred = self.model(tensor)[0]

        dets = non_max_suppression_face(pred, self.conf_thres, self.iou_thres)[0]
        if dets is None or len(dets) == 0:
            return []

        results: list[Detection] = []

        for row in dets:
            xyxy = row[:4].unsqueeze(0)          # (1,4)
            conf = float(row[4].item())
            landmarks = row[5:13].unsqueeze(0)   # (1,8)
            class_id = int(row[13].item())

            xyxy = self._scale_back(xyxy, ratio, pad, (h0, w0))[0].cpu().numpy()
            landmarks = self._scale_back(landmarks, ratio, pad, (h0, w0))[0].cpu().numpy()

            corners = landmarks.reshape(4, 2)
            box = tuple(float(v) for v in xyxy)

            width = max(box[2] - box[0], 1e-6)
            height = max(box[3] - box[1], 1e-6)
            is_double, region = self._interpret(class_id, corners, width / height)

            results.append(
                Detection(
                    bbox=box,                      # type: ignore[arg-type]
                    corners=corners,
                    confidence=conf,
                    is_double=is_double,
                    region=region,
                )
            )

        return results
