"""
抽象層與資料結構。

設計原則（參考 fast-alpr 的 40 行 base.py）：
- 只有 2 個 ABC 擴展點：BaseDetector / BaseOCR
- 結果物件是 frozen dataclass，不可變
- region 欄位是一等公民（第一輪調研發現所有開源項目都沒有）
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class Region(str, Enum):
    """車牌所屬地區（四地 + 未知）"""

    MAINLAND = "mainland"  # 內地（藍/黃/綠/白）
    GANGAO = "gangao"      # 粵Z 兩地牌（黑底白字，粵Z…港/澳）
    HK = "hk"              # 香港本地牌（前白後黃）
    MO = "mo"              # 澳門本地牌（黑底白字）
    UNKNOWN = "unknown"    # 舊權重（只有單/雙層）時的佔位

    @classmethod
    def from_class_id(cls, class_id: int) -> "Region":
        """4 類別模型的 class_id → Region"""
        mapping = {0: cls.MAINLAND, 1: cls.GANGAO, 2: cls.HK, 3: cls.MO}
        return mapping.get(class_id, cls.UNKNOWN)


class PlateColor(str, Enum):
    """車牌顏色（識別模型的 5 類輸出）"""

    BLACK = "黑"
    BLUE = "藍"
    GREEN = "綠"
    WHITE = "白"
    YELLOW = "黃"


@dataclass(frozen=True)
class Detection:
    """
    檢測結果。

    Attributes:
        bbox: (x1, y1, x2, y2) 軸對齊邊界框，原圖座標
        corners: (4, 2) 四個角點，順序 tl, tr, br, bl
        confidence: 檢測置信度
        is_double: 是否雙行車牌
        region: 地區（4 類別模型可直接得出；舊權重為 UNKNOWN）
    """

    bbox: tuple[float, float, float, float]
    corners: np.ndarray
    confidence: float
    is_double: bool = False
    region: Region = Region.UNKNOWN

    @property
    def width(self) -> float:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]

    @property
    def aspect_ratio(self) -> float:
        """寬高比。單行車牌 ~3.1-3.5，方形雙行 ~0.85"""
        h = max(self.height, 1e-6)
        return self.width / h


@dataclass(frozen=True)
class OcrResult:
    """
    OCR 結果（對應 fast-alpr 的 OcrResult）。

    Attributes:
        text: 識別出的車牌文字（CTC 解碼後，未經規則校驗）
        confidence: 整體置信度（字符置信度均值）
        char_confidences: 每個字符的置信度
        color: 車牌顏色
        color_confidence: 顏色分類置信度
    """

    text: str
    confidence: float
    char_confidences: list[float] = field(default_factory=list)
    color: PlateColor | None = None
    color_confidence: float | None = None


@dataclass(frozen=True)
class PlateResult:
    """
    單一車牌的完整結果（檢測 + 識別 + 後處理）。

    Attributes:
        detection: 檢測結果
        ocr: 識別結果（識別失敗時為 None）
        region: 最終地區判定
        raw_text: 後處理前的原始 OCR 文字
        text: 後處理後的文字
        is_valid: 是否符合該地區車牌格式
        validation_note: 校驗說明（不通過時的原因）
        correction_applied: 是否應用了字符糾正
    """

    detection: Detection
    ocr: OcrResult | None
    region: Region
    raw_text: str = ""
    text: str = ""
    is_valid: bool = False
    validation_note: str = ""
    correction_applied: bool = False

    @property
    def confidence(self) -> float:
        return self.ocr.confidence if self.ocr else 0.0

    @property
    def color(self) -> PlateColor | None:
        return self.ocr.color if self.ocr else None

    def to_dict(self) -> dict:
        """轉為可序列化的 dict（給 API / Web UI 用）"""
        return {
            "text": self.text,
            "raw_text": self.raw_text,
            "region": self.region.value,
            "confidence": round(self.confidence, 4),
            "is_double": self.detection.is_double,
            "is_valid": self.is_valid,
            "validation_note": self.validation_note,
            "correction_applied": self.correction_applied,
            "color": self.color.value if self.color else None,
            "color_confidence": (
                round(self.ocr.color_confidence, 4)
                if self.ocr and self.ocr.color_confidence is not None
                else None
            ),
            "bbox": [round(v, 1) for v in self.detection.bbox],
            "corners": self.detection.corners.round(1).tolist(),
            "detect_confidence": round(self.detection.confidence, 4),
        }


class BaseDetector(ABC):
    """檢測器擴展點。實作只需回傳 Detection 列表。"""

    #: 模型的類別語義，決定 class_id 如何解釋
    class_mode: str = "single_double"  # "single_double" | "region"

    @abstractmethod
    def predict(self, image: np.ndarray) -> list[Detection]:
        """
        Args:
            image: BGR uint8 影像 (H, W, 3)

        Returns:
            Detection 列表（可能為空）
        """


class BaseOCR(ABC):
    """識別器擴展點。實作只需回傳 OcrResult。"""

    @abstractmethod
    def predict(self, plate_crop: np.ndarray) -> OcrResult | None:
        """
        Args:
            plate_crop: 已校正的車牌小圖，BGR uint8

        Returns:
            OcrResult，識別失敗時回傳 None
        """
