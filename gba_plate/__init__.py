"""
GBA Plate Recognition — 粵港澳大灣區四地車牌識別

分層架構（抽象設計參考 fast-alpr）：
    BaseDetector / BaseOCR 為擴展點，ALPR 只負責組合。

    from gba_plate import ALPR
    alpr = ALPR()
    for r in alpr.process(cv2.imread("plate.jpg")):
        print(r.region, r.text, r.confidence)
"""

from gba_plate.base import (
    BaseDetector,
    BaseOCR,
    Detection,
    OcrResult,
    PlateResult,
    Region,
    PlateColor,
)
from gba_plate.alpr import ALPR

__version__ = "0.1.0"

__all__ = [
    "ALPR",
    "BaseDetector",
    "BaseOCR",
    "Detection",
    "OcrResult",
    "PlateResult",
    "Region",
    "PlateColor",
]
