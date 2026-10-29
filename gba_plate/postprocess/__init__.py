"""後處理：四地車牌格式校驗與字符糾正"""

from gba_plate.postprocess.rules import (
    correct_confusables,
    detect_region_from_text,
    normalize,
    validate,
)

__all__ = ["normalize", "correct_confusables", "detect_region_from_text", "validate"]
