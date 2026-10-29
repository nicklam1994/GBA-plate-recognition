"""
ALPR 組合類 —— 唯一對外入口。

設計（參考 fast-alpr）：
    這個類只做一件事：把 detector → preprocess → ocr → postprocess 串起來。
    所有模型細節都在 BaseDetector / BaseOCR 的子類裡。

    alpr = ALPR()
    results = alpr.process(image)
    for r in results:
        print(r.region.value, r.text, f"{r.confidence:.2f}", "合法" if r.is_valid else r.validation_note)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from gba_plate.base import (
    BaseDetector,
    BaseOCR,
    Detection,
    PlateResult,
    Region,
)
from gba_plate.postprocess.rules import (
    correct_confusables,
    detect_region_from_text,
    normalize,
    validate,
)
from gba_plate.preprocess.double_line import looks_double_line, split_merge_double
from gba_plate.preprocess.perspective import crop_by_bbox, warp_plate


class ALPR:
    """四地車牌識別管線。"""

    def __init__(
        self,
        detector: BaseDetector | None = None,
        ocr: BaseOCR | None = None,
        *,
        detect_model: str | Path = "weights/plate_detect.pt",
        rec_model: str | Path = "weights/plate_rec_color.pth",
        device: str | None = None,
        class_mode: str = "single_double",
        conf_thres: float = 0.3,
        apply_rules: bool = True,
        min_confidence: float = 0.0,
    ) -> None:
        """
        Args:
            detector: 自訂檢測器（None 則用默認 YOLO 權重）
            ocr: 自訂識別器（None 則用默認 CRNN 權重）
            detect_model: 默認檢測權重路徑
            rec_model: 默認識別權重路徑
            device: torch 設備
            class_mode: "single_double"（舊權重）或 "region"（4 類別新權重）
            conf_thres: 檢測置信度閾值
            apply_rules: 是否套用格式校驗與字符糾正
            min_confidence: 識別置信度低於此值時丟棄結果
        """
        if detector is None:
            from gba_plate.detectors.yolo_plate import YOLOPlateDetector

            detector = YOLOPlateDetector(
                weights=detect_model,
                device=device,
                conf_thres=conf_thres,
                class_mode=class_mode,
            )
        if ocr is None:
            from gba_plate.ocr.crnn import CRNNPlateOCR

            ocr = CRNNPlateOCR(weights=rec_model, device=device)

        self.detector = detector
        self.ocr = ocr
        self.apply_rules = apply_rules
        self.min_confidence = min_confidence

    # ------------------------------------------------------------------ 內部

    def _prepare_crop(self, image: np.ndarray, det: Detection) -> np.ndarray:
        """透視校正 → 雙行拼接"""
        if det.corners is not None and np.asarray(det.corners).shape == (4, 2):
            crop = warp_plate(image, det.corners)
        else:
            crop = crop_by_bbox(image, det.bbox)

        is_double = det.is_double or looks_double_line(crop)
        if is_double:
            crop = split_merge_double(crop)
        return crop

    def _postprocess(self, raw_text: str, region: Region) -> tuple[str, bool, bool, str]:
        """
        Returns:
            (final_text, is_valid, correction_applied, note)
        """
        text = normalize(raw_text)
        corrected, changed = correct_confusables(text, region)
        ok, note = validate(corrected, region)
        return corrected, ok, changed, note

    # ------------------------------------------------------------------ 對外

    def process(self, image: np.ndarray) -> list[PlateResult]:
        """對單張影像做完整識別"""
        if image is None or image.size == 0:
            return []

        detections = self.detector.predict(image)
        results: list[PlateResult] = []

        for det in detections:
            try:
                crop = self._prepare_crop(image, det)
                ocr_result = self.ocr.predict(crop)
            except Exception:
                # 單個車牌失敗不影響其他車牌
                ocr_result = None

            if ocr_result is None:
                results.append(
                    PlateResult(
                        detection=det,
                        ocr=None,
                        region=det.region,
                        is_valid=False,
                        validation_note="識別失敗",
                    )
                )
                continue

            if ocr_result.confidence < self.min_confidence:
                continue

            region = det.region
            if region == Region.UNKNOWN:
                color_val = ocr_result.color.value if ocr_result.color else None
                region = detect_region_from_text(ocr_result.text, color_val)

            if self.apply_rules:
                text, is_valid, corrected, note = self._postprocess(ocr_result.text, region)
            else:
                text, is_valid, corrected, note = ocr_result.text, True, False, "rules disabled"

            results.append(
                PlateResult(
                    detection=det,
                    ocr=ocr_result,
                    region=region,
                    raw_text=ocr_result.text,
                    text=text,
                    is_valid=is_valid,
                    validation_note=note,
                    correction_applied=corrected,
                )
            )

        # 按置信度降序（方便調用方取 top-1）
        results.sort(key=lambda r: r.confidence, reverse=True)
        return results

    def process_file(self, path: str | Path) -> list[PlateResult]:
        """讀檔並識別（支援中文路徑）"""
        import cv2

        data = np.fromfile(str(path), dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"無法讀取影像: {path}")
        return self.process(image)

    def __call__(self, image: np.ndarray) -> list[PlateResult]:
        return self.process(image)
