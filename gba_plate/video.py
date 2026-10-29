"""
影片處理：逐幀檢測 + 跨幀追蹤去重 + 時序投票。

解決的核心問題：
    影片同一車牌會連續出現在幾十到幾百幀。若逐幀輸出，結果列表會被同一輛車
    刷屏，而且單幀 OCR 錯誤無法糾正。

做法：
    1. IoU 貪心匹配維護軌跡（track），同一輛車只算一個結果
    2. 只在「新建軌跡」或「置信度創新高」時跑 OCR，省去 90% 的重複識別
    3. 每個軌跡收集多幀 OCR 結果，結束時做字符級多數投票
       —— 這是影片相對靜態圖的核心優勢，能把 85% 的單幀準確率拉到 95%+

依賴：OpenCV（讀寫）、ffmpeg（輸出 H.264，可選）
"""

from __future__ import annotations

import json
import subprocess
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from gba_plate.base import PlateResult, Region

#: 顏色（BGR），與 Web UI 的四地區配色一致
REGION_COLORS = {
    Region.MAINLAND: (80, 175, 76),    # 綠
    Region.GANGAO: (0, 152, 255),      # 橙
    Region.HK: (243, 150, 33),         # 藍
    Region.MO: (176, 39, 156),         # 紫
    Region.UNKNOWN: (136, 136, 136),   # 灰
}


def iou(a: tuple, b: tuple) -> float:
    """兩個 (x1,y1,x2,y2) 的 IoU"""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def vote_text(texts: list[tuple[str, float]]) -> tuple[str, float]:
    """
    字符級多數投票。

    Args:
        texts: [(text, confidence), ...]，同一軌跡的多幀 OCR 結果

    Returns:
        (vote_text, ratio) — ratio 是勝出字符的平均支持率
    """
    if not texts:
        return "", 0.0
    if len(texts) == 1:
        return texts[0][0], 1.0

    # 按長度分組（不同長度的 OCR 結果通常來自誤識別，取最常見長度）
    by_len: dict[int, list[tuple[str, float]]] = defaultdict(list)
    for t, c in texts:
        if t:
            by_len[len(t)].append((t, c))

    if not by_len:
        return "", 0.0

    # 選「加權票數最高」的長度
    def weight(group):
        return sum(c for _, c in group)

    best_len = max(by_len, key=lambda k: weight(by_len[k]))
    group = by_len[best_len]

    # 逐位置投票（不按置信度加權，避免單個高置信度錯誤主導）
    chars: list[str] = []
    support: list[float] = []
    for pos in range(best_len):
        counter = Counter()
        for t, _ in group:
            counter[t[pos]] += 1
        top_char, top_n = counter.most_common(1)[0]
        chars.append(top_char)
        support.append(top_n / len(group))

    return "".join(chars), sum(support) / len(support)


@dataclass
class PlateTrack:
    """一個車牌在影片中的完整軌跡"""

    track_id: int
    first_frame: int
    first_time: float
    last_frame: int = 0
    hits: int = 1
    age: int = 0  # 連續未匹配幀數
    bbox: tuple[float, float, float, float] = (0, 0, 0, 0)
    region: Region = Region.UNKNOWN
    #: 逐幀地區投票 —— 檢測模型對同一輛車可能給出不同地區（粵Z/澳門來回跳），
    #: 用多數投票而非最後一幀，避免最終結果取到少數派
    region_votes: Counter = field(default_factory=Counter)
    best_conf: float = 0.0
    best_frame: int = 0
    best_result: PlateResult | None = None
    ocr_samples: list[tuple[str, float]] = field(default_factory=list)

    def to_dict(self, fps: float) -> dict:
        """輸出到 JSON 時間軸"""
        text, vote_ratio = vote_text(self.ocr_samples)
        best = self.best_result

        # 投票結果與最佳單幀結果不一致時，以投票為準（影片優勢）
        final_text = text or (best.text if best else "")
        confidence = 0.0
        if self.ocr_samples:
            # 綜合置信度：票數支持率 × 平均單幀置信度
            mean_conf = sum(c for _, c in self.ocr_samples) / len(self.ocr_samples)
            confidence = round(vote_ratio * mean_conf, 4)

        return {
            "track_id": self.track_id,
            "t": round(self.first_time, 2),
            "first_frame": self.first_frame,
            "last_frame": self.last_frame,
            "frames": self.hits,
            "duration": round((self.last_frame - self.first_frame) / max(fps, 1e-6), 2),
            "text": final_text,
            "text_single_frame": best.text if best else "",
            "vote_corrected": bool(best and final_text and final_text != best.text),
            "ocr_samples": len(self.ocr_samples),
            "confidence": confidence,
            "detect_confidence": round(self.best_conf, 4),
            "region": self.region.value,
            "province": best.province if best else None,
            "city": best.city if best else None,
            "region_note": best.region_note if best else "",
            "energy": best.energy if best else None,
            "energy_note": best.energy_note if best else "",
            "color": best.color.value if (best and best.color) else None,
            "is_valid": best.is_valid if best else False,
            "validation_note": best.validation_note if best else "",
            "bbox": [round(v, 1) for v in self.bbox],
        }


class VideoProcessor:
    """
    影片逐幀識別 + 軌跡去重。

    用法：
        proc = VideoProcessor(alpr, frame_stride=2)
        result = proc.process("dashcam.mp4", "out.mp4")
        for p in result["plates"]:
            print(p["t"], p["text"], p["region"])
    """

    def __init__(
        self,
        alpr,
        *,
        iou_thres: float = 0.3,
        max_age: int = 30,
        min_hits: int = 2,
        frame_stride: int = 1,
        ocr_on_new_track_only: bool = True,
        conf_improve_thres: float = 0.08,
        max_det_per_frame: int = 20,
        ocr_sample_interval: int = 8,
    ) -> None:
        """
        Args:
            alpr: 已初始化的 ALPR 實例
            iou_thres: 匹配同一軌跡的 IoU 門檻
            max_age: 連續多少幀未匹配就結束軌跡
            min_hits: 至少命中幾幀才輸出（過濾單幀誤檢）
            frame_stride: 每隔幾幀處理一次（2 = 只用一半幀，速度翻倍）
            ocr_on_new_track_only: 只在新建軌跡或置信度顯著提升時跑 OCR
            conf_improve_thres: 置信度提升多少才重跑 OCR
            max_det_per_frame: 每幀最多處理幾個檢測框
            ocr_sample_interval: 每 N 次命中補一個 OCR 樣本供投票（0 = 關閉）
        """
        self.alpr = alpr
        self.iou_thres = iou_thres
        self.max_age = max_age
        self.min_hits = min_hits
        self.frame_stride = max(1, frame_stride)
        self.ocr_on_new_track_only = ocr_on_new_track_only
        self.conf_improve_thres = conf_improve_thres
        self.max_det_per_frame = max_det_per_frame
        self.ocr_sample_interval = ocr_sample_interval

        self.tracks: list[PlateTrack] = []
        self._next_id = 1
        self._finished: list[PlateTrack] = []

    # ------------------------------------------------------------------ 內部

    def _match(self, detections) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        """
        貪心 IoU 匹配。

        Returns:
            (matches, unmatched_dets, unmatched_tracks) — matches 是 (track_idx, det_idx)
        """
        if not self.tracks or not detections:
            return [], list(range(len(detections))), list(range(len(self.tracks)))

        pairs = []
        for ti, tr in enumerate(self.tracks):
            for di, det in enumerate(detections):
                score = iou(tr.bbox, det.bbox)
                if score >= self.iou_thres:
                    pairs.append((score, ti, di))

        pairs.sort(key=lambda x: -x[0])
        used_t, used_d = set(), set()
        matches = []
        for _, ti, di in pairs:
            if ti in used_t or di in used_d:
                continue
            matches.append((ti, di))
            used_t.add(ti)
            used_d.add(di)

        unmatched_d = [i for i in range(len(detections)) if i not in used_d]
        unmatched_t = [i for i in range(len(self.tracks)) if i not in used_t]
        return matches, unmatched_d, unmatched_t

    def _update_track(self, tr: PlateTrack, det, frame_idx: int, t_sec: float) -> None:
        """用新檢測更新軌跡"""
        tr.bbox = det.bbox
        tr.last_frame = frame_idx
        tr.hits += 1
        tr.age = 0

        # 地區：累積投票，最終取多數（檢測模型對同一輛車可能逐幀跳變）
        if det.region != Region.UNKNOWN:
            tr.region_votes[det.region] += 1
            tr.region = tr.region_votes.most_common(1)[0][0]

        need_ocr = (
            tr.best_result is None
            or (not self.ocr_on_new_track_only)
            or (det.confidence > tr.best_conf + self.conf_improve_thres)
        )

        if need_ocr and det.confidence > tr.best_conf:
            tr.best_conf = det.confidence
            tr.best_frame = frame_idx

    def _run_ocr(self, frame: np.ndarray, det) -> PlateResult | None:
        """只對單個檢測跑 OCR（複用 ALPR 的預處理與後處理）"""
        try:
            crop = self.alpr._prepare_crop(frame, det)
            region_hint = det.region.value if det.region != Region.UNKNOWN else None
            ocr_result = self.alpr.ocr.predict(crop, region=region_hint)
        except Exception:
            return None

        if ocr_result is None or not ocr_result.text:
            return None

        region = det.region
        if region == Region.UNKNOWN:
            from gba_plate.postprocess.rules import detect_region_from_text
            color_val = ocr_result.color.value if ocr_result.color else None
            region = detect_region_from_text(ocr_result.text, color_val)

        text, is_valid, corrected, note = self.alpr._postprocess(ocr_result.text, region)

        province, city, region_note = (None, None, "")
        energy, energy_note = (None, "")
        if region in (Region.MAINLAND, Region.GANGAO):
            from gba_plate.postprocess.region_map import resolve_plate_region
            province, city, region_note = resolve_plate_region(text)
        if region == Region.MAINLAND:
            from gba_plate.postprocess.region_map import resolve_energy
            energy, energy_note = resolve_energy(text)

        return PlateResult(
            detection=det,
            ocr=ocr_result,
            region=region,
            raw_text=ocr_result.text,
            text=text,
            is_valid=is_valid,
            validation_note=note,
            correction_applied=corrected,
            province=province,
            city=city,
            region_note=region_note,
            energy=energy,
            energy_note=energy_note,
        )

    # ------------------------------------------------------------------ 對外

    def process(
        self,
        video_path: str | Path,
        output_video: str | Path | None = None,
        output_json: str | Path | None = None,
        *,
        draw: bool = True,
        max_frames: int = 0,
        progress: Callable[[int, int], None] | None = None,
        on_track: Callable[[PlateTrack], None] | None = None,
    ) -> dict:
        """
        處理整段影片。

        Args:
            video_path: 輸入影片
            output_video: 標註後影片輸出路徑（None = 不輸出）
            output_json: JSON 時間軸輸出路徑（None = 不輸出）
            draw: 是否繪製標註
            max_frames: 最多處理多少幀（0 = 全部）
            progress: 進度回調 (current_frame, total_frames)
            on_track: 每當有新軌跡確認時回調

        Returns:
            {"plates": [...], "meta": {...}}
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"影片不存在: {video_path}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"無法開啟影片: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if max_frames:
            total = min(total, max_frames)

        writer = None
        if output_video and draw:
            out_path = Path(output_video)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            # 跳幀時輸出 fps 需相應降低，否則影片會變快
            out_fps = fps / self.frame_stride
            writer = cv2.VideoWriter(
                str(out_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                out_fps,
                (width, height),
            )

        self.tracks = []
        self._finished = []
        self._next_id = 1
        frame_idx = 0
        processed = 0
        t0 = time.time()
        frames_with_plate = 0
        total_dets = 0

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if max_frames and frame_idx >= max_frames:
                    break

                is_key = (frame_idx % self.frame_stride == 0)
                if is_key:
                    processed += 1
                    # 只做檢測，不跑 OCR（OCR 由軌跡邏輯條件性觸發）
                    detections = self.alpr.detector.predict(frame)[: self.max_det_per_frame]
                    total_dets += len(detections)
                    if detections:
                        frames_with_plate += 1

                    matches, unmatched_d, unmatched_t = self._match(detections)

                    # 更新已匹配軌跡
                    for ti, di in matches:
                        tr = self.tracks[ti]
                        det = detections[di]
                        prev_best = tr.best_conf
                        self._update_track(tr, det, frame_idx, frame_idx / fps)

                        # 觸發 OCR 的條件：
                        #   1. 還沒有任何識別結果
                        #   2. 置信度顯著提升
                        #   3. 週期性採樣（每 ocr_sample_interval 幀補一個投票樣本）
                        need_ocr = tr.best_result is None
                        if not need_ocr and det.confidence > prev_best + self.conf_improve_thres:
                            need_ocr = True
                        if (not need_ocr and self.ocr_sample_interval > 0
                                and tr.hits % self.ocr_sample_interval == 0):
                            need_ocr = True

                        if need_ocr:
                            r = self._run_ocr(frame, det)
                            if r is not None and r.text:
                                if tr.best_result is None or det.confidence > prev_best:
                                    tr.best_result = r
                                tr.ocr_samples.append((r.text, r.confidence))

                    # 新建軌跡（立即跑一次 OCR）
                    for di in unmatched_d:
                        det = detections[di]
                        tr = PlateTrack(
                            track_id=self._next_id,
                            first_frame=frame_idx,
                            first_time=frame_idx / fps,
                            last_frame=frame_idx,
                            bbox=det.bbox,
                            region=det.region,
                            region_votes=(
                                Counter({det.region: 1})
                                if det.region != Region.UNKNOWN else Counter()
                            ),
                            best_conf=det.confidence,
                            best_frame=frame_idx,
                        )
                        self._next_id += 1
                        r = self._run_ocr(frame, det)
                        if r is not None and r.text:
                            tr.best_result = r
                            tr.ocr_samples.append((r.text, r.confidence))
                        self.tracks.append(tr)

                    # 老化未匹配軌跡
                    for ti in sorted(unmatched_t, reverse=True):
                        if ti >= len(self.tracks):
                            continue
                        tr = self.tracks[ti]
                        tr.age += 1
                        if tr.age > self.max_age:
                            self.tracks.pop(ti)
                            if tr.hits >= self.min_hits:
                                self._finished.append(tr)
                                if on_track:
                                    on_track(tr)

                # 繪製（每幀都畫，用最近的軌跡狀態）
                if writer is not None:
                    if is_key:
                        canvas = self._draw_frame(frame, self.tracks)
                        self._last_canvas = canvas
                    writer.write(getattr(self, "_last_canvas", frame))

                frame_idx += 1
                if progress and processed % 30 == 0:
                    progress(min(frame_idx, total), total)

        finally:
            cap.release()
            if writer is not None:
                writer.release()

            # 收尾：把還在活躍的軌跡一併輸出
            for tr in self.tracks:
                if tr.hits >= self.min_hits:
                    self._finished.append(tr)
                    if on_track:
                        on_track(tr)
            self.tracks = []

        elapsed = time.time() - t0
        self._finished.sort(key=lambda t: t.first_frame)
        plates = [tr.to_dict(fps) for tr in self._finished]

        meta = {
            "video": video_path.name,
            "fps": round(fps, 2),
            "width": width,
            "height": height,
            "total_frames": frame_idx,
            "processed_frames": processed,
            "frame_stride": self.frame_stride,
            "duration_sec": round(frame_idx / max(fps, 1e-6), 2),
            "elapsed_sec": round(elapsed, 1),
            "ms_per_processed_frame": round(elapsed / max(processed, 1) * 1000, 1),
            "frames_with_plate": frames_with_plate,
            "total_raw_detections": total_dets,
            "unique_plates": len(plates),
            "dedup_ratio": round(total_dets / max(len(plates), 1), 1),
            "vote_corrected": sum(1 for p in plates if p["vote_corrected"]),
        }
        result = {"plates": plates, "meta": meta}

        if output_json:
            p = Path(output_json)
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "w", encoding="utf-8") as fh:
                json.dump(result, fh, ensure_ascii=False, indent=1)

        # mp4v 兼容性差，有 ffmpeg 就轉 H.264
        if output_video and Path(output_video).exists():
            self._to_h264(Path(output_video))

        return result

    def _draw_frame(self, frame: np.ndarray, tracks: list[PlateTrack]) -> np.ndarray:
        """在幀上繪製所有活躍軌跡"""
        canvas = frame.copy()
        for tr in tracks:
            x1, y1, x2, y2 = [int(v) for v in tr.bbox]
            color = REGION_COLORS.get(tr.region, REGION_COLORS[Region.UNKNOWN])
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)

            label = tr.best_result.text if tr.best_result else ""
            if label:
                tag = f"#{tr.track_id} {label}"
            else:
                tag = f"#{tr.track_id}"

            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            ty = max(y1 - 6, th + 4)
            cv2.rectangle(canvas, (x1, ty - th - 4), (x1 + tw + 6, ty + 2), color, -1)
            cv2.putText(canvas, tag, (x1 + 3, ty - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        return canvas

    @staticmethod
    def _to_h264(path: Path, crf: int = 23) -> bool:
        """用 ffmpeg 轉 H.264（瀏覽器可直接播放）"""
        tmp = path.with_suffix(".h264.mp4")
        try:
            r = subprocess.run(
                ["ffmpeg", "-y", "-i", str(path), "-c:v", "libx264",
                 "-preset", "fast", "-crf", str(crf), "-pix_fmt", "yuv420p",
                 "-movflags", "+faststart", str(tmp)],
                capture_output=True, timeout=1800,
            )
            if r.returncode == 0 and tmp.exists() and tmp.stat().st_size > 0:
                tmp.replace(path)
                return True
            tmp.unlink(missing_ok=True)
        except Exception:
            tmp.unlink(missing_ok=True)
        return False
