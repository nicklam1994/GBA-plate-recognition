#!/usr/bin/env python3
"""
影片離線識別：輸出標註影片 + JSON 時間軸。

用法：
    python scripts/process_video.py video.mp4
    python scripts/process_video.py dashcam.mp4 -o out/ --stride 2
    python scripts/process_video.py dashcam.mp4 --no-video   # 只出 JSON
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser(description="影片離線車牌識別")
    ap.add_argument("video", help="輸入影片檔")
    ap.add_argument("-o", "--out-dir", default="video_out", help="輸出目錄")
    ap.add_argument("--weights", default=None,
                    help="檢測權重（默認自動：有 yolov8 四類就用它）")
    ap.add_argument("--stride", type=int, default=1, help="每隔幾幀處理（2=速度翻倍）")
    ap.add_argument("--max-frames", type=int, default=0, help="最多處理多少幀（測試用）")
    ap.add_argument("--min-hits", type=int, default=2, help="軌跡至少命中幾幀才輸出")
    ap.add_argument("--max-age", type=int, default=30, help="軌跡消失幾幀後結束")
    ap.add_argument("--conf", type=float, default=0.25, help="檢測置信度門檻")
    ap.add_argument("--no-video", action="store_true", help="不輸出標註影片")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    from gba_plate.alpr import ALPR
    from gba_plate.video import VideoProcessor

    # 自動選權重：優先用四類模型
    weights = args.weights
    if weights is None:
        candidates = [
            ROOT / "runs/detect/gba_4class_yolov8s/weights/best.pt",
            ROOT / "weights/plate_detect.pt",
        ]
        for c in candidates:
            if c.exists():
                weights = str(c)
                break
        if weights is None:
            raise SystemExit("找不到可用權重")

    is_region = "yolov8" in Path(weights).name.lower() or "gba_4class" in str(weights)
    print("=" * 68)
    print("影片離線車牌識別")
    print("=" * 68)
    print(f"影片:   {args.video}")
    print(f"權重:   {weights}  ({'四類地區' if is_region else '單/雙行'})")
    print(f"stride: {args.stride}   min_hits: {args.min_hits}   max_age: {args.max_age}")
    print()

    print("載入模型...", flush=True)
    alpr = ALPR(
        detect_model=weights,
        class_mode="region" if is_region else "single_double",
        conf_thres=args.conf,
        device=args.device,
    )

    proc = VideoProcessor(
        alpr,
        frame_stride=args.stride,
        min_hits=args.min_hits,
        max_age=args.max_age,
    )

    vpath = Path(args.video)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = vpath.stem

    out_video = None if args.no_video else out_dir / f"{stem}_annotated.mp4"
    out_json = out_dir / f"{stem}_plates.json"

    def on_progress(cur, total):
        pct = 100 * cur / max(total, 1)
        print(f"\r  處理中 {cur}/{total} ({pct:.0f}%)", end="", flush=True)

    print("開始處理...", flush=True)
    result = proc.process(
        vpath,
        output_video=out_video,
        output_json=out_json,
        max_frames=args.max_frames,
        progress=on_progress,
    )
    print()

    meta = result["meta"]
    plates = result["plates"]

    print()
    print("=" * 68)
    print("處理統計")
    print("=" * 68)
    print(f"影片長度:     {meta['duration_sec']} 秒 ({meta['total_frames']} 幀 @ {meta['fps']}fps)")
    print(f"實際處理:     {meta['processed_frames']} 幀（stride={meta['frame_stride']}）")
    print(f"耗時:         {meta['elapsed_sec']} 秒 ({meta['ms_per_processed_frame']} ms/幀)")
    print(f"原始檢測框:   {meta['total_raw_detections']}")
    print(f"去重後車牌:   {meta['unique_plates']}  (壓縮比 {meta['dedup_ratio']}×)")
    print(f"時序投票糾正: {meta['vote_corrected']} 個")

    print()
    print("=" * 68)
    print(f"識別結果（{len(plates)} 個車牌）")
    print("=" * 68)
    if not plates:
        print("  未識別到車牌")
    else:
        header = f"{'時間':>7} {'車牌':<13} {'地區':<9} {'省 · 市':<20} {'幀數':>4} {'置信':>5}"
        print(header)
        print("-" * 74)
        region_names = {"mainland": "內地", "gangao": "粵Z", "hk": "香港",
                        "mo": "澳門", "unknown": "未知"}
        for p in plates:
            t = f"{int(p['t']//60):02d}:{p['t']%60:05.2f}"
            loc = ""
            if p["province"]:
                loc = p["province"] + (" · " + p["city"] if p["city"] else "")
                if p["region_note"]:
                    loc += f"（{p['region_note']}）"
            rn = region_names.get(p["region"], p["region"])
            mark = " ✎" if p["vote_corrected"] else ""
            print(f"{t:>7} {p['text'] or '(未識別)':<13} {rn:<9} {loc:<20} "
                  f"{p['frames']:>4} {p['confidence']:>5.2f}{mark}")

        # 投票糾正明細
        corrected = [p for p in plates if p["vote_corrected"]]
        if corrected:
            print()
            print("時序投票糾正明細：")
            for p in corrected:
                print(f"  {p['text_single_frame']} → {p['text']}  "
                      f"（{p['ocr_samples']} 幀投票，{p['confidence']:.2f}）")

    print()
    print(f"JSON 時間軸: {out_json}")
    if out_video and Path(out_video).exists():
        size_mb = Path(out_video).stat().st_size / 1024 / 1024
        print(f"標註影片:    {out_video}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
