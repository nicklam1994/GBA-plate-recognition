#!/usr/bin/env python3
"""
生成測試用影片：把一張車牌圖做成平移動畫，模擬車輛駛過。

用途：驗證 VideoProcessor 的跨幀追蹤與去重，不需要真實行車記錄儀素材。
"""
import glob
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path("/home/nicklam-ai/repos/GBA-plate-recognition")


def make_video(src_img: Path, out: Path, frames: int = 120, fps: int = 25,
               canvas: tuple[int, int] = (1280, 720)):
    img = cv2.imdecode(np.fromfile(str(src_img), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"無法讀取 {src_img}")

    W, H = canvas
    # 把原圖縮到高度 60% 放進畫布
    scale = (H * 0.6) / img.shape[0]
    im = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)),
                    interpolation=cv2.INTER_AREA)
    ih, iw = im.shape[:2]

    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    for i in range(frames):
        # 背景：漸變灰
        frame = np.full((H, W, 3), 60, dtype=np.uint8)
        for y in range(H):
            frame[y, :, :] = 40 + int(60 * y / H)

        # 水平移動 + 輕微垂直浮動
        progress = i / max(frames - 1, 1)
        x = int((W - iw) * progress)
        y = int((H - ih) / 2 + 30 * np.sin(progress * 6.28))

        x2, y2 = x + iw, y + ih
        # 邊界裁剪
        sx1, sy1 = max(0, -x), max(0, -y)
        dx1, dy1 = max(0, x), max(0, y)
        dx2, dy2 = min(W, x2), min(H, y2)
        if dx2 > dx1 and dy2 > dy1:
            frame[dy1:dy2, dx1:dx2] = im[sy1:sy1 + (dy2 - dy1), sx1:sx1 + (dx2 - dx1)]

        writer.write(frame)

    writer.release()
    print(f"生成 {out}  ({frames} 幀 @ {fps}fps, {W}x{H})")


def main():
    out_dir = ROOT / "video_test"
    out_dir.mkdir(exist_ok=True)

    # 選幾張不同地區的測試圖
    imgs = sorted(glob.glob(str(ROOT / "imgs" / "*.jpg")))
    if not imgs:
        # 退回用 Rreal 數據集的圖片
        imgs = sorted(glob.glob(str(ROOT / "test_data/real/hk/*.jpg")))[:3]

    picks = imgs[:3] if len(imgs) >= 3 else imgs
    for i, src in enumerate(picks):
        make_video(Path(src), out_dir / f"test_move_{i}.mp4", frames=90, fps=25)
        print(f"  來源: {Path(src).name}")


if __name__ == "__main__":
    main()
