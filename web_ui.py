#!/usr/bin/env python3
"""
車牌識別 Web UI 服務器（gba_plate 版）

與舊版的差別：不再自己實作 letterbox / NMS / 透視 / 識別，
全部交給 gba_plate.ALPR。本檔案只負責 Flask 路由、WebSocket 收發、畫框。

用法:
    python web_ui.py --host 0.0.0.0 --port 5000
    瀏覽器訪問 http://localhost:5000
"""

from __future__ import annotations

import argparse
import base64
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, render_template
from flask_socketio import SocketIO, emit

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

app = Flask(__name__)
app.config["SECRET_KEY"] = "gba-plate-recognition-web"
socketio = SocketIO(app, cors_allowed_origins="*")

#: 全局 ALPR 實例（延遲初始化）
alpr = None

#: 地區 → 邊框顏色（BGR）
REGION_COLORS = {
    "mainland": (0, 200, 0),     # 綠
    "gangao": (0, 165, 255),     # 橙（粵Z 兩地牌）
    "hk": (255, 180, 0),         # 藍
    "mo": (200, 0, 200),         # 紫
    "unknown": (160, 160, 160),  # 灰
}


def load_models(device: str | None = None, class_mode: str = "single_double") -> None:
    """載入 ALPR 管線（冪等）"""
    global alpr
    if alpr is not None:
        return

    from gba_plate import ALPR

    print("正在載入 gba_plate.ALPR ...")
    alpr = ALPR(
        detect_model=ROOT / "weights/plate_detect.pt",
        rec_model=ROOT / "weights/plate_rec_color.pth",
        device=device,
        class_mode=class_mode,
    )
    print("模型載入完成")


def draw_results(img: np.ndarray, results: list) -> np.ndarray:
    """在影像上繪製檢測框、角點、標籤"""
    out = img.copy()

    for r in results:
        d = r.to_dict()
        x1, y1, x2, y2 = [int(v) for v in d["bbox"]]
        region_color = REGION_COLORS.get(d["region"], REGION_COLORS["unknown"])
        box_color = region_color if d["is_valid"] else (0, 0, 255)

        cv2.rectangle(out, (x1, y1), (x2, y2), box_color, 2)
        for cx, cy in d["corners"]:
            cv2.circle(out, (int(cx), int(cy)), 3, (0, 255, 255), -1)

        label = f"{d['text']} [{d['region']}] {d['confidence']:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale, thick = 0.55, 1
        (tw, th), baseline = cv2.getTextSize(label, font, scale, thick)

        ty = max(y1 - 6, th + 6)
        cv2.rectangle(
            out, (x1, ty - th - 4), (x1 + tw + 4, ty + baseline), box_color, -1
        )
        cv2.putText(
            out, label, (x1 + 2, ty), font, scale, (255, 255, 255), thick, cv2.LINE_AA
        )

    return out


def encode_jpeg_base64(img: np.ndarray, quality: int = 80) -> str:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG 編碼失敗")
    return base64.b64encode(buf).decode("utf-8")


def decode_base64_image(data: str) -> np.ndarray | None:
    """base64 → BGR ndarray"""
    try:
        raw = base64.b64decode(data)
        arr = np.frombuffer(raw, np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception:
        return None


@app.route("/")
def index():
    return render_template("index.html")


@socketio.on("connect")
def handle_connect():
    print("客戶端已連接")
    emit("status", {"message": "已連接到服務器"})


@socketio.on("disconnect")
def handle_disconnect():
    print("客戶端已斷開")


@socketio.on("frame")
def handle_frame(data):
    """WebCam 連續幀"""
    if alpr is None:
        emit("error", {"message": "模型未載入"})
        return

    img = decode_base64_image(data.get("image", ""))
    if img is None:
        return

    t0 = time.perf_counter()
    try:
        results = alpr.process(img)
    except Exception as exc:  # noqa: BLE001
        print(f"處理幀錯誤: {exc}")
        emit("error", {"message": str(exc)})
        return
    process_ms = (time.perf_counter() - t0) * 1000

    emit(
        "result",
        {
            "image": encode_jpeg_base64(draw_results(img, results), 80),
            "plates": [r.to_dict() for r in results],
            "process_time": round(process_ms, 2),
            "num_plates": len(results),
        },
    )


@socketio.on("test_image")
def handle_test_image(data):
    """單張上傳圖片"""
    if alpr is None:
        emit("error", {"message": "模型未載入"})
        return

    img = decode_base64_image(data.get("image", ""))
    if img is None:
        emit("error", {"message": "無法解碼圖片"})
        return

    t0 = time.perf_counter()
    try:
        results = alpr.process(img)
    except Exception as exc:  # noqa: BLE001
        print(f"測試圖片錯誤: {exc}")
        emit("error", {"message": str(exc)})
        return
    process_ms = (time.perf_counter() - t0) * 1000

    emit(
        "test_result",
        {
            "image": encode_jpeg_base64(draw_results(img, results), 90),
            "plates": [r.to_dict() for r in results],
            "process_time": round(process_ms, 2),
            "num_plates": len(results),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="GBA 車牌識別 Web UI")
    parser.add_argument("--host", default="0.0.0.0", help="監聽地址")
    parser.add_argument("--port", type=int, default=5000, help="監聽端口")
    parser.add_argument("--device", default=None, help="torch 設備（cuda / cpu）")
    parser.add_argument(
        "--class-mode",
        default="single_double",
        choices=["single_double", "region"],
        help="檢測模型類別語義：舊權重用 single_double，4 類別新權重用 region",
    )
    parser.add_argument("--debug", action="store_true", help="Flask 調試模式")
    args = parser.parse_args()

    load_models(device=args.device, class_mode=args.class_mode)

    print(f"\n啟動 Web UI: http://localhost:{args.port}")
    print("按 Ctrl+C 停止\n")
    socketio.run(
        app,
        host=args.host,
        port=args.port,
        debug=args.debug,
        allow_unsafe_werkzeug=True,
    )


if __name__ == "__main__":
    main()
