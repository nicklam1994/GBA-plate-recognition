#!/usr/bin/env python3
"""
車牌識別 Web UI 服務器
支持 WebCam 實時識別

用法:
    python web_ui.py --host 0.0.0.0 --port 5000
    
然後在瀏覽器訪問: http://localhost:5000
"""
import argparse
import base64
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, render_template, Response, jsonify, request
from flask_socketio import SocketIO, emit
import torch

# 添加項目根目錄
sys.path.insert(0, str(Path(__file__).parent))

from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression_face, scale_coords
from utils.datasets import letterbox
from plate_recognition.plate_rec import get_plate_result, init_model
from plate_recognition.double_plate_split_merge import get_split_merge

# Flask 應用
app = Flask(__name__)
app.config['SECRET_KEY'] = 'plate-recognition-web'
socketio = SocketIO(app, cors_allowed_origins="*")

# 全局模型
detect_model = None
rec_model = None
device = None
img_size = 640


def load_models():
    """加載檢測和識別模型"""
    global detect_model, rec_model, device
    
    print("正在加載模型...")
    
    # 選擇設備
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用設備: {device}")
    
    # 加載檢測模型
    detect_model_path = 'weights/plate_detect.pt'
    detect_model = attempt_load(detect_model_path, map_location=device)
    detect_model.eval()
    print(f"檢測模型已加載: {detect_model_path}")
    
    # 加載識別模型
    rec_model_path = 'weights/plate_rec_color.pth'
    rec_model = init_model(device, rec_model_path, is_color=True)
    print(f"識別模型已加載: {rec_model_path}")
    
    print("模型加載完成！")


def detect_and_recognize(img_bgr):
    """檢測並識別車牌"""
    global detect_model, rec_model, device, img_size
    
    results = []
    
    try:
        # 預處理
        img = letterbox(img_bgr, new_shape=img_size)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, HWC to CHW
        img = img.copy()  # 解決負步長問題
        img = torch.from_numpy(img).to(device)
        img = img.float() / 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        
        # 檢測
        with torch.no_grad():
            pred = detect_model(img)[0]
        
        # NMS
        det = non_max_suppression_face(pred, conf_thres=0.3, iou_thres=0.5)[0]
        
        if det is not None and len(det):
            # 縮放座標
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img_bgr.shape).round()
            
            for *xyxy, conf, landmarks, class_num in det:
                # 獲取車牌區域
                x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                
                # 獲取車牌小圖（使用 bbox 而非 landmarks）
                plate_img = img_bgr[y1:y2, x1:x2]
                
                if plate_img.size == 0:
                    continue
                
                # 識別
                plate_text, conf_val, color, color_conf = get_plate_result(
                    plate_img, device, rec_model, is_color=True
                )
                
                if plate_text:
                    # 提取標量值（conf_val 是每個字符的置信度數組）
                    import numpy as np
                    conf_scalar = float(np.mean(conf_val)) if hasattr(conf_val, '__len__') else float(conf_val)
                    color_conf_scalar = float(color_conf.item()) if hasattr(color_conf, 'item') else float(color_conf)
                    
                    results.append({
                        'bbox': [x1, y1, x2, y2],
                        'plate': plate_text,
                        'confidence': conf_scalar,
                        'color': color,
                        'color_confidence': color_conf_scalar,
                    })
    
    except Exception as e:
        print(f"識別錯誤: {e}")
        import traceback
        traceback.print_exc()
    
    return results


@app.route('/')
def index():
    """主頁面"""
    return render_template('index.html')


@socketio.on('connect')
def handle_connect():
    """WebSocket 連接"""
    print('客戶端已連接')
    emit('status', {'message': '已連接到服務器'})


@socketio.on('disconnect')
def handle_disconnect():
    """WebSocket 斷開"""
    print('客戶端已斷開')


@socketio.on('frame')
def handle_frame(data):
    """處理視頻幀"""
    try:
        # 解碼 base64 圖片
        img_data = base64.b64decode(data['image'])
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return
        
        # 檢測並識別
        start_time = time.time()
        results = detect_and_recognize(img)
        process_time = (time.time() - start_time) * 1000
        
        # 在圖片上繪製結果
        for result in results:
            x1, y1, x2, y2 = result['bbox']
            plate = result['plate']
            conf = result['confidence']
            
            # 繪製邊框
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 繪製文字
            text = f"{plate} ({conf:.2f})"
            cv2.putText(img, text, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # 編碼為 JPEG
        _, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # 返回結果
        emit('result', {
            'image': img_base64,
            'plates': results,
            'process_time': process_time,
            'num_plates': len(results)
        })
    
    except Exception as e:
        print(f"處理幀錯誤: {e}")
        emit('error', {'message': str(e)})


@socketio.on('test_image')
def handle_test_image(data):
    """處理單張測試圖片"""
    try:
        img_data = base64.b64decode(data['image'])
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            emit('error', {'message': '無法解碼圖片'})
            return
        
        results = detect_and_recognize(img)
        
        # 繪製結果
        for result in results:
            x1, y1, x2, y2 = result['bbox']
            plate = result['plate']
            conf = result['confidence']
            color = result['color']
            
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
            text = f"{plate} [{color}]"
            cv2.putText(img, text, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        _, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        emit('test_result', {
            'image': img_base64,
            'plates': results,
            'num_plates': len(results)
        })
    
    except Exception as e:
        print(f"測試圖片錯誤: {e}")
        emit('error', {'message': str(e)})


def main():
    parser = argparse.ArgumentParser(description='車牌識別 Web UI')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='服務器地址')
    parser.add_argument('--port', type=int, default=5000, help='服務器端口')
    parser.add_argument('--debug', action='store_true', help='調試模式')
    
    args = parser.parse_args()
    
    # 加載模型
    load_models()
    
    # 啟動服務器
    print(f"\n啟動 Web UI 服務器...")
    print(f"訪問地址: http://localhost:{args.port}")
    print(f"按 Ctrl+C 停止服務器\n")
    
    socketio.run(app, host=args.host, port=args.port, debug=args.debug, allow_unsafe_werkzeug=True)


if __name__ == '__main__':
    main()
