#!/usr/bin/env python3
"""
測試當前模型在真實圖片上的效果
"""
import cv2
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from models.experimental import attempt_load
from utils.general import non_max_suppression_face, scale_coords
from utils.datasets import letterbox
from plate_recognition.plate_rec import get_plate_result, init_model
import torch
import numpy as np


def test_model():
    """測試模型識別效果"""
    
    # 加載模型
    print("正在加載模型...")
    device = torch.device('cpu')
    
    detect_model = attempt_load('weights/plate_detect.pt', map_location=device)
    detect_model.eval()
    
    rec_model = init_model(device, 'weights/plate_rec_color.pth', is_color=True)
    
    print("模型加載完成！\n")
    
    # 測試圖片
    test_images = [
        'imgs/hongkang1.jpg',      # 粵Z牌
        'imgs/single_blue.jpg',    # 藍牌
        'imgs/single_green.jpg',   # 新能源
        'imgs/double_yellow.jpg',  # 雙層黃牌
        'imgs/police.jpg',         # 警車牌
        'imgs/xue.jpg',            # 教練車牌
    ]
    
    print("=" * 70)
    print("模型識別測試")
    print("=" * 70)
    
    total_plates = 0
    correct_plates = 0
    
    for img_path in test_images:
        img = cv2.imread(img_path)
        if img is None:
            print(f"\n❌ 無法讀取: {img_path}")
            continue
        
        print(f"\n📷 {img_path}")
        
        # 預處理
        img_size = 640
        img_resized = letterbox(img, new_shape=img_size)[0]
        img_resized = img_resized[:, :, ::-1].transpose(2, 0, 1).copy()
        img_tensor = torch.from_numpy(img_resized).to(device).float() / 255.0
        if img_tensor.ndimension() == 3:
            img_tensor = img_tensor.unsqueeze(0)
        
        # 檢測
        with torch.no_grad():
            pred = detect_model(img_tensor)[0]
        
        det = non_max_suppression_face(pred, conf_thres=0.3, iou_thres=0.5)[0]
        
        if det is not None and len(det):
            det[:, :4] = scale_coords(img_tensor.shape[2:], det[:, :4], img.shape).round()
            
            for *xyxy, conf, landmarks, class_num in det:
                x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                plate_img = img[y1:y2, x1:x2]
                
                if plate_img.size == 0:
                    continue
                
                # 識別
                plate_text, conf_val, color, color_conf = get_plate_result(
                    plate_img, device, rec_model, is_color=True
                )
                
                if plate_text:
                    conf_scalar = float(np.mean(conf_val)) if hasattr(conf_val, '__len__') else float(conf_val)
                    print(f"  ✅ {plate_text} (置信度: {conf_scalar:.2f}, 顏色: {color})")
                    total_plates += 1
                    if conf_scalar > 0.9:
                        correct_plates += 1
        else:
            print(f"  ❌ 未檢測到車牌")
    
    print("\n" + "=" * 70)
    print(f"測試結果:")
    print(f"  總計車牌: {total_plates}")
    print(f"  高置信度: {correct_plates}")
    print(f"  準確率: {correct_plates/total_plates*100:.1f}%" if total_plates > 0 else "  準確率: N/A")
    print("=" * 70)


if __name__ == '__main__':
    test_model()
