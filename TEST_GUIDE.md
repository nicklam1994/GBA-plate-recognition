# 四地車牌識別 - 測試流程指南

## 📊 測試數據來源

### 1. 合成數據（已生成）
- **位置**: `./dataset_100k/`
- **數量**: 100,000 張
- **分佈**: 內地 50% | 粵Z 15% | 香港 20% | 澳門 15%
- **用途**: 訓練 + 驗證

### 2. 真實測試圖片
- **位置**: `./test_data/`
- **來源**: we0091234 項目自帶 + 開源數據集
- **數量**: 15 張（包含粵Z牌）
- **用途**: 基準測試

### 3. 開源數據集（可選下載）

#### Roboflow 香港車牌數據集
- **URL**: https://universe.roboflow.com/hong-kong/license-plate-recognition-rxg4e-7lvtq
- **數量**: 10,125 張
- **格式**: YOLOv8
- **下載**: 需要 Roboflow API key

#### CCPD 中國車牌數據集
- **URL**: https://opendatalab.com/OpenDataLab/CCPD/download
- **數量**: 250,000+ 張
- **格式**: 自定義
- **大小**: 13.11 GB
- **注意**: 主要包含內地車牌，無港澳

#### OpenITS 功能車牌圖像庫
- **URL**: http://www.openits.cn/openData4/569.jhtml
- **包含**: 香港出入境車牌 31 張 + 澳門出入境車牌 28 張
- **格式**: VOC/YOLO
- **下載**: 免費（需註冊）

---

## 🧪 測試流程

### 階段 1: 基準測試（使用現有模型）

```bash
cd ~/repos/chinese-plate-recognition
source venv/bin/activate

# 測試 15 張真實圖片
python detect_plate.py \
    --detect_model weights/plate_detect.pt \
    --rec_model weights/plate_rec_color.pth \
    --image_path test_data/hk \
    --output test_results/baseline

# 查看結果
cat test_results/baseline/*.txt
```

**預期結果**:
- 內地車牌：準確率 > 95%
- 粵Z牌：準確率 > 80%（已有字符支持）
- 香港/澳門本地牌：準確率 < 50%（需要微調）

### 階段 2: 微調檢測模型

```bash
# 等待數據集生成完成
watch -n 5 'ls dataset_100k/images/ | wc -l'

# 微調檢測模型（4 類別）
python finetune_detect.py \
    --weights weights/plate_detect.pt \
    --data ./dataset_100k/data.yaml \
    --epochs 50 \
    --batch 16 \
    --device 0  # GPU 設備

# 最佳模型位置
ls runs/finetune/plate_detect_4region/weights/best.pt
```

### 階段 3: 評估微調效果

```bash
# 使用微調後的模型測試
python detect_plate.py \
    --detect_model runs/finetune/plate_detect_4region/weights/best.pt \
    --rec_model weights/plate_rec_color.pth \
    --image_path test_data/hk \
    --output test_results/finetuned

# 對比結果
diff test_results/baseline/ test_results/finetuned/
```

### 階段 4: 識別模型微調（可選）

如果檢測效果提升但識別不準確，需要微調識別模型：

```bash
# 準備識別訓練數據
python prepare_rec_data.py --synthetic ./dataset_100k --output ./rec_dataset

# 微調識別模型
cd ~/repos/chinese-plate-recognition/plate_recognition
python train_rec.py \
    --data ./rec_dataset \
    --epochs 100 \
    --batch 64
```

---

## 📈 評估指標

### 檢測模型指標
- **mAP@0.5**: 目標 > 0.90
- **mAP@0.5:0.95**: 目標 > 0.70
- **Precision**: 目標 > 0.90
- **Recall**: 目標 > 0.85
- **FPS**: 目標 > 30 (GPU)

### 識別模型指標
- **字符準確率**: 目標 > 95%
- **車牌準確率**: 目標 > 90%
- **推理時間**: 目標 < 50ms (GPU)

### 四地覆蓋指標
| 地區 | 檢測 mAP | 識別準確率 | 目標 |
|------|---------|-----------|------|
| 內地 | > 0.95 | > 95% | ✅ 已達標 |
| 粵Z | > 0.90 | > 85% | 🔄 微調中 |
| 香港 | > 0.85 | > 80% | 🔄 微調中 |
| 澳門 | > 0.85 | > 80% | 🔄 微調中 |

---

## 🔍 錯誤分析

### 常見錯誤類型

1. **檢測失敗**
   - 原因：車牌角度過大、遮擋嚴重、光照不足
   - 解決：增加數據增強（透視變換、光照變化）

2. **識別錯誤**
   - 原因：字符相似（0/O、1/I）、字體不匹配
   - 解決：增加對應字符的訓練樣本

3. **地區分類錯誤**
   - 原因：粵Z牌與香港牌混淆
   - 解決：增加兩地牌的區分特徵訓練

### 錯誤分析工具

```python
# 分析錯誤案例
python analyze_errors.py \
    --pred test_results/finetuned/ \
    --gt test_data/annotations.txt \
    --output error_analysis/

# 生成錯誤報告
cat error_analysis/error_report.txt
```

---

## 🚀 部署測試

### CPU 部署
```bash
# 導出 ONNX
python export.py --weights best.pt --include onnx

# CPU 推理
python onnx_infer.py \
    --detect_model best.onnx \
    --rec_model plate_rec_color.onnx \
    --image_path test_data/hk
```

### GPU 部署（TensorRT）
```bash
# 導出 TensorRT
python export.py --weights best.pt --include engine

# TensorRT 推理
python trt_infer.py \
    --detect_model best.engine \
    --rec_model plate_rec_color.engine \
    --image_path test_data/hk
```

---

## 📝 測試日誌模板

```markdown
## 測試日期: 2026-10-29

### 環境
- GPU: RTX 4090
- PyTorch: 2.15.0
- CUDA: 12.x

### 數據集
- 訓練集: 100,000 張（合成）
- 驗證集: 1,000 張（合成）
- 測試集: 15 張（真實）

### 結果
| 指標 | 基準 | 微調後 | 提升 |
|------|------|--------|------|
| mAP@0.5 | 0.75 | 0.92 | +22.7% |
| 內地準確率 | 95% | 96% | +1% |
| 粵Z準確率 | 60% | 88% | +46.7% |
| 香港準確率 | 30% | 82% | +173% |
| 澳門準確率 | 25% | 79% | +216% |

### 結論
- 微調後四地覆蓋率顯著提升
- 粵Z牌效果最好（字符集相同）
- 香港/澳門牌需要更多真實數據

### 下一步
- [ ] 收集更多真實港澳車牌圖片
- [ ] 微調識別模型
- [ ] 優化後處理規則
```

---

## 📚 參考資源

- we0091234 訓練文檔: `readme/`
- YOLOv5 官方文檔: https://docs.ultralytics.com/
- Roboflow 教程: https://blog.roboflow.com/
- CCPD 論文: https://arxiv.org/abs/1806.10782
