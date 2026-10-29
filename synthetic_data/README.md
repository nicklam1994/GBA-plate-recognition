# 四地車牌合成數據生成器

生成內地、粵Z兩地牌、香港、澳門四地車牌的合成訓練數據。

## 功能

- ✅ **四地車牌覆蓋**：內地（藍/黃/綠/白/雙層黃）、粵Z（單層/雙層）、香港（前白/後黃）、澳門（黑底白字）
- ✅ **自動字體縮放**：字符過多時自動縮小字體，確保不裁切
- ✅ **數據增強**：噪聲、模糊、陰影、雨滴、光暈、JPEG 壓縮、天氣模擬
- ✅ **多格式輸出**：圖片 + labels.txt + YOLO 格式標籤
- ✅ **並行生成**：多進程加速，~350 img/s

## 快速開始

```bash
# 激活虛擬環境
cd ~/repos/chinese-plate-recognition
source venv/bin/activate

# 生成 1000 張（默認分佈：內地50% 粵Z15% 香港20% 澳門15%）
python synthetic_data/generate.py --num 1000 --output ./dataset

# 自定義地區權重
python synthetic_data/generate.py --num 5000 \
    --mainland 40 --gangao 20 --hk 25 --mo 15 \
    --output ./dataset_large

# 關閉增強（生成乾淨樣本）
python synthetic_data/generate.py --num 100 --no-augment --output ./dataset_clean

# 指定圖片尺寸
python synthetic_data/generate.py --num 500 --img-width 640 --img-height 480 --output ./dataset_640
```

## 輸出結構

```
dataset/
├── images/              # 車牌圖片
│   ├── mainland_000000_mainland_blue.jpg
│   ├── hk_000001_hk_rear.jpg
│   ├── gangao_000002_gangao_single.jpg
│   └── mo_000003_mo_plate.jpg
├── labels.txt           # 標註（文件名 車牌號 地區 規格）
├── labels_yolo/         # YOLO 格式標籤（用於檢測訓練）
│   ├── mainland_000000_mainland_blue.txt
│   └── ...
└── dataset_config.json  # 數據集配置
```

## 車牌規格

| 地區 | 規格 | 尺寸 | 配色 | 格式示例 |
|------|------|------|------|---------|
| 內地 | 藍牌 | 440×140 | 藍底白字 | 京A12345 |
| 內地 | 黃牌 | 440×140 | 黃底黑字 | 滬B67890 |
| 內地 | 綠牌（新能源）| 440×140 | 綠底白字 | 粵BD12345 |
| 內地 | 白牌（警車）| 440×140 | 白底黑字 | 鄂F8888警 |
| 內地 | 雙層黃牌 | 220×280 | 黃底黑字 | 寧R\n7691 |
| 粵Z | 單層 | 440×140 | 黑底白字 | 粤ZAB12港 |
| 粵Z | 雙層 | 220×280 | 黑底白字 | 粤ZP9\nRAZ澳 |
| 香港 | 前牌 | 370×130 | 白底黑字 | AB 1234 |
| 香港 | 後牌 | 370×130 | 黃底黑字 | HI 7835 |
| 澳門 | 民用 | 370×130 | 黑底白字 | MK 16 18 |

## YOLO 訓練標籤

類別 ID 映射：
```json
{
    "0": "mainland",
    "1": "gangao",
    "2": "hk",
    "3": "mo"
}
```

YOLO 格式（每行）：`class_id x_center y_center width height`（歸一化 0-1）

## 數據增強效果

| 增強類型 | 描述 | 默認概率 |
|---------|------|---------|
| 高斯噪聲 | 模擬传感器噪聲 | 30-70% |
| 運動模糊 | 模擬車輛移動 | 30-70% |
| 亮度/對比度 | 模擬不同光照 | 30-70% |
| 陰影 | 遮擋效果 | 15-35% |
| 雨滴 | 雨天場景 | 10-25% |
| 光暈 | 逆光場景 | 15-35% |
| JPEG 壓縮 | 低質量傳輸 | 15-35% |
| 天氣模擬 | 晴天/夜間/雨天/霧天/逆光 | 50% |
| 遮擋 | 污漬/劃痕/貼紙 | 15% |

## 與 we0091234 訓練格式兼容

生成的 YOLO 標籤可直接用於 we0091234 的檢測模型訓練：

```bash
# 1. 生成數據
python synthetic_data/generate.py --num 10000 --output ./yolov8_dataset

# 2. 複製到 yolov8-plate 項目
cp -r ./yolov8_dataset/images ~/repos/yolov8-plate/datasets/plates/
cp -r ./yolov8_dataset/labels_yolo ~/repos/yolov8-plate/datasets/plates/labels/

# 3. 訓練檢測模型
cd ~/repos/yolov8-plate
python train.py --data datasets/plates/data.yaml
```

## API 使用

```python
from synthetic_data.plate_specs import generate_plate
from synthetic_data.plate_renderer import render_plate
from synthetic_data.augmentations import augment_plate

# 生成車牌
text, spec, region = generate_plate("hk")  # mainland/gangao/hk/mo/random

# 渲染
img = render_plate(text, spec)

# 增強
img_aug = augment_plate(img, level="medium")

# 保存
import cv2
cv2.imwrite("plate.jpg", img_aug)
```

## 文件結構

```
synthetic_data/
├── __init__.py           # 包初始化
├── plate_specs.py        # 車牌規格定義 + 車牌號生成器
├── plate_renderer.py     # Pillow 渲染器
├── augmentations.py      # 數據增強模塊
├── generate.py           # 批量生成腳本
└── README.md             # 本文件
```

## 下一步

1. **生成大規模數據集**：10 萬張以上
2. **微調 we0091234 檢測模型**：使用合成數據 + 少量真實數據
3. **評估識別準確率**：在真實港澳車牌圖片上測試
4. **迭代優化**：根據錯誤案例調整生成參數
