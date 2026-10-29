# 粵港澳大灣區車牌識別系統
## GBA Plate Recognition

> 四地車牌檢測與識別：**內地車牌 · 粵Z兩地牌 · 香港車牌 · 澳門車牌**

**狀態**：真實數據積累階段 — 暫緩合成數據生成與模型微調

---

## 🎯 項目定位

針對粵港澳大灣區跨境車輛場景，實現四類車牌的統一檢測與識別。

| 地區 | 車牌類型 | 顏色 | 格式 | 標準尺寸 |
|------|---------|------|------|---------|
| **內地** | 小型汽車 | 藍底白字 | 省+字母+5位 | 440×140mm |
| 內地 | 大型汽車 | 黃底黑字 | 省+字母+5位 | 440×140mm |
| 內地 | 新能源 | 綠底黑字 | 省+字母+6位 | 480×140mm |
| **粵Z** | 港澳入出內地 | 黑底白字 | 粵Z+4位+港/澳 | 440×140mm |
| **香港** | 前牌 | 白底黑字 | 1-2字母+1-4數字 | 520×110mm |
| 香港 | 後牌 | 黃底黑字 | 1-2字母+1-4數字 | 520×110mm |
| **澳門** | 民用 | 黑底白字 | M+字母+2數字+2數字 | 460×100mm |

### 格式規則（已核實）

- **內地車牌** → 永遠單行，不會分兩行
- **粵Z 兩地牌** → 永遠單行
- **香港車牌** → 以單行為主；方形雙行（上字母 / 下數字）為少數車型
- **澳門車牌** → 單行 `MA-12-34` 為主；雙行 `MA / 12-34` 用於電單車等空間受限車型
- **香港禁用字母**：`I`、`O`、`Q`；數字部分不以 `0` 開頭

---

## 📊 當前狀態

**已完成**
- [x] 技術方案調研，確定以 we0091234 為代碼基礎
- [x] CCPD 數據集下載、解壓（355,015 張真實內地車牌）
- [x] 三地牌同車實拍樣本（11 張）
- [x] 香港本地牌參考圖（3 張）、澳門本地牌參考圖（14 張）
- [x] 基礎管線可運行，粵Z 牌可識別（預訓練模型）
- [x] PyTorch 2.6+ 兼容性修復
- [x] Web UI（WebCam 實時識別）

**暫緩**
- ⏸️ 合成數據生成器 — 待真實數據充足、規格完全核實後再啟動
- ⏸️ 模型微調 — 待真實測試集建立

**待收集**
- [ ] 香港本地車牌實拍（目標 300+ 張）
- [ ] 澳門本地車牌實拍（目標 200+ 張）
- [ ] 粵Z 兩地牌實拍（目標 300+ 張）

---

## 📁 數據集

真實數據放在 `test_data/real/`（**不納入版本控制**）。

| 目錄 | 內容 | 數量 | 來源 |
|------|------|------|------|
| `ccpd/` | 內地車牌，多場景 | 355,015 張 | CCPD |
| `three_region/` | 三地牌同車實拍 | 11 張 | 自採 |
| `hk/` | 香港本地牌參考 | 3 張 | 自採 |
| `mo/` | 澳門本地牌參考 | 14 張 | 自採 |

**CCPD 子集分佈**

| 子集 | 數量 | 特點 |
|------|------|------|
| `ccpd_base` | 199,996 | 標準場景 |
| `ccpd_challenge` | 50,003 | 挑戰場景 |
| `ccpd_tilt` | 30,216 | 傾斜 |
| `ccpd_fn` | 20,967 | 遠距離 / 小目標 |
| `ccpd_blur` | 20,611 | 模糊 |
| `ccpd_db` | 10,132 | 雙層 |
| `ccpd_rotate` | 10,053 | 旋轉 |
| `ccpd_weather` | 9,999 | 雨雪霧 |
| `ccpd_np` | 3,036 | 無車牌（負樣本） |

### 獲取 CCPD

- Google Drive: `https://drive.google.com/open?id=1rdEsCUcIUaYOVRkx5IMTRNA7PcGMmSgc`
- 百度網盤: `https://pan.baidu.com/s/1i5AOjAbtkwb17Zy-NQGqkw`（提取碼 `hm0u`）

```bash
cd test_data/real/ccpd
tar xf CCPD2019.tar.xz
```

---

## 🏗️ 代碼結構

```
GBA-plate-recognition/
├── models/                  # YOLOv5 模型定義
├── plate_recognition/       # CRNN 識別網絡
│   ├── plateNet.py          # 網絡結構（78 類字符）
│   ├── plate_rec.py         # 推理 + CTC 解碼
│   └── double_plate_split_merge.py  # 雙行車牌分割
├── weights/                 # 預訓練權重
│   ├── plate_detect.pt      # 檢測 0.45M 參數
│   └── plate_rec_color.pth  # 識別 0.18M 參數
├── utils/                   # 通用工具
├── scripts/                 # 數據處理腳本
├── synthetic_data/          # 合成生成器（暫緩）
├── templates/               # Web UI 前端
├── web_ui.py                # WebCam 實時識別界面
├── detect_plate.py          # 命令行推理入口
├── train.py                 # 訓練腳本
├── export.py                # ONNX / TensorRT 導出
├── docs/                    # 報告與指引
└── test_data/               # 真實數據（不入庫）
```

### 字符集（78 類）

```
blank + 31 省份簡稱 + 學警港澳挂使領民航危險品 + 0-9 + A-Z（無 I / O）
```

字典已含 **港**、**澳**，粵Z 牌可直接識別。

---

## 🚀 快速開始

```bash
# 1. 環境
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. 權重（隨倉庫提供，或執行）
bash weights/download_weights.sh

# 3. 命令行推理
python detect_plate.py \
    --detect_model weights/plate_detect.pt \
    --rec_model weights/plate_rec_color.pth \
    --image_path imgs \
    --output result

# 4. Web UI — WebCam 實時識別
python web_ui.py --host 0.0.0.0 --port 5000
# 瀏覽器打開 http://localhost:5000
```

### 已知兼容性修復

PyTorch 2.6+ 改變了 `torch.load` 的 `weights_only` 默認值，需顯式傳 `weights_only=False`（已修復於 `models/experimental.py`、`plate_recognition/plate_rec.py`）。

---

## 📈 基準測試

11 張三地牌實拍，we0091234 預訓練模型，CPU：

| 車牌類型 | 檢測率 | 備註 |
|---------|--------|------|
| 粵Z 兩地牌 | 71% (5/7) | 黑底白字可識別 |
| 香港本地牌 | 50% (2/4) | 部分漏檢 |
| 澳門本地牌 | 0% (0/3) | 訓練數據缺失 |
| **整體** | **64% (7/11)** | 待微調 |

詳見 `docs/BASELINE_TEST_REPORT.md`。

---

## 🗺️ 路線圖

1. **真實數據收集**（當前）
   - 港澳本地牌實拍
   - 粵Z 牌多場景實拍
   - 建立標註規範

2. **數據標註**
   - 檢測框 + 四角點
   - 車牌類型標籤
   - 字符轉錄

3. **模型微調**
   - 檢測：四類別（內地 / 粵Z / 香港 / 澳門）
   - 識別：港澳字符集與格式
   - 雙行車牌處理

4. **合成數據補強**（有真實基準後）
   - 以真實數據為錨，合成數據為補充

5. **部署**
   - ONNX / TensorRT
   - 邊緣設備（RK3588 / Jetson）

---

## 🙏 致謝

- **we0091234/Chinese_license_plate_detection_recognition** — 代碼基礎（YOLOv5 檢測 + CRNN 識別，支持 12 種中文車牌含港澳粵Z）
- **detectRecog/CCPD** — CCPD 車牌數據集（ECCV 2018）
- **szad670401/HyperLPR** — 中文車牌識別框架參考
- **ankandrew/fast-plate-ocr** — 多國車牌 OCR 方案參考
- **pcb9382/PlateRecognition** — 多地區車牌識別架構參考

原始代碼許可證見 `LICENSE`，原始 README 見 `docs/README_UPSTREAM.md`。

---

**維護者**：Nick（尼克🦊）
**倉庫**：https://github.com/nicklam1994/GBA-plate-recognition
