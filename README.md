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

**已完成（重構）**
- [x] `gba_plate/` 分層架構（抽象層 + 檢測 + 前處理 + OCR + 後處理）
- [x] 四地格式校驗規則引擎（開源空白區，本項目獨創）
- [x] 76 個單元測試全綠
- [x] Web UI 遷移至新架構（消除與 detect_plate.py 的重複推理）

**暫緩**
- ⏸️ 合成數據生成器 — 待真實數據充足、規格完全核實後再啟動
- ⏸️ 模型微調 — 待真實測試集建立（港澳數據已下載 6,311 張）

**待收集**
- [ ] 香港本地車牌實拍（目標 300+ 張）
- [ ] 澳門本地車牌實拍（目標 200+ 張）
- [ ] 粵Z 兩地牌實拍（目標 300+ 張）

---

## 📁 數據集

真實數據放在 `test_data/real/`（**不納入版本控制**）。

| 目錄 | 內容 | 圖片 | 標註框 | 來源 |
|------|------|------|--------|------|
| `ccpd/` | 內地車牌，多場景 | 355,018 | ~355k | CCPD2019 |
| `macau_merged/` | 澳門為主，含 GDM/HKF/HKR/MEV/MO/MTP/OTH | 4,629 | 7,356 | Roboflow (CC BY 4.0) |
| `macau_clear/` | 澳門車牌 | 769 | 822 | Roboflow (CC BY 4.0) |
| `macau_shh5r/` | 澳門車牌 | 490 | 521 | Roboflow (CC BY 4.0) |
| `macau_parkinglot/` | 澳門停車場 | 239 | 239 | Roboflow (CC BY 4.0) |
| `hk_alan_carplate/` | 三地牌同車 | 134 | 166 | Roboflow (CC BY 4.0) |
| `hk_lpr/` | 香港車牌 | 50 | 88 | Roboflow (CC BY 4.0) |
| `three_region/` | 三地牌同車實拍 | 11 | 手工 | 自採 |
| `hk/` | 香港本地牌參考 | 3 | 手工 | 自採 |
| `mo/` | 澳門本地牌參考 | 14 | 手工 | 自採 |

**港澳真實數據合計**：6,311 張 / 9,192 標註框

### 港澳標註類別分佈（實測）

| 類別 | 標註數 | 說明 |
|------|--------|------|
| 澳門本地牌 | ~7,900 | MO 6,328 + Licence-Plate 1,343 + MO 239 |
| 粵Z 澳牌 (GDM) | 368 | Roboflow 專用類別 |
| 香港本地牌 | 326 | HKF 103 + HKR 87 + hk_carplate 135 + 1 |
| 其他/臨時 | 255 | OTH 217 + MTP 38 |

> **注意**：澳門數據充足，香港與粵Z 偏少，需持續自採補充。

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
├── gba_plate/                    # ★ 分層架構（新，推薦使用）
│   ├── base.py                   #   ABC 抽象 + 結果 dataclass（Region/Detection/OcrResult/PlateResult）
│   ├── alpr.py                   #   ALPR 組合類（唯一對外入口）
│   ├── detectors/
│   │   └── yolo_plate.py         #   YOLOv5 檢測器（支援 single_double / region 兩種類別語義）
│   ├── preprocess/
│   │   ├── perspective.py        #   4 角點排序 + 透視校正
│   │   └── double_line.py        #   雙行 5/12 + 1/3 hstack 拼接
│   ├── ocr/
│   │   ├── vocab.py              #   78/79 類字符表（79 = 78 + 「-」）
│   │   └── crnn.py               #   CRNN + CTC 識別（180K 參數）
│   └── postprocess/
│       └── rules.py              #   四地格式校驗 + 混淆字符糾正 + 地區推斷
├── tests/                        # pytest（76 個測試）
├── scripts/                      # 數據處理 / 驗證腳本
├── models/                       # YOLOv5 網絡定義（底層依賴）
├── plate_recognition/            # CRNN 網絡定義（底層依賴）
├── weights/
│   ├── plate_detect.pt           #   檢測 1.12 MB
│   └── plate_rec_color.pth       #   識別 0.72 MB（180,435 參數）
├── web_ui.py                     # WebCam 實時識別（218 行，只調 ALPR）
├── templates/index.html          # Web UI 前端
├── detect_plate.py               # 舊版單體腳本（保留對照，新代碼請用 gba_plate）
└── test_data/                    # 真實數據（不入庫）
```

### 分層設計理由

| 層 | 來源 | 借鑒的關鍵點 |
|----|------|-------------|
| 抽象接口 | fast-alpr | 40 行 ABC 把擴展點定死；`OcrResult` 內建 region |
| 檢測權重 | we0091234 | 唯一公開且可微調，輸出 4 角點 |
| 雙行處理 | zhahoi-ncnn / pcb9382 | 5/12 + 1/3 疊加 8.3%，防字符跨行被切斷 |
| 識別模型 | we0091234 | 180K 參數，比 LPRNet 小 2.5 倍 |
| 字符集 | we0091234 + HuKai97 | 78 類含港澳 + 「-」= 79 類 |
| 多類別語義 | 本項目獨創 | 開源項目全部只有 single/double，無地區分類 |
| 後處理規則 | 本項目獨創 | 港澳格式校驗是開源空白區 |

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

# 2. 權重（隨倉庫提供）
bash weights/download_weights.sh

# 3. 新架構 API（推薦）
python -c "
from gba_plate import ALPR
alpr = ALPR()
for r in alpr.process_file('imgs/hongkang1.jpg'):
    print(r.region.value, r.text, f'{r.confidence:.2f}', r.is_valid)
"

# 4. Web UI — WebCam 實時識別
python web_ui.py --port 5000
# 瀏覽器打開 http://localhost:5000
```

### Python API

```python
from gba_plate import ALPR
import cv2

alpr = ALPR(
    detect_model="weights/plate_detect.pt",
    rec_model="weights/plate_rec_color.pth",
    class_mode="single_double",   # 舊權重；4 類別新權重用 "region"
)

results = alpr.process(cv2.imread("test.jpg"))
for r in results:
    print(r.region.value)          # mainland / gangao / hk / mo
    print(r.text)                  # 後處理後文字
    print(r.province, r.city)      # 廣東省 深圳市（僅內地/粵Z 牌）
    print(r.energy, r.energy_note) # D 純電動 / F 非純電動（僅新能源牌）
    print(r.confidence)            # 識別置信度
    print(r.is_valid, r.validation_note)   # 格式校驗結果
    print(r.detection.corners)     # 4 角點
    d = r.to_dict()                # 可直接餵給 JSON API
```

### 省 · 市 · 能源解析

內地車牌第 2 位字母是發牌機關代碼，對應地級市；新能源牌另有能源標識。

| 車牌 | 省 | 地級市 | 能源 |
|------|-----|--------|------|
| 粤A12345 | 廣東省 | 廣州市 | — |
| 粤B12345 | 廣東省 | 深圳市 | — |
| 粤S12345 | 廣東省 | 東莞市 | — |
| 粤Z1234港 | 廣東省 | 港澳入出內地車輛 | —（香港入出）|
| 沪AF71017 | 上海市 | 上海市區 | F · 非純電動（混動/燃料電池）|
| 京AD12345 | 北京市 | 北京市區 | D · 純電動 |

能源代碼規則（92 式新能源號牌 8 位）：
- **小型車**：能源字母在第 3 位；**大型車**：在末位
- `D`/`A`/`B`/`C`/`E`/`P`/`Q`/`R`/`S`/`T` → 純電動（D 為主，號段用盡後依序啟用其餘）
- `F`/`G`/`H`/`J`/`K`/`L`/`M`/`N` → 非純電動（插電混動、燃料電池）

覆蓋範圍：31 個省級行政區 + 港澳，共 382 個發牌機關代碼。

```python
from gba_plate.postprocess.region_map import describe_plate, resolve_energy
describe_plate("沪AF71017")   # '上海市 · 上海市區 · 非純電動（混動/燃料電池）'
resolve_energy("粤BD12345")   # ('D', '純電動')
```

### 擴展方式

想換檢測器或識別器，只需實作兩個方法：

```python
from gba_plate import BaseDetector, BaseOCR, Detection, OcrResult

class MyDetector(BaseDetector):
    class_mode = "region"
    def predict(self, image):
        return [Detection(bbox=(x1,y1,x2,y2), corners=..., confidence=0.9, region=Region.HK)]

class MyOCR(BaseOCR):
    def predict(self, plate_crop):
        return OcrResult(text="AB1234", confidence=0.95)

alpr = ALPR(detector=MyDetector(), ocr=MyOCR())
```

### 測試

```bash
python -m pytest tests/ -q          # 76 個測試，<1 秒
python scripts/verify_refactor.py   # 端到端驗證（需模型）
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

---

## 📚 文檔索引

| 文檔 | 內容 |
|------|------|
| `docs/OPENSOURCE_RESEARCH.md` | 開源方案調研報告（數據集 / 代碼 / 論文）|
| `docs/ARCHITECTURE_COMPARISON.md` | 架構對比與融合方案（第一輪，5 個 repo）|
| `docs/OPENSOURCE_SEARCH_ROUND2.md` | 全量搜索與架構對比（第二輪，265 repo）|
| `docs/OPENSOURCE_SEARCH_ROUND3.md` | 3 個待評估項目深挖（第三輪，最新調研）|
| `docs/REFACTOR_NOTES.md` | **重構實作筆記（分層架構、79 類字符、校驗規則）** |
| `docs/BASELINE_TEST_REPORT.md` | 三地牌基準測試（11 張實拍）|
| `docs/TEST_REPORT.md` | CCPD 樣本測試 |
| `docs/DOWNLOAD_GUIDE.md` | 數據集下載指引 |
| `docs/README_UPSTREAM.md` | 原始項目 README（we0091234）|

### 待下載的港澳真實數據

Roboflow 上有 4 個 CC BY 4.0 港澳車牌數據集（約 2,900 張），需免費 API key：

```bash
# 1. 取得 key: https://app.roboflow.com/settings/api
export ROBOFLOW_API_KEY=rf_xxxxx

# 2. 一條命令下載全部
python scripts/download_hk_mo_datasets.py --output test_data/real
```