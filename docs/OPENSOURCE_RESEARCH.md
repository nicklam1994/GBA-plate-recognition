# 粵港澳大灣區車牌識別 — 開源方案調研報告

**調研日期**：2026-10-29  
**調研範圍**：GitHub、Roboflow Universe、學術論文、商業方案  
**目標**：內地 / 粵Z 兩地牌 / 香港 / 澳門 四地車牌全覆盖

---

## 一、核心結論

| 問題 | 結論 |
|------|------|
| 有沒有一個現成項目能直接做四地？ | **沒有**。pcb9382 宣稱支援港澳台日韓美歐，但權重不公開 |
| 港澳真實數據有嗎？ | **有**。Roboflow 上有 3 個 CC BY 4.0 的港澳數據集，合計約 2,900 張 |
| 港澳車牌識別難在哪？ | 不是字體，是**訓練樣本**。字符集（港/澳）和網絡結構都夠用 |
| 最省力的路線？ | 用 we0091234 做骨幹 → 補 Roboflow 港澳真實數據 → 微調 + 分地區後處理 |

---

## 二、數據集調研（關鍵發現）

### ✅ 可直接獲取的港澳真實數據

| # | 數據集 | 圖片數 | 類別 | 許可 | 價值 |
|---|--------|--------|------|------|------|
| 1 | **V2 Merged Macau License Plate**<br>`fyp-dmovy/v2-merged-macau-license-plate` | **2,495** | `GDM`, `HKF`, `HKR`, `MEV`, `MO` | CC BY 4.0 | ⭐⭐⭐⭐⭐ |
| 2 | **alan carplate**<br>`hk-license-plates/alan-carplate` | 134 | `china_carplate`, `hk_carplate`, `hk-car-license-plate`, `macau_carplate`, `carplate` | CC BY 4.0 | ⭐⭐⭐⭐⭐ |
| 3 | Macau_License_Plate_ParkingLot<br>`macaocarplaterecognitiontraining/macau_license_plate_parkinglot` | 239 | `MO` | CC BY 4.0 | ⭐⭐⭐⭐ |
| 4 | HK License Plate Recognition<br>`hong-kong-license-plate-detect/hk-license-plate-recognition` | 33 | `License-Plate` | — | ⭐⭐ |
| 5 | License Plate Recognition (HK workspace)<br>`hong-kong/license-plate-recognition-rxg4e-7lvtq` | 10,125 | 需核實是否含港澳牌 | — | ⭐⭐⭐（待核實）|

#### 為何 #1 與 #2 特別有價值

**#1 `GDM / HKF / HKR / MEV / MO` 的推測含義**
- `HKF` = Hong Kong Front（香港前牌，白底）
- `HKR` = Hong Kong Rear（香港後牌，黃底）
- `GDM` = Guangdong–Macau（粵Z 澳牌）
- `MEV` = Macau EV / Macau 特定類
- `MO` = Macau 本地牌

**這正是本項目需要的分類粒度** — 香港前後牌分開，粵澳牌獨立成類。

**#2 `alan carplate` 的價值**
類別同時包含 `china_carplate` + `hk_carplate` + `macau_carplate`，
即**一車掛三牌的實拍場景標註**，與我們 `test_data/real/three_region/` 的場景完全一致。

### ✅ 已有（內地）

| 數據集 | 數量 | 狀態 |
|--------|------|------|
| CCPD2019 | 355,015 | ✅ 已下載解壓，9 個子集 |
| CBLPRD-330k | 330,000 | 未下載（GAN 生成，非純真實）|

### ❌ 找不到的

- **香港本地牌大規模真實數據集**（公開的只有幾百張量級）
- 粵Z 牌獨立數據集（混在港澳數據集內）
- 港澳車牌的**字符級轉錄**（多數只有檢測框，無車牌文字標註）

---

## 三、代碼方案調研

### 第一梯隊

#### 1. we0091234/Chinese_license_plate_detection_recognition ⭐1.9k（當前基礎）

| 項目 | 內容 |
|------|------|
| 架構 | YOLOv5s 檢測 + CRNN(CTC) 識別 |
| 字符集 | 78 類（含港、澳） |
| 支援 | 12 種中文車牌，含港澳粵Z |
| 權重 | ✅ 完全公開（檢測 1.1MB + 識別 733KB） |
| 訓練代碼 | ✅ 完整 |
| 許可 | GPL-3.0 |
| 衍生 | yolov7_plate / yolov8-plate / yolo26-plate |

**實測**：預訓練模型在 11 張三地牌實拍上整體檢測率 64%（粵Z 71% / 香港 50% / 澳門 0%）。
結論：**骨幹可用，缺港澳訓練數據**。

#### 2. pcb9382/PlateRecognition ⭐582

| 項目 | 內容 |
|------|------|
| 架構 | YOLOv5/v7/v8 檢測 + **PP-OCRv5 / PP-OCRv6 / LPRNet 三後端可切換** |
| 平台 | Linux/Win/Android/iOS/**HarmonyOS**/RK3588/海思 |
| 性能 | RTX3090 450μs；驍龍865 CPU 70ms；精度宣稱 99%+ |
| 權重 | ❌ **不公開**（微信索取，數據 30 元） |
| 許可 | MIT（僅代碼） |

**參考價值**：多後端切換架構 + C++/TensorRT 部署，是邊緣部署的最佳參考。
**不可用作基礎**：沒有權重，無法微調。

#### 3. zhahoi/PlateRecognition_ncnn ⭐51

pcb9382 的 **NCNN 移植版**，MIT 許可，README 技術文檔異常詳盡：

| 技術點 | 值 |
|--------|-----|
| 檢測輸入 | 640×640 letterbox（填充值 114）|
| 檢測輸出 | [25200, 15]：bbox(4) + obj(1) + 4 關鍵點(8) + 類別(2) |
| 類別 | 0=單層, 1=雙層 |
| 識別輸入 | 168×48，mean=149.94, norm=1/(0.193×255) |
| 識別輸出 | [21, 78] 字符 + [5] 顏色 |
| 雙層處理 | 上層取前 5/12 高度，下層取後 2/3，resize 後水平拼接 |
| 顏色定義 | 0=黑（使領館）1=藍（小型車）2=綠（新能源）3=白（軍警）4=黃（大型車）|

**參考價值**：️⃣ 雙層車牌分割的具體比例（5/12 + 2/3）、正規化參數、CTC 解碼細節，可直接對照我們實作。

### 第二梯隊

| 項目 | ⭐ | 價值 |
|------|-----|------|
| **ankandrew/fast-alpr** | — | 完整 ALPR 框架（檢測+OCR 一體），模組化好 |
| **ankandrew/fast-plate-ocr** | 780 | 65 國 OCR 模型，22 萬樣本，CCT 架構，0.67ms/plate |
| **HyperLPR3** | 6.3k | 中文車牌框架，港澳支援有限（官方明言樣本不均衡） |
| **morsetechlab/yolov11-license-plate-detection** | HF 45👍 | YOLOv11 車牌檢測，但數據集有 train/test 污染 |
| EasyPR | 6k | 傳統 CV 方案，已過時 |

### 學術前沿

| 論文 | 方案 | 亮點 |
|------|------|------|
| **Light-Edge** (J Real-Time Image Proc 2025) | ResNet18+FPN 單階段 + CTC | 38MB，Jetson Nano **14 FPS @ 4.8W**，mAP 90.2% |
| CSCM-YOLOv8 + CSM-LPRNet (PLoS One 2026) | 兩階段改進 | 複雜場景（低光/模糊/遮擋） |
| A Single Neural Network for Mixed Style LP (2023) | ALPRNet | **HZM 數據集**（港珠澳大橋實拍 1,376 張，三地風格）— 私有 |

**Light-Edge 的啟示**：如果最終要上邊緣設備，單階段 + CTC 比兩階段更省資源。但訓練門檻高。

### 商業方案（可作對標）

| 方案 | 說明 |
|------|------|
| **JIESHUN 捷順** | 已落地港澳車牌識別停車場方案，支援多牌同時識別 |
| **Plate Recognizer** | 提供香港/中國 ALPR API |

---

## 四、港澳車牌規格核實（用於後處理規則）

### 香港

| 項目 | 規格 |
|------|------|
| 前牌 | 白底黑字，520×110mm |
| 後牌 | 黃底黑字，520×110mm |
| 格式 | 1-2 字母 + 1-4 數字（`AB 1234`、`X 567`）|
| **禁用字母** | `I`、`O`、`Q` |
| **數字不以 0 開頭** | `01`、`002` 非法 |
| 字體 | Charles Wright（BS AU 145a）|
| 特殊前綴 | `AM` 政府、`LC` 立法會、`ZG` 駐港解放軍、`VV` 鄉村車、`T` 臨時牌（白底紅字）|
| 雙行 | 少數車型用方形雙行（上字母/下數字）|

### 澳門

| 項目 | 規格 |
|------|------|
| 顏色 | 民用黑底白字；免稅車黑底黃字 |
| 尺寸 | 460×100mm（歐式單行）；方形（美/日式雙行）|
| 格式 | `M` + 1 字母 + `-` + 2 數字 + `-` + 2 數字 |
| 示例 | `MA-12-34`、`MG-99-88`、`MW-00-01` |
| 字母序列 | MA → MB → ... → MZ（已到 MZ 附近）|
| 雙行 | `MA` / `12-34`（電單車等空間受限車型）|

### 粵Z 兩地牌

| 項目 | 規格 |
|------|------|
| 顏色 | 黑底白字，白框線 |
| 尺寸 | 440×140mm |
| 格式 | `粵Z` + 4 位（數字或 3 位 + 1 字母）+ `港`/`澳` |
| 示例 | `粵Z·1234港`、`粵Z·AB12澳` |
| 特點 | 與港澳本地牌**同時懸掛**，一車多牌 |

### 後處理規則建議

```python
HK_PATTERN   = r'^[A-HJ-NP-Z]{0,2}\s?[1-9]\d{0,3}$'   # 排除 I/O/Q，禁 0 開頭
MO_PATTERN   = r'^M[A-Z]{1}-?\d{2}-?\d{2}$'
GANGAO_PATTERN = r'^粵Z-?\d{4}[港澳]$|^粵Z-?[A-Z0-9]{2,4}[港澳]$'
```

---

## 五、推薦路線

### 方案：we0091234 骨幹 + Roboflow 港澳數據微調

```
階段 1（數據）
  ├─ 已有：CCPD 355k（內地）
  ├─ 下載：Roboflow 港澳 3 個數據集（約 2,870 張）
  │        ├─ V2 Merged Macau（2,495）← HKF/HKR/GDM/MO 分類現成
  │        ├─ alan carplate（134）← 三地牌同車
  │        └─ Macau ParkingLot（239）
  └─ 自採：三地牌實拍 11 張（已有）+ 持續補充

階段 2（檢測模型）
  └─ 改 4 類別輸出：mainland / gangao / hk / mo
     用 CCPD + Roboflow 港澳混合訓練
     → 解決「澳門牌 0% 檢測率」的核心問題

階段 3（識別模型）
  ├─ 現有 78 類字符集已含港澳字符，無需擴充
  ├─ 需解決：香港純英數組合、澳門連字符
  └─ 策略：用港澳真實數據微調識別頭

階段 4（後處理）
  ├─ 分地區格式校驗（見上）
  ├─ 易混淆字符糾正（0↔O, 1↔I, 5↔S, 8↔B）
  └─ 一車多牌的配對邏輯（粵Z + 港澳本地牌同時出現）

階段 5（合成數據，有真實基準後）
  └─ 用真實數據校準生成器參數，補長尾場景
```

### 為何不換基礎項目

| 考量 | we0091234 | pcb9382 | fast-alpr |
|------|-----------|---------|-----------|
| 權重公開 | ✅ | ❌ | ✅ |
| 可微調 | ✅ | ❌ | ✅ |
| 中文車牌優化 | ✅ | ✅ | ❌ |
| 78 類字符集現成 | ✅ | ✅ | ❌ |
| 改造成本 | 低 | 高 | 高 |

---

## 六、下一步（需要人工介入）

### 需要你做

1. **註冊 Roboflow 免費帳號**，取得 API key
   - 地址：https://app.roboflow.com/settings/api
   - 免費額度足夠下載公開數據集

2. **提供 API key** 給我，我一條命令下載全部港澳數據集

### 我來做（拿到 key 後）

```bash
pip install roboflow
export ROBOFLOW_API_KEY=rf_xxxxx

# 三個港澳數據集
roboflow download -f yolov8 fyp-dmovy/v2-merged-macau-license-plate/<version>
roboflow download -f yolov8 hk-license-plates/alan-carplate/<version>
roboflow download -f yolov8 macaocarplaterecognitiontraining/macau_license_plate_parkinglot/<version>
```

版本號需先用 API 查詢確認。

---

## 附錄：完整資源清單

### 數據集

| 名稱 | URL |
|------|-----|
| V2 Merged Macau | https://universe.roboflow.com/fyp-dmovy/v2-merged-macau-license-plate |
| alan carplate | https://universe.roboflow.com/hk-license-plates/alan-carplate |
| Macau ParkingLot | https://universe.roboflow.com/macaocarplaterecognitiontraining/macau_license_plate_parkinglot |
| HK LP Recognition | https://universe.roboflow.com/hong-kong-license-plate-detect/hk-license-plate-recognition |
| HK workspace LP | https://universe.roboflow.com/hong-kong/license-plate-recognition-rxg4e-7lvtq |
| CCPD | https://github.com/detectRecog/CCPD |
| CBLPRD-330k | https://github.com/SunlifeV/CBLPRD-330k |

### 代碼

| 名稱 | URL | 許可 |
|------|-----|------|
| we0091234 | https://github.com/we0091234/Chinese_license_plate_detection_recognition | GPL-3.0 |
| pcb9382 | https://github.com/pcb9382/PlateRecognition | MIT |
| zhahoi NCNN 移植 | https://github.com/zhahoi/PlateRecognition_ncnn | MIT |
| fast-alpr | https://github.com/ankandrew/fast-alpr | MIT |
| fast-plate-ocr | https://github.com/ankandrew/fast-plate-ocr | MIT |
| HyperLPR | https://github.com/szad670401/HyperLPR | Apache-2.0 |
| Pytorch-LPRNet | https://github.com/sirius-ai/LPRNet_Pytorch | Apache-2.0 |

### 論文

| 標題 | 連結 |
|------|------|
| Light-Edge (2025) | https://link.springer.com/article/10.1007/s11554-025-01738-3 |
| CSCM-YOLOv8 + CSM-LPRNet (2026) | https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0339649 |
| HZM 三地牌數據集 (2023) | 私有，需聯繫作者 |

---

**撰寫**：2026-10-29  
**維護者**：Nick（尼克🦊）
