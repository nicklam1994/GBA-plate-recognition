# 架構對比與融合方案

**日期**：2026-10-29  
**方法**：clone 各項目 → codegraph 索引 → 讀核心代碼 → 實測 / 交叉驗證  
**原則**：不沿用先前調研結論，以代碼事實為依據

---

## 一、當前項目架構診斷

**GBA-plate-recognition**（基於 we0091234）實測結構：

| 文件 | 行數 | 職責 |
|------|------|------|
| `detect_plate.py` | 356 | 模型加載 + 預處理 + NMS + 透視變換 + 雙層處理 + 識別 + 繪圖 + CLI |
| `train.py` | 602 | 訓練 |
| `web_ui.py` | 259 | **重複實現一套推理**（與 detect_plate.py 邏輯重複）|
| `plate_recognition/plate_rec.py` | 119 | 識別 + CTC 解碼（字典硬編碼 L24）|
| `plate_recognition/plateNet.py` | 202 | 網絡結構 |
| `plate_recognition/double_plate_split_merge.py` | 15 | 雙行拼接 |

### 已識別的問題

| # | 問題 | 證據 |
|---|------|------|
| 1 | **無抽象層** | 不存在 BaseDetector / BaseOCR 接口，換模型要改多處 |
| 2 | **推理邏輯重複** | `detect_plate.py` 與 `web_ui.py` 各有一套 NMS + 識別流程 |
| 3 | **檢測類別不含地區** | `detect_plate.py:102` → `class_num` 只表示「單層/雙層」，非地區 |
| 4 | **字符集硬編碼** | `plate_rec.py:24` 的 `plateName` 字串直接寫死 |
| 5 | **無地區輸出** | 結果只有 plate + color + confidence，缺 region |
| 6 | **單體函數過長** | `detect_Recognition_plate()` 承擔 7 種職責 |

**結論**：可作訓練/推理基礎，但**架構不適合擴展到四地分類**，需重構分層。

---

## 二、候選項目實證對比

### 對比總表

| 維度 | we0091234（當前）| fast-alpr | pcb9382 | zhahoi-ncnn |
|------|------------------|-----------|---------|-------------|
| **架構分層** | ❌ 單體腳本 | ✅ ABC 抽象（40 行 base.py）| ✅ 類封裝 | ✅ 類封裝 |
| **檢測/識別解耦** | ❌ 硬耦合 | ✅ 完全可插拔 | ✅ 可切換 | ✅ 可切換 |
| **推理後端** | ⚠️ ONNX/TRT 各自腳本 | ✅ ort 6 種 provider | ✅ TRT8.2 / NCNN / CoreML | ✅ NCNN |
| **地區分類輸出** | ❌ 無 | ✅ `OcrResult.region` | ❌ 無 | ❌ 無 |
| **中文車牌字符集** | ✅ 78 類含港澳 | ❌ 需自配 | ✅ 78 類 | ✅ 78 類 |
| **雙行車牌** | ✅ 5/12 + 1/3 | ❌ | ✅ 5/12 + 1/3 | ✅ 5/12 + 1/3 |
| **權重公開** | ✅ 1.9MB | ✅ ONNX | ❌ 微信索取 | ❌ |
| **訓練代碼** | ✅ 完整 | ⚠️ 識別可訓練 | ❌ | ❌ |
| **單元測試** | ❌ | ✅ pytest | ❌ | ❌ |
| **文檔** | ⚠️ 簡略 | ✅ mkdocs 完整 | ⚠️ | ✅ 13KB 詳盡 |

### 各自的核心精華

#### fast-alpr：抽象設計（最值得抄）

```python
# base.py — 僅 40 行，但把擴展點定死了
@dataclass(frozen=True)
class OcrResult:
    text: str
    confidence: float | list[float]
    region: str | None = None          # ← 地區分類，正是我們要的
    region_confidence: float | None = None

class BaseDetector(ABC):
    @abstractmethod
    def predict(self, frame) -> list[DetectionResult]: ...

class BaseOCR(ABC):
    @abstractmethod
    def predict(self, cropped_plate) -> OcrResult | None: ...
```

`ALPR` 類只做一件事：組合 detector + ocr。所有模型細節都在子類。

**可借鑒**：`region` / `region_confidence` 欄位設計 + ABC 接口。

#### zhahoi-ncnn / pcb9382：雙行處理（同一份邏輯）

```cpp
// plate_detection_recognition.cpp:218
upper = img[0 : 5/12*h, :]              // 上層取前 41.7%
lower = img[1/3*h : h, :]               // 下層從 33.3% 開始
upper = resize(upper, lower.size())     // 上層縮放到與下層同高
out = hstack([upper, lower])            // 水平拼接 → 交給單行 CRNN
```

**關鍵洞察**：不是「分成兩張圖分別識別」，而是**拼成一張寬圖給單行 CRNN**。
`5/12`(41.7%) 與 `1/3`(33.3%) 刻意重疊 8.3%，避免字符跨行時被切斷。

Python 版（we0091234）邏輯完全相同，已驗證一致。

#### pcb9382：多後端切換架構

```cpp
enum class RecognitionBackend { PPOCRV5, PPOCRV6, LPRNet };
// 運行時可選，共用同一份 NCC/前處理
```

**可借鑒**：識別器不是寫死，而是枚舉 + 工廠。

#### we0091234：78 類字符集（不可替代的資產）

```
#京沪津...粤...学警港澳挂使领民航危0123456789ABCDEFGHJKLMNPQRSTUVWXYZ险品
```

- 33 個中文（含港、澳） + 24 字母（去 I/O） + 10 數字 + blanks
- 權重僅 0.18M 參數，CPU 單張 ~30ms

**這是唯一同時具備「中文字符 + 港澳字符 + 公開權重」的現成方案**。

---

## 三、2026 年新選項評估

### PP-OCRv6（PaddleOCR 3.7.0，2026-06）

| 項目 | 數值 |
|------|------|
| 參數 | 34.5M |
| 語言 | 50 種單模型（中英日 + 46 拉丁）|
| 對比 | 比 PP-OCRv5_server 檢測 +4.6%、識別 +5.1%；聲稱超越 Qwen3-VL-235B / GPT-5.5 |
| 許可 | Apache-2.0 |

**評估**：通用 OCR 強，但**不適合直接做車牌識別**：
- 34.5M 參數 vs 我們 0.18M → 200 倍體積，延遲不可接受
- 通用字典（數千字）vs 車牌 78 字符 → 冗餘
- 不理解車牌格式約束（如香港禁 I/O/Q）

**結論**：可作為**實驗對照**或**極端場景 fallback**，不作主線。

### fast-plate-ocr（CCT 架構）

- 65 國模型，region recognition 內建
- 但字符集無中文，港澳車牌無法直接用
- 訓練框架完整，可自定義 `plate_config.plate_regions = [mainland, gangao, hk, mo]`

**結論**：訓練框架設計值得參考（region 標籤與識別聯合訓練），模型不適用。

### Light-Edge（學術，2025）

- ResNet18 + FPN 單階段 + CTC，38MB，Jetson Nano 14 FPS @ 4.8W
- 單階段端到端，省掉兩階段的中間開銷

**結論**：若最終要走邊緣設備，這是架構方向。但訓練門檻高（需自建數據管線），先不採用。

---

## 四、融合方案（取各家精華）

```
┌──────────────────────────────────────────────────────────┐
│  架構層：fast-alpr 式 ABC 抽象                            │
│  ├─ BaseDetector.predict(frame) -> list[Detection]       │
│  ├─ BaseOCR.predict(crop) -> OcrResult(text, region, ...)│
│  └─ ALPR(detector, ocr).process(image) -> list[ALPRResult]│
└──────────────────────────────────────────────────────────┘
                          ↓
┌──────────────────────────────────────────────────────────┐
│  檢測層：we0091234 權重 + 改 4 類別                        │
│  classes: 0=mainland 1=gangao 2=hk 3=mo                   │
│  保留 4 角點回歸（透視校正用）                             │
│  訓練數據：CCPD 355k + 港澳 6.3k                          │
└──────────────────────────────────────────────────────────┘
                          ↓
┌──────────────────────────────────────────────────────────┐
│  預處理層：                                            │
│  ├─ 透視變換（4 角點 → 正矩形）                          │
│  └─ 雙行檢測 → 5/12+1/3 hstack 拼接（zhahoi/pcb 同源）    │
└──────────────────────────────────────────────────────────┘
                          ↓
┌──────────────────────────────────────────────────────────┐
│  識別層：78 類 CRNN（we0091234 權重，0.18M）              │
│  ├─ 字符集：現有 + 「-」（澳門連字符）                     │
│  ├─ 可切換後端（pcb9382 式枚舉）：CRNN | PP-OCRv6 | LPRNet│
│  └─ region 由檢測類別直接得出（比 OCR 猜更可靠）           │
└──────────────────────────────────────────────────────────┘
                          ↓
┌──────────────────────────────────────────────────────────┐
│  後處理層：規則引擎（新增）                                │
│  ├─ 香港：^[A-HJ-NP-Z]{0,2}\\s?[1-9]\\d{0,3}$（禁 I/O/Q，禁 0 開頭）│
│  ├─ 澳門：^M[A-Z]-?\\d{2}-?\\d{2}$                        │
│  ├─ 粵Z：^粵Z-?\\d{4}[港澳]$                              │
│  └─ 易混淆字符糾正（0↔O、1↔I、5↔S、8↔B）                 │
└──────────────────────────────────────────────────────────┘
                          ↓
┌──────────────────────────────────────────────────────────┐
│  部署層：pcb9382 式多後端                                 │
│  ONNX → TensorRT(GPU) / NCNN(ARM) / OpenVINO(Intel)       │
└──────────────────────────────────────────────────────────┘
```

### 各層來源對照

| 層 | 來源 | 理由 |
|----|------|------|
| 抽象接口 | fast-alpr | 40 行定死擴展點，最少代碼最大靈活性 |
| 檢測權重 | we0091234 | 唯一公開 + 可微調 + 4 角點 |
| 雙行處理 | zhahoi-ncnn / pcb9382 | 5/12+1/3 拼接法，經兩項目驗證 |
| 識別模型 | we0091234 | 78 類含港澳，0.18M 極輕量 |
| 多後端枚舉 | pcb9382 | 運行時切換識別器 |
| region 輸出 | fast-alpr 概念 + 檢測類別 | 檢測已分類，無需 OCR 再猜 |
| 後處理規則 | 自行設計 | 無人做過港澳格式校驗 |
| 部署 | pcb9382 | TRT/NCNN/CoreML 全平台 |

---

## 五、具體重構計劃

### 目標結構

```
GBA-plate-recognition/
├── gba_plate/
│   ├── base.py              # ABC: BaseDetector, BaseOCR, 結果 dataclass
│   ├── alpr.py              # ALPR 組合類
│   ├── detectors/
│   │   └── yolo_plate.py    # we0091234 權重 + 4 類別
│   ├── ocr/
│   │   ├── crnn78.py        # 78 類 CRNN（默認）
│   │   └── ppocr.py         # PP-OCRv6（可選後端）
│   ├── preprocess/
│   │   ├── perspective.py   # 4 角點透視變換
│   │   └── double_line.py   # 5/12+1/3 拼接
│   └── postprocess/
│       └── rules.py         # 四地格式校驗 + 字符糾正
├── scripts/                  # 訓練/數據處理
├── tests/                    # pytest
├── docs/
└── web_ui.py                 # 只調用 gba_plate.ALPR
```

### 分階段

| 階段 | 內容 | 產出 |
|------|------|------|
| 1 | 建 `gba_plate/base.py` + `alpr.py`（抄 fast-alpr 抽象）| 可插拔骨架 |
| 2 | 把現有 detect_plate.py 邏輯搬進 `detectors/` + `ocr/` | 功能等價，結構解耦 |
| 3 | 加 `preprocess/double_line.py` + `postprocess/rules.py` | 雙行 + 港澳格式校驗 |
| 4 | 用 CCPD+港澳數據重訓檢測為 4 類別 | 澳門牌可檢測 |
| 5 | 參考 pcb9382 加多後端（ONNX/TensorRT/NCNN）| 可部署 |

---

## 六、與先前調研結論的差異

| 先前結論（不可信）| 本次實證 |
|------------------|---------|
| we0091234 是最佳底子 | **部分正確**：權重/字符集是最佳，但**架構最差**，必須重構 |
| pcb9382 功能最全不採用 | **代碼架構確實最好**（多後端、類封裝），只是權重不公開。架構要抄 |
| fast-alpr 只是「參考」 | **抽象設計應直接採用**，這是整個重構的地基 |
| 需要重新設計字典 | **不需要**：78 類已含港澳，只需加「-」 |
| PP-OCRv4/v5 可作識別引擎 | **不建議**：34.5M 參數對車牌過重，專用 78 類 CRNN 更優 |

---

**撰寫**：2026-10-29  
**方法**：代碼實證，非二手資料轉述
