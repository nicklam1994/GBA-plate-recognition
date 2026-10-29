# 重構實作筆記

**日期**：2026-10-29  
**範圍**：把單體腳本改為 `gba_plate` 分層包（階段 1-2）

---

## 一、為何重構

三輪調研診斷出的問題（`docs/OPENSOURCE_SEARCH_ROUND*.md`）：

| 問題 | 證據 |
|------|------|
| 推理邏輯重複 | `detect_plate.py` 356 行與 `web_ui.py` 259 行各有一套 NMS + 識別 |
| 無抽象層 | 無 BaseDetector / BaseOCR，換模型要改多處 |
| 檢測類別不含地區 | `class_num` 只表示單層/雙層 |
| 字符集硬編碼 | `plate_rec.py:24` 字串直接寫死 |
| 單函數過長 | `detect_Recognition_plate()` 擔 7 種職責 |

---

## 二、新架構

```
gba_plate/
├── base.py          183 行   ABC 抽象 + 4 個結果 dataclass
├── alpr.py          179 行   組合類（唯一對外入口）
├── detectors/
│   └── yolo_plate.py  165 行  YOLOv5 檢測器（雙類別語義）
├── preprocess/
│   ├── perspective.py  88 行  角點排序 + 透視校正
│   └── double_line.py  65 行  5/12 + 1/3 拼接
├── ocr/
│   ├── vocab.py        77 行  78/79 類字符表
│   └── crnn.py        180 行  CRNN + CTC
└── postprocess/
    └── rules.py       237 行  四地格式校驗 + 糾正 + 推斷
```

`web_ui.py`：259 → 218 行，移除自己實作的推理，只調 `ALPR`。

---

## 三、關鍵設計決策

### 1. 為何選 fast-alpr 式 ABC（而非 pcb9382 式類封裝）

fast-alpr 的 `base.py` 只有 40 行就把擴展點定死了：

```python
@dataclass(frozen=True)
class OcrResult:
    text: str
    confidence: float
    region: str | None = None          # ← 地區欄位，其他項目都沒有
    region_confidence: float | None = None

class BaseDetector(ABC):
    @abstractmethod
    def predict(self, frame) -> list[DetectionResult]: ...

class BaseOCR(ABC):
    @abstractmethod
    def predict(self, cropped_plate) -> OcrResult | None: ...
```

`ALPR` 類只做組合。這比 pcb9382 的 C++ 類封裝更適合 Python，且強迫分離關注點。

### 2. 為何檢測器要支援兩種 class_mode

現有權重 `plate_detect.pt` 的 class_id 是 **0=單層 / 1=雙層**，不是地區。
未來重訓的權重是 **0=mainland / 1=gangao / 2=hk / 3=mo**。

兩種語義共存的做法：

```python
class_mode = "single_double"   # 舊權重：class_id → is_double，region=UNKNOWN
class_mode = "region"          # 新權重：class_id → region，is_double 靠寬高比判斷
```

好處：新權重訓好後**只需換檔 + 改一個參數**，調用方代碼不動。

### 3. 為何 79 類而不是重新設計字典

| 方案 | 問題 |
|------|------|
| 重新設計字典 | 現有 78 類權重報廢，需從零訓練 |
| 78 類 + `-` = 79 類 | 只需微調最後一層，其餘權重可複用 |

`-` 來自 HuKai97/YOLOv5-LPRNet 的字符集（它的 68 類含 `-` 但缺港澳）。
兩者互補，融合後 79 類覆蓋四地全部格式。

**實現細節**：`crnn.py` 從權重的 `newCnn.weight.shape[0]` 自動讀類別數，
所以 78 類權重和 79 類權重都能載入，不需改代碼。

### 4. 為何地區推斷要「粵Z 優先」

`detect_region_from_text` 的判斷順序至關重要：

```python
# ❌ 錯誤順序：粵Z 會被省份規則吃掉
if is_province(text[0]):      # 粤 是省份 → MAINLAND
    return Region.MAINLAND
if text.startswith("粤Z"):    # 永遠到不了
    return Region.GANGAO

# ✅ 正確順序
if text.startswith("粤Z"):    # 先判斷粵Z
    return Region.GANGAO
if is_province(text[0]):
    return Region.MAINLAND
```

**踩過的坑**：第一版把「粵Z」寫成繁體「粵」(U+7CB5)，但字符表裡是簡體「粤」(U+7CA4)，
導致 `startswith` 永遠失敗，所有粵Z 牌被誤判為 mainland。
現在 `normalize()` 會強制把繁體「粵」轉成簡體。

### 5. 內地格式校驗的三種規則

不能只用一條 `省+字母+5~6位`：

```python
MAINLAND_RE          = 省 + 字母 + 5 位                    # 普通牌
MAINLAND_NE_SMALL_RE = 省 + 字母 + D/F + 5 位              # 小型新能源（D/F 在第 3 位）
MAINLAND_NE_LARGE_RE = 省 + 字母 + 5 位 + D/F              # 大型新能源（D/F 在末位）
MAINLAND_SPECIAL_RE  = 省 + 字母 + 數字 + [警学领使挂港澳试超]  # 特殊牌
```

如果只用寬鬆的 `[A-Z0-9]{5,6}`，像「京AB12345」（B 不是 D/F）這種非法組合也會通過。
測試 `test_mainland[京AB12345-False]` 就是防止這個回歸。

---

## 四、驗證結果

### 單元測試

```
76 passed in 0.96s
```

覆蓋：字符表（78/79 一致性、港澳字符、無 I/O）、CTC 解碼（blank/折疊/分隔重複/越界）、
四地格式校驗（含邊界案例）、字符糾正、地區推斷、透視變換、雙行拼接。

### 端到端驗證

`scripts/verify_refactor.py` 對 6 張測試圖：

```
hongkang1.jpg      gangao    粤ZR066港      1.00 黑  OK  ok
single_blue.jpg    mainland  辽DU4356       1.00 藍  OK  ok
single_blue.jpg    mainland  浙B2V9L7       1.00 藍  OK  ok
single_green.jpg   mainland  沪AF71017      1.00 綠  OK  ok(新能源小型)
double_yellow.jpg  mainland  京EA5331       1.00 黃  OK  ok
police.jpg         mainland  鄂F8888警      1.00 白  OK  ok(特殊後綴)
xue.jpg            mainland  琼A6716学      1.00 黃  OK  ok(特殊後綴)

共 7 個車牌，格式校驗通過 7 個 (100%)
地區分佈: {'gangao': 1, 'mainland': 6}
```

識別文字與舊管線完全一致（功能等價），新增了地區判定與格式校驗。

### Web UI

HTTP 200，`/` 正常返回；`alpr.process()` → `draw_results()` → base64 往返驗證通過。

---

## 五、待辦（階段 3+）

| 階段 | 內容 | 前置條件 |
|------|------|---------|
| 3 | 用 CCPD + 港澳 6.3k 重訓檢測為 4 類別 | 已具備數據 |
| 4 | 識別模型擴展到 79 類（加 `-`） | 需造帶連字符的澳門訓練樣本 |
| 5 | 參考 pcb9382 加多後端（ONNX/TensorRT/NCNN） | 階段 3 完成後 |
| 6 | 可選：加入 STN 對齊層（55K 參數） | 階段 4 完成後 |

---

**撰寫**：2026-10-29
