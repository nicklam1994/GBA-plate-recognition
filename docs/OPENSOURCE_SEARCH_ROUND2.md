# 開源項目全量搜索與架構對比（第二輪）

**日期**：2026-10-29  
**背景**：第一輪報告只對比了先前 session 已 clone 的 5 個項目，覆蓋不全。本輪用 GitHub API 做系統性全量搜索重做。

---

## 一、搜索方法（可複現）

```bash
# 16 個查詢，每個取 top 30-40，按 star 排序
gh search repos "<query>" --sort stars --limit 40 \
  --json fullName,stargazersCount,updatedAt,description
gh api -X GET search/repositories \
  -f q="<query>" -f sort=stars -f per_page=30
```

**查詢詞**：`license plate recognition` / `license plate detection` / `ALPR` / `车牌识别` / `車牌識別` / `hong kong license plate` / `macau license plate` / `license plate ocr` / `license plate ANPR` / `plate detection onnx` / `plate recognition tensorrt` / `awesome license plate` / `车牌 检测 识别 yolo` / `hongkong carplate`

**結果**：去重後 **265 個 repo**（第一輪只看了 5 個，覆蓋率 <2%）

### 搜索的局限（誠實說明）

| 局限 | 影響 |
|------|------|
| `macau license plate` 返回 0 結果 | GitHub 上幾乎沒有澳門專門項目 |
| `hongkong carplate` 0 結果 | 同上，港澳是空白區 |
| 中文查詢首次失敗（shell 編碼）| 改用 `gh api -f q=` 修復，結果已補齊 |
| 只搜 GitHub，未搜 Gitee/CSDN 私有倉 | 可能有遺漏的中文項目 |

---

## 二、本輪新發現的項目（第一輪完全沒提）

| 項目 | ★ | 語言 | 為何重要 |
|------|---|------|---------|
| **openalpr/openalpr** | 11,472 | C++ | 歷史最悠久 ALPR 庫，但傳統 CV（非深度學習），已過時 |
| **sergiomsilva/alpr-unconstrained** | 1,771 | Python | WPOD-NET + CRNN，學術經典，但無中文 |
| **sirius-ai/LPRNet_Pytorch** | 1,160 | Python | LPRNet 官方 PyTorch 實現，輕量識別事實標準 |
| **DataXujing/vehicle-license-plate-recognition** | 879 | Python | 中國車牌，但**傳統字符分割**（card_seg.py），非 CTC |
| **lqian/light-LPR** | 782 | C++ | 商業級嵌入式 LPR（有硬件產品），**無港澳支援** |
| **DoubangoTelecom/ultimateALPR-SDK** | 747 | C++ | 商業級最快 ANPR，閉源算法 |
| **xuexingyu24/License_Plate_Detection_Pytorch** | 685 | Python | 兩階段輕量方案 |
| **parkpow/deep-license-plate-recognition** | 671 | Python | deep ALPR 實現 |
| **HuKai97/YOLOv5-LPRNet-Licence-Recognition** | 622 | Python | **YOLOv5 + LPRNet，字符集含 `-`** |
| **we0091234/yolov8-plate** | 518 | Python | 同作者 YOLOv8 版（149 py 文件，比 v5 版複雜） |
| **we0091234/yolov7_plate** | 394 | Python | YOLOv7 版 |
| **we0091234/yolo26-plate** | 155 | Python | YOLO26 最新版 |
| **Nenger/chinese_licence_plate_generator** | 73 | Python | **真實字符貼片生成器** ← 重大發現 |
| **zxx1218/LicensePlateDetection** | 96 | Python | 注意力機制改進 YOLO |
| **bobhyun/TS-ANPR** | 46 | Python | TensorFlow ANPR |

---

## 三、本輪實證發現（第一輪漏掉的關鍵點）

### 發現 1：字符生成器不該用字體渲染 —— `cn-plate-generator`

這是本輪最重要的發現，直接解決我們「合成車牌字體不真實」的問題。

**Nenger/chinese_licence_plate_generator 的方法**：

```
fake_resource/
├── chinese/     33 個（省份漢字，從真實車牌裁下）
├── numbers/     10 個
├── letters/     26 個
├── plate_background_use/   4 個（真實車牌底板 1.5MB BMP）
└── prepare_character.py    ← 從真實車牌提取字符的腳本
```

**流程**（`fake_plate_generator.py`）：

```python
# 1. 用真字符圖（帶 alpha 通道）貼到真車牌底板上
character = self.chinese["京"]        # 真實裁片，非字體渲染
mask = threshold(character[:,:,3], 100, 255)
overlay_img(character, plate_bg, mask, x, y)

# 2. 字符位置硬編碼（藍牌標準排版）
self.character_position_x_list_part_1 = [43, 111]        # 省 + 字母
self.character_position_x_list_part_2 = [205, 269, 330, 395, 464]  # 5 位序號

# 3. 三階段生成
#    Stage 1: 車牌（真字符 + 真底板）
#    Stage 2: 背景世界（純色 或 真實場景圖）
#    Stage 3: 貼到背景 + 加負樣本（hard negative mining）
#             位置信息寫入文件名作為 ground truth
```

**與我們之前的做法對比**：

| 做法 | 真實度 | 問題 |
|------|--------|------|
| 我們（Pillow + platech.ttf）| 低 | 字體不像真車牌，vision 分析直接判定「一眼看出偽造」 |
| Nenger（真字符貼片）| 高 | 字符本身就是從真車牌裁的，無法區分 |

**結論**：等真實數據充足後，生成器應改為「真實字符貼片 + 真實底板」路線，而非字體渲染。

### 發現 2：LPRNet vs CRNN 實測參數量

```python
LPRNet      class_num= 68  params=   446,976  (1.71 MB fp32)
myNet_ocr   class_num= 78  params= 1,925,142  (7.34 MB fp32)  # 默認 cfg
```

但實際權重用的是小 cfg：

```python
# weights/plate_rec_color.pth 實際 checkpoint
cfg: [8, 8, 16, 16, 'M', 32, 32, 'M', 48, 48, 'M', 64, 128]
實際參數量: 180,435  (0.69 MB fp32)
```

| 模型 | 類別數 | 參數 | 大小 | 結構 |
|------|--------|------|------|------|
| **we0091234 CRNN（實際權重）** | 78 | **180,435** | 0.69 MB | CNN + BiLSTM + CTC |
| HuKai97 LPRNet | 68 | 446,976 | 1.71 MB | 純 CNN + CTC，無 RNN |

**結論**：we0091234 的 CRNN 比 LPRNet **小 2.5 倍**。LPRNet 雖無 RNN（推理鏈路短），但參數更多。**繼續用 CRNN 是正確的**，第一輪的判斷成立，但當時沒有實測數據支撐。

### 發現 3：字符集互補 —— 79 類融合方案

| 來源 | 類別數 | 省份 | 字母 | 數字 | 港澳 | `-` |
|------|--------|------|------|------|------|-----|
| we0091234 | 78 | ✅ 31 | ✅ 24（無 I/O）| ✅ 10 | ✅ 港澳 | ❌ |
| HuKai97 | 68 | ✅ 31 | ✅ 26（含 I/O）| ✅ 10 | ❌ | **✅** |

**互補關係明確**：
- we0091234 有港澳字符，缺 `-`
- HuKai97 有 `-`，缺港澳字符

**融合**：78 + `-` = **79 類**，同時覆蓋四地所有格式（含澳門 `MA-12-34` 的連字符）。
字母處理：保留 we0091234 的「無 I/O」（避免與 1/0 混淆），因為車牌場景不需要 I/O。

### 發現 4：we0091234/yolov8-plate 架構並未改善

clone 後檢查：149 個 py 文件（v5 版只有 ~30 個），是完整 ultralytics YOLOv8 倉庫 + 車牌改動。
**問題依舊**：仍是單體腳本，無抽象層，`detect_plate.py` 依然混合 7 種職責。

**結論**：換 YOLOv8/v11 不會改善架構，只是換檢測器。架構重構是獨立任務。

### 發現 5：商業/學術項目都不支援港澳

| 項目 | 港澳支援 |
|------|---------|
| openalpr | ❌ 歐美為主 |
| alpr-unconstrained | ❌ 巴西/歐洲 |
| light-LPR | ❌ 中國大陸/台灣/韓國 |
| ultimateALPR-SDK | ❌（商業版可能支援）|
| LPRNet 系列 | ❌ 中國大陸 |

**再次確認：港澳車牌識別是開源空白區，必須自己訓練。**

---

## 四、修正後的項目評級

### 採用（架構/代碼參考）

| 項目 | 採用什麼 | 理由 |
|------|---------|------|
| **fast-alpr** | ABC 抽象層（40 行）| 擴展點設計最乾淨 |
| **pcb9382** | 多後端枚舉（PPOCRv5/v6/LPRNet）| 運行時切換識別器 |
| **zhahoi-ncnn** | 雙行 5/12+1/3 hstack | 兩項目交叉驗證一致 |
| **cn-plate-generator** | 真實字符貼片生成法 | 解決字體不真實的根因 |
| **we0091234** | 78 類 CRNN 權重（180K 參數）| 唯一公開+最小+含港澳 |

### 不採用（明確排除）

| 項目 | 排除理由 |
|------|---------|
| openalpr (11.5k★) | 傳統 CV（Haar + 字符分割），深度學習時代已過時 |
| LPRNet 系列 | 參數 2.5 倍於現有 CRNN，字符集缺港澳 |
| light-LPR | C++ 閉源模型，無港澳 |
| DataXujing | 字符分割方案，非 CTC，對港澳格式不適配 |
| ultimateALPR-SDK | 商業閉源 |
| PP-OCRv5/v6 | 34.5M 參數，對車牌過重 200 倍 |

### 待評估（有價值但未分析到底）

| 項目 | 待評估點 |
|------|---------|
| xuexingyu24/License_Plate_Detection_Pytorch | 兩階段輕量設計是否優於單階段 |
| zxx1218/LicensePlateDetection | 注意力機制對小車牌的增益 |
| we0091234/yolo26-plate | YOLO26 是否改善小目標檢測 |

---

## 五、最終融合方案（第二版）

```
┌───────────────────────────────────────────────────────────┐
│ 1. 抽象層 ← fast-alpr                                      │
│    BaseDetector / BaseOCR / ALPRResult(region, region_conf)│
└───────────────────────────────────────────────────────────┘
                          ↓
┌───────────────────────────────────────────────────────────┐
│ 2. 檢測層 ← we0091234 權重 + 改 4 類別                     │
│    classes: 0=mainland 1=gangao 2=hk 3=mo                 │
│    訓練：CCPD 355k + 港澳 6.3k（Roboflow, 已下載）         │
└───────────────────────────────────────────────────────────┘
                          ↓
┌───────────────────────────────────────────────────────────┐
│ 3. 預處理 ← zhahoi/pcb9382                                 │
│    4 角點透視變換 → 雙行 5/12+1/3 hstack 拼接              │
└───────────────────────────────────────────────────────────┘
                          ↓
┌───────────────────────────────────────────────────────────┐
│ 4. 識別層 ← we0091234 CRNN（180K 參數）                    │
│    字符集：78 類 + '-' = 79 類（融合 HuKai97 的連字符）     │
│    後端可切換 ← pcb9382 枚舉模式                            │
└───────────────────────────────────────────────────────────┘
                          ↓
┌───────────────────────────────────────────────────────────┐
│ 5. 後處理（新增，無人做過）                                 │
│    香港 ^[A-HJ-NP-Z]{0,2}\s?[1-9]\d{0,3}$                  │
│    澳門 ^M[A-Z]-?\d{2}-?\d{2}$                             │
│    粵Z  ^粵Z-?\d{4}[港澳]$                                 │
│    混淆糾正 0↔O / 1↔I / 5↔S / 8↔B                          │
└───────────────────────────────────────────────────────────┘
                          ↓
┌───────────────────────────────────────────────────────────┐
│ 6. 部署 ← pcb9382                                          │
│    ONNX → TensorRT / NCNN / OpenVINO                      │
└───────────────────────────────────────────────────────────┘

合成數據（後期）← cn-plate-generator
  真實字符貼片 + 真實底板，取代字體渲染
```

---

## 六、與第一輪報告的差異總結

| 項目 | 第一輪 | 第二輪（實證）|
|------|--------|-------------|
| 搜索範圍 | 5 個 repo（沿用舊 clone）| **265 個 repo（GitHub API 全量）** |
| 生成器方案 | 字體渲染（Pillow + TTF）| **真實字符貼片 + 真實底板** ← 修正 |
| LPRNet 評估 | 未評估 | **實測 447K vs CRNN 180K，CRNN 更優** |
| 字符集 | 78 類 + 「-」 | **確認可從 HuKai97 借用「-」的處理經驗** |
| 架構重構必要性 | 需要 | 確認需要，且換 YOLOv8/v11 無法解決 |
| 港澳支援 | 開源空白 | **再次確認空白（5 個主流項目全不支援）** |

---

**撰寫**：2026-10-29  
**搜索方法**：GitHub Search API，16 查詢，265 repo 去重  
**分析深度**：5 個新項目源碼級閱讀 + 參數量實測
