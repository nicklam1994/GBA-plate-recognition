# 真實車牌測試圖片下載指引

## ✅ 已下載的樣本

### CCPD 樣本（5 張）
- 位置：`test_data/real/ccpd/`
- 文件：`ccpd_demo_0.jpg` 到 `ccpd_demo_4.jpg`
- 來源：CCPD GitHub 倉庫的 demo 圖片

---

## 📥 手動下載指引

### 1️⃣ Roboflow 香港車牌數據集（33 張）

**步驟：**
1. 訪問：https://universe.roboflow.com/hong-kong-license-plate-detect/hk-license-plate-recognition
2. 點擊 "Download" 按鈕
3. 選擇格式：`YOLOv8` 或 `Raw`
4. 下載 ZIP 文件
5. 解壓到：`test_data/real/hk_roboflow/`

**注意：**
- 需要註冊 Roboflow 帳號（免費）
- 數據集包含 33 張香港車牌圖片
- 包含 YOLO 格式標註

---

### 2️⃣ CCPD 完整數據集（13GB，25萬+ 張）

**下載方式 A：Google Drive（推薦）**
1. 訪問：https://drive.google.com/open?id=1rdEsCUcIUaYOVRkx5IMTRNA7PcGMmSgc
2. 下載 `CCPD2019.tar.xz`
3. 解壓命令：
   ```bash
   cd ~/repos/chinese-plate-recognition/test_data/real/ccpd
   tar xf CCPD2019.tar.xz
   ```

**下載方式 B：百度網盤**
1. 訪問：https://pan.baidu.com/s/1i5AOjAbtkwb17Zy-NQGqkw
2. 提取碼：`hm0u`
3. 下載後解壓到：`test_data/real/ccpd/`

**數據集內容：**
- 25萬+ 張真實車牌圖片
- 包含多種場景（模糊、傾斜、雨天等）
- 文件名包含完整標註信息
- 主要為內地車牌（藍牌）

**CCPD-Green（新能源車牌）：**
- Google Drive：https://drive.google.com/file/d/1m8w1kFxnCEiqz_-t2vTcgrgqNIv986PR/view
- 百度網盤：https://pan.baidu.com/s/1JSpc9BZXFlPkXxRK4qUCyw（提取碼：ol3j）

---

### 3️⃣ CBLPRD-330k 數據集（33萬張）

**下載步驟：**
1. 訪問：https://github.com/SunlifeV/CBLPRD-330k
2. 按照 README 指引下載數據集
3. 解壓到：`test_data/real/cblprd/`

**數據集內容：**
- 33萬張中國車牌圖片
- 包含多種車牌類型（藍/黃/綠/黑）
- 適合大規模訓練

---

### 4️⃣ 粵Z兩地牌（需手動收集）

**建議來源：**
1. 搜索引擎圖片搜索：
   - "粤Z 車牌"
   - "中港車牌"
   - "两地牌"
2. 汽車論壇：
   - 愛卡汽車
   - 太平洋汽車網
3. 社交媒體：
   - 微博搜索 #粵Z#
   - 小紅書搜索 #中港車牌#

**下載後保存到：** `test_data/real/gangao/`

---

### 5️⃣ 澳門車牌（需手動收集）

**建議來源：**
1. 搜索引擎：
   - "澳門車牌"
   - "Macau license plate"
   - "MG 車牌" "MA 車牌"
2. 維基百科：
   - https://zh.wikipedia.org/wiki/澳門車輛號牌
3. 澳門交通事務局：
   - https://www.dsat.gov.mo/dsat/plate.aspx

**下載後保存到：** `test_data/real/mo/`

---

## 🧪 測試已下載的樣本

```bash
cd ~/repos/chinese-plate-recognition
source venv/bin/activate

# 測試 CCPD 樣本
python detect_plate.py \
    --detect_model weights/plate_detect.pt \
    --rec_model weights/plate_rec_color.pth \
    --image_path test_data/real/ccpd \
    --output test_results/ccpd
```

---

## 📊 數據集優先級

| 優先級 | 數據集 | 規模 | 用途 |
|--------|--------|------|------|
| ⭐⭐⭐⭐⭐ | Roboflow HK | 33 張 | 香港車牌快速測試 |
| ⭐⭐⭐⭐ | CCPD | 25萬+ 張 | 內地車牌基準測試 |
| ⭐⭐⭐ | CBLPRD-330k | 33萬張 | 大規模訓練 |
| ⭐⭐⭐⭐⭐ | 粵Z手動收集 | 需收集 | 兩地牌測試 |
| ⭐⭐⭐⭐ | 澳門手動收集 | 需收集 | 澳門車牌測試 |

---

## 🎯 下一步

1. ✅ 下載 Roboflow HK（33 張，10 分鐘）
2. ⏳ 下載 CCPD 完整數據集（13GB，1-2 小時）
3. ⏳ 手動收集粵Z和澳門車牌圖片
4. ⏳ 驗證所有測試圖片並建立基準
