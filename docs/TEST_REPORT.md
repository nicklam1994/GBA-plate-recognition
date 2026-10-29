# 真實車牌測試報告

**測試日期**：2026-10-29  
**測試模型**：we0091234 預訓練模型  
**測試設備**：CPU

---

## 📊 測試結果

### CCPD 樣本測試（5 張）

| 圖片 | 識別結果 | 顏色 | 置信度 | 狀態 |
|------|---------|------|--------|------|
| ccpd_demo_0.jpg | 皖A195K9 | 藍色 | - | ✅ 成功 |
| ccpd_demo_1.jpg | 皖AUJ299 | 藍色 | - | ✅ 成功 |
| ccpd_demo_2.jpg | 皖AT022C | 藍色 | - | ✅ 成功 |
| ccpd_demo_3.jpg | 皖AMK620 | 藍色 | - | ✅ 成功 |
| ccpd_demo_4.jpg | 皖A256R2 | 藍色 | - | ✅ 成功 |

**統計：**
- 總計圖片：5 張
- 成功識別：5 張
- 失敗：0 張
- **準確率：100%**
- 平均處理時間：31ms/張

---

## 📁 已下載數據集

### 1. CCPD 樣本（5 張）
- **位置**：`test_data/real/ccpd/`
- **來源**：CCPD GitHub 倉庫 demo 圖片
- **狀態**：✅ 已下載並測試
- **內容**：5 張內地藍牌（安徽）

### 2. Roboflow 香港車牌（待下載）
- **位置**：`test_data/real/hk_roboflow/`
- **來源**：Roboflow Universe
- **狀態**：⏳ 需手動下載
- **內容**：33 張香港車牌

### 3. CCPD 完整數據集（待下載）
- **位置**：`test_data/real/ccpd/`
- **來源**：Google Drive / 百度網盤
- **狀態**：⏳ 需手動下載
- **內容**：25萬+ 張內地車牌
- **大小**：13GB

---

## 🎯 測試發現

### ✅ 優勢
1. **內地藍牌識別效果優秀**：5/5 準確率 100%
2. **處理速度快**：CPU 模式下 31ms/張
3. **顏色識別準確**：全部正確識別為藍色

### ⚠️ 待驗證
1. **香港車牌**：需下載 Roboflow 數據集測試
2. **澳門車牌**：需手動收集測試圖片
3. **粵Z兩地牌**：需手動收集測試圖片
4. **複雜場景**：需下載 CCPD 完整數據集測試（模糊、傾斜、雨天等）

---

## 📥 下一步行動

### 立即執行（10 分鐘）
1. ✅ 下載 Roboflow 香港車牌數據集
   - 訪問：https://universe.roboflow.com/hong-kong-license-plate-detect/hk-license-plate-recognition
   - 下載 33 張香港車牌
   - 解壓到：`test_data/real/hk_roboflow/`

### 短期執行（1-2 小時）
2. ⏳ 下載 CCPD 完整數據集
   - Google Drive：https://drive.google.com/open?id=1rdEsCUcIUaYOVRkx5IMTRNA7PcGMmSgc
   - 或百度網盤：https://pan.baidu.com/s/1i5AOjAbtkwb17Zy-NQGqkw（提取碼：hm0u）
   - 解壓到：`test_data/real/ccpd/`

### 中期執行（需手動收集）
3. ⏳ 收集粵Z兩地牌圖片
   - 搜索引擎：「粤Z 車牌」「中港車牌」
   - 目標：50-100 張
   - 保存到：`test_data/real/gangao/`

4. ⏳ 收集澳門車牌圖片
   - 搜索引擎：「澳門車牌」「Macau license plate」
   - 目標：30-50 張
   - 保存到：`test_data/real/mo/`

---

## 📈 預期測試結果

| 車牌類型 | 預期準確率 | 備註 |
|---------|-----------|------|
| 內地藍牌 | > 95% | 已驗證 100% |
| 內地綠牌 | > 90% | 需測試 |
| 香港車牌 | > 80% | 待測試 |
| 澳門車牌 | > 75% | 待測試 |
| 粵Z兩地牌 | > 85% | 待測試 |

---

## 🔗 相關資源

- **下載指引**：`test_data/DOWNLOAD_GUIDE.md`
- **CCPD 論文**：https://github.com/detectRecog/CCPD
- **Roboflow HK**：https://universe.roboflow.com/hong-kong-license-plate-detect/hk-license-plate-recognition

---

**最後更新**：2026-10-29  
**維護者**：Nick（尼克🦊）
