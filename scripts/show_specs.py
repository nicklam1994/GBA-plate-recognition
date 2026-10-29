#!/usr/bin/env python3
"""四地區車牌規格總覽：列出尺寸/比例/顏色，並渲染樣本對照圖。"""
import sys

sys.path.insert(0, "/home/nicklam-ai/repos/GBA-plate-recognition")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from synthetic_data import plate_specs as ps  # noqa: E402
from synthetic_data.plate_renderer import render_plate  # noqa: E402

# (分組, 規格變數名, 說明, 樣本文字)
ITEMS = [
    ("內地", "MAINLAND_BLUE",      "藍牌 小型車 7位",      "粤B·12345"),
    ("內地", "MAINLAND_YELLOW",    "黃牌 大型車前牌 7位",   "粤B·12345"),
    ("內地", "MAINLAND_DOUBLE_YELLOW", "黃牌 大型車後牌(雙行)", "粤B\n12345"),
    ("內地", "MAINLAND_GREEN",     "綠牌 小型新能源 8位(漸變)", "粤A·D12345"),
    ("內地", "MAINLAND_GREEN_LARGE", "綠牌 大型新能源(黃綠雙拼)", "粤A·D12345"),
    ("內地", "MAINLAND_COACH",     "學牌 教練車",          "粤B1234學"),
    ("內地", "MAINLAND_WHITE",     "警牌 白底紅警",        "粤B1234警"),
    ("內地", "MAINLAND_BLACK",     "使館 224·009使",       "224·009使"),
    ("內地", "MAINLAND_BLACK",     "領館 粵204·01領",      "粤204·01領"),
    ("內地", "MAINLAND_MILITARY",  "軍車 VA12345",         "VA12345"),
    ("內地", "MAINLAND_ARMED_POLICE", "武警 WJ·A00001",     "WJ·A00001"),
    ("內地", "MAINLAND_RESCUE",    "消防救援 8位",          "粤X1234应急"),
    ("粵Z",  "GANGAO_SINGLE",      "粵Z 兩地牌 7位",        "粤Z·1234港"),
    ("香港", "HK_FRONT",           "前牌 白底",            "AB1234"),
    ("香港", "HK_REAR",            "後牌 黃底",            "AB1234"),
    ("香港", "HK_FRONT_DOUBLE",    "前牌 方牌(雙行)",       "AB\n1234"),
    ("香港", "HK_REAR_DOUBLE",     "後牌 方牌(雙行)",       "AB\n1234"),
    ("澳門", "MO_A",               "A款 520×120",         "MZ-12-34"),
    ("澳門", "MO_B",               "B款 460×100",         "MZ-12-34"),
    ("澳門", "MO_C",               "C款 440×100",         "MZ-12-34"),
    ("澳門", "MO_D",               "D款 雙行 340×230",     "MZ-12\n34"),
    ("澳門", "MO_E",               "E款 雙行 330×175",     "MZ-12\n34"),
    ("澳門", "MO_MOTO",            "電單車 230×50",        "MZ-12-34"),
    ("澳門", "MO_ZA",              "駐澳解放軍 ZA",         "ZA-12-34"),
    ("澳門", "MO_A_TAXFREE",       "免稅 黑底黃字",        "MZ-12-34"),
    ("澳門", "MO_A_TEMP",          "臨時T牌 黃底黑字",      "MZ-12-34"),
]

print("=" * 88)
print(f"{'地區':<6} {'規格':<22} {'尺寸(mm)':<12} {'比例':<8} {'行數':<6} 說明")
print("=" * 88)
for region, name, desc, text in ITEMS:
    spec = getattr(ps, name)
    h = "雙行" if spec.is_double else "單行"
    print(f"{region:<6} {spec.name:<22} {spec.width:>4}×{spec.height:<6} "
          f"{spec.width/spec.height:>6.2f}:1 {h:<6} {desc}")

# ---- 渲染對照圖 ----
print()
print("渲染樣本…")
tiles = []
for region, name, desc, text in ITEMS:
    spec = getattr(ps, name)
    img = render_plate(text, spec)
    if img is None:
        print(f"  ! {name} 渲染失敗")
        continue
    arr = np.array(img)
    if arr.ndim == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
    else:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)   # render_plate 返回 BGR
    p = Image.fromarray(arr)
    # 保持寬高比，統一縮放到目標寬 300
    r = 300 / p.width
    p = p.resize((300, max(24, int(p.height * r))), Image.LANCZOS)
    tiles.append((region, desc, p))

pad = 10
col_w = 320
rows = (len(tiles) + 1) // 2
row_h = max(t[2].height for t in tiles) + 34
canvas = Image.new("RGB", (col_w * 2 + pad * 3, row_h * rows + pad), (24, 24, 32))
draw = ImageDraw.Draw(canvas)
for i, (region, desc, p) in enumerate(tiles):
    cx = pad + (i % 2) * (col_w + pad)
    cy = pad + (i // 2) * row_h
    canvas.paste(p, (cx, cy))
    draw.text((cx, cy + p.height + 4), f"[{region}] {desc}", fill=(150, 200, 235))

canvas.save("/tmp/spec_all.png")
print(f"  saved /tmp/spec_all.png {canvas.size}")
