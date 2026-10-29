#!/usr/bin/env python3
"""
預標註流水線 步驟 2：PaddleOCR 識別 + 規則後處理。

執行環境：~/tools/paddleocr/venv/bin/python3
輸入：<out>/crops + index.json（步驟 1 產物）
輸出：<out>/labeled.json —— 每筆含 PaddleOCR 原始結果、清洗後文字、置信度、是否通過格式校驗

用法：
    ~/tools/paddleocr/venv/bin/python3 scripts/prelabel_stage2_ocr.py --out prelabel/hk
"""
from __future__ import annotations

import argparse
import json
import re
from difflib import SequenceMatcher
from pathlib import Path

import cv2
import numpy as np
from paddleocr import PaddleOCR

ROOT = Path(__file__).resolve().parent.parent

#: 內地車牌可用的漢字：31 省簡稱（簡繁雙收）+ 使領警學掛港澳
PROVINCE_CHARS = (
    "京津冀晋蒙辽吉黑沪苏浙皖闽赣鲁豫鄂湘粤桂琼渝川贵云藏陕甘青宁新"
    "京津滬晉蒙遼吉黑滬蘇浙皖閩贛魯豫鄂湘粵桂瓊渝川貴雲藏陝甘青寧新"
    "使领領警学學挂掛港澳"
)

# --- 格式校驗（與 gba_plate.postprocess.rules 保持一致的核心規則） ---
# 先前寫死「粵」，其他 30 個省份全部判為不合格，內地樣本通過率因此為 0。
RE_MAINLAND = re.compile(
    rf"^[{PROVINCE_CHARS}][A-HJ-NP-Z][A-HJ-NP-Z0-9]{{4,6}}$")
RE_HK = re.compile(r"^[A-Z]{1,2}\d{1,4}$")
RE_HK_ZM = re.compile(r"^Z[MNP]\d{3,4}$")          # 澳門車入港的香港牌
RE_HK_FV = re.compile(r"^F[VU]\d{3,4}$")           # 內地車入港的香港牌
RE_MO = re.compile(r"^M[A-Z]\d{4}$")               # 澳門本地 M?-??-??
RE_MO_HK = re.compile(r"^HK\d{4}$")                # 澳門發給香港配額車的號段
RE_GANGAO = re.compile(r"^粵Z[A-Z0-9]{3,5}[港澳]$")


def strip_noise(s: str) -> str:
    """去掉空格、連字符、間隔符、全半角差異，統一為比對用形式。"""
    s = s.replace("澳", "澳").replace("粵", "粵")
    s = re.sub(r"[\s\-·・．.．]", "", s)
    return s.upper()


def normalize_gangao(s: str) -> str:
    """粵Z 牌常見 OCR 問題：'Z·7366澳粤' 這種上下行顛倒。"""
    if "粵Z" in s:
        return s
    if "粤Z" in s:
        return s.replace("粤", "粵")
    # 「Z7366澳粵」→「粵Z7366澳」
    m = re.match(r"^Z([A-Z0-9]{3,5}[港澳])粵$", s)
    if m:
        return "粵Z" + m.group(1)
    # 「864D港粵Z」→「粵Z864D港」
    m = re.match(r"^([A-Z0-9]{3,5}[港澳])粵Z$", s)
    if m:
        return "粵Z" + m.group(1)
    return s


#: PaddleOCR 的最佳輸入寬度。裁圖存 560 寬供人眼核對，
#: 但餵給 OCR 要縮回 320 —— 實測 560 時雙行牌的字母行會讀不到
#: （TK7687 → 7687），320 時正確。
OCR_TARGET_W = 320


def prep_for_ocr(path: Path):
    """讀圖並等比縮到 OCR 友善寬度。"""
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return None
    w = img.shape[1]
    if w > OCR_TARGET_W + 20:
        s = OCR_TARGET_W / float(w)
        img = cv2.resize(img, (OCR_TARGET_W, max(1, int(img.shape[0] * s))),
                         interpolation=cv2.INTER_AREA)
    return img


def find_split_line(img) -> int:
    """
    用水平投影找雙行牌兩行之間的分界線（該處筆劃最少）。

    固定用 50% 切會切歪 —— 實測上排 XM 與下排 2969 的分界在 60% 左右，
    按 52% 切會把下排頂部留在上排圖裡。
    """
    h = img.shape[0]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 0, 255,
                          cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    row_ink = bw.sum(axis=1) / 255.0
    lo, hi = int(h * 0.30), int(h * 0.72)
    if hi <= lo + 2:
        return h // 2
    band = row_ink[lo:hi]
    return lo + int(band.argmin())


def ocr_lines(img, ocr) -> tuple[list[str], list[float]]:
    """
    雙行牌：按投影分界切成上下兩半，各自 OCR 後按「上→下」合併。

    比拼接成一張圖可靠，因為 PaddleOCR 對單行文字最擅長；
    拼接會引入殘留，且縮圖後上排字母容易整個丟失。
    """
    split = find_split_line(img)
    parts = [img[:split + 2], img[max(0, split - 2):]]
    texts: list[str] = []
    scores: list[float] = []
    for part in parts:
        if part.size == 0 or part.shape[0] < 6:
            continue
        try:
            r = ocr.predict(part)[0]
            d = r["res"] if isinstance(r, dict) and "res" in r else r
            if not isinstance(d, dict):
                continue
            ts = d.get("rec_texts") or []
            ss = d.get("rec_scores") or []
            texts.extend(ts)
            scores.extend(float(x) for x in ss)
        except Exception:
            continue
    return texts, scores


#: 常見車標／車身文字，PaddleOCR 常與車牌一起框到
_CAR_TEXT = frozenset({
    "TOYOTA", "HONDA", "NISSAN", "BENZ", "BMW", "LEXUS", "BENTLEY",
    "ROLLS", "ROYCE", "AUDI", "MAZDA", "SUBARU", "MITSUBISHI", "PORSCHE",
    "FERRARI", "MASERATI", "VOLVO", "FORD", "CHEVROLET", "HYUNDAI", "KIA",
    "POLICE", "TAXI", "TAIWAN", "CROWN", "COMFORT", "ALPHARD", "VELLFIRE",
    "MODEL", "PLATE", "HONG", "KONG",
})


def sort_by_line(texts: list[str], polys) -> list[int]:
    """
    按「先行後列」排序文字塊，回傳索引順序。

    先前只按 y 座標排序，同一行內的 MY 與 1153（y 幾乎相同）
    次序不穩定，實測被拼成 11-53MY。車牌是橫書，同行必須按 x 排。
    """
    n = len(texts)
    if n <= 1 or polys is None or len(polys) != n:
        return list(range(n))

    boxes = []
    for i in range(n):
        pts = np.array(polys[i], dtype=np.float32).reshape(-1, 2)
        boxes.append((i, float(pts[:, 0].min()), float(pts[:, 1].min()),
                      float(pts[:, 1].max())))

    # 以中位高度的一半作為「同一行」容差
    heights = [b[3] - b[2] for b in boxes]
    med_h = float(np.median(heights)) if heights else 10.0
    tol = max(4.0, med_h * 0.5)

    lines: list[list] = []
    for b in sorted(boxes, key=lambda b: b[2]):   # 先按上緣 y
        placed = False
        for line in lines:
            if abs(b[2] - line[0][2]) <= tol:
                line.append(b)
                placed = True
                break
        if not placed:
            lines.append([b])

    order: list[int] = []
    for line in lines:
        for b in sorted(line, key=lambda b: b[1]):   # 行內按左緣 x
            order.append(b[0])
    return order


def pick_plate(texts: list[str]) -> tuple[str, list[str]]:
    """
    從 PaddleOCR 的多個文字塊中挑出車牌號。

    實測典型情況：
        ['YS', 'TOYOTA', 'Crowm hholors Ld.', '8696']  → 應為 YS8696
        ['XM', '2969', 'dO1OYOTA']                     → 應為 XM2969
    車標與車身文字混在結果裡，且字母、數字常被切成兩塊，故：
      1. 丟棄含小寫字母的塊（車標說明文字多為混合大小寫）
      2. 丟棄命中常見品牌詞的塊
      3. 只保留純大寫字母數字或含粵港澳的塊
      4. 按原順序拼接
    """
    parts: list[str] = []
    dropped: list[str] = []
    for t in texts:
        c = strip_noise(t)
        if not c:
            continue
        # 含小寫 → 車標或說明文字
        if any(ch.islower() for ch in t):
            dropped.append(t)
            continue
        if len(c) > 8:
            dropped.append(t)
            continue
        if c in _CAR_TEXT or any(b in c for b in _CAR_TEXT if len(b) > 3):
            dropped.append(t)
            continue
        # 先前只允許「粵港澳」三個漢字，導致所有內地車牌（粵B12345 之外的
        # 京/沪/苏/浙…）整個被丟掉 —— 內地樣本通過率因此為 0。
        if not re.fullmatch(rf"[A-Z0-9{PROVINCE_CHARS}]+", c):
            dropped.append(t)
            continue
        parts.append(c)
    return "".join(parts), dropped


def validate(text: str, region: str) -> tuple[bool, str]:
    """回傳 (是否通過, 命中的格式名)。"""
    t = text
    # 粵Z 才是兩地牌；粵B／粵A 等是內地牌，detector 偶爾誤判為 gangao
    if t.startswith(("粵Z", "粤Z")):
        ok = bool(RE_GANGAO.match(t) or RE_GANGAO.match(t.replace("粤", "粵")))
        return ok, "粵Z"
    if RE_MAINLAND.match(t):
        return True, "MAINLAND"
    if region == "gangao":
        return bool(RE_GANGAO.match(t)), "粵Z"
    if region == "mainland":
        # detector 判內地但號碼不合規則：可能是省份讀漏或字符誤讀
        return False, "MAINLAND"
    if region == "mo" or region == "hk":
        if RE_HK_ZM.match(t):
            return True, "HK-ZM"
        if RE_HK_FV.match(t):
            return True, "HK-FV"
        if RE_MO_HK.match(t):
            return True, "MO-HK"
        if RE_MO.match(t):
            return True, "MO"
        if RE_HK.match(t):
            return True, "HK"
    return False, ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="prelabel/hk")
    ap.add_argument("--batch", type=int, default=0, help="只跑前 N 個（0=全部）")
    args = ap.parse_args()

    out = (ROOT / args.out) if not Path(args.out).is_absolute() else Path(args.out)
    crops = out / "crops"
    index = {r["file"]: r for r in json.load(open(out / "index.json"))}

    files = sorted(crops.glob("*.jpg"))
    if args.batch:
        files = files[: args.batch]
    print(f"待識別 {len(files)} 張")

    ocr = PaddleOCR(
        lang="ch",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        text_det_limit_side_len=960,
        text_det_limit_type="max",
        text_det_thresh=0.2,
        text_det_box_thresh=0.4,
    )

    results = []
    for i, f in enumerate(files, 1):
        meta = index.get(f.name, {})
        try:
            small = prep_for_ocr(f)
            texts, scores = [], []
            if small is not None:
                # 一律整塊 OCR，不預先切上下半。
                # 教訓：曾有兩塊獨立車牌上下緊貼（MY1153 / ZM2089），detector
                # 給出兩塊重疊、偏高的框，「雙行判斷」再把它當成單塊雙行牌
                # 從中間切開，結果讀成 MY + M200。
                # 交給 PaddleOCR 自己分塊 + 按 y 排序，兩塊牌都能各自讀對。
                r = ocr.predict(small)[0]
                d = r["res"] if isinstance(r, dict) and "res" in r else r
                if isinstance(d, dict):
                    texts = d.get("rec_texts") or []
                    scores = d.get("rec_scores") or []
                    polys = d.get("rec_polys") or d.get("dt_polys") or []
                    if len(polys) == len(texts) and len(texts) > 1:
                        order = sort_by_line(texts, polys)
                        texts = [texts[k] for k in order]
                        scores = [scores[k] for k in order] if len(scores) == len(order) else scores
        except Exception as e:
            texts, scores = [], []
            print(f"  ! {f.name}: {e}")

        clean, dropped = pick_plate(texts)
        clean = normalize_gangao(clean)
        region = meta.get("region_det", "hk")
        ok, fmt = validate(clean, region)

        results.append({
            **meta,
            "paddle_raw": texts,
            "paddle_dropped": dropped,
            "paddle_confidence": round(float(min(scores)), 3) if scores else 0.0,
            "paddle_mean_conf": round(float(sum(scores) / len(scores)), 3) if scores else 0.0,
            "text_clean": clean,
            "valid": ok,
            "format": fmt,
            "lines": len(texts),
        })
        if i % 100 == 0:
            print(f"  {i}/{len(files)}", flush=True)

    json.dump(results, open(out / "labeled.json", "w"), ensure_ascii=False, indent=1)

    n = len(results)
    n_valid = sum(1 for r in results if r["valid"])
    n_empty = sum(1 for r in results if not r["text_clean"])
    confs = [r["paddle_mean_conf"] for r in results if r["paddle_mean_conf"] > 0]
    print(f"\n完成 {n} 筆")
    print(f"  通過格式校驗 {n_valid} ({100*n_valid/max(1,n):.1f}%)")
    print(f"  空結果       {n_empty} ({100*n_empty/max(1,n):.1f}%)")
    if confs:
        confs.sort()
        print(f"  平均置信度   {sum(confs)/len(confs):.3f}")
        print(f"  置信度中位數 {confs[len(confs)//2]:.3f}")
    from collections import Counter
    print("  格式分佈:", dict(Counter(r["format"] or "(未通過)" for r in results)))
    print(f"\n輸出 {out/'labeled.json'}")


if __name__ == "__main__":
    main()
