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

# --- 格式校驗（與 gba_plate.postprocess.rules 保持一致的核心規則） ---
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


#: 常見車標／車身文字，PaddleOCR 常與車牌一起框到
_CAR_TEXT = frozenset({
    "TOYOTA", "HONDA", "NISSAN", "BENZ", "BMW", "LEXUS", "BENTLEY",
    "ROLLS", "ROYCE", "AUDI", "MAZDA", "SUBARU", "MITSUBISHI", "PORSCHE",
    "FERRARI", "MASERATI", "VOLVO", "FORD", "CHEVROLET", "HYUNDAI", "KIA",
    "POLICE", "TAXI", "TAIWAN", "CROWN", "COMFORT", "ALPHARD", "VELLFIRE",
    "MODEL", "PLATE", "HONG", "KONG",
})


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
        if not re.fullmatch(r"[A-Z0-9粵港澳]+", c):
            dropped.append(t)
            continue
        parts.append(c)
    return "".join(parts), dropped


RE_MAINLAND = re.compile(r"^粵[A-HJ-NP-Z][A-Z0-9]{4,6}$")   # 粵B·12345 等


def validate(text: str, region: str) -> tuple[bool, str]:
    """回傳 (是否通過, 命中的格式名)。"""
    t = text
    # 粵Z 才是兩地牌；粵B／粵A 等是內地牌，detector 偶爾誤判為 gangao
    if t.startswith("粵Z"):
        return bool(RE_GANGAO.match(t)), "粵Z"
    if RE_MAINLAND.match(t):
        return True, "MAINLAND"
    if region == "gangao":
        return bool(RE_GANGAO.match(t)), "粵Z"
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
                r = ocr.predict(small)[0]
                d = r["res"] if isinstance(r, dict) and "res" in r else r
                if isinstance(d, dict):
                    texts = d.get("rec_texts") or []
                    scores = d.get("rec_scores") or []
                    # 雙行牌要先讀上排再讀下排，PaddleOCR 原始順序不可靠
                    # （實測 XM2969 被輸出成 2969XM）
                    polys = d.get("rec_polys") or d.get("dt_polys") or []
                    if len(polys) == len(texts) and len(texts) > 1:
                        order = sorted(
                            range(len(texts)),
                            key=lambda i: float(np.array(polys[i])[:, 1].mean()))
                        texts = [texts[i] for i in order]
                        scores = [scores[i] for i in order] if len(scores) == len(order) else scores
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
