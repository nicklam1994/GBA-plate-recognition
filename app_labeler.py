#!/usr/bin/env python3
"""
車牌標註人工核對 WebUI。

用途：YOLO 鎖框 + PaddleOCR 預識別之後，人工逐張確認或修正，
把結果存成可用的訓練標註。

啟動：
    ./venv/bin/python app_labeler.py --port 8791

資料：
    <data>/labeled.json    步驟 2 產物（OCR 預標註）
    <data>/crops/*.jpg     步驟 1 產物（車牌裁圖，已含雙行拼接）
    <data>/verified.json   本工具產物（人工核對結果）

排序策略：先顯示「未通過格式校驗」與低置信度的，因為那些最可能出錯；
已核對的排到最後，避免重複勞動。
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import cv2
import numpy as np

from flask import Flask, jsonify, request, send_file, render_template_string

from gba_plate.evaluate import evaluate_pairs
from gba_plate.base import Region
from gba_plate.postprocess.region_map import resolve_plate_region, resolve_energy
from gba_plate.postprocess.rules import normalize, validate

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "prelabel" / "hk"

app = Flask(__name__)

STATE = {"data_dir": DATA}

#: 省份簡稱（簡繁雙收）+ 特殊用途漢字，供類型判斷與正則組裝
PROVINCE_SRC = (
    "京津冀晋蒙辽吉黑沪苏浙皖闽赣鲁豫鄂湘粤桂琼渝川贵云藏陕甘青宁新"
    "京津滬晉蒙遼吉黑滬蘇浙皖閩贛魯豫鄂湘粵桂瓊渝川貴雲藏陝甘青寧新"
    "使领領警学學挂掛港澳"
)


# ------------------------------------------------------- 車牌底色與類型判斷

def detect_plate_color(img: np.ndarray) -> str:
    """
    判斷車牌底色：blue / yellow / green / white / unknown。

    取裁圖四周條帶的中位色 —— 車牌底色是整塊純色，
    文字筆劃只佔中間區域，取邊框能避開文字干擾。
    """
    if img is None or img.size == 0:
        return "unknown"
    h, w = img.shape[:2]
    strips = [
        img[: max(1, int(h * 0.12))],
        img[int(h * 0.88):],
        img[:, : max(1, int(w * 0.08))],
        img[:, int(w * 0.92):],
    ]
    px = np.vstack([s.reshape(-1, 3) for s in strips if s.size])
    if not len(px):
        return "unknown"
    b, g, r = (float(v) for v in np.median(px, axis=0))
    if b > 85 and b > r * 1.30 and b > g * 1.10:
        return "blue"
    if r > 105 and g > 85 and b < 95 and r > b * 1.4:
        return "yellow"
    if g > 90 and g > r * 1.12 and g > b * 1.08:
        return "green"
    if min(b, g, r) > 115:
        return "white"
    return "unknown"


#: 底色 → 車型
_COLOR_TYPE = {
    "blue": "小型汽車（藍牌）",
    "yellow": "大型汽車（黃牌）",
    "green": "新能源汽車（綠牌）",
    "white": "警用／軍用／港澳入出（白牌）",
}


def classify_plate_type(text: str, region: str,
                        color: str = "unknown") -> str:
    """
    判斷車牌類型（中文名）。

    內地牌依 GA36-2018：底色決定車型，號碼結構決定細分類。
    港澳牌結構單一，只區分一般／粵Z。
    """
    t = (text or "").upper().strip()
    if not t:
        return ""

    if region == "gangao":
        if t.startswith("粵Z"):
            return "粵Z 港澳入出內地車輛"
        return "港澳入出內地車輛"

    if region == "hk":
        if t.startswith(("FV", "FU")):
            return "FV/FU 內地入出香港車輛"
        if t.startswith(("ZM", "ZN", "ZL")):
            return "澳門私家車往來港澳常規配額（香港牌）"
        return "香港車牌"

    if region == "mo":
        return "澳門車牌"

    if region == "mainland":
        # 特殊牌先判，避免被車型覆蓋
        if t.startswith("WJ"):
            return "武警車牌"
        if len(t) >= 2 and t[0] in "VKHBSLJNGZY" and t[1].isalpha() is False:
            pass
        if re.match(r"^[A-Z]{2}\d{4,5}$", t) and t[0] in "VKHBSLJNGZ":
            return "軍隊車牌"
        # 號段常見「皖AH0008」這種序號含字母的形式，先前只允許 \d{4}，
        # 導致所有帶字母號段都退化成「民用汽車」，省市也跟著解析不出來。
        if re.match(rf"^[{PROVINCE_SRC}][A-Z][A-Z0-9]{{4,6}}$", t):
            if len(t) == 8:
                # 8 位＝新能源：第 3 位是能源類型字母
                if t[2] in "DABCEFPQRST":
                    return "新能源小型汽車（綠牌·純電動）"
                return "新能源小型汽車（綠牌·非純電）"
            if len(t) == 7:
                if color == "yellow":
                    return "大型汽車（黃牌·燃油）"
                if color == "white":
                    return "警用／軍用／港澳入出（白牌）"
                if color == "green":
                    return "新能源汽車（綠牌）"
                # 藍牌＝小型燃油車，實測最常見
                return "小型汽車（藍牌·燃油）"
            return _COLOR_TYPE.get(color, "民用汽車")
        if re.match(rf"^[{PROVINCE_SRC}][A-Z][A-Z0-9]{{5}}$", t):
            # 大型車後牌 7 位（省份+字母+5位，舊式）
            return "大型汽車（黃牌·燃油）" if color == "yellow" else "民用汽車"
        # 只有粵Z 開頭且以港/澳結尾才是兩地牌；粵A/B/C… 是普通內地牌
        if t.startswith(("粵Z", "粤Z")) and re.search(r"[港澳]$", t):
            return "港澳入出內地車輛"
        return _COLOR_TYPE.get(color, "民用汽車")

    return ""


# ---------------------------------------------------------------- 格式規則
#
# 這裡刻意不再自己維護一套正則。
# 先前 app_labeler / prelabel_stage2 / gba_plate 各有一套 RE_*，三份互相漂移：
# 例如 XM9112 在 labeled.json 是 valid=True，UI 卻可能顯示格式不符；
# 內地牌更因為只寫死「粵」而讓 30 個省份全軍覆沒。
# 現在統一呼叫 gba_plate.postprocess.rules.validate()，只有一份真相。

def check_format(text: str, region: str | None = None) -> tuple[bool, str]:
    """
    校驗車牌格式。有地區時按該地區規則驗，否則逐個地區試。

    Returns:
        (是否通過, 命中的格式名)
    """
    if not text:
        return False, ""
    t = normalize(text)
    if not t:
        return False, ""

    order = []
    if region:
        try:
            order.append(Region(region))
        except ValueError:
            pass
    for r in (Region.GANGAO, Region.MAINLAND, Region.HK, Region.MO):
        if r not in order:
            order.append(r)

    #: validate 只回 "ok"，對人工核對沒辨識度，換成看得懂的地區名
    label = {
        Region.GANGAO: "粵Z",
        Region.MAINLAND: "內地",
        Region.HK: "香港",
        Region.MO: "澳門",
    }

    def _hit(r, fmt: str) -> tuple[bool, str]:
        if fmt == "ok" or fmt.startswith("ok("):
            return True, label.get(r, fmt)
        return True, fmt

    for r in order:
        try:
            ok, fmt = validate(t, r)
        except Exception:
            continue
        if ok:
            return _hit(r, fmt)

    # 連字符 fallback：港澳牌不寫連字符，但澳門牌是 MA-12-34。
    # 使用者可能把香港牌打成 XM-9112，去掉連字符再驗一次。
    if "-" in t:
        bare = t.replace("-", "")
        for r in order:
            try:
                ok, fmt = validate(bare, r)
            except Exception:
                continue
            if ok:
                return _hit(r, fmt)
    return False, ""


CLARITY_CACHE: dict[str, float] = {}

#: 拉普拉斯方差門檻。實測：完全模糊 ≈ 5~30，可辨識 > 100
CLARITY_BLUR = 40.0
CLARITY_OK = 100.0


def compute_clarity(data_dir: Path) -> dict[str, float]:
    """
    計算每張裁圖的清晰度（拉普拉斯方差），用於把廢圖排除在人工核對之外。

    模糊圖人工也讀不出號碼，讓人去核對純屬浪費時間。
    """
    global CLARITY_CACHE
    if CLARITY_CACHE:
        return CLARITY_CACHE
    cache_file = data_dir / "clarity.json"
    if cache_file.exists():
        CLARITY_CACHE = load_json(cache_file, {})
        return CLARITY_CACHE

    result: dict[str, float] = {}
    for p in sorted((data_dir / "crops").glob("*.jpg")):
        img = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if img is None:
            result[p.name] = 0.0
            continue
        # 統一縮放到寬 320 再算拉普拉斯方差。
        # 直接算會被尺寸干擾：CCPD 車牌裁圖只有 ~100px 寬，
        # 同樣清晰的圖數值天然比香港 200px 的低，導致 129/260 被誤判為廢圖。
        w = img.shape[1]
        if w and w != 320:
            s = 320.0 / w
            img = cv2.resize(img, (320, max(1, int(img.shape[0] * s))),
                             interpolation=cv2.INTER_AREA if s < 1
                             else cv2.INTER_LANCZOS4)
        result[p.name] = round(float(cv2.Laplacian(img, cv2.CV_64F).var()), 1)
    save_json(cache_file, result)
    CLARITY_CACHE = result
    return result


def load_json(path: Path, default):
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return default


def save_json(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=1)
    tmp.replace(path)


COLOR_CACHE: dict[str, str] = {}


def compute_colors(data_dir: Path) -> dict[str, str]:
    """算出每張裁圖的底色，快取到 colors.json（車牌底色＝車型的最強線索）。"""
    global COLOR_CACHE
    if COLOR_CACHE:
        return COLOR_CACHE
    cache_file = data_dir / "colors.json"
    if cache_file.exists():
        COLOR_CACHE = load_json(cache_file, {})
        return COLOR_CACHE

    result: dict[str, str] = {}
    for p in sorted((data_dir / "crops").glob("*.jpg")):
        img = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)
        result[p.name] = detect_plate_color(img)
    save_json(cache_file, result)
    COLOR_CACHE = result
    return result


def build_items() -> list[dict]:
    """合併預標註與人工結果，按「優先處理」排序。"""
    d = STATE["data_dir"]
    labeled = load_json(d / "labeled.json", [])
    verified = load_json(d / "verified.json", {})
    clarity = compute_clarity(d)
    colors = compute_colors(d)

    items = []
    for r in labeled:
        f = r["file"]
        v = verified.get(f)
        text = (v or {}).get("text") or r.get("text_clean") or ""
        region_now = (v or {}).get("region") or r.get("region_det", "")
        ok, fmt = check_format(text, region_now)
        # 內地牌的省份／地市，以及四地共用的車牌類型
        prov, city, note = (None, None, "")
        if region_now in ("mainland", "gangao") and text:
            prov, city, note = resolve_plate_region(text)
        ptype = classify_plate_type(text, region_now, colors.get(f, "unknown"))
        items.append({
            "file": f,
            "src": r.get("src", ""),
            "region_det": r.get("region_det", ""),
            "det_conf": r.get("det_conf", 0),
            "ocr_conf": r.get("paddle_mean_conf", 0),
            "ocr_raw": r.get("paddle_raw", []),
            "ocr_text": r.get("text_clean", ""),
            "text": text,
            "format": fmt,
            "format_ok": ok,
            "double_line": bool(r.get("double_line_merged")),
            "verified": v is not None,
            "action": (v or {}).get("action", ""),
            "bbox": r.get("bbox"),
            "clarity": clarity.get(f, 0.0),
            "region_final": region_now,
            "region_edited": bool((v or {}).get("region")),
            "province": prov,
            "city": city,
            "region_note": note,
            "plate_type": ptype,
            "plate_color": colors.get(f, "unknown"),
            "energy": (resolve_energy(text)[1] if region_now in
                       ("mainland", "gangao") and text else ""),
        })

    # 同圖多牌：把同一張原圖的所有車牌框附上，供原圖上全部畫出
    by_src: dict[str, list[dict]] = {}
    for it in items:
        by_src.setdefault(it["src"], []).append(it)
    for src, group in by_src.items():
        sibs = [{"file": g["file"], "bbox": g["bbox"],
                 "text": g["text"], "region": g["region_final"],
                 "verified": g["verified"], "action": g["action"]}
                for g in group]
        for g in group:
            g["siblings"] = sibs

    # 排序策略（依人工效率而非單純分數）：
    #   1. 未核對優先
    #   2. 「有 OCR 文字但格式不符」最優先 —— 只需改一兩個字符，處理最快
    #   3. 空結果排到最後 —— 要從零輸入，成本最高
    #   4. 同組內 OCR 置信度低的優先（更可能錯）
    def key(it):
        has_text = 1 if it["text"] else 0
        fixable = 1 if (has_text and not it["format_ok"]) else 0
        # 模糊圖（清晰度 < 門檻）排最後：人眼也讀不出，核對無意義
        blurry = 1 if it["clarity"] < CLARITY_BLUR else 0
        return (
            blurry,
            1 if it["verified"] else 0,
            0 if fixable else 1,
            0 if has_text else 1,
            it["ocr_conf"],
            it["file"],
        )
    items.sort(key=key)
    return items


def stats() -> dict:
    items = build_items()
    total = len(items)
    verified = sum(1 for i in items if i["verified"])
    ok = sum(1 for i in items if i["format_ok"])
    corrected = sum(1 for i in items
                    if i["verified"] and i["action"] == "correct")
    not_plate = sum(1 for i in items if i["action"] == "not_plate")
    unusable = sum(1 for i in items if i["action"] == "unusable")
    blurry = sum(1 for i in items if i["clarity"] < CLARITY_BLUR)
    usable = total
    return {
        "total": total,
        "verified": verified,
        "remaining": total - verified,
        "format_ok": ok,
        "format_bad": total - ok,
        "corrected": corrected,
        "not_plate": not_plate,
        "unusable": unusable,
        "blurry": blurry,
        "usable": usable,
        "usable_remaining": usable - verified,
        "low_clarity": blurry,
        "progress": round(100 * verified / max(1, usable), 1),
    }


# ---------------------------------------------------------------- 路由

@app.get("/")
def index():
    return render_template_string(PAGE)


@app.get("/api/items")
def api_items():
    items = build_items()
    only = request.args.get("only", "")
    # 不把「模糊」當成硬性排除條件 —— 拉普拉斯方差對小尺寸裁圖失真
    # （CCPD 車牌只有 ~100px 寬，數值天然偏低），曾誤殺 124/260 張
    # 人眼其實能讀的圖。改為只排序靠後，讓使用者自己決定。
    if only == "todo":
        items = [i for i in items if not i["verified"]]
    elif only == "bad":
        items = [i for i in items if not i["format_ok"] and not i["verified"]]
    elif only == "clear":
        items = [i for i in items if i["clarity"] >= CLARITY_OK]
    elif only == "blurry":
        items = [i for i in items if i["clarity"] < CLARITY_BLUR]
    elif only == "notplate":
        items = [i for i in items if i["action"] == "not_plate"]
    return jsonify({"items": items, "stats": stats()})


@app.get("/api/stats")
def api_stats():
    return jsonify(stats())


@app.post("/api/verify")
def api_verify():
    """action: pass / correct / skip / unusable（看不清）/ not_plate（根本不是車牌）"""
    body = request.get_json(force=True) or {}
    f = body.get("file")
    action = body.get("action", "pass")
    text = (body.get("text") or "").strip()

    if not f:
        return jsonify({"ok": False, "error": "缺少 file"}), 400
    d = STATE["data_dir"]
    verified = load_json(d / "verified.json", {})

    if action == "pass":
        # 使用者直接按 Enter 確認。但若他先在輸入框改了字再按 Enter，
        # 以前會用伺服器上的舊值蓋掉他的修改（改了等於白改）。
        # 現在：前端傳來的值與伺服器不同，就視為「已修正」。
        cur = next((i for i in build_items() if i["file"] == f), None)
        cur_text = (cur or {}).get("text", "")
        if text and text != cur_text:
            action = "correct"          # 保留使用者的修改
        else:
            text = cur_text
    elif action in ("unusable", "not_plate"):
        # unusable = 是車牌但糊到看不清；not_plate = 檢測框錯了，畫面裡沒車牌
        # 兩者都清空 text，但分開記錄 —— not_plate 是 detector 的負樣本，
        # 可用來降低誤檢，不能跟「看不清」混在一起。
        text = ""

    region = (body.get("region") or "").strip() or None
    verified[f] = {
        "text": text,
        "action": action,
        "region": region,
        "ts": int(time.time()),
    }
    save_json(d / "verified.json", verified)

    ok, fmt = check_format(text, region) if text else (False, "")
    return jsonify({"ok": True, "text": text, "format": fmt,
                    "format_ok": ok, "stats": stats()})


@app.post("/api/check")
def api_check():
    """
    即時校驗輸入框內容。

    先前按鈕的「正確／格式不符」只依頁面載入時的 text 算一次，
    使用者手動把 912 改成 XM9112 後，按鈕仍顯示格式不符 ——
    明明輸入對了卻被提示錯，會讓人以為校驗壞了。
    """
    body = request.get_json(force=True) or {}
    text = (body.get("text") or "").strip()
    region = (body.get("region") or "").strip() or None
    file = (body.get("file") or "").strip()
    ok, fmt = check_format(text, region) if text else (False, "")

    # 省市與類型也必須跟著輸入即時重算 —— 先前只在頁面載入時算一次，
    # 使用者補上省份「皖」之後，chips 仍停在 AH0008 的失敗狀態。
    prov = city = note = energy = ptype = ""
    if text and region in ("mainland", "gangao"):
        p_, c_, n_ = resolve_plate_region(text)
        prov = p_ or ""
        city = c_ or ""
        note = n_ or ""
        if region == "mainland" or text.startswith(("粵Z", "粤Z")):
            energy = resolve_energy(text)[1] or ""

    color = "unknown"
    if file:
        color = compute_colors(STATE["data_dir"]).get(file, "unknown")
    if text:
        ptype = classify_plate_type(text, region or "", color)

    return jsonify({
        "ok": True, "format_ok": ok, "format": fmt,
        "province": prov, "city": city, "region_note": note,
        "plate_type": ptype, "plate_color": color, "energy": energy,
    })


@app.get("/api/eval")
def api_eval():
    """
    拿人工核對過的真值，檢驗 OCR 預測的準確率。

    真值 = verified.json 的 text（人工確認過的）
    預測 = labeled.json 的 text_clean（PaddleOCR 原始輸出）
    只統計「已核對且非無法辨識」的樣本。
    """
    d = STATE["data_dir"]
    labeled = load_json(d / "labeled.json", [])
    verified = load_json(d / "verified.json", {})

    pairs, type_map = [], {}
    for r in labeled:
        v = verified.get(r["file"])
        if not v:
            continue
        gt = (v.get("text") or "").strip()
        if not gt:                      # 無法辨識／空白不列入
            continue
        pred = (r.get("text_clean") or "").strip()
        region = v.get("region") or r.get("region_det", "")
        pairs.append((gt, pred, region))
        t = classify_plate_type(gt, region, "unknown")
        if t:
            type_map[gt] = t

    if not pairs:
        return jsonify({"ok": True, "empty": True,
                        "message": "尚未核對任何樣本，先核對幾張再看評測"})

    rep = evaluate_pairs(pairs, by_type=type_map).as_dict()
    rep["ok"] = True
    rep["empty"] = False
    return jsonify(rep)


@app.post("/api/reset")
def api_reset():
    """清除某筆的核對結果（重做）。"""
    body = request.get_json(force=True) or {}
    f = body.get("file")
    d = STATE["data_dir"]
    verified = load_json(d / "verified.json", {})
    verified.pop(f, None)
    save_json(d / "verified.json", verified)
    return jsonify({"ok": True, "stats": stats()})


@app.get("/crop/<path:name>")
def crop(name: str):
    p = (STATE["data_dir"] / "crops" / name).resolve()
    base = (STATE["data_dir"] / "crops").resolve()
    if not str(p).startswith(str(base)) or not p.exists():
        return "", 404
    return send_file(p, mimetype="image/jpeg")


@app.get("/srcimg/<path:rel>")
def srcimg(rel: str):
    """原圖，用於查看車牌在車身上的上下文。"""
    p = (ROOT / rel).resolve()
    if not str(p).startswith(str(ROOT)) or not p.exists():
        return "", 404
    return send_file(p)


PAGE = r"""<!DOCTYPE html>
<html lang="zh-Hant">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>車牌標註核對</title>
<style>
  * { box-sizing: border-box; }
  body {
    margin: 0; background: #15161c; color: #e6e9ef;
    font-family: -apple-system, "Segoe UI", "Microsoft JhengHei", sans-serif;
  }
  header {
    display: flex; align-items: center; gap: 16px;
    padding: 10px 18px; background: #1d1f28;
    border-bottom: 1px solid #2c2f3c; position: sticky; top: 0; z-index: 10;
  }
  h1 { font-size: 15px; margin: 0; font-weight: 600; letter-spacing: .5px; }
  .bar { flex: 1; height: 6px; background: #2c2f3c; border-radius: 3px; overflow: hidden; }
  .bar > i { display: block; height: 100%; background: #3ea86b; transition: width .2s; }
  .stat { font-size: 12px; color: #9aa3b2; font-variant-numeric: tabular-nums; }
  .stat b { color: #e6e9ef; font-size: 13px; }

  main { padding: 18px; max-width: 900px; margin: 0 auto; }

  .card {
    background: #1d1f28; border: 1px solid #2c2f3c; border-radius: 10px;
    padding: 16px; margin-bottom: 14px;
  }
  .row { display: flex; gap: 16px; align-items: flex-start; }
  .plate-wrap {
    flex: 0 0 auto; background: #0f1015; border-radius: 8px;
    padding: 8px; display: flex; align-items: center; justify-content: center;
    position: relative; align-self: flex-start;
  }
  .plate-wrap img {
    display: block; max-width: 520px; max-height: 210px;
    cursor: zoom-in; transition: transform .15s ease;
  }
  .plate-wrap img.zoom {
    transform: scale(1.8); cursor: zoom-out;
    z-index: 50; position: relative;
    box-shadow: 0 8px 32px rgba(0,0,0,.8);
  }
  .meta { flex: 0 0 38%; font-size: 12px; color: #9aa3b2; line-height: 1.9; }
  .meta b { color: #cfd6e3; }
  .tag {
    display: inline-block; padding: 1px 7px; border-radius: 4px;
    font-size: 11px; margin-right: 5px;
  }
  .t-ok   { background: #1d3b2a; color: #5ec98a; }
  .t-bad  { background: #3d2024; color: #ef7a86; }
  .t-warn { background: #3b3420; color: #e0b44e; }

  /* 評測面板 */
  .evalgrid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin: 6px 0 16px; }
  .evalcard { background: #14161e; border: 1px solid #262b3a; border-radius: 8px; padding: 12px 14px; }
  .evalcard .k { font-size: 11px; color: #6d7688; }
  .evalcard .v { font-size: 22px; font-weight: 700; color: #e6e9ef; margin-top: 2px; }
  .evalcard .v.good { color: #5ec98a; }
  .evalcard .v.mid { color: #e0b44e; }
  .evalcard .v.bad { color: #ef7a86; }
  table.ev { width: 100%; border-collapse: collapse; font-size: 12px; margin-bottom: 16px; }
  table.ev th { text-align: left; color: #6d7688; font-weight: 500;
                padding: 6px 8px; border-bottom: 1px solid #262b3a; }
  table.ev td { padding: 5px 8px; border-bottom: 1px solid #1b1e28; color: #cfd6e3; }
  table.ev tr:hover td { background: #171a23; }
  .evh { font-size: 13px; color: #9aa3b2; margin: 14px 0 6px; font-weight: 600; }

  label { display: block; font-size: 12px; color: #9aa3b2; margin: 12px 0 6px; }
  input[type=text] {
    width: 100%; padding: 12px 14px; font-size: 26px; letter-spacing: 3px;
    font-family: ui-monospace, "SF Mono", Consolas, monospace;
    background: #0f1015; color: #e6e9ef;
    border: 2px solid #2c2f3c; border-radius: 8px; outline: none;
  }
  input[type=text]:focus { border-color: #4a7fd4; }
  input.edited { border-color: #d4a24a; }

  .hint { font-size: 11px; color: #6d7688; margin-top: 6px; }

  .actions { display: flex; gap: 10px; margin-top: 14px; flex-wrap: wrap; }
  button {
    padding: 11px 20px; font-size: 14px; border-radius: 8px; cursor: pointer;
    border: 1px solid #2c2f3c; background: #262933; color: #e6e9ef;
    font-family: inherit;
  }
  button:hover { background: #2f3340; }
  button.primary { background: #2e7d4f; border-color: #3ea86b; }
  button.primary:hover { background: #35905b; }
  button.warn { background: #5a2a2f; border-color: #7d3b42; }
  button.caution { background: #6b4f1a; border-color: #a8812b; }
  button.caution:hover { background: #7d5d1f; }
  button.ghost { background: transparent; }
  kbd {
    display: inline-block; padding: 1px 6px; border-radius: 4px;
    background: #0f1015; border: 1px solid #3a3f4d;
    font-size: 11px; font-family: ui-monospace, monospace; color: #9aa3b2;
  }

  .filters { display: flex; gap: 8px; margin-bottom: 12px; }
  .filters button { padding: 6px 12px; font-size: 12px; }
  .filters button.on { background: #2f3340; border-color: #4a7fd4; }

  .done { text-align: center; padding: 60px 20px; color: #6d7688; font-size: 15px; }
  .srcbox { margin-top: 14px; }
  .srcbox img { max-width: 100%; border-radius: 6px; border: 1px solid #2c2f3c; display: block; }
  details summary { cursor: pointer; font-size: 12px; color: #6d7688; }

  /* 原圖 + YOLO 框 */
  .srcwrap { position: relative; display: inline-block; max-width: 100%; line-height: 0; }
  /* 不能用 object-fit: contain 或 width:100% —— 圖片會被 letterbox 留黑邊，
     而框的座標是按 clientWidth/naturalWidth 換算的，黑邊會讓框整個偏寬。
     讓 img 自然尺寸縮放，clientWidth 才等於實際顯示寬度。 */
  .srcwrap img { max-width: 100%; max-height: 62vh; display: block; border-radius: 6px; }
  /* 內地牌資訊 */
  .minfo { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 10px; }
  .chip {
    font-size: 12px; padding: 4px 10px; border-radius: 5px;
    background: #191c26; border: 1px solid #2c3346; color: #cfd6e3;
  }
  .chip b { color: #8fb8f0; font-weight: 600; }
  .chip.col-blue { border-color: #2f4a80; background: #16233d; }
  .chip.col-yellow { border-color: #7a6428; background: #2b2415; }
  .chip.col-green { border-color: #2b6b45; background: #14291d; }
  .chip.col-white { border-color: #4a4f5c; background: #1d1f26; }
  /* 上一張 / 下一張 */
  .navrow { display: flex; gap: 8px; margin-top: 10px; }
  .navrow button { flex: 0 0 auto; padding: 8px 16px; }
  .yolobox {
    position: absolute; border: 2px solid #4ade80;
    box-shadow: 0 0 0 1px rgba(0,0,0,.7), inset 0 0 0 1px rgba(0,0,0,.5);
    pointer-events: none;
  }
  .yolobox::after {
    content: attr(data-label); position: absolute; left: -2px; top: -19px;
    background: #4ade80; color: #0b1a10; font-size: 10px; font-weight: 700;
    padding: 1px 5px; border-radius: 3px; white-space: nowrap;
  }
  /* 同圖中的其他車牌：灰色細框 */
  .yolobox.other {
    border: 1.5px dashed #8a93a6; box-shadow: none;
  }
  .yolobox.other::after {
    background: #495062; color: #cfd6e3; font-weight: 500;
  }
  .yolobox.cur { border-color: #4ade80; border-style: solid; }
  /* 同圖中「已核對過」的其他牌：淡化，一眼看出還剩幾塊沒處理 */
  .yolobox.other.done {
    opacity: .35; border-color: #3d4452; border-style: dotted;
  }
  select {
    padding: 8px 10px; font-size: 13px; border-radius: 6px;
    background: #0f1015; color: #e6e9ef; border: 1px solid #2c2f3c;
    font-family: inherit; outline: none;
  }
  select:focus { border-color: #4a7fd4; }
  select.edited { border-color: #d4a24a; }
  .regrow { display: flex; align-items: center; gap: 10px; margin-top: 12px; }
</style>
</head>
<body>
<header>
  <h1>車牌標註核對</h1>
  <div class="bar"><i id="pbar" style="width:0%"></i></div>
  <div class="stat">已核 <b id="s-done">0</b>/<b id="s-usable">0</b>
    · 待核 <b id="s-left">0</b>
    · 已改 <b id="s-fix">0</b>
    · 非車牌 <b id="s-np" style="color:#e0b44e">0</b>
    · 廢圖 <b id="s-blur">0</b></div>
</header>

<main>
  <div class="filters">
    <button data-only="todo" class="on">待核對（清晰）</button>
    <button data-only="bad">未通過校驗</button>
    <button data-only="clear">全部清晰</button>
    <button data-only="blurry">模糊廢圖</button>
    <button data-only="notplate">不是車牌</button>
    <button data-only="all">全部</button>
    <button id="btn-eval" style="margin-left:auto;border-color:#2f4a80;color:#8fb8f0">📊 檢驗準確率</button>
  </div>

  <div id="view"></div>
</main>

<script>
let ITEMS = [];
let IDX = 0;
let ONLY = 'todo';

const $ = id => document.getElementById(id);
const esc = s => String(s == null ? '' : s)
  .replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')
  .replace(/"/g,'&quot;');

function stats(s) {
  $('s-done').textContent = s.verified;
  $('s-usable').textContent = s.usable;
  $('s-left').textContent = s.usable_remaining;
  $('s-fix').textContent = s.corrected;
  $('s-blur').textContent = s.blurry;
  if ($('s-np')) $('s-np').textContent = s.not_plate || 0;
  $('pbar').style.width = s.progress + '%';
}

async function load() {
  const r = await fetch('/api/items?only=' + ONLY);
  const j = await r.json();
  ITEMS = j.items;
  IDX = 0;
  stats(j.stats);
  render();
}

function render() {
  const v = $('view');
  if (!ITEMS.length) {
    v.innerHTML = '<div class="done">這個篩選下沒有項目了 🎉</div>';
    return;
  }
  if (IDX >= ITEMS.length) IDX = 0;
  const it = ITEMS[IDX];

  const fmtTag = it.format_ok
    ? '<span class="tag t-ok">' + esc(it.format) + '</span>'
    : '<span class="tag t-bad">格式不符</span>';
  const dblTag = it.double_line
    ? '<span class="tag t-warn">雙行已拼接</span>' : '';
  const ACTION_ZH = { pass: '正確', correct: '已改', skip: '跳過',
                      unusable: '看不清', not_plate: '非車牌' };
  const verTag = it.verified
    ? '<span class="tag ' + (it.action === 'not_plate' ? 't-warn' : 't-ok') + '">'
      + esc(ACTION_ZH[it.action] || it.action) + '</span>' : '';

  v.innerHTML = `
    <div class="card">
      <div class="srcbox" style="margin-top:0">
        <div class="srcwrap" id="srcwrap">
          <img id="srcimg" src="/srcimg/${encodeURIComponent(it.src)}" alt="">
        </div>
        <div class="hint" style="margin-top:6px">綠實線＝正在核對這塊；灰虛線＝同一張圖的<b>其他車牌</b>
          ${(it.siblings||[]).length > 1
            ? '（共 ' + (it.siblings||[]).length + ' 個都要核對，已核對的會淡化打 ✓，點框可切換）'
            : ''}</div>
      </div>

      <div class="navrow">
        <button id="btn-prev">← 上一張<kbd>←</kbd></button>
        <button id="btn-next">下一張 →<kbd>→</kbd></button>
      </div>

      <div class="row" style="margin-top:14px">
        <div class="plate-wrap"><img id="plate" src="/crop/${encodeURIComponent(it.file)}" alt=""></div>
        <div class="meta">
          <div>${fmtTag}${dblTag}${verTag}</div>
          <div>檢測地區 <b>${esc(it.region_det)}</b> · 檢測置信 <b>${(it.det_conf||0).toFixed(2)}</b></div>
          <div>OCR 置信 <b>${(it.ocr_conf||0).toFixed(3)}</b>
            · 清晰度 <b style="color:${it.clarity < 40 ? '#ef7a86' : (it.clarity < 100 ? '#e0b44e' : '#5ec98a')}">${(it.clarity||0).toFixed(0)}</b></div>
          <div>OCR 原始 <b>${esc((it.ocr_raw||[]).join(' | ') || '(空)')}</b></div>
          <div class="hint">第 ${IDX+1} / ${ITEMS.length} 張</div>
        </div>
      </div>

      <div class="minfo" id="minfo">
        ${it.province ? `<span class="chip">📍 <b>${esc(it.province)}</b>${it.city && it.city !== it.province ? ' · ' + esc(it.city) : ''}</span>` : ''}
        ${it.region_note ? `<span class="chip">${esc(it.region_note)}</span>` : ''}
        ${it.plate_type ? `<span class="chip col-${esc(it.plate_color||'unknown')}">🚗 <b>${esc(it.plate_type)}</b></span>` : ''}
        ${it.energy ? `<span class="chip col-green">⚡ ${esc(it.energy)}</span>` : ''}
      </div>

      ${it.clarity < 40 ? '<div style="margin-top:10px;padding:8px 12px;background:#3d2024;border-radius:6px;font-size:12px;color:#ef7a86">⚠️ 此圖極度模糊（清晰度 ' + (it.clarity||0).toFixed(0) + '），可能無法辨識，建議按「無法辨識」跳過</div>' : ''}
      ${it.action === 'not_plate' ? '<div style="margin-top:10px;padding:8px 12px;background:#3b3420;border-radius:6px;font-size:12px;color:#e0b44e">已標記為「不是車牌」—— 此框屬檢測誤報，將作為負樣本，不進入 OCR 訓練集</div>' : ''}

      <label>車牌號碼（可直接編輯）</label>
      <input type="text" id="txt" value="${esc(it.text)}" spellcheck="false" autocomplete="off">

      <div class="actions">
        <button class="${it.format_ok ? 'primary' : 'caution'}" id="btn-pass">
        ${it.format_ok ? '✅ 正確，下一張' : '⚠️ 格式不符，仍提交'} <kbd>Enter</kbd></button>
        <button id="btn-save">✏️ 提交修正 <kbd>Ctrl+Enter</kbd></button>
        <button class="ghost" id="btn-skip">⏭ 跳過</button>
        <button class="ghost" id="btn-np">❌ 不是車牌</button>
        <button class="warn" id="btn-bad">🚫 無法辨識</button>
      </div>
      <div class="hint">提示：直接打字修正後按 <kbd>Ctrl+Enter</kbd> 提交；
        <kbd>Enter</kbd> 表示 OCR 已正確；<kbd>←</kbd> 上一張；
        <kbd>Alt+X</kbd> 標記「不是車牌」（框錯了，畫面裡沒有車牌）</div>

      <div class="regrow">
        <span style="font-size:12px;color:#9aa3b2">地區</span>
        <select id="reg">
          <option value="hk"${it.region_final==='hk'?' selected':''}>香港</option>
          <option value="mo"${it.region_final==='mo'?' selected':''}>澳門</option>
          <option value="mainland"${it.region_final==='mainland'?' selected':''}>內地</option>
          <option value="gangao"${it.region_final==='gangao'?' selected':''}>粵Z</option>
          <option value="unknown"${it.region_final==='unknown'?' selected':''}>未知</option>
        </select>
        <span class="hint" style="margin:0">檢測判定為 <b>${esc(it.region_det)}</b>${it.region_det!==it.region_final?' → 已改為 '+esc(it.region_final):''}</span>
      </div>

    </div>`;

  const plate = $('plate');
  if (plate) plate.onclick = () => plate.classList.toggle('zoom');

  // 在原圖上畫出「同一張圖的所有車牌框」，當前那塊高亮
  const si = $('srcimg'), wrap = $('srcwrap');
  if (si && wrap && it.bbox) {
    const draw = () => {
      const nw = si.naturalWidth, nh = si.naturalHeight;
      if (!nw || !nh) return;
      // 清掉舊框（保留 img）
      wrap.querySelectorAll('.yolobox').forEach(n => n.remove());
      const sx = si.clientWidth / nw, sy = si.clientHeight / nh;
      const sibs = (it.siblings && it.siblings.length) ? it.siblings
                    : [{ file: it.file, bbox: it.bbox, text: it.text,
                         region: it.region_final }];
      sibs.forEach((s, k) => {
        if (!s.bbox) return;
        const isCur = (s.file === it.file);
        const box = document.createElement('div');
        box.className = 'yolobox' + (isCur ? ' cur' : ' other')
                        + (!isCur && s.verified ? ' done' : '');
        const [x1, y1, x2, y2] = s.bbox;
        box.style.left = (x1 * sx) + 'px';
        box.style.top = (y1 * sy) + 'px';
        box.style.width = Math.max(2, (x2 - x1) * sx) + 'px';
        box.style.height = Math.max(2, (y2 - y1) * sy) + 'px';
        box.dataset.label = (isCur ? '▶ ' : (s.verified ? '✓ ' : ''))
          + (s.action === 'not_plate' ? '非車牌'
             : (s.text || '(空)') + ' · ' + (s.region || '?'));
        if (!isCur) {
          box.style.pointerEvents = 'auto';
          box.style.cursor = 'pointer';
          box.onclick = () => {
            const idx = ITEMS.findIndex(x => x.file === s.file);
            if (idx >= 0) { IDX = idx; render(); }
          };
        }
        wrap.appendChild(box);
      });
    };
    if (si.complete) draw(); else si.onload = draw;
    window.addEventListener('resize', draw);
    window.addEventListener('load', draw);
  }

  const reg = $('reg');

  // 即時校驗：輸入框或地區一改，按鈕狀態立刻跟著變
  let checkTimer = null;
  const revalidate = () => {
    clearTimeout(checkTimer);
    checkTimer = setTimeout(async () => {
      const t = $('txt').value.trim();
      const r = ($('reg') && $('reg').value) || '';
      const res = await fetch('/api/check', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: t, region: r, file: it.file }),
      });
      const j = await res.json();
      const bp = $('btn-pass');
      if (!bp) return;
      bp.className = j.format_ok ? 'primary' : 'caution';
      bp.innerHTML = (j.format_ok ? '✅ 正確，下一張' : '⚠️ 格式不符，仍提交')
        + ' <kbd>Enter</kbd>';
      const tag = document.querySelector('.card .tag');
      if (tag) {
        tag.className = 'tag ' + (j.format_ok ? 't-ok' : 't-bad');
        tag.textContent = j.format_ok ? j.format : '格式不符';
      }
      // 省市 / 車型 / 能源 chips 一併重繪
      const mi = $('minfo');
      if (mi) {
        const chips = [];
        if (j.province) {
          chips.push('<span class="chip">📍 <b>' + esc(j.province) + '</b>'
            + (j.city && j.city !== j.province ? ' · ' + esc(j.city) : '') + '</span>');
        }
        if (j.region_note) chips.push('<span class="chip">' + esc(j.region_note) + '</span>');
        if (j.plate_type) {
          chips.push('<span class="chip col-' + esc(j.plate_color || 'unknown') + '">🚗 <b>'
            + esc(j.plate_type) + '</b></span>');
        }
        if (j.energy) chips.push('<span class="chip col-green">⚡ ' + esc(j.energy) + '</span>');
        mi.innerHTML = chips.join('');
      }
    }, 220);
  };
  if ($('txt')) $('txt').addEventListener('input', revalidate);
  if (reg) reg.onchange = () => {
    reg.classList.toggle('edited', reg.value !== it.region_final);
    revalidate();
  };

  const txt = $('txt');
  txt.focus();
  txt.select();
  txt.addEventListener('input', () => {
    txt.classList.toggle('edited', txt.value !== it.text);
  });

  $('btn-pass').onclick = () => submit('pass');
  $('btn-save').onclick = () => submit('correct');
  $('btn-skip').onclick = () => { IDX++; render(); };
  $('btn-bad').onclick = () => submit('unusable');
  if ($('btn-np')) $('btn-np').onclick = () => submit('not_plate');

  // 左右切換照片（不改變核對狀態，純瀏覽）
  const bp = $('btn-prev'), bn = $('btn-next');
  if (bp) bp.onclick = () => { IDX = (IDX - 1 + ITEMS.length) % ITEMS.length; render(); };
  if (bn) bn.onclick = () => { IDX = (IDX + 1) % ITEMS.length; render(); };
}

async function submit(action) {
  const it = ITEMS[IDX];
  if (!it) return;
  const text = $('txt').value.trim();
  const region = ($('reg') && $('reg').value) || '';

  // 防誤送：格式不符又沒改過，要求確認，避免把 OCR 錯值當正確送出
  if (action === 'pass' && text === it.text && !it.format_ok) {
    const ok = confirm(
      '此號碼未通過格式校驗：\n\n  ' + (text || '(空)') +
      '\n\n仍要標記為正確嗎？\n（按「取消」可先修正）');
    if (!ok) { $('txt').focus(); return; }
  }
  const r = await fetch('/api/verify', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ file: it.file, action, text, region }),
  });
  const j = await r.json();
  if (!j.ok) { alert('保存失敗：' + (j.error || '')); return; }
  stats(j.stats);

  // 不從陣列移除已處理項 —— 先前 splice(IDX,1) 會讓陣列位移，
  // IDX 卻原地不動，最後一張時又歸零，看起來就是「整個跳過」。
  // 改為標記 verified 後向後找下一個未核對項，位置永遠穩定。
  it.verified = true;
  it.action = action;
  it.text = text;
  it.format = j.format;
  it.format_ok = j.format_ok;
  it.region_final = region || it.region_final;
  // 同步更新同圖其他框的顯示狀態
  (it.siblings || []).forEach(s => {
    if (s.file === it.file) { s.verified = true; s.text = text; s.region = it.region_final; }
  });

  // 「無法辨識」只影響當前這一塊牌，同圖其他牌仍要逐個核對
  if (!nextUnverified(IDX)) {
    $('view').innerHTML =
      '<div class="done">全部核對完成 🎉<br><span style="font-size:13px;color:#6d7688">'
      + '可用 ' + (j.stats.usable || 0) + ' 張，已核 ' + (j.stats.verified || 0) + ' 張</span></div>';
    return;
  }
  render();
}

/** 從 from 之後找第一個未核對的項；全部核對過回傳 false。 */
function nextUnverified(from) {
  const n = ITEMS.length;
  for (let k = 1; k <= n; k++) {
    const j = (from + k) % n;
    if (!ITEMS[j].verified) { IDX = j; return true; }
  }
  return false;
}

function pct(x) { return (x * 100).toFixed(1) + '%'; }
function cls(v, good, bad) { return v >= good ? 'good' : (v >= bad ? 'mid' : 'bad'); }

async function showEval() {
  const v = $('view');
  v.innerHTML = '<div class="done">計算中…</div>';
  const j = await (await fetch('/api/eval')).json();
  if (j.empty) {
    v.innerHTML = '<div class="done">' + esc(j.message || '') +
      '<br><span style="font-size:13px;color:#6d7688">核對幾張後再回來看</span></div>';
    return;
  }
  const rows = (obj) => Object.entries(obj || {})
    .sort((a, b) => b[1].n - a[1].n)
    .map(([k, s]) => `<tr><td>${esc(k)}</td><td>${s.n}</td>
      <td style="color:${s.em_rate>=0.9?'#5ec98a':(s.em_rate>=0.7?'#e0b44e':'#ef7a86')}">${pct(s.em_rate)}</td>
      <td>${pct(s.cer)}</td></tr>`).join('');

  v.innerHTML = `
    <div class="card">
      <div class="evh" style="margin-top:0">識別準確率（真值＝你核對過的 ${j.total} 個車牌）</div>
      <div class="evalgrid">
        <div class="evalcard"><div class="k">精確匹配 EM</div>
          <div class="v ${cls(j.em_rate, .9, .7)}">${pct(j.em_rate)}</div></div>
        <div class="evalcard"><div class="k">完全正確</div>
          <div class="v">${j.em} / ${j.total}</div></div>
        <div class="evalcard"><div class="k">字符錯誤率 CER</div>
          <div class="v ${j.cer<=0.05?'good':(j.cer<=0.15?'mid':'bad')}">${pct(j.cer)}</div></div>
        <div class="evalcard"><div class="k">錯最多的地方</div>
          <div class="v" style="font-size:14px;color:#e0b44e">${(j.top_errors||[]).length?(esc(j.top_errors[0][0])+' ×'+j.top_errors[0][1]):'—'}</div></div>
      </div>

      <div class="evh">依地區</div>
      <table class="ev"><tr><th>地區</th><th>樣本</th><th>精確匹配</th><th>CER</th></tr>${rows(j.by_region)}</table>

      <div class="evh">依車牌類型</div>
      <table class="ev"><tr><th>類型</th><th>樣本</th><th>精確匹配</th><th>CER</th></tr>${rows(j.by_type)}</table>

      ${(j.top_errors||[]).length ? `
      <div class="evh">常見字符錯誤</div>
      <div class="minfo">${j.top_errors.map(([k, n]) =>
        `<span class="chip">${esc(k)} <b>×${n}</b></span>`).join('')}</div>` : ''}

      ${(j.failures||[]).length ? `
      <div class="evh">錯誤樣本（依錯誤字數排序）</div>
      <table class="ev"><tr><th>真值</th><th>預測</th><th>地區</th><th>錯字數</th></tr>
        ${j.failures.slice(0, 25).map(f => `<tr>
          <td style="color:#5ec98a">${esc(f.gt)}</td>
          <td style="color:#ef7a86">${esc(f.pred || '(空)')}</td>
          <td>${esc(f.region)}</td><td>${f.dist}</td></tr>`).join('')}
      </table>` : '<div class="hint">全部正確 🎉</div>'}

      <div class="navrow"><button id="btn-back">← 回去核對</button></div>
    </div>`;
  $('btn-back').onclick = () => render();
}

document.querySelectorAll('.filters button').forEach(b => {
  b.onclick = () => {
    document.querySelectorAll('.filters button').forEach(x => x.classList.remove('on'));
    b.classList.add('on');
    ONLY = b.dataset.only;
    load();
  };
});

document.addEventListener('keydown', e => {
  // 焦點在輸入框／下拉時，左右鍵要留給游標移動與選項切換，
  // 不能拿來翻頁 —— 否則修改車牌號碼時按方向鍵，照片會突然跳走。
  const tag = (e.target && e.target.tagName || '').toLowerCase();
  const inField = tag === 'input' || tag === 'textarea' || tag === 'select'
                  || (e.target && e.target.isContentEditable);
  if (inField && (e.key === 'ArrowLeft' || e.key === 'ArrowRight')) {
    return;                       // 不 preventDefault，讓瀏覽器正常處理
  }

  if (e.key === 'Enter' && !e.ctrlKey && !e.metaKey) {
    e.preventDefault(); submit('pass');
  } else if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
    e.preventDefault(); submit('correct');
  } else if (e.key === 'ArrowLeft') {
    e.preventDefault(); IDX = (IDX - 1 + ITEMS.length) % ITEMS.length; render();
  } else if (e.key === 'ArrowRight') {
    e.preventDefault(); IDX = (IDX + 1) % ITEMS.length; render();
  } else if (e.key === 'Escape') {
    $('btn-skip') && $('btn-skip').click();
  } else if (e.altKey && (e.key === 'x' || e.key === 'X')) {
    e.preventDefault(); submit('not_plate');
  }
});

load();
document.addEventListener('click', e => {
  if (e.target && e.target.id === 'btn-eval') showEval();
});
</script>
</body>
</html>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8795)  # 固定端口，勿再另開
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--data", default=str(DATA))
    args = ap.parse_args()

    STATE["data_dir"] = Path(args.data)
    if not (STATE["data_dir"] / "labeled.json").exists():
        raise SystemExit(f"找不到 {STATE['data_dir']/'labeled.json'}"
                         "，先跑 prelabel_stage2_ocr.py")

    s = stats()
    print(f"標註核對 WebUI")
    print(f"  資料目錄 {STATE['data_dir']}")
    print(f"  共 {s['total']} 個車牌，已核 {s['verified']}，待核 {s['remaining']}")
    print(f"  http://127.0.0.1:{args.port}")
    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
