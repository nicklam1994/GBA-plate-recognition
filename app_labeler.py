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

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "prelabel" / "hk"

app = Flask(__name__)

STATE = {"data_dir": DATA}

# ---------------------------------------------------------------- 格式規則

RE_HK = re.compile(r"^[A-Z]{1,2}\d{1,4}$")
RE_HK_ZM = re.compile(r"^Z[MNP]\d{3,4}$")
RE_HK_FV = re.compile(r"^F[VU]\d{3,4}$")
RE_MO = re.compile(r"^M[A-Z]\d{4}$")
RE_MO_HK = re.compile(r"^HK\d{4}$")
RE_GANGAO = re.compile(r"^粵Z[A-Z0-9]{3,5}[港澳]$")
RE_MAINLAND = re.compile(r"^粵[A-HJ-NP-Z][A-Z0-9]{4,6}$")

FORMAT_RULES = [
    ("粵Z 兩地牌", RE_GANGAO, "粵Z + 3~5 位英數 + 港/澳"),
    ("內地粵牌", RE_MAINLAND, "粵 + 市字母 + 4~6 位"),
    ("香港 ZM 牌", RE_HK_ZM, "ZM/ZN/ZP + 3~4 位"),
    ("香港 FV/FU 牌", RE_HK_FV, "FV/FU + 3~4 位"),
    ("澳門 HK 牌", RE_MO_HK, "HK + 4 位（澳門發給港車）"),
    ("澳門本地牌", RE_MO, "M + 字母 + 4 位"),
    ("香港本地牌", RE_HK, "1~2 字母 + 1~4 位數字"),
]


def check_format(text: str) -> tuple[bool, str]:
    for name, rx, _ in FORMAT_RULES:
        if rx.match(text):
            return True, name
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


def build_items() -> list[dict]:
    """合併預標註與人工結果，按「優先處理」排序。"""
    d = STATE["data_dir"]
    labeled = load_json(d / "labeled.json", [])
    verified = load_json(d / "verified.json", {})
    clarity = compute_clarity(d)

    items = []
    for r in labeled:
        f = r["file"]
        v = verified.get(f)
        text = (v or {}).get("text") or r.get("text_clean") or ""
        ok, fmt = check_format(text)
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
        })

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
    blurry = sum(1 for i in items if i["clarity"] < CLARITY_BLUR)
    usable = total - blurry
    return {
        "total": total,
        "verified": verified,
        "remaining": total - verified,
        "format_ok": ok,
        "format_bad": total - ok,
        "corrected": corrected,
        "blurry": blurry,
        "usable": usable,
        "usable_remaining": usable - verified,
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
    if only == "todo":
        items = [i for i in items if not i["verified"]
                 and i["clarity"] >= CLARITY_BLUR]
    elif only == "bad":
        items = [i for i in items if not i["format_ok"] and not i["verified"]
                 and i["clarity"] >= CLARITY_BLUR]
    elif only == "clear":
        items = [i for i in items if i["clarity"] >= CLARITY_OK]
    elif only == "blurry":
        items = [i for i in items if i["clarity"] < CLARITY_BLUR]
    return jsonify({"items": items, "stats": stats()})


@app.get("/api/stats")
def api_stats():
    return jsonify(stats())


@app.post("/api/verify")
def api_verify():
    """action: pass（原樣通過） / correct（改為 text） / skip / unusable"""
    body = request.get_json(force=True) or {}
    f = body.get("file")
    action = body.get("action", "pass")
    text = (body.get("text") or "").strip()

    if not f:
        return jsonify({"ok": False, "error": "缺少 file"}), 400
    d = STATE["data_dir"]
    verified = load_json(d / "verified.json", {})

    if action == "pass":
        # 用當前顯示的 OCR 文字作為最終值
        cur = next((i for i in build_items() if i["file"] == f), None)
        text = (cur or {}).get("text", text)
    elif action == "unusable":
        text = ""

    verified[f] = {
        "text": text,
        "action": action,
        "ts": int(time.time()),
    }
    save_json(d / "verified.json", verified)

    ok, fmt = check_format(text) if text else (False, "")
    return jsonify({"ok": True, "text": text, "format": fmt,
                    "format_ok": ok, "stats": stats()})


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
  .srcbox { margin-top: 10px; }
  .srcbox img { max-width: 100%; border-radius: 6px; border: 1px solid #2c2f3c; }
  details summary { cursor: pointer; font-size: 12px; color: #6d7688; }
</style>
</head>
<body>
<header>
  <h1>車牌標註核對</h1>
  <div class="bar"><i id="pbar" style="width:0%"></i></div>
  <div class="stat">已核 <b id="s-done">0</b>/<b id="s-usable">0</b>
    · 待核 <b id="s-left">0</b>
    · 已改 <b id="s-fix">0</b>
    · 廢圖 <b id="s-blur">0</b></div>
</header>

<main>
  <div class="filters">
    <button data-only="todo" class="on">待核對（清晰）</button>
    <button data-only="bad">未通過校驗</button>
    <button data-only="clear">全部清晰</button>
    <button data-only="blurry">模糊廢圖</button>
    <button data-only="all">全部</button>
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
  const verTag = it.verified
    ? '<span class="tag t-ok">已核 ' + esc(it.action) + '</span>' : '';

  v.innerHTML = `
    <div class="card">
      <div class="row">
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

      ${it.clarity < 40 ? '<div style="margin-top:10px;padding:8px 12px;background:#3d2024;border-radius:6px;font-size:12px;color:#ef7a86">⚠️ 此圖極度模糊（清晰度 ' + (it.clarity||0).toFixed(0) + '），可能無法辨識，建議按「無法辨識」跳過</div>' : ''}

      <label>車牌號碼（可直接編輯）</label>
      <input type="text" id="txt" value="${esc(it.text)}" spellcheck="false" autocomplete="off">

      <div class="actions">
        <button class="${it.format_ok ? 'primary' : 'caution'}" id="btn-pass">
        ${it.format_ok ? '✅ 正確，下一張' : '⚠️ 格式不符，仍提交'} <kbd>Enter</kbd></button>
        <button id="btn-save">✏️ 提交修正 <kbd>Ctrl+Enter</kbd></button>
        <button class="ghost" id="btn-skip">⏭ 跳過</button>
        <button class="warn" id="btn-bad">🚫 無法辨識</button>
      </div>
      <div class="hint">提示：直接打字修正後按 <kbd>Ctrl+Enter</kbd> 提交；
        <kbd>Enter</kbd> 表示 OCR 已正確；<kbd>←</kbd> 上一張</div>

      <details class="srcbox"><summary>查看原圖上下文</summary>
        <img src="/srcimg/${encodeURIComponent(it.src)}" alt="">
      </details>
    </div>`;

  const plate = $('plate');
  if (plate) plate.onclick = () => plate.classList.toggle('zoom');

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
}

async function submit(action) {
  const it = ITEMS[IDX];
  if (!it) return;
  const text = $('txt').value.trim();

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
    body: JSON.stringify({ file: it.file, action, text }),
  });
  const j = await r.json();
  if (!j.ok) { alert('保存失敗：' + (j.error || '')); return; }
  stats(j.stats);
  if (ONLY === 'todo') {
    ITEMS.splice(IDX, 1);          // 待核對模式下移除已處理項
    if (IDX >= ITEMS.length) IDX = 0;
  } else {
    it.verified = true; it.action = action; it.text = text;
    it.format = j.format; it.format_ok = j.format_ok;
    IDX++;
  }
  render();
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
  if (e.key === 'Enter' && !e.ctrlKey && !e.metaKey) {
    e.preventDefault(); submit('pass');
  } else if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
    e.preventDefault(); submit('correct');
  } else if (e.key === 'ArrowLeft') {
    e.preventDefault(); if (IDX > 0) { IDX--; render(); }
  } else if (e.key === 'Escape') {
    $('btn-skip') && $('btn-skip').click();
  }
});

load();
</script>
</body>
</html>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8791)
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
