#!/usr/bin/env python3
"""
用 PaddleOCR 識別港澳真實車牌，對照人工真值評估能力。

執行環境：~/tools/paddleocr/venv/bin/python3
輸入：ocr_bench/gba_crops/（由 scripts/prep_gba_paddle_test.py 產生的放大裁圖）
輸出：ocr_bench/paddle_results.json
"""
import glob
import json
import re
import sys
from pathlib import Path

from paddleocr import PaddleOCR

ROOT = Path("/home/nicklam-ai/repos/GBA-plate-recognition")
CROPS = ROOT / "ocr_bench" / "gba_crops"
OUT = ROOT / "ocr_bench" / "paddle_results.json"

index = {r["file"]: r for r in json.load(open(CROPS / "index.json"))}

ocr = PaddleOCR(
    lang="ch",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    # 車牌是單行／雙行短文字，det 需要放寬才抓得到
    text_det_limit_side_len=960,
    text_det_limit_type="max",
    text_det_thresh=0.2,
    text_det_box_thresh=0.4,
)


def clean(s: str) -> str:
    """去掉空白與分隔符，便於與真值比對。"""
    return re.sub(r"[\s\-·・]", "", s).upper()


results = []
for f in sorted(glob.glob(str(CROPS / "*.jpg"))):
    name = Path(f).name
    meta = index.get(name, {})
    try:
        r = ocr.predict(f)[0]
        d = r["res"] if isinstance(r, dict) and "res" in r else r
        texts = d.get("rec_texts") or []
        scores = d.get("rec_scores") or []
    except Exception as e:
        texts, scores = [], []
        print(f"  ! {name}: {e}", file=sys.stderr)

    joined = "".join(texts)
    rec = {
        "file": name,
        "region": meta.get("region"),
        "truth": meta.get("truth"),
        "paddle_raw": texts,
        "paddle_scores": [round(float(s), 3) for s in scores],
        "paddle_joined": joined,
        "paddle_clean": clean(joined),
        "lines": len(texts),
    }
    if rec["truth"]:
        t = clean(rec["truth"])
        rec["exact"] = rec["paddle_clean"] == t
        # 字符級相似度
        import difflib
        rec["similarity"] = round(
            difflib.SequenceMatcher(None, rec["paddle_clean"], t).ratio(), 3)
    results.append(rec)

json.dump(results, open(OUT, "w"), ensure_ascii=False, indent=1)

print(f"\n識別 {len(results)} 張\n" + "=" * 86)
print(f"{'檔案':<40} {'真值':<14} {'PaddleOCR':<20} {'相似度':<7} 完全對")
print("=" * 86)
judged = [r for r in results if r.get("truth")]
for r in judged:
    mark = "✓" if r.get("exact") else ("~" if r.get("similarity", 0) >= 0.6 else "✗")
    print(f"{r['file'][:38]:<40} {r['truth']:<14} "
          f"{r['paddle_joined'][:18]:<20} {r.get('similarity',0):<7.2f} {mark}")

if judged:
    exact = sum(1 for r in judged if r.get("exact"))
    sim_avg = sum(r.get("similarity", 0) for r in judged) / len(judged)
    print("=" * 86)
    print(f"有真值樣本 {len(judged)} 張：完全正確 {exact} ({100*exact/len(judged):.0f}%)  "
          f"平均字符相似度 {sim_avg:.3f}")
print(f"\n（無真值樣本 {len(results)-len(judged)} 張，已存入 {OUT.name} 待人工核對）")
