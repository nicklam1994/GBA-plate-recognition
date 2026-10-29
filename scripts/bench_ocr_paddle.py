#!/usr/bin/env python3
"""
用 PaddleOCR 跑車牌 OCR 基準測試（通用中文 OCR，未針對車牌微調）。

必須用 paddleocr 的 venv 執行：
    ~/tools/paddleocr/venv/bin/python3 scripts/bench_ocr_paddle.py
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path("/home/nicklam-ai/repos/GBA-plate-recognition")


def main():
    from paddleocr import PaddleOCR

    bench = ROOT / "ocr_bench"
    truth = json.load(open(bench / "truth.json", encoding="utf-8"))

    print("載入 PaddleOCR（首次會下載模型）...", flush=True)
    ocr = PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        lang="ch",
    )

    exact = 0
    char_total = 0
    char_correct = 0
    fails = []
    t0 = time.time()

    for name, gt in truth.items():
        img_path = str(bench / "crops" / name)
        try:
            result = ocr.predict(img_path)
        except Exception as e:
            fails.append((gt, f"ERR:{str(e)[:20]}", 0))
            continue

        # 提取文字
        pred = ""
        conf = 0.0
        if result:
            r0 = result[0]
            # PaddleOCR 3.x: 用 dict-like 訪問
            texts = None
            scores = None
            try:
                texts = r0["rec_texts"]
                scores = r0["rec_scores"]
            except Exception:
                try:
                    texts = r0.get("rec_texts")
                    scores = r0.get("rec_scores")
                except Exception:
                    texts = None

            if texts:
                # 多個文本框時選最長（車牌通常是一個框）
                pairs = list(zip(texts, scores if scores else [0] * len(texts)))
                pairs.sort(key=lambda x: -len(str(x[0])))
                pred = str(pairs[0][0]).replace(" ", "")
                conf = float(pairs[0][1])

        if pred == gt:
            exact += 1
        else:
            fails.append((gt, pred, round(conf, 2)))

        char_total += max(len(gt), len(pred))
        for i in range(min(len(gt), len(pred))):
            if gt[i] == pred[i]:
                char_correct += 1

    elapsed = time.time() - t0
    n = len(truth)

    print()
    print("=" * 62)
    print("PaddleOCR 3.x（通用中文，未微調）")
    print("=" * 62)
    print(f"樣本:       {n}")
    print(f"整串正確:   {exact} ({100*exact/max(n,1):.1f}%)")
    print(f"字符級正確: {char_correct}/{char_total} ({100*char_correct/max(char_total,1):.1f}%)")
    print(f"耗時:       {elapsed:.1f}s ({elapsed/max(n,1)*1000:.0f} ms/張)")
    print()
    print(f"錯誤樣本（前 20）:")
    for gt, pred, conf in fails[:20]:
        print(f"  真值 {gt:<10} → 預測 {pred:<10} (conf {conf})")


if __name__ == "__main__":
    main()
