#!/usr/bin/env python3
"""
識別準確率評測。

用途：拿人工核對過的真值（verified.json），回頭檢驗預測（PaddleOCR / CRNN）
到底準不準 —— 這是把標註工具從「只會收集數據」升級成「能驗證模型」的關鍵。

指標：
    精確匹配率 EM   預測與真值完全一致的比例（車牌場景真正關心的數字）
    CER            字符錯誤率 = 編輯距離 / 真值長度（看得出「差多少」）
    各地區/各類型    分組統計，找出模型在哪一類最弱

用法：
    from gba_plate.evaluate import evaluate_pairs, summarize
    rep = evaluate_pairs(pairs)          # pairs = [(gt, pred, region), ...]
    print(rep["em"], rep["cer"])
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field


def edit_distance(a: str, b: str) -> int:
    """Levenshtein 距離（滾動陣列，O(min(n,m)) 空間）。"""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(
                prev[j] + 1,          # 刪除
                cur[j - 1] + 1,       # 插入
                prev[j - 1] + (ca != cb),  # 替換
            ))
        prev = cur
    return prev[-1]


def char_errors(gt: str, pred: str) -> Counter:
    """
    粗略統計錯在哪些字符上。

    用對齊後的位置比較，而非單純集合差 —— 車牌是定長字串，
    位置資訊比「少了哪個字符」更有診斷價值。
    """
    out: Counter = Counter()
    if not gt or not pred:
        return out
    if len(gt) == len(pred):
        for g, p in zip(gt, pred):
            if g != p:
                out[f"{g}→{p}"] += 1
        return out
    # 長度不同時退化為字符多重集差
    gs, ps = Counter(gt), Counter(pred)
    for ch in gs:
        miss = gs[ch] - ps.get(ch, 0)
        if miss > 0:
            out[f"{ch}→(漏)"] += miss
    for ch in ps:
        extra = ps[ch] - gs.get(ch, 0)
        if extra > 0:
            out[f"(多)→{ch}"] += extra
    return out


@dataclass
class Report:
    """評測結果。"""

    total: int = 0
    em: int = 0                       # 精確匹配數
    cer_sum: float = 0.0              # CER 累加
    by_region: dict = field(default_factory=dict)
    by_type: dict = field(default_factory=dict)
    errors: Counter = field(default_factory=Counter)
    failures: list = field(default_factory=list)

    @property
    def em_rate(self) -> float:
        return self.em / self.total if self.total else 0.0

    @property
    def cer(self) -> float:
        return self.cer_sum / self.total if self.total else 0.0

    def as_dict(self) -> dict:
        return {
            "total": self.total,
            "em": self.em,
            "em_rate": round(self.em_rate, 4),
            "cer": round(self.cer, 4),
            "by_region": self.by_region,
            "by_type": self.by_type,
            "top_errors": self.errors.most_common(15),
            "failures": self.failures[:60],
        }


def evaluate_pairs(pairs, by_type: dict[str, str] | None = None) -> Report:
    """
    評測一組 (真值, 預測) 對。

    Args:
        pairs: [(gt, pred, region), ...]，region 可為空字串
        by_type: {gt: 類型名} 可選，用於分組統計
    """
    rep = Report()
    by_type = by_type or {}
    reg_acc = defaultdict(lambda: {"n": 0, "em": 0, "cer": 0.0})
    typ_acc = defaultdict(lambda: {"n": 0, "em": 0, "cer": 0.0})

    for gt, pred, region in pairs:
        gt = (gt or "").strip().upper()
        pred = (pred or "").strip().upper()
        if not gt:
            continue
        rep.total += 1
        dist = edit_distance(gt, pred)
        cer = dist / len(gt)
        exact = int(dist == 0)
        rep.em += exact
        rep.cer_sum += cer

        rk = region or "unknown"
        reg_acc[rk]["n"] += 1
        reg_acc[rk]["em"] += exact
        reg_acc[rk]["cer"] += cer

        tk = by_type.get(gt) or "未分類"
        typ_acc[tk]["n"] += 1
        typ_acc[tk]["em"] += exact
        typ_acc[tk]["cer"] += cer

        if not exact:
            rep.errors.update(char_errors(gt, pred))
            if len(rep.failures) < 60:
                rep.failures.append({
                    "gt": gt, "pred": pred, "region": rk,
                    "dist": dist, "cer": round(cer, 3),
                })

    def fmt(d):
        return {
            k: {
                "n": v["n"],
                "em_rate": round(v["em"] / v["n"], 4) if v["n"] else 0.0,
                "cer": round(v["cer"] / v["n"], 4) if v["n"] else 0.0,
            }
            for k, v in d.items()
        }

    rep.by_region = fmt(reg_acc)
    rep.by_type = fmt(typ_acc)
    rep.failures.sort(key=lambda x: -x["dist"])
    return rep


if __name__ == "__main__":
    import sys

    # 簡易自測：python -m gba_plate.evaluate
    cases = [("AB1234", "AB1234"), ("AB1234", "AB1235"),
             ("粵B12345", "B12345"), ("TK7687", "7687")]
    r = evaluate_pairs([(g, p, "hk") for g, p in cases])
    print(f"EM {r.em}/{r.total} = {r.em_rate:.1%}   CER {r.cer:.3f}")
    print("錯誤:", r.errors.most_common(5))
    sys.exit(0)
