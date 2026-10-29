"""
港澳車牌格式後處理。

背景：
    OCR 即使加了字符集約束，仍可能輸出不符合格式的結果
    （長度超限、字母數字位置錯亂、開頭為 0）。

    格式規則是強先驗，能修正相當比例的錯誤 —— 而且零成本。

支援：
    香港   1-2 字母 + 1-4 數字，總長 ≤ 6，禁用 I/O/Q，數字不以 0 開頭
    澳門   M + 1 字母 + 4 數字（可選連字符）
    粵Z    粵 Z + 4 位英數 + 港/澳
"""

from __future__ import annotations

import re

#: 香港車牌：1-2 字母 + 1-4 數字
HK_RE = re.compile(r"^([A-Z]{1,2})(\d{1,4})$")
#: 澳門車牌：M + 字母 + 4 位數字
MO_RE = re.compile(r"^M([A-Z])(\d{4})$")
#: 粵Z 兩地牌：粵Z + 4 位英數 + 港/澳
GANGAO_RE = re.compile(r"^[粵粤]Z([A-Z0-9]{4})([港澳])$")

#: 香港禁用字母（與 1/0 混淆）
HK_FORBIDDEN = {"I", "O", "Q"}

#: 易混淆字符對（用於格式修正）
CONFUSE = {
    "0": "O", "O": "0",
    "1": "I", "I": "1",
    "5": "S", "S": "5",
    "2": "Z", "Z": "2",
    "8": "B", "B": "8",
    "6": "G", "G": "6",
}

ALPHA = set("ABCDEFGHJKLMNPQRSTUVWXYZ")
DIGITS = set("0123456789")


def normalize_hk(text: str) -> str | None:
    """
    把 OCR 原始輸出修正成合法香港車牌。

    處理的典型錯誤：
        "0P390"  → 開頭是 0，數字部分不以 0 開頭 → 嘗試 O→Q 替換或丟棄
        "W58696" → 6 位，超出 2+4 上限 → 截斷數字
        "6X7"    → 開頭是數字 → 數字移後
        "A2A7"   → 字母數字交錯 → 取前面的字母 + 後面的數字

    Returns:
        合法車牌字串；無法修正時回傳 None
    """
    if not text:
        return None
    s = "".join(c for c in text.upper() if c in ALPHA or c in DIGITS)
    if not s:
        return None

    # 已合法
    if HK_RE.match(s):
        letters, digits = HK_RE.match(s).groups()  # type: ignore[union-attr]
        if not (set(letters) & HK_FORBIDDEN) and not digits.startswith("0"):
            return s

    # 嘗試：取所有字母（最多 2）在前，所有數字（最多 4）在後
    letters = "".join(c for c in s if c in ALPHA)[:2]
    digits = "".join(c for c in s if c in DIGITS)[:4]

    if not letters:
        return None

    # 去掉禁用字母
    letters = "".join(c for c in letters if c not in HK_FORBIDDEN)
    if not letters:
        return None

    # 數字不能以 0 開頭：裁掉前導 0
    digits = digits.lstrip("0")
    if not digits:
        return None

    candidate = letters + digits
    if len(candidate) <= 6 and HK_RE.match(candidate):
        return candidate
    return None


def normalize_mo(text: str) -> str | None:
    """修正澳門車牌（M + 字母 + 4 數字）"""
    if not text:
        return None
    s = "".join(c for c in text.upper() if c in ALPHA or c in DIGITS)
    if not s:
        return None

    if MO_RE.match(s):
        return s

    # 補 M 前綴
    if not s.startswith("M"):
        if s[0] in ALPHA and len(s) >= 5:
            s = "M" + s
        else:
            return None

    letters = "".join(c for c in s[1:] if c in ALPHA)[:1]
    digits = "".join(c for c in s[1:] if c in DIGITS)[:4]
    if not letters or len(digits) < 3:
        return None
    return f"M{letters}{digits}"


def normalize_gangao(text: str) -> str | None:
    """修正粵Z 兩地牌"""
    if not text:
        return None
    s = text.strip().upper().replace("粤", "粵")

    m = GANGAO_RE.match(s)
    if m:
        return "粵Z" + m.group(1) + m.group(2)

    # 沒有後綴時不強制補（後綴是重要判別特徵）
    return None


def apply_format_rules(text: str, region: str) -> tuple[str, bool, str]:
    """
    對港澳車牌套用格式修正。

    Args:
        text: OCR 輸出
        region: "hk" / "mo" / "gangao"

    Returns:
        (修正後文字, 是否改動, 說明)
    """
    if not text:
        return text, False, "空結果"

    if region == "hk":
        fixed = normalize_hk(text)
    elif region == "mo":
        fixed = normalize_mo(text)
    elif region == "gangao":
        fixed = normalize_gangao(text)
    else:
        return text, False, ""

    if fixed is None:
        return text, False, "格式無法修正"
    if fixed != text:
        return fixed, True, f"格式修正 {text} → {fixed}"
    return fixed, False, ""
