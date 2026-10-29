"""
四地車牌格式校驗與字符糾正。

這是三輪調研確認的空白區 —— 所有開源項目只有「識別出字符」，
沒有人做車牌格式合法性校驗。港澳格式規則尤其缺失。

⚠️ 字符一律用**簡體**（與識別模型的 78 類字符表一致）：
   粤 (U+7CA4) 而非 粵 (U+7CB5)
   学 (U+5B66) 而非 學
   领 / 使 / 警 / 港 / 澳 均為字符表原字

格式規範（已核實）：
    內地  省 + 字母 + 5 位（新能源 6 位）+ 可選特殊後綴（警/学/领/使/挂/港/澳）
    粵Z   粤Z + 4 位 + 港/澳（永遠單行）
    香港  1-2 字母 + 1-4 數字；禁用 I/O/Q；數字不以 0 開頭
    澳門  M + 1 字母 + 2 數字 + 2 數字（單行）；雙行 MA / 12-34
"""

from __future__ import annotations

import re

from gba_plate.base import Region
from gba_plate.ocr.vocab import is_province

# --------------------------------------------------------------------- 格式

#: 內地普通牌：省 + 字母 + 5 位英數
#: 字母池排除 I 與 O —— GA36-2018 明訂序號編碼不使用這兩個字母
#: （I/1、O/0 肉眼與電子檢控都易混淆）。四地規定一致，港澳另加 Q/V。
ALNUM_NO_IO = "A-HJ-NP-Z0-9"
ALPHA_NO_IO = "A-HJ-NP-Z"

MAINLAND_RE = re.compile(rf"^[\u4e00-\u9fa5][{ALPHA_NO_IO}][{ALNUM_NO_IO}]{{5}}$")

#: 內地新能源小型車：省 + 字母 + D/F + 5 位（D/F 在第 3 位）
MAINLAND_NE_SMALL_RE = re.compile(r"^[\u4e00-\u9fa5][A-Z][DF][A-Z0-9]{5}$")

#: 內地新能源大型車：省 + 字母 + 5 位 + D/F（D/F 在末位）
MAINLAND_NE_LARGE_RE = re.compile(r"^[\u4e00-\u9fa5][A-Z][A-Z0-9]{5}[DF]$")

#: 內地特殊牌：省 + 字母 + 數字 + 中文後綴（警/学/领/使/挂/港/澳/试/超）
MAINLAND_SPECIAL_RE = re.compile(
    r"^[\u4e00-\u9fa5][A-Z][A-Z0-9]{0,5}[警学领使挂港澳试超]$"
)

#: 粵Z：粤Z + 4 位英數 + 港/澳（注意用簡體「粤」）
GANGAO_RE = re.compile(r"^粤Z[A-Z0-9]{4}[港澳]$")

#: 粵Z 寬鬆版：預留給 OCR 漏字/多字情況
GANGAO_LOOSE_RE = re.compile(r"^粤Z[A-Z0-9]{2,5}[港澳]$")

#: 香港：1-2 字母 + 1-4 數字；字母不含 I/O/Q；數字首位非 0
# 香港字母池剔除 I、O、Q（不禁 V）—— 寫成 A-HJ-NPR-Z，
# 不能用 P-Z，否則 Q 會被範圍包含進去
HK_ALPHA = "A-HJ-NPR-Z"
HK_RE = re.compile(rf"^[{HK_ALPHA}]{{1,2}}[1-9]\d{{0,3}}$")

#: 澳門單行：M + 字母 + 2 數字 + 2 數字（連字符可有可無）
#: 澳門車牌 M 後面的字母池剔除 I、O、Q、V
#: （DSAT 為避免與 1/0 混淆而移除）。實務上 MH 發完直接跳 MJ，MN 之後跳 MP。
# 注意 P-R 是範圍會把 Q 也包進來，必須寫成 P,R-T 才真的排除 Q
MO_ALPHA = "A-HJ-NPR-TUW-Z"
MO_RE = re.compile(rf"^M[{MO_ALPHA}]-?\d{{2}}-?\d{{2}}$")

#: 澳門雙行（OCR 後拼成一行）：MA12-34 或 MA1234
MO_DOUBLE_RE = re.compile(rf"^M[{MO_ALPHA}]\d{{2}}[-\s]?\d{{2}}$")

#: 澳門發給「香港私家車往來港澳常規配額」的車牌：HK + 4 位純流水數字。
#: 由 HK-00-01 起順序發放，所以前導 0 是正常現象（HK_RE 不允許前導 0，
#: 那是香港本地牌自己的規則，兩者不能混用）。
#: 流水號不含字母，自然排除了 I/O/Q。
MO_HK_RE = re.compile(r"^HK\d{4}$")

#: 香港車牌禁用字母（易與數字混淆）
HK_FORBIDDEN_LETTERS = frozenset("IOQ")

#: 粵Z 前綴（簡體「粤」，與字符表一致）
GANGAO_PREFIX = "粤Z"

#: 易混淆：數字 → 字母（用於省份後的字母位）
_DIGIT_TO_LETTER = {"0": "O", "1": "I", "5": "S", "8": "B", "2": "Z"}

#: 易混淆：字母 → 數字（用於香港/澳門的數字位）
_LETTER_TO_DIGIT = {"O": "0", "I": "1", "S": "5", "B": "8", "Z": "2", "Q": "0"}


# --------------------------------------------------------------------- 正規化

def normalize(text: str) -> str:
    """
    正規化 OCR 原始輸出。

    - 全形字母數字轉半形
    - 繁体「粵」統一為簡體「粤」（模型輸出是簡體）
    - 去除空白與雜訊符號（連字符保留給澳門）
    """
    if not text:
        return ""

    trans = str.maketrans(
        "０１２３４５６７８９ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ",
        "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ",
    )
    text = text.translate(trans)

    # 繁体 → 簡體（字符表用的是簡體）
    text = text.replace("粵", "粤").replace("學", "学")

    # 去掉分隔符與空白（連字符 '-' 保留給澳門 MA-12-34）
    text = re.sub(r"[\s·.,，。、_]+", "", text)

    # 字母統一轉大寫。先前漏了這步，使用者手動輸入 xm9112 會判定格式不符，
    # 而 OCR 輸出本身是大寫，所以只有人工核對時才暴露這個問題。
    text = text.upper()
    return text.strip()


# --------------------------------------------------------------------- 糾正

def correct_confusables(text: str, region: Region | None = None) -> tuple[str, bool]:
    """
    糾正易混淆字符。

    Args:
        text: 正規化後的文字
        region: 已知地區（None 用保守策略）

    Returns:
        (糾正後文字, 是否有改動)
    """
    if not text:
        return text, False

    chars = list(text)
    changed = False

    if region == Region.HK:
        # 香港：字母段把 I/O/Q 換回數字，數字段把字母換回數字
        for i, ch in enumerate(chars):
            if ch.isdigit():
                continue
            if ch in _LETTER_TO_DIGIT:
                # 只有在「已經出現過字母」之後才視為數字段
                if i >= 1:
                    chars[i] = _LETTER_TO_DIGIT[ch]
                    changed = True

    elif region in (Region.MAINLAND, Region.GANGAO):
        # 內地第 2 位、粵Z 第 3 位應為字母/數字混合，只修第 2 位
        if len(chars) >= 2 and chars[1].isdigit() and chars[1] in _DIGIT_TO_LETTER:
            chars[1] = _DIGIT_TO_LETTER[chars[1]]
            changed = True

    return "".join(chars), changed


# --------------------------------------------------------------------- 校驗

def validate(text: str, region: Region) -> tuple[bool, str]:
    """
    校驗車牌文字是否符合該地區格式。

    Args:
        text: 待校驗文字（已正規化）
        region: 地區

    Returns:
        (是否合法, 說明)
    """
    if not text:
        return False, "空字串"

    if region == Region.MAINLAND:
        if not is_province(text[0]):
            return False, f"首字 '{text[0]}' 不是省級簡稱"
        if MAINLAND_RE.match(text):
            return True, "ok"
        if MAINLAND_NE_SMALL_RE.match(text):
            return True, "ok(新能源小型)"
        if MAINLAND_NE_LARGE_RE.match(text):
            return True, "ok(新能源大型)"
        if MAINLAND_SPECIAL_RE.match(text):
            return True, "ok(特殊後綴)"
        return False, "不符合「省+字母+5位」或新能源/特殊格式"

    if region == Region.GANGAO:
        if GANGAO_RE.match(text):
            return True, "ok"
        if GANGAO_LOOSE_RE.match(text):
            return True, "ok(寬鬆)"
        if not text.startswith(GANGAO_PREFIX):
            return False, "未以「粤Z」開頭"
        if text[-1] not in "港澳":
            return False, "未以「港」或「澳」結尾"
        return False, "不符合「粤Z+4位+港/澳」格式"

    if region == Region.HK:
        if len(text) > 8:
            return False, f"長度 {len(text)} 超出香港車牌上限 8"
        # HK + 4 位純數字有兩種可能，不能一概而論：
        #   HK0001（前導 0）→ 必為澳門發的配額車，因為香港本地牌禁止前導 0
        #   HK1393（無前導 0）→ 可能是香港本地牌（白/黃底），也可能是配額車（黑底）
        # 後者要靠底色區分（detect_region_from_text 的 color 參數），
        # validate 只拿到 text 與 region，不能擅自斷定來源。
        if re.match(r"^HK0\d{3}$", text):
            return True, "ok(澳門HK字頭)"
        if HK_RE.match(text):
            return True, "ok"
        bad = [c for c in text if c in HK_FORBIDDEN_LETTERS]
        if bad:
            return False, f"含香港禁用字母 {bad}"
        if re.match(r"^[A-HJ-NP-Z]{1,2}0\d{0,3}$", text):
            return False, "數字部分以 0 開頭"
        return False, "不符合「1-2字母+1-4數字」格式"

    if region == Region.MO:
        # 澳門發給香港配額車的 HK 字頭，要排在本地 M 字頭之前判斷，
        # 否則 HK1393 會先被 Region.HK 的一般規則吃掉（見 validate 的 HK 分支）
        if MO_HK_RE.match(text):
            return True, "ok(澳門HK字頭)"
        if MO_RE.match(text) or MO_DOUBLE_RE.match(text):
            return True, "ok"
        if text.startswith("HK"):
            return False, "HK 字頭須為 4 位純數字（如 HK1393）"
        return False, "不符合「M+字母+2數字+2數字」格式"

    # UNKNOWN
    if len(text) < 2:
        return False, "長度過短"
    return True, "region 未知，未做格式校驗"


# --------------------------------------------------------------------- 推斷

def detect_region_from_text(text: str, color: str | None = None) -> Region:
    """
    從文字 + 顏色推斷地區（檢測器未提供地區時的 fallback）。

    優先級（關鍵：粵Z 必須先判斷，否則會被省份規則吃掉）：
        1. 粤Z 前綴          → GANGAO
        2. 首字是省份        → MAINLAND
        3. ZM/ZN/ZL 開頭    → HK     澳門車入港，香港發的牌
        4. FV/FU 開頭       → HK     內地車入港，香港發的牌
        5. HK 開頭 + 黑底   → MO     澳門發給香港配額車的專用號段
        6. M + 字母開頭      → MO     澳門本地牌
        7. 純英數           → HK

    第 3~5 條處理「字頭與發牌地不符」的情況 —— 這是跨境車牌最容易誤判處：
        ZM1234  是香港牌（不是澳門）  Zhuhai-Macao 入境香港專用
        FV1234  是香港牌（不是內地）  內地企業車入港用
        HK-12-34 是澳門牌（不是香港） 澳門發給港車的配額號段

    Args:
        text: 正規化後的文字
        color: 車牌顏色（PlateColor.value，如 "黑"/"白"/"黃"）

    Returns:
        推斷的地區，無法判斷時 UNKNOWN
    """
    if not text:
        return Region.UNKNOWN

    # 1. 粵Z 兩地牌（必須最先判斷）
    if text.startswith(GANGAO_PREFIX):
        return Region.GANGAO

    # 2. 內地（首字省份）
    if is_province(text[0]):
        return Region.MAINLAND

    # 3. ZM/ZN/ZL —— 澳門私家車往來港澳常規配額，經港珠澳大橋入港，領香港號牌
    if re.match(r"^Z[MNL]", text):
        return Region.HK

    # 4. FV/FU —— 內地企業車入香港，領的是香港號牌（前白後黃）
    if re.match(r"^F[VU]", text):
        return Region.HK

    # 5. HK-xx-xx —— 澳門發給香港配額車的專用號段，黑底白字
    #    與香港本地可能的 "HK1234" 靠顏色區分（澳門牌前後皆黑底）
    if re.match(r"^HK\d", text) and color == "黑":
        return Region.MO

    # 6. 澳門（M + 字母）
    if re.match(r"^M[A-Z]", text):
        return Region.MO

    # 7. 香港（純英數）
    if re.match(r"^[A-Z0-9]+$", text):
        return Region.HK

    return Region.UNKNOWN
