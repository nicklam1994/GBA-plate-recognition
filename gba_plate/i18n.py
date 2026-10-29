"""
後端 i18n：繁簡轉換與語言正規化。

前端 i18n（static/js/i18n.js）負責 UI 靜態文字；
後端負責「隨資料返回的動態文字」——省市名、能源說明、車牌顏色。

依賴 zhconv（純 Python，約 200KB）。未安裝時優雅退化為原文。
"""

from __future__ import annotations

from typing import Any

#: 支援的語言
SUPPORTED_LANGS = ("zh-TW", "zh-CN", "en")

#: 預設語言（繁體，符合項目定位）
DEFAULT_LANG = "zh-TW"

#: 語言別名 → 標準碼
_LANG_ALIASES = {
    "zh": "zh-TW",
    "zh-tw": "zh-TW",
    "zh-hant": "zh-TW",
    "zh-hk": "zh-TW",
    "zh-mo": "zh-TW",
    "tw": "zh-TW",
    "cht": "zh-TW",
    "zh-cn": "zh-CN",
    "zh-hans": "zh-CN",
    "zh-sg": "zh-CN",
    "cn": "zh-CN",
    "chs": "zh-CN",
    "en": "en",
    "en-us": "en",
    "en-gb": "en",
}

#: zhconv 是否可用（延遲偵測）
_zhconv_ready: bool | None = None


def _get_convert():
    """延遲載入 zhconv，未安裝回傳 None"""
    global _zhconv_ready
    if _zhconv_ready is False:
        return None
    try:
        from zhconv import convert

        _zhconv_ready = True
        return convert
    except ImportError:
        _zhconv_ready = False
        return None


def normalize_lang(lang: str | None) -> str:
    """
    把各種語言標記正規化為標準碼。

    Examples:
        >>> normalize_lang("zh-CN")
        'zh-CN'
        >>> normalize_lang("zh_CN")
        'zh-CN'
        >>> normalize_lang("zh-Hans")
        'zh-CN'
        >>> normalize_lang(None)
        'zh-TW'
        >>> normalize_lang("fr")
        'zh-TW'
    """
    if not lang:
        return DEFAULT_LANG
    key = str(lang).strip().lower().replace("_", "-")
    if key in _LANG_ALIASES:
        return _LANG_ALIASES[key]
    # 前綴匹配（如 "zh-CN-x"）
    for alias, code in _LANG_ALIASES.items():
        if key.startswith(alias):
            return code
    return DEFAULT_LANG


def to_simplified(text: str) -> str:
    """
    繁體 → 簡體。zhconv 不可用時原樣返回。
    """
    if not text:
        return text
    convert = _get_convert()
    if convert is None:
        return text
    try:
        return convert(text, "zh-cn")
    except Exception:
        return text


def to_traditional(text: str) -> str:
    """
    簡體 → 繁體。zhconv 不可用時原樣返回。
    """
    if not text:
        return text
    convert = _get_convert()
    if convert is None:
        return text
    try:
        return convert(text, "zh-tw")
    except Exception:
        return text


def localize_text(text: str | None, lang: str) -> str | None:
    """
    按語言本地化單一字串。

    - zh-TW：原文（後端資料本來就是繁體）
    - zh-CN：轉簡體
    - en：地名/專有名詞保留中文（無英文對照表），僅 UI 文字由前端翻譯
    """
    if text is None:
        return None
    code = normalize_lang(lang)
    if code == "zh-CN":
        return to_simplified(text)
    return text


#: 隨結果返回、需要本地化的欄位
_LOCALIZABLE_FIELDS = ("province", "city", "region_note", "energy_note", "color")


def localize_plate_dict(data: dict[str, Any], lang: str) -> dict[str, Any]:
    """
    就地本地化 PlateResult.to_dict() 的結果（回傳新 dict）。

    Args:
        data: PlateResult.to_dict() 的輸出
        lang: 語言標記（任意別名，會自動正規化）

    Returns:
        本地化後的 dict
    """
    code = normalize_lang(lang)
    if code != "zh-CN":
        # zh-TW 與 en 都不需改動後端資料
        out = dict(data)
        out["lang"] = code
        return out

    out = dict(data)
    for field in _LOCALIZABLE_FIELDS:
        if field in out and isinstance(out[field], str):
            out[field] = to_simplified(out[field])
    out["lang"] = code
    return out


if __name__ == "__main__":
    cases = ["廣東省", "深圳市", "遼寧省 · 撫順市", "非純電動（混動/燃料電池）", "藍", "綠"]
    print(f"zhconv 可用: {_get_convert() is not None}")
    print()
    for lang in SUPPORTED_LANGS:
        print(f"[{lang}]")
        for c in cases:
            print(f"  {localize_text(c, lang)}")
        print()
    print("normalize_lang 測試:", [normalize_lang(x) for x in ["zh-CN", "zh_CN", "zh-Hans", None, "fr", "en-US"]])
