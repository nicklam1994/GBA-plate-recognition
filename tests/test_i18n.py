"""後端 i18n 測試"""

import pytest

from gba_plate.i18n import (
    DEFAULT_LANG,
    SUPPORTED_LANGS,
    localize_plate_dict,
    localize_text,
    normalize_lang,
    to_simplified,
    to_traditional,
)


class TestNormalizeLang:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("zh-TW", "zh-TW"),
            ("zh-tw", "zh-TW"),
            ("zh_TW", "zh-TW"),
            ("zh-Hant", "zh-TW"),
            ("zh-HK", "zh-TW"),
            ("zh-MO", "zh-TW"),
            ("tw", "zh-TW"),
            ("cht", "zh-TW"),
            ("zh-CN", "zh-CN"),
            ("zh_CN", "zh-CN"),
            ("zh-Hans", "zh-CN"),
            ("zh-SG", "zh-CN"),
            ("cn", "zh-CN"),
            ("chs", "zh-CN"),
            ("en", "en"),
            ("en-US", "en"),
            ("en-GB", "en"),
        ],
    )
    def test_aliases(self, raw, expected):
        assert normalize_lang(raw) == expected

    @pytest.mark.parametrize("raw", [None, "", "fr", "ja", "xx-YY", "  "])
    def test_fallback_to_default(self, raw):
        assert normalize_lang(raw) == DEFAULT_LANG

    def test_supported_langs_declared(self):
        assert set(SUPPORTED_LANGS) == {"zh-TW", "zh-CN", "en"}


class TestConversion:
    def test_to_simplified_basic(self):
        assert to_simplified("廣東省") == "广东省"
        assert to_simplified("非純電動") == "非纯电动"

    def test_to_traditional_basic(self):
        assert to_traditional("广东省") == "廣東省"

    def test_simplified_idempotent(self):
        """已是簡體再轉簡體不變"""
        text = "深圳市"
        assert to_simplified(to_simplified(text)) == to_simplified(text)

    def test_empty_safe(self):
        assert to_simplified("") == ""
        assert to_traditional("") == ""
        assert to_simplified(None) is None  # type: ignore[arg-type]

    def test_mixed_content(self):
        """混合內容（含數字、符號）不能出錯"""
        src = "上海市區（F 非純電動）"
        out = to_simplified(src)
        assert "上海" in out
        assert "F" in out


class TestLocalizeText:
    def test_zh_tw_keeps_traditional(self):
        assert localize_text("廣東省", "zh-TW") == "廣東省"

    def test_zh_cn_converts(self):
        assert localize_text("廣東省", "zh-CN") == "广东省"

    def test_en_keeps_original(self):
        """英文模式下地名保留中文（無英文對照表）"""
        assert localize_text("廣東省", "en") == "廣東省"

    def test_none_passthrough(self):
        assert localize_text(None, "zh-CN") is None


class TestLocalizePlateDict:
    @pytest.fixture
    def sample(self):
        return {
            "text": "粤AF71017",
            "province": "上海市",
            "city": "上海市區",
            "region": "mainland",
            "region_note": "",
            "energy": "F",
            "energy_note": "非純電動（混動/燃料電池）",
            "color": "綠",
            "confidence": 0.99,
            "is_valid": True,
        }

    def test_zh_tw_unchanged(self, sample):
        out = localize_plate_dict(sample, "zh-TW")
        assert out["province"] == "上海市"
        assert out["city"] == "上海市區"
        assert out["color"] == "綠"
        assert out["lang"] == "zh-TW"

    def test_zh_cn_all_fields_converted(self, sample):
        out = localize_plate_dict(sample, "zh-CN")
        assert out["city"] == "上海市区"
        assert out["color"] == "绿"
        assert out["energy_note"] == "非纯电动（混动/燃料电池）"
        assert out["lang"] == "zh-CN"

    def test_text_field_not_converted(self, sample):
        """車牌號碼本身不能繁簡轉換（否則字符對不上）"""
        out = localize_plate_dict(sample, "zh-CN")
        assert out["text"] == "粤AF71017"

    def test_original_dict_not_mutated(self, sample):
        before = dict(sample)
        localize_plate_dict(sample, "zh-CN")
        assert sample == before

    def test_en_passthrough(self, sample):
        out = localize_plate_dict(sample, "en")
        assert out["city"] == "上海市區"
        assert out["lang"] == "en"

    def test_alias_lang(self, sample):
        out = localize_plate_dict(sample, "zh_Hans")
        assert out["lang"] == "zh-CN"
        assert out["color"] == "绿"

    def test_none_fields_safe(self):
        data = {"text": "AB1234", "province": None, "city": None, "region": "hk"}
        out = localize_plate_dict(data, "zh-CN")
        assert out["province"] is None
        assert out["city"] is None
