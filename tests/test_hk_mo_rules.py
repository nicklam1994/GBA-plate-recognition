"""港澳車牌格式後處理測試"""

import pytest

from gba_plate.postprocess.hk_mo_rules import (
    apply_format_rules,
    normalize_gangao,
    normalize_hk,
    normalize_mo,
)


class TestHongKong:
    @pytest.mark.parametrize("text,expected", [
        ("AB1234", "AB1234"),
        ("A123", "A123"),
        ("KA1", "KA1"),
        ("FM814", "FM814"),
        ("WW4958", "WW4958"),
    ])
    def test_valid_passthrough(self, text, expected):
        assert normalize_hk(text) == expected

    @pytest.mark.parametrize("text,expected", [
        # 6 位超限 → 截斷數字（2 字母 + 4 數字 = 6，剛好合法）
        ("W58696", "W5869"),
        # 數字開頭 → 字母提前
        ("6X7", "X67"),
        # 字母數字交錯 → 取字母在前
        ("A2A7", "AA27"),
        # 含空格
        ("AB 1234", "AB1234"),
        # 小寫
        ("ab1234", "AB1234"),
    ])
    def test_correctable(self, text, expected):
        assert normalize_hk(text) == expected

    @pytest.mark.parametrize("text", [
        "0000",      # 全數字且以 0 開頭
        "1234",      # 無字母
        "",          # 空
        None,        # None
        "II1234",    # 禁用字母
        "OQ99",      # 全禁用字母
    ])
    def test_uncorrectable(self, text):
        assert normalize_hk(text) is None

    def test_forbidden_letters_stripped(self):
        # I/O/Q 應被剔除
        out = normalize_hk("AI123")
        assert out is not None
        assert "I" not in out


class TestMacau:
    @pytest.mark.parametrize("text,expected", [
        ("MA1234", "MA1234"),
        ("MG5678", "MG5678"),
        ("MA 12 34", "MA1234"),
    ])
    def test_valid(self, text, expected):
        assert normalize_mo(text) == expected

    def test_add_prefix(self):
        # 缺 M 前綴時補上
        assert normalize_mo("A1234") == "MA1234"

    def test_uncorrectable(self):
        assert normalize_mo("") is None
        assert normalize_mo(None) is None
        assert normalize_mo("M") is None


class TestGangao:
    @pytest.mark.parametrize("text,expected", [
        ("粵Z1234港", "粵Z1234港"),
        ("粵Z1234澳", "粵Z1234澳"),
        ("粤Z1234港", "粵Z1234港"),   # 簡體粵 → 繁體
        ("粤ZAK75澳", "粵ZAK75澳"),
    ])
    def test_valid(self, text, expected):
        assert normalize_gangao(text) == expected

    @pytest.mark.parametrize("text", [
        "粵Z1234",    # 缺後綴
        "Z1234港",    # 缺粵
        "",
        None,
    ])
    def test_invalid(self, text):
        assert normalize_gangao(text) is None


class TestApplyFormatRules:
    def test_changed_flag(self):
        text, changed, note = apply_format_rules("W58696", "hk")
        assert changed is True
        assert text == "W5869"
        assert "格式修正" in note

    def test_unchanged_flag(self):
        text, changed, note = apply_format_rules("AB1234", "hk")
        assert changed is False
        assert text == "AB1234"

    def test_unknown_region(self):
        text, changed, note = apply_format_rules("任意文字", "mainland")
        assert changed is False
        assert text == "任意文字"

    def test_uncorrectable_keeps_original(self):
        text, changed, note = apply_format_rules("1234", "hk")
        assert changed is False
        assert text == "1234"
        assert "無法修正" in note
