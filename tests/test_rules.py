"""後處理規則測試（純邏輯，不需模型）"""

import pytest

from gba_plate.base import Region
from gba_plate.postprocess.rules import (
    correct_confusables,
    detect_region_from_text,
    normalize,
    validate,
)


class TestNormalize:
    def test_removes_whitespace(self):
        assert normalize("京 A 12345") == "京A12345"

    def test_removes_dots(self):
        assert normalize("京A·12345") == "京A12345"
        assert normalize("京A.12345") == "京A12345"

    def test_fullwidth_to_halfwidth(self):
        assert normalize("京Ａ１２３４５") == "京A12345"

    def test_traditional_yue_to_simplified(self):
        """繁體「粵」必須統一成簡體「粤」（模型輸出是簡體）"""
        assert normalize("粵Z1234港") == "粤Z1234港"

    def test_traditional_xue_to_simplified(self):
        assert normalize("琼A6716學") == "琼A6716学"

    def test_keeps_hyphen(self):
        """連字符要保留（澳門格式需要）"""
        assert normalize("MA-12-34") == "MA-12-34"

    def test_empty(self):
        assert normalize("") == ""
        assert normalize(None) == ""


class TestValidate:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("京A12345", True),
            ("粤B12345", True),
            ("沪AF71017", True),      # 新能源 6 位
            ("鄂F8888警", True),       # 警車
            ("琼A6716学", True),       # 教練車
            ("沪A1707领", True),       # 領館
            ("A12345", False),        # 無省份
            ("京AB12345", False),     # 字母位錯
        ],
    )
    def test_mainland(self, text, expected):
        ok, _ = validate(text, Region.MAINLAND)
        assert ok is expected

    @pytest.mark.parametrize(
        "text,expected",
        [
            ("粤Z1234港", True),
            ("粤ZAB12澳", True),
            ("粤ZR066港", True),
            ("粤Z1234", False),       # 缺後綴
            ("京A12345", False),      # 不是粵Z
        ],
    )
    def test_gangao(self, text, expected):
        ok, _ = validate(text, Region.GANGAO)
        assert ok is expected

    @pytest.mark.parametrize(
        "text,expected",
        [
            ("AB1234", True),
            ("A567", True),
            ("XY99", True),
            ("DT88", True),
            ("AB12345", False),       # 數字超 4 位
            ("AI1234", False),        # 含禁用字母 I
            ("AB0123", False),        # 數字 0 開頭
        ],
    )
    def test_hk(self, text, expected):
        ok, _ = validate(text, Region.HK)
        assert ok is expected

    @pytest.mark.parametrize(
        "text,expected",
        [
            ("MA-12-34", True),
            ("MA1234", True),
            ("MG-99-88", True),
            ("MW-00-01", True),
            ("MX99-11", True),
            ("A1234", False),         # 不以 M 開頭
        ],
    )
    def test_mo(self, text, expected):
        ok, _ = validate(text, Region.MO)
        assert ok is expected

    def test_empty_always_invalid(self):
        for region in Region:
            ok, _ = validate("", region)
            assert ok is False

    def test_note_is_meaningful(self):
        ok, note = validate("ABC", Region.HK)
        assert not ok
        assert note  # 必須有說明，不能空


class TestDetectRegion:
    def test_gangao_before_province(self):
        """粵Z 必須優先於省份判斷（回歸測試：曾因繁體字誤判為 mainland）"""
        assert detect_region_from_text("粤Z1234港") == Region.GANGAO
        assert detect_region_from_text("粤ZR066港") == Region.GANGAO

    def test_mainland(self):
        assert detect_region_from_text("京A12345") == Region.MAINLAND
        assert detect_region_from_text("粤B12345") == Region.MAINLAND

    def test_mo(self):
        assert detect_region_from_text("MA1234") == Region.MO
        assert detect_region_from_text("MG9988") == Region.MO

    def test_hk(self):
        assert detect_region_from_text("AB1234") == Region.HK
        assert detect_region_from_text("DT88", "黃") == Region.HK

    def test_empty(self):
        assert detect_region_from_text("") == Region.UNKNOWN


class TestCorrectConfusables:
    def test_mainland_second_char_digit_to_letter(self):
        """內地第 2 位應為字母，0→O"""
        fixed, changed = correct_confusables("京012345", Region.MAINLAND)
        assert changed
        assert fixed[1] == "O"

    def test_hk_forbidden_letter_to_digit(self):
        """香港字母段之後的 I/O/Q 應換回數字"""
        fixed, changed = correct_confusables("AB12O4", Region.HK)
        assert changed
        assert "O" not in fixed

    def test_no_change_when_clean(self):
        fixed, changed = correct_confusables("京A12345", Region.MAINLAND)
        assert not changed
        assert fixed == "京A12345"

    def test_empty(self):
        fixed, changed = correct_confusables("", None)
        assert fixed == ""
        assert not changed


class TestCrossBorderPrefixes:
    """
    跨境車牌中「字頭與發牌地不符」的三種情況。

    ZM/ZN/ZP 是香港牌（澳門車入港用）
    FV/FU    是香港牌（內地車入港用）
    HK-xx-xx 是澳門牌（香港車入澳用）
    """

    def test_zm_is_hong_kong(self):
        """ZM 是香港發給澳門車的牌，不是澳門本地牌。"""
        assert detect_region_from_text("ZM6371") == Region.HK
        assert detect_region_from_text("ZN1234") == Region.HK
        assert detect_region_from_text("ZP9999") == Region.HK

    def test_zm_wrong_order_not_hk(self):
        """OCR 若把 ZM 讀成 MZ，會落進澳門規則 —— 記錄此已知風險。"""
        assert detect_region_from_text("MZ6371") == Region.MO

    def test_fv_fu_is_hong_kong(self):
        """FV/FU 是香港發給內地車的牌，不是內地牌。"""
        assert detect_region_from_text("FV1234") == Region.HK
        assert detect_region_from_text("FU8888") == Region.HK

    def test_hk_black_is_macau(self):
        """HK 字頭 + 黑底 = 澳門發給香港配額車的牌。"""
        assert detect_region_from_text("HK1234", color="黑") == Region.MO

    def test_hk_white_is_hong_kong(self):
        """HK 字頭 + 白/黃底 = 香港本地牌。"""
        assert detect_region_from_text("HK1234", color="白") == Region.HK
        assert detect_region_from_text("HK8888", color="黃") == Region.HK

    def test_macau_local_prefix(self):
        """M + 字母是澳門本地牌。"""
        assert detect_region_from_text("MA1234") == Region.MO
        assert detect_region_from_text("MW8888") == Region.MO

    def test_gangao_still_first(self):
        """粵Z 判斷優先於一切。"""
        assert detect_region_from_text("粤Z1234港") == Region.GANGAO
