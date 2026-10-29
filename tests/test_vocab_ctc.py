"""字表與 CTC 解碼測試（純邏輯，不需模型）"""

import numpy as np
import pytest

from gba_plate.ocr.vocab import (
    BASE_VOCAB_V78,
    VOCAB_V79,
    blank_index,
    build_index_map,
    build_vocab,
    has_gangao_suffix,
    is_province,
    vocab_size,
)
from gba_plate.ocr.crnn import ctc_greedy_decode


class TestVocab:
    def test_v78_size(self):
        assert vocab_size(False) == 78
        assert len(BASE_VOCAB_V78) == 78

    def test_v79_size(self):
        assert vocab_size(True) == 79
        assert len(VOCAB_V79) == 79

    def test_v79_extends_v78(self):
        """79 類必須是 78 類加一個字符，保持索引兼容"""
        assert VOCAB_V79[:78] == BASE_VOCAB_V78
        assert VOCAB_V79[78] == "-"

    def test_blank_is_zero(self):
        assert BASE_VOCAB_V78[0] == "#"
        assert blank_index() == 0

    def test_contains_gangao_chars(self):
        """港澳字符必須在表內（粵Z 與澳門牌依賴）"""
        assert "港" in BASE_VOCAB_V78
        assert "澳" in BASE_VOCAB_V78

    def test_uses_simplified_yue(self):
        """必須是簡體「粤」，不是繁體「粵」"""
        assert "粤" in BASE_VOCAB_V78
        assert "粵" not in BASE_VOCAB_V78

    def test_no_letter_i_o(self):
        """車牌字母不含 I/O（避免與 1/0 混淆）"""
        assert "I" not in BASE_VOCAB_V78
        assert "O" not in BASE_VOCAB_V78

    def test_provinces(self):
        assert is_province("京")
        assert is_province("粤")
        assert not is_province("A")
        assert not is_province("港")

    def test_index_map_is_bijective(self):
        m = build_index_map()
        assert len(m) == 78
        assert len(set(m.values())) == 78

    def test_gangao_suffix(self):
        assert has_gangao_suffix("粤Z1234港")
        assert has_gangao_suffix("粤Z1234澳")
        assert not has_gangao_suffix("京A12345")


class TestCTC:
    def test_basic_decode(self):
        """index 1..N → 對應字符"""
        vocab = BASE_VOCAB_V78
        indices = np.array([1, 2, 3])       # 京 沪 津
        probs = np.array([0.9, 0.8, 0.7])
        text, probs_out = ctc_greedy_decode(indices, probs, vocab)
        assert text == "京沪津"
        assert len(probs_out) == 3

    def test_skip_blank(self):
        """blank(0) 必須被跳過"""
        vocab = BASE_VOCAB_V78
        indices = np.array([0, 1, 0, 2, 0])
        probs = np.array([0.9, 0.9, 0.9, 0.9, 0.9])
        text, _ = ctc_greedy_decode(indices, probs, vocab)
        assert text == "京沪"

    def test_collapse_repeats(self):
        """連續相同索引只保留一次（CTC 折疊）"""
        vocab = BASE_VOCAB_V78
        indices = np.array([1, 1, 1, 0, 2])
        probs = np.array([0.5, 0.6, 0.7, 0.9, 0.8])
        text, probs_out = ctc_greedy_decode(indices, probs, vocab)
        assert text == "京沪"
        # 保留的是第一次出現时的概率
        assert probs_out == [0.5, 0.8]

    def test_repeat_separated_by_blank(self):
        """被 blank 隔開的相同索引要保留兩個字符"""
        vocab = BASE_VOCAB_V78
        indices = np.array([1, 0, 1])
        probs = np.array([0.9, 0.9, 0.9])
        text, _ = ctc_greedy_decode(indices, probs, vocab)
        assert text == "京京"

    def test_all_blank(self):
        vocab = BASE_VOCAB_V78
        indices = np.array([0, 0, 0])
        probs = np.array([0.9, 0.9, 0.9])
        text, probs_out = ctc_greedy_decode(indices, probs, vocab)
        assert text == ""
        assert probs_out == []

    def test_out_of_range_ignored(self):
        """索引超出詞表長度時忽略，不拋異常"""
        vocab = BASE_VOCAB_V78
        indices = np.array([1, 999])
        probs = np.array([0.9, 0.9])
        text, _ = ctc_greedy_decode(indices, probs, vocab)
        assert text == "京"
