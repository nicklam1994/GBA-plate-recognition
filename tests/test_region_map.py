"""省/市/能源解析測試（純邏輯，不需模型）"""

import pytest

from gba_plate.postprocess.region_map import (
    CITY_CODES,
    ENERGY_LABELS,
    PROVINCE_NAMES,
    describe_plate,
    format_plate_region,
    resolve_energy,
    resolve_plate_region,
)


class TestProvinceNames:
    def test_covers_all_31_mainland_provinces(self):
        """31 個省級行政區 + 港澳 = 33"""
        assert len(PROVINCE_NAMES) == 33
        assert "港" in PROVINCE_NAMES
        assert "澳" in PROVINCE_NAMES

    def test_all_provinces_in_vocab(self):
        """對照表的省份簡稱必須都在識別字符集裡"""
        from gba_plate.ocr.vocab import PROVINCES

        for ch in PROVINCES:
            assert ch in PROVINCE_NAMES, f"{ch} 缺省名"

    def test_city_table_covers_all_provinces(self):
        """每個省都要有地級市表（港澳除外）"""
        missing = set(PROVINCE_NAMES) - set(CITY_CODES) - {"港", "澳"}
        assert not missing, f"缺地級市表: {missing}"

    def test_guangdong_is_complete(self):
        """廣東是粵港澳重點，必須完整（含 X 順德 / Y 南海 / Z 港澳）"""
        gd = CITY_CODES["粤"]
        assert gd["A"] == "廣州市"
        assert gd["B"] == "深圳市"
        assert gd["C"] == "珠海市"
        assert gd["S"] == "東莞市"
        assert gd["X"] == "佛山市順德區"
        assert gd["Y"] == "佛山市南海區"
        assert gd["Z"] == "港澳入出內地車輛"
        # 粵I 與 粵O 不存在（避免與 1/0 混淆）
        assert "I" not in gd
        assert "O" not in gd


class TestResolvePlateRegion:
    @pytest.mark.parametrize(
        "text,province,city",
        [
            ("粤A12345", "廣東省", "廣州市"),
            ("粤B12345", "廣東省", "深圳市"),
            ("粤C12345", "廣東省", "珠海市"),
            ("粤S12345", "廣東省", "東莞市"),
            ("京A12345", "北京市", "北京市區"),
            ("沪B12345", "上海市", "上海市郊區"),
            ("川A12345", "四川省", "成都市"),
            ("浙B12345", "浙江省", "寧波市"),
            ("辽D12345", "遼寧省", "撫順市"),
            ("鄂F12345", "湖北省", "襄陽市"),
            ("琼A12345", "海南省", "海口市"),
            ("新A12345", "新疆維吾爾自治區", "烏魯木齊市"),
        ],
    )
    def test_mainland(self, text, province, city):
        p, c, _ = resolve_plate_region(text)
        assert p == province
        assert c == city

    def test_gangao_with_hk_suffix(self):
        p, c, note = resolve_plate_region("粤Z1234港")
        assert p == "廣東省"
        assert c == "港澳入出內地車輛"
        assert note == "香港入出"

    def test_gangao_with_mo_suffix(self):
        p, c, note = resolve_plate_region("粤ZAB12澳")
        assert p == "廣東省"
        assert c == "港澳入出內地車輛"
        assert note == "澳門入出"

    @pytest.mark.parametrize("text", ["AB1234", "MA1234", "AB 1234"])
    def test_non_mainland_returns_none(self, text):
        p, c, note = resolve_plate_region(text)
        assert p is None
        assert c is None
        assert note  # 要有原因說明

    def test_too_short(self):
        p, _, note = resolve_plate_region("京")
        assert p is None
        assert "過短" in note

    def test_invalid_second_char(self):
        """第 2 位必須是字母"""
        p, _, note = resolve_plate_region("京112345")
        assert p is None
        assert "不是字母" in note

    def test_nonexistent_code(self):
        """粵I / 粵O 不存在"""
        p, c, note = resolve_plate_region("粤I12345")
        assert p == "廣東省"
        assert c is None
        assert "非民用" in note


class TestResolveEnergy:
    @pytest.mark.parametrize(
        "text,code,expect_pure",
        [
            ("沪AF71017", "F", False),
            ("京AD12345", "D", True),
            ("粤BD12345", "D", True),
            ("粤BA12345", "A", True),   # A 是純電動號段用盡後的備用
            ("粤BG12345", "G", False),  # G 是非純電動備用
        ],
    )
    def test_new_energy_small(self, text, code, expect_pure):
        got_code, note = resolve_energy(text)
        assert got_code == code
        # 注意不能用 `"純電動" in note`，「非純電動」也含該子字串
        assert note.startswith("純電動") is expect_pure

    @pytest.mark.parametrize("text", ["京A12345", "粤B12345", "鄂F8888警", "AB1234", "MA1234"])
    def test_non_new_energy(self, text):
        """7 位普通牌 / 帶中文後綴 / 港澳牌都沒有能源標識"""
        code, note = resolve_energy(text)
        assert code is None
        assert note == ""

    def test_energy_labels_complete(self):
        """純電動與非純電動代碼集合不重疊"""
        from gba_plate.postprocess.region_map import ENERGY_NON_PURE, ENERGY_PURE_ELECTRIC

        assert not (ENERGY_PURE_ELECTRIC & ENERGY_NON_PURE)
        assert len(ENERGY_LABELS) == len(ENERGY_PURE_ELECTRIC) + len(ENERGY_NON_PURE)

    def test_large_vehicle_energy_at_last_position(self):
        """大型新能源：能源字母在末位"""
        code, note = resolve_energy("京A12345D")
        assert code == "D"
        assert "純電動" in note
        assert "大型" in note


class TestFormat:
    def test_format_plate_region(self):
        assert format_plate_region("粤B12345") == "廣東省 · 深圳市"

    def test_format_gangao(self):
        assert format_plate_region("粤Z1234港") == "廣東省 · 港澳入出內地車輛（香港入出）"

    def test_format_non_mainland(self):
        assert format_plate_region("AB1234") == ""

    def test_describe_with_energy(self):
        out = describe_plate("沪AF71017")
        assert "上海市" in out
        assert "上海市區" in out
        assert "非純電動" in out

    def test_describe_without_energy(self):
        assert describe_plate("粤B12345") == "廣東省 · 深圳市"
