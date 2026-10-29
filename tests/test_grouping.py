"""車輛級聚合（一車多牌）測試。"""
import pytest

from gba_plate.base import Detection, PlateResult, Region
from gba_plate.grouping import (
    VehicleResult,
    check_combo,
    group_vehicles,
    same_vehicle,
    x_overlap_ratio,
)


def mk(text, bbox, region=Region.UNKNOWN, conf=0.9):
    """造一個 PlateResult。bbox = (x1, y1, x2, y2)"""
    det = Detection(bbox=tuple(float(v) for v in bbox), confidence=conf,
                    corners=None, is_double=False)
    return PlateResult(detection=det, ocr=None, region=region, text=text)


# 三地牌車實測佈局（ScreenShot_101020）：三塊牌垂直堆疊、水平對齊
TRI = [
    mk("粤Z7366澳", (240, 537, 463, 602), Region.GANGAO),   # 粵Z
    mk("MI1388",    (233, 596, 465, 647), Region.MO),        # 澳門
    mk("ZM6371",    (220, 641, 479, 696), Region.HK),        # 香港 ZM
]


class TestGeometry:
    def test_x_overlap_full(self):
        a = mk("A", (100, 100, 300, 150))
        b = mk("B", (100, 160, 300, 210))
        assert x_overlap_ratio(a, b) == pytest.approx(1.0)

    def test_x_overlap_none(self):
        a = mk("A", (100, 100, 200, 150))
        b = mk("B", (400, 160, 500, 210))
        assert x_overlap_ratio(a, b) == 0.0

    def test_x_overlap_narrower_denominator(self):
        a = mk("A", (100, 100, 300, 150))   # 寬 200
        b = mk("B", (250, 160, 350, 210))   # 寬 100，重疊 50
        assert x_overlap_ratio(a, b) == pytest.approx(0.5)


class TestSameVehicle:
    def test_stacked_plates_are_same_vehicle(self):
        assert same_vehicle(TRI[0], TRI[1])
        assert same_vehicle(TRI[1], TRI[2])
        assert same_vehicle(TRI[0], TRI[2])

    def test_far_apart_not_same(self):
        a = mk("A", (100, 100, 300, 150))
        b = mk("B", (100, 400, 300, 450))   # 垂直距離 300px，牌寬 200
        assert not same_vehicle(a, b)

    def test_horizontal_neighbours_not_same(self):
        """左右並排的兩個牌 x 重疊為 0，不應聚為同車。"""
        a = mk("A", (100, 100, 300, 150))
        b = mk("B", (320, 100, 520, 150))
        assert not same_vehicle(a, b)

    def test_scale_mismatch_not_same(self):
        """遠處小牌與近處大牌尺度差太多，不聚類。"""
        big = mk("A", (100, 100, 500, 200))    # 寬 400
        small = mk("B", (120, 210, 220, 240))  # 寬 100，比例 4.0
        assert not same_vehicle(big, small)


class TestGrouping:
    def test_single_plate(self):
        vs = group_vehicles([mk("A123", (0, 0, 100, 40), Region.HK)])
        assert len(vs) == 1
        assert not vs[0].is_multi_plate

    def test_empty(self):
        assert group_vehicles([]) == []

    def test_three_plate_single_vehicle(self):
        vs = group_vehicles(TRI)
        assert len(vs) == 1
        v = vs[0]
        assert v.is_multi_plate
        assert len(v.plates) == 3
        # 由上而下排序
        assert v.texts == ["粤Z7366澳", "MI1388", "ZM6371"]

    def test_two_vehicles_separated(self):
        a = [mk("粤Z1港", (100, 100, 300, 150), Region.GANGAO),
             mk("HK1", (100, 160, 300, 210), Region.HK)]
        b = [mk("粤Z2港", (100, 600, 300, 650), Region.GANGAO),
             mk("HK2", (100, 660, 300, 710), Region.HK)]
        vs = group_vehicles(a + b)
        assert len(vs) == 2
        # 每組兩塊牌，且粵Z 在上（texts[0]）
        assert all(len(v.plates) == 2 for v in vs)
        assert sorted(v.texts[0] for v in vs) == ["粤Z1港", "粤Z2港"]
        assert sorted(v.texts[1] for v in vs) == ["HK1", "HK2"]

    def test_primary_prefers_gangao(self):
        vs = group_vehicles(TRI)
        assert vs[0].primary.region == Region.GANGAO
        assert vs[0].primary.text == "粤Z7366澳"


class TestCombo:
    def test_combo_label(self):
        v = group_vehicles(TRI)[0]
        assert v.combo == "粤Z+澳門+香港"

    def test_entry_direction_ao(self):
        v = group_vehicles(TRI)[0]
        assert v.entry_direction == "澳"

    def test_entry_direction_gang(self):
        plates = [mk("粤Z288K港", (100, 100, 300, 150), Region.GANGAO),
                  mk("TB8888", (100, 160, 300, 210), Region.HK)]
        v = group_vehicles(plates)[0]
        assert v.entry_direction == "港"

    def test_no_direction_for_plain_plate(self):
        v = group_vehicles([mk("AB1234", (0, 0, 100, 40), Region.HK)])[0]
        assert v.entry_direction is None


class TestCheckCombo:
    def test_gang_suffix_without_hk_flagged(self):
        plates = [mk("粤Z1234港", (100, 100, 300, 150), Region.GANGAO),
                  mk("MI1388", (100, 160, 300, 210), Region.MO)]
        v = group_vehicles(plates)[0]
        notes = check_combo(v)
        assert any("未見香港牌" in n for n in notes)

    def test_ao_suffix_with_mo_ok(self):
        notes = check_combo(group_vehicles(TRI)[0])
        assert not any("未見澳門牌" in n for n in notes)

    def test_zm_without_mo_flagged(self):
        plates = [mk("ZM6371", (100, 100, 300, 150), Region.HK),
                  mk("AB1234", (100, 160, 300, 210), Region.HK)]
        v = group_vehicles(plates)[0]
        notes = check_combo(v)
        assert any("澳門本地牌" in n for n in notes)

    def test_single_plate_no_notes(self):
        v = group_vehicles([mk("AB1234", (0, 0, 100, 40), Region.HK)])[0]
        assert check_combo(v) == []


class TestDescribe:
    def test_describe_multi(self):
        v = group_vehicles(TRI)[0]
        s = v.describe()
        assert "[多牌]" in s
        assert "粤Z:粤Z7366澳" in s


class TestCrossBorderCombos:
    """跨境組合校驗：FV/FU 與 HK-xx-xx。"""

    def test_fv_without_mainland_flagged(self):
        plates = [mk("FV1234", (100, 100, 300, 150), Region.HK),
                  mk("AB8888", (100, 160, 300, 210), Region.HK)]
        v = group_vehicles(plates)[0]
        notes = check_combo(v)
        assert any("內地牌" in n for n in notes)

    def test_fv_with_mainland_ok(self):
        plates = [mk("FV1234", (100, 100, 300, 150), Region.HK),
                  mk("粵A12345", (100, 160, 300, 210), Region.MAINLAND)]
        v = group_vehicles(plates)[0]
        assert not any("內地牌" in n for n in check_combo(v))

    def test_hk_macau_plate_without_hk_flagged(self):
        plates = [mk("HK1234", (100, 100, 300, 150), Region.MO),
                  mk("MA1234", (100, 160, 300, 210), Region.MO)]
        v = group_vehicles(plates)[0]
        notes = check_combo(v)
        assert any("香港牌" in n for n in notes)

    def test_hk_macau_plate_with_hk_ok(self):
        plates = [mk("HK1234", (100, 100, 300, 150), Region.MO),
                  mk("AB8888", (100, 160, 300, 210), Region.HK)]
        v = group_vehicles(plates)[0]
        assert not any("未見香港牌" in n for n in check_combo(v))
