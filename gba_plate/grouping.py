"""
車輛級聚合：把同一輛車上的多個車牌歸為一組。

背景
----
港澳路面上大量兩地牌／三地牌車輛同時懸掛多塊牌照，例如：

    ┌──────────────────┐
    │ 粵Z·7366澳       │  內地粵Z兩地牌（澳門入出）
    │ MI-13-88         │  澳門本地牌
    │ ZM 6371          │  香港發給澳門車的 ZM 牌
    └──────────────────┘

實測 test_data/real/three_region 的 11 張圖，一輛車固定懸掛 3 塊牌，
三塊牌在圖像中**垂直堆疊**、水平方向高度對齊：

    兩兩中心距離 = 0.20 ~ 0.70 × 牌寬
    水平中心偏差 = 0 ~ 0.50 × 牌寬
    x 區間重疊率 = 0.55 ~ 1.00

檢測端能穩定抓齊 3 個框（11/11 張全部 3 個），但結果是一堆互不相關的
`PlateResult`。本模組負責把它們聚成「車輛」，並推斷組合類型。

設計要點
--------
1. **不用純距離**：停車場多車場景下不同車輛的牌也可能很近，純距離會誤聚。
   改以「x 區間重疊率」為主判據 —— 同一輛車的牌必然水平高度對齊。
2. **尺度約束**：同一輛車的牌在同一景深，寬度比不會偏離太多。
3. **不強制垂直**：部分車型左右並排（少見），只要 x 重疊高且距離近就允許。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Iterable, Sequence

from gba_plate.base import PlateResult, Region

# --- 聚類閾值（依 three_region 實測分佈校準） ---
X_OVERLAP_MIN = 0.45      # x 區間重疊率下限（實測 0.55~1.00）
DIST_RATIO_MAX = 1.30     # 中心距離上限（× 平均牌寬，實測 0.20~0.82）
WIDTH_RATIO_MIN = 0.55    # 牌寬比下限
WIDTH_RATIO_MAX = 1.85    # 牌寬比上限

#: 主牌優先級：粵Z 信息量最大（含入出方向），其次內地，再按港澳
_REGION_PRIORITY = {
    Region.GANGAO: 0,
    Region.MAINLAND: 1,
    Region.HK: 2,
    Region.MO: 3,
    Region.UNKNOWN: 9,
}

#: 地區中文標籤
REGION_LABEL = {
    Region.MAINLAND: "內地",
    Region.GANGAO: "粤Z",
    Region.HK: "香港",
    Region.MO: "澳門",
    Region.UNKNOWN: "未知",
}


def _bbox(r: PlateResult) -> tuple[float, float, float, float]:
    b = r.detection.bbox
    return float(b[0]), float(b[1]), float(b[2]), float(b[3])


def _center(r: PlateResult) -> tuple[float, float]:
    x1, y1, x2, y2 = _bbox(r)
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _width(r: PlateResult) -> float:
    x1, _, x2, _ = _bbox(r)
    return max(1.0, x2 - x1)


def _height(r: PlateResult) -> float:
    _, y1, _, y2 = _bbox(r)
    return max(1.0, y2 - y1)


def x_overlap_ratio(a: PlateResult, b: PlateResult) -> float:
    """兩個框在水平方向的投影重疊率（以較窄的一方為分母）。"""
    ax1, _, ax2, _ = _bbox(a)
    bx1, _, bx2, _ = _bbox(b)
    lo, hi = max(ax1, bx1), min(ax2, bx2)
    if hi <= lo:
        return 0.0
    return (hi - lo) / min(_width(a), _width(b))


def same_vehicle(a: PlateResult, b: PlateResult) -> bool:
    """判斷兩個車牌是否掛在同一輛車上。"""
    wa, wb = _width(a), _width(b)
    avg_w = (wa + wb) / 2.0

    if not (WIDTH_RATIO_MIN < wa / wb < WIDTH_RATIO_MAX):
        return False

    ca, cb = _center(a), _center(b)
    dist = ((ca[0] - cb[0]) ** 2 + (ca[1] - cb[1]) ** 2) ** 0.5
    if dist > DIST_RATIO_MAX * avg_w:
        return False

    return x_overlap_ratio(a, b) >= X_OVERLAP_MIN


def _cluster(plates: Sequence[PlateResult]) -> list[list[int]]:
    """連通分量聚類，回傳每組的索引列表。"""
    n = len(plates)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    for i, j in combinations(range(n), 2):
        if same_vehicle(plates[i], plates[j]):
            union(i, j)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    # 每組內按 y 由小到大（即由上而下）
    for idx in groups.values():
        idx.sort(key=lambda k: _center(plates[k])[1])
    # 按組內首個成員的 y 排序，保持閱讀順序
    return sorted(groups.values(), key=lambda g: _center(plates[g[0]])[1])


@dataclass
class VehicleResult:
    """
    一輛車的識別結果（可能掛多塊牌照）。

    Attributes:
        vehicle_id: 畫面內的車輛序號（0 起）
        plates: 該車的所有車牌，按圖像由上而下排序
    """

    vehicle_id: int
    plates: list[PlateResult] = field(default_factory=list)

    # ---------------------------------------------------------------- 便捷屬性

    @property
    def is_multi_plate(self) -> bool:
        return len(self.plates) > 1

    @property
    def primary(self) -> PlateResult | None:
        """主牌：粵Z 優先，其次內地，再按港澳。"""
        if not self.plates:
            return None
        return min(self.plates,
                   key=lambda r: (_REGION_PRIORITY.get(r.region, 9),
                                  -r.confidence))

    @property
    def texts(self) -> list[str]:
        return [r.text for r in self.plates if r.text]

    @property
    def regions(self) -> list[str]:
        return [REGION_LABEL.get(r.region, "未知") for r in self.plates]

    @property
    def combo(self) -> str:
        """
        組合類型描述，如「粵Z+澳門+香港」。

        跨越同一地區的多塊牌只記一次（例如前後牌都是香港）。
        """
        seen: list[str] = []
        for r in self.plates:
            label = REGION_LABEL.get(r.region, "未知")
            if label not in seen:
                seen.append(label)
        return "+".join(seen)

    @property
    def entry_direction(self) -> str | None:
        """粵Z 牌的入出方向：港 / 澳 / None。"""
        for r in self.plates:
            if r.region == Region.GANGAO and r.text and r.text[-1] in "港澳":
                return r.text[-1]
        return None

    def describe(self) -> str:
        """單行摘要，供日誌與 UI table 使用。"""
        parts = []
        for r in self.plates:
            label = REGION_LABEL.get(r.region, "未知")
            parts.append(f"{label}:{r.text or '(空)'}")
        flag = " [多牌]" if self.is_multi_plate else ""
        return f"#{self.vehicle_id}{flag} " + " | ".join(parts)


def group_vehicles(plates: Iterable[PlateResult]) -> list[VehicleResult]:
    """
    把一堆 PlateResult 按所屬車輛聚類。

    Args:
        plates: 任意順序的車牌結果

    Returns:
        按畫面位置排序的 VehicleResult 列表
    """
    items = list(plates)
    if not items:
        return []

    vehicles = [
        VehicleResult(vehicle_id=i, plates=[items[k] for k in idx])
        for i, idx in enumerate(_cluster(items))
    ]
    return vehicles


# ------------------------------------------------------------------ 組合校驗

def check_combo(vehicle: VehicleResult) -> list[str]:
    """
    檢查多牌組合是否符合地理邏輯，回傳問題描述列表。

    規則（依粵港澳三套跨境牌制度，見 docs/PLATE_COMBINATIONS.md）：
      - 粵Z 後綴「港」→ 應同時有香港牌
      - 粵Z 後綴「澳」→ 應同時有澳門牌
      - ZM／ZN／ZL（澳門車入港，香港發牌）→ 應伴隨澳門本地牌
      - FV／FU（內地車入港，香港發牌）→ 應伴隨內地牌
      - HK-xx-xx（香港配額車入澳，澳門發牌）→ 應伴隨香港牌
    """
    notes: list[str] = []
    if not vehicle.is_multi_plate:
        return notes

    regions = {r.region for r in vehicle.plates}
    direction = vehicle.entry_direction

    if direction == "港" and Region.HK not in regions:
        notes.append("粵Z 後綴為「港」但未見香港牌")
    if direction == "澳" and Region.MO not in regions:
        notes.append("粵Z 後綴為「澳」但未見澳門牌")

    for r in vehicle.plates:
        t = (r.text or "").upper()
        # 澳門車入港：ZM/ZN/ZL 是香港牌，須有澳門本地牌
        if t.startswith(("ZM", "ZN", "ZL")) and Region.MO not in regions:
            notes.append(f"{t} 為澳門車入港牌，但未見澳門本地牌")
        # 內地車入港：FV/FU 是香港牌，須有內地牌
        if t.startswith(("FV", "FU")) and Region.MAINLAND not in regions:
            notes.append(f"{t} 為內地車入港牌，但未見內地牌")
        # 香港車入澳：HK 字頭的黑底牌是澳門配額牌，須有香港牌
        if (t.startswith("HK") and r.region == Region.MO
                and Region.HK not in regions):
            notes.append(f"{t} 為香港車入澳的澳門配額牌，但未見香港牌")

    return notes
