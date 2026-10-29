"""
四地車牌規格定義
- 內地（單層藍/黃/綠/白）
- 粵Z兩地牌（單層/雙層）
- 香港（前白後黃，1-8位）
- 澳門（黑底白字，M/MA + 數字）
"""
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import random
import string


@dataclass
class PlateSpec:
    """車牌規格"""
    name: str                    # 規格名稱
    width: int                   # 寬度 (px)
    height: int                  # 高度 (px)
    bg_color: Tuple[int, int, int]  # 背景色 RGB
    text_color: Tuple[int, int, int]  # 文字色 RGB
    font_size: int               # 字體大小
    char_spacing: int            # 字符間距
    margin_x: int                # 左右邊距
    margin_y: int                # 上下邊距
    border_width: int = 0        # 邊框寬度
    border_color: Optional[Tuple[int, int, int]] = None
    is_double: bool = False      # 是否雙層
    top_height: int = 0          # 雙層上半高度
    bottom_height: int = 0       # 雙層下半高度
    separator_color: Optional[Tuple[int, int, int]] = None  # 雙層分隔線顏色
    #: 漸變底色（上, 下）。設定時忽略 bg_color，用於新能源漸變綠牌
    bg_gradient: Optional[Tuple[Tuple[int, int, int], Tuple[int, int, int]]] = None
    #: 尾字特殊顏色。目前僅「警」為紅色（2017 版外交牌的「使」「領」已改白字）
    suffix_color: Optional[Tuple[int, int, int]] = None
    #: 前綴特殊顏色，用於軍車／武警／消防的字頭（紅色）
    prefix_color: Optional[Tuple[int, int, int]] = None
    #: 前綴紅色字符數（軍車 2、武警 3、消防 4）
    prefix_len: int = 0
    #: 上下雙色拼接（上, 下）。用於大型新能源的黃綠雙拼
    bg_split: Optional[Tuple[Tuple[int, int, int], Tuple[int, int, int]]] = None


# ============================================================
# 內地車牌規格
# ============================================================

# 單層藍牌: 440×140 px (標準比例)
MAINLAND_BLUE = PlateSpec(
    name="mainland_blue",
    width=440, height=140,
    bg_color=(15, 75, 180),       # 藍色
    text_color=(255, 255, 255),   # 白色文字
    font_size=120,                # 字高 90mm / 牌高 140mm = 64%
    char_spacing=6,
    margin_x=30, margin_y=25,
    border_width=4,
    border_color=(255, 255, 255),
)

# 單層黃牌（大型車前牌）
MAINLAND_YELLOW = PlateSpec(
    name="mainland_yellow",
    width=440, height=140,
    bg_color=(230, 190, 30),      # 黃色
    text_color=(0, 0, 0),         # 黑色文字
    font_size=120,
    char_spacing=6,
    margin_x=30, margin_y=25,
    border_width=4,
    border_color=(0, 0, 0),
)

# 新能源車牌（綠牌）—— 外廓 480×140mm（GA36-2018），比普通牌長 40mm
# 小型新能源為「漸變綠」：上淺下深
MAINLAND_GREEN = PlateSpec(
    name="mainland_green",
    width=480, height=140,
    bg_color=(60, 170, 95),          # 佔位，實際由 bg_gradient 覆蓋
    text_color=(0, 0, 0),            # 黑字
    font_size=120,
    char_spacing=6,
    margin_x=30, margin_y=25,
    border_width=4,
    border_color=(20, 20, 20),
    bg_gradient=((208, 236, 198), (32, 126, 72)),   # 上淺綠 → 下深綠
)

# 大型新能源車牌 —— 黃綠雙拼（上半黃、下半綠）
MAINLAND_GREEN_LARGE = PlateSpec(
    name="mainland_green_large",
    width=480, height=140,
    bg_color=(200, 180, 60),
    text_color=(0, 0, 0),
    font_size=120,
    char_spacing=6,
    margin_x=30, margin_y=25,
    border_width=4,
    border_color=(20, 20, 20),
    bg_split=((235, 200, 70), (45, 150, 90)),       # 黃 / 綠
)

# 白色警車牌 —— 白底黑字，尾字「警」為紅色
MAINLAND_WHITE = PlateSpec(
    name="mainland_white",
    width=440, height=140,
    bg_color=(250, 250, 250),     # 白色
    text_color=(0, 0, 0),         # 黑色文字
    font_size=120,
    char_spacing=6,
    margin_x=30, margin_y=25,
    border_width=4,
    border_color=(0, 0, 0),
    suffix_color=(200, 30, 30),   # 「警」紅色
)

# 使領館車牌（2017 版《外交車輛管理辦法》）—— 黑底白字
#   大使館：224·009使   6 位數字（3 位國家碼 + 3 位編號）+ 白「使」
#   領事館：粵204·01領  省簡稱 + 3 位國家碼 + 2 位編號 + 白「領」
# 「使」「領」已於 2017 由紅色改為白色並移至末位，不再用 suffix_color
MAINLAND_BLACK = PlateSpec(
    name="mainland_black",
    width=440, height=140,
    bg_color=(18, 18, 18),        # 黑色
    text_color=(245, 245, 245),   # 白色文字（含「使」「領」）
    font_size=110,
    char_spacing=5,
    margin_x=25, margin_y=25,
    border_width=4,
    border_color=(245, 245, 245),
)

# 解放軍車輛號牌 —— 白底黑字，前牌字頭+字頭號為紅色
#   格式：字頭（軍種字母）+ 字頭號 + 5 位序號，例 VA12345
#   V=中央軍委、K=空軍、H=海軍；車尾僅字頭紅色
MAINLAND_MILITARY = PlateSpec(
    name="mainland_military",
    width=440, height=140,
    bg_color=(248, 248, 248),
    text_color=(20, 20, 20),
    font_size=115,
    char_spacing=5,
    margin_x=25, margin_y=25,
    border_width=4,
    border_color=(20, 20, 20),
    prefix_color=(200, 30, 30),
    prefix_len=2,                 # 字頭 + 字頭號紅色
)

# 武警部隊號牌 —— 白底黑字，WJ + 單位字母為紅色
#   2019 式：WJ·A00001（與解放軍同式樣），5 位數字
MAINLAND_ARMED_POLICE = PlateSpec(
    name="mainland_armed_police",
    width=440, height=140,
    bg_color=(248, 248, 248),
    text_color=(20, 20, 20),
    font_size=115,
    char_spacing=4,
    margin_x=22, margin_y=25,
    border_width=4,
    border_color=(20, 20, 20),
    prefix_color=(200, 30, 30),
    prefix_len=3,                 # WJ + 單位字母紅色
)

# 國家綜合性消防救援隊伍號牌 —— 白底黑字，8 位，前 4 位隊伍代號為紅色
MAINLAND_RESCUE = PlateSpec(
    name="mainland_rescue",
    width=480, height=140,
    bg_color=(248, 248, 248),
    text_color=(20, 20, 20),
    font_size=110,
    char_spacing=4,
    margin_x=22, margin_y=25,
    border_width=4,
    border_color=(20, 20, 20),
    prefix_color=(200, 30, 30),
    prefix_len=4,                 # 救援隊伍代號紅色
)

# 教練車（學牌）—— 黃底黑字，尾字「學」
MAINLAND_COACH = PlateSpec(
    name="mainland_coach",
    width=440, height=140,
    bg_color=(230, 190, 30),
    text_color=(0, 0, 0),
    font_size=120,
    char_spacing=6,
    margin_x=30, margin_y=25,
    border_width=4,
    border_color=(0, 0, 0),
)

# 大型車後牌（雙行）—— 440×220mm（前牌才是 440×140）
MAINLAND_DOUBLE_YELLOW = PlateSpec(
    name="mainland_double_yellow",
    width=440, height=220,
    bg_color=(230, 190, 30),
    text_color=(0, 0, 0),
    font_size=110,
    char_spacing=6,
    margin_x=25, margin_y=15,
    border_width=3,
    border_color=(0, 0, 0),
    is_double=True,
    top_height=110,
    bottom_height=110,
    separator_color=(0, 0, 0),
)

# ============================================================
# 粵Z 兩地牌規格
# ============================================================
# 港澳入出內地車輛專用號牌：黑底白字、白色框線、外廓 440×140mm。
# 真實樣本目視驗證（24 張）全部為單行，不存在雙行版本。

# 粵Z 單層（黑底白字 + 白框）
GANGAO_SINGLE = PlateSpec(
    name="gangao_single",
    width=440, height=140,
    bg_color=(20, 20, 20),        # 黑色底
    text_color=(255, 255, 255),   # 白色字
    font_size=120,
    char_spacing=6,
    margin_x=30, margin_y=25,
    border_width=5,
    border_color=(255, 255, 255),
)

#: 已廢棄：真實粵Z 車牌沒有雙行版本。保留符號僅為兼容舊引用。
GANGAO_DOUBLE = GANGAO_SINGLE

# ============================================================
# 香港車牌規格
# ============================================================

# 香港前牌（白底黑字）- 單行
HK_FRONT = PlateSpec(
    name="hk_front",
    width=520, height=110,        # 大長牌 520×110mm（比例 4.73:1）
    bg_color=(245, 245, 245),     # 白底（車頭）
    text_color=(20, 20, 20),      # 黑字
    font_size=100,                # 字高 ≈ 80/110 = 73%（自動縮放兜底）
    char_spacing=8,
    margin_x=22, margin_y=8,
    border_width=3,
    border_color=(30, 30, 30),
)

# 香港後牌（黃底黑字）- 單行
HK_REAR = PlateSpec(
    name="hk_rear",
    width=520, height=110,        # 同上尺寸，車尾黃底
    bg_color=(240, 200, 20),      # 黃底
    text_color=(20, 20, 20),
    font_size=100,
    char_spacing=8,
    margin_x=22, margin_y=8,
    border_width=3,
    border_color=(30, 30, 30),
)

# 香港前牌（白底黑字）- 雙行方形（大方牌 330×170mm，比例 1.94:1）
# 法例 374E 附表4：兩行時字母必須在上、數字必須在下
HK_FRONT_DOUBLE = PlateSpec(
    name="hk_front_double",
    width=330, height=170,
    bg_color=(245, 245, 245),
    text_color=(20, 20, 20),
    font_size=90,
    char_spacing=6,
    margin_x=18, margin_y=8,
    border_width=3,
    border_color=(30, 30, 30),
    is_double=True,
    top_height=85,
    bottom_height=85,
    separator_color=None,         # 無分隔線
)

# 香港後牌（黃底黑字）- 雙行方形（大方牌 330×170mm）
HK_REAR_DOUBLE = PlateSpec(
    name="hk_rear_double",
    width=330, height=170,
    bg_color=(240, 200, 20),
    text_color=(20, 20, 20),
    font_size=90,
    char_spacing=6,
    margin_x=18, margin_y=8,
    border_width=3,
    border_color=(30, 30, 30),
    is_double=True,
    top_height=85,
    bottom_height=85,
    separator_color=None,         # 無分隔線
)

# ============================================================
# 澳門車牌規格（DSAT《車輛註冊號牌技術規格表》2023）
# ============================================================
# 汽車（第一類）字高 70mm、筆劃 12mm、字距牌邊 ≥15mm：
#   A 款 520×120  標準歐洲長牌
#   B 款 460×100  較短長牌
#   C 款 440×100  緊湊長牌
#   D 款 340×230  雙行正方形牌
#   E 款 330×175  雙行中型方形牌
# 電單車（第二類）230×50，字高 35mm、字闊 25mm、筆劃 6mm、邊距 7.5mm
# 顏色（前後相同，與香港「前白後黃」不同）：
#   民用        黑底白字
#   免稅/政府   黑底黃字
#   臨時 (T)    黃底黑字
#
# 尺寸以 1px = 1mm 繪製，字高 70mm → PIL font_size ≈ 100（大寫高約 0.7em）

_MO_BG, _MO_FG = (14, 14, 14), (245, 245, 245)          # 民用黑底白字

# A 款：標準長牌
MO_A = PlateSpec(
    name="mo_a",
    width=520, height=120,
    bg_color=_MO_BG, text_color=_MO_FG,
    font_size=100,
    char_spacing=10,
    margin_x=15, margin_y=25,      # 字高 70 → 上下各留 25mm
    border_width=3,
    border_color=_MO_FG,
)

# B 款：較短長牌
MO_B = PlateSpec(
    name="mo_b",
    width=460, height=100,
    bg_color=_MO_BG, text_color=_MO_FG,
    font_size=100,
    char_spacing=9,
    margin_x=15, margin_y=15,
    border_width=3,
    border_color=_MO_FG,
)

# C 款：緊湊長牌
MO_C = PlateSpec(
    name="mo_c",
    width=440, height=100,
    bg_color=_MO_BG, text_color=_MO_FG,
    font_size=100,
    char_spacing=9,
    margin_x=15, margin_y=15,
    border_width=3,
    border_color=_MO_FG,
)

# D 款：雙行正方形牌
MO_D = PlateSpec(
    name="mo_d",
    width=340, height=230,
    bg_color=_MO_BG, text_color=_MO_FG,
    font_size=100,
    char_spacing=8,
    margin_x=15, margin_y=15,
    border_width=3,
    border_color=_MO_FG,
    is_double=True,
    top_height=115,
    bottom_height=115,
    separator_color=None,
)

# E 款：雙行中型方形牌
MO_E = PlateSpec(
    name="mo_e",
    width=330, height=175,
    bg_color=_MO_BG, text_color=_MO_FG,
    font_size=95,
    char_spacing=8,
    margin_x=15, margin_y=10,
    border_width=3,
    border_color=_MO_FG,
    is_double=True,
    top_height=87,
    bottom_height=88,
    separator_color=None,
)

# 電單車：230×50，字高 35mm
MO_MOTO = PlateSpec(
    name="mo_moto",
    width=230, height=50,
    bg_color=_MO_BG, text_color=_MO_FG,
    font_size=50,
    char_spacing=5,
    margin_x=8, margin_y=8,
    border_width=2,
    border_color=_MO_FG,
)

# 免稅／政府車：黑底黃字（A 款尺寸）
MO_A_TAXFREE = PlateSpec(
    name="mo_a_taxfree",
    width=520, height=120,
    bg_color=(14, 14, 14), text_color=(240, 200, 30),
    font_size=100,
    char_spacing=10,
    margin_x=15, margin_y=25,
    border_width=3,
    border_color=(240, 200, 30),
)

# 臨時制度 T 牌：黃底黑字
MO_A_TEMP = PlateSpec(
    name="mo_a_temp",
    width=520, height=120,
    bg_color=(235, 195, 25), text_color=(20, 20, 20),
    font_size=100,
    char_spacing=10,
    margin_x=15, margin_y=25,
    border_width=3,
    border_color=(20, 20, 20),
)

#: 汽車單行款式（A/B/C，按真實安裝比例加權）
MO_SINGLE_SPECS = [(MO_A, 0.55), (MO_B, 0.15), (MO_C, 0.15)]
#: 汽車雙行款式（D/E）
MO_DOUBLE_SPECS = [(MO_D, 0.5), (MO_E, 0.5)]

# 駐澳解放軍車牌（ZA）—— 尺寸字體照大陸號牌格式，顏色與字元格式照澳門規格
#   黑底白字，格式 ZA-nn-nn（如 ZA-12-34），不受交通事務局管理
MO_ZA = PlateSpec(
    name="mo_za",
    width=440, height=140,        # 參照大陸汽車號牌尺寸
    bg_color=(14, 14, 14),
    text_color=(245, 245, 245),
    font_size=115,
    char_spacing=6,
    margin_x=25, margin_y=25,
    border_width=3,
    border_color=(245, 245, 245),
)

#: 兼容舊引用：默認 = A 款民用
MO_PLATE = MO_A


# ============================================================
# 字符集定義
# ============================================================

# 內地省份簡稱
PROVINCES = list("京沪津渝冀晋蒙辽吉黑苏浙皖闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新")

# 內地字母（不含 I/O，避免與 1/0 混淆）
MAINLAND_ALPHA = "ABCDEFGHJKLMNPQRSTUVWXYZ"

# 粵Z後綴
GANGAO_SUFFIX = ["港", "澳"]


# ============================================================
# 車牌號生成器
# ============================================================

def generate_mainland_plate() -> Tuple[str, PlateSpec]:
    """生成內地車牌號"""
    plate_type = random.choices(
        ["blue", "yellow", "green", "white", "double_yellow"],
        weights=[50, 15, 20, 5, 10],
        k=1
    )[0]

    province = random.choice(PROVINCES)
    alpha = random.choice(MAINLAND_ALPHA)

    if plate_type == "blue":
        # 藍牌: 省份+字母+5位英數
        suffix = ''.join(random.choices(MAINLAND_ALPHA + string.digits, k=5))
        return f"{province}{alpha}{suffix}", MAINLAND_BLUE

    elif plate_type == "yellow":
        # 黃牌: 省份+字母+5位英數
        suffix = ''.join(random.choices(MAINLAND_ALPHA + string.digits, k=5))
        return f"{province}{alpha}{suffix}", MAINLAND_YELLOW

    elif plate_type == "green":
        # 新能源: 省份+字母+6位（小型車 D/F + 5位，大型車 6位 + D/F）
        if random.random() < 0.5:
            # 小型新能源
            special = random.choice(['D', 'F'])
            rest = ''.join(random.choices(MAINLAND_ALPHA + string.digits, k=5))
            return f"{province}{alpha}{special}{rest}", MAINLAND_GREEN
        else:
            # 大型新能源
            rest = ''.join(random.choices(MAINLAND_ALPHA + string.digits, k=5))
            special = random.choice(['D', 'F'])
            return f"{province}{alpha}{rest}{special}", MAINLAND_GREEN

    elif plate_type == "white":
        # 白牌: 省份+字母+4位+警/學
        suffix = ''.join(random.choices(string.digits, k=4))
        special = random.choice(["警", "学"])
        return f"{province}{alpha}{suffix}{special}", MAINLAND_WHITE

    else:
        # 雙層黃牌
        top = f"{province}{alpha}"
        bottom = ''.join(random.choices(string.digits, k=4))
        return f"{top}\n{bottom}", MAINLAND_DOUBLE_YELLOW


def generate_gangao_plate() -> Tuple[str, PlateSpec]:
    """
    生成粵Z兩地牌。

    真實樣本（24 張目視驗證）：全部單行黑底白字，
    格式為「粵Z + 3~4 位英數 + 港/澳」，卡片上有分隔點（粵Z·T67港）。
    原生成器有 30% 機率產生雙行版本，但那在真實中並不存在。
    """
    suffix = random.choice(GANGAO_SUFFIX)
    n_mid = random.choices([3, 4], weights=[30, 70], k=1)[0]
    mid = "".join(random.choices(MAINLAND_ALPHA + string.digits, k=n_mid))
    return f"粤Z{mid}{suffix}", GANGAO_SINGLE


def generate_hk_plate() -> Tuple[str, PlateSpec]:
    """
    生成香港車牌。

    真實分佈依據 plate.hk 202,605 筆統計：
        雙字母+數字   75.9%   （新式主流，如 AB1234）
        純字母組合     7.5%   （個性化，如 ASTRA）
        英數混合      15.5%   （個性化，如 TR0UBLE）
        純數字         1.0%   （早期車牌，如 1314）
        單字母+數字    <0.1%  （早期車牌，如 A1）

    法例 374E 附表4：
        - 只能一行或兩行，禁止三行以上
        - 兩行時「字母在上、數字在下」，禁止顛倒
        - 單行用大長牌 520×110mm；兩行用大方牌 330×170mm
    """
    spec_pick = random.random()

    if spec_pick < 0.02:
        # 早期純數字車牌（1-4 位）
        n = random.choices([1, 2, 3, 4], weights=[5, 25, 40, 30], k=1)[0]
        digits = "".join(random.choices("0123456789", k=n))
        text, spec = digits, random.choice([HK_FRONT, HK_REAR])
    elif spec_pick < 0.05:
        # 早期單字母 + 數字（如 A1、X567）
        letter = random.choice("ABCDEFGHJKLMNPQRSTUVWXYZ")
        n = random.choices([1, 2, 3, 4], weights=[20, 40, 30, 10], k=1)[0]
        digits = "".join(random.choices("0123456789", k=n))
        text, spec = f"{letter} {digits}", random.choice([HK_FRONT, HK_REAR])
    else:
        # 主流：雙字母 + 1-4 數字（數字位數按真實權重）
        letters = "".join(
            random.choices("ABCDEFGHJKLMNPQRSTUVWXYZ", k=2)  # 排除 I/O/Q
        )
        n = random.choices([1, 2, 3, 4], weights=[2, 12, 42, 44], k=1)[0]
        digits = "".join(random.choices("0123456789", k=n))
        text = f"{letters} {digits}"
        spec = random.choice([HK_FRONT, HK_REAR])

    # 12% 用兩行方牌：字母在上、數字在下（法例要求，不可顛倒）
    if random.random() < 0.12:
        parts = text.split()
        if len(parts) == 2 and parts[0].isalpha() and parts[1].isdigit():
            is_rear = spec.name == "hk_rear"
            return (
                f"{parts[0]}\n{parts[1]}",
                HK_REAR_DOUBLE if is_rear else HK_FRONT_DOUBLE,
            )
    return text, spec


def generate_mo_plate() -> Tuple[str, PlateSpec]:
    """
    生成澳門車牌（DSAT 規格）。

    格式：M + 英文字母 + 4 位數字，顯示為 M?-??-??（分隔符為牌面規定的一部分）。
    款式：單行 A/B/C 款與雙行 D/E 款；雙行按字符順序斷開（上排 M?-??，下排 ??）。
    顏色：民用黑底白字為主，少量免稅（黑底黃字）與臨時 T 牌（黃底黑字）。
    """
    letter = random.choice("ABCDEFGHJKLMNPQRSTUVWXYZ")
    num1 = "".join(random.choices(string.digits, k=2))
    num2 = "".join(random.choices(string.digits, k=2))

    r = random.random()
    if r < 0.02:
        # 臨時制度 T 牌（黃底黑字）
        return f"M{letter}-{num1}-{num2}", MO_A_TEMP
    if r < 0.05:
        # 免稅／政府車（黑底黃字）
        return f"M{letter}-{num1}-{num2}", MO_A_TAXFREE
    if r < 0.10:
        # 電單車牌（230×50）
        return f"M{letter}-{num1}-{num2}", MO_MOTO

    # 汽車：88% 單行（A/B/C），12% 雙行（D/E）
    if random.random() < 0.12:
        spec = random.choices([s for s, _ in MO_DOUBLE_SPECS],
                              weights=[w for _, w in MO_DOUBLE_SPECS], k=1)[0]
        return f"M{letter}-{num1}\n{num2}", spec

    spec = random.choices([s for s, _ in MO_SINGLE_SPECS],
                          weights=[w for _, w in MO_SINGLE_SPECS], k=1)[0]
    return f"M{letter}-{num1}-{num2}", spec


def generate_zm_plate() -> Tuple[str, PlateSpec]:
    """
    生成 ZM 車牌 —— 澳門登記車輛經港珠澳大橋進入香港時掛的香港牌。

    格式：ZM + 4 位數字，白底黑字（HK_FRONT 規格）。
    香港路面常見，但原生成器完全沒有這個類別。
    """
    digits = "".join(random.choices(string.digits, k=4))
    return f"ZM {digits}", HK_FRONT


def generate_plate(region: str = "random") -> Tuple[str, PlateSpec, str]:
    """
    生成車牌
    region: mainland / gangao / hk / mo / random
    返回: (車牌文字, 規格, 地區標籤)
    """
    if region == "random":
        region = random.choices(
            ["mainland", "gangao", "hk", "mo"],
            weights=[50, 15, 20, 15],
            k=1
        )[0]

    if region == "mainland":
        text, spec = generate_mainland_plate()
    elif region == "gangao":
        text, spec = generate_gangao_plate()
    elif region == "hk":
        text, spec = generate_hk_plate()
    elif region == "mo":
        text, spec = generate_mo_plate()
    else:
        raise ValueError(f"Unknown region: {region}")

    return text, spec, region


if __name__ == "__main__":
    # 測試生成
    for _ in range(20):
        text, spec, region = generate_plate()
        print(f"[{region:8s}] [{spec.name:25s}] {text}")
