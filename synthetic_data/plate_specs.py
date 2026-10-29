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


# ============================================================
# 內地車牌規格
# ============================================================

# 單層藍牌: 440×140 px (標準比例)
MAINLAND_BLUE = PlateSpec(
    name="mainland_blue",
    width=440, height=140,
    bg_color=(15, 75, 180),       # 藍色
    text_color=(255, 255, 255),   # 白色文字
    font_size=80,
    char_spacing=6,
    margin_x=30, margin_y=25,
    border_width=4,
    border_color=(255, 255, 255),
)

# 單層黃牌
MAINLAND_YELLOW = PlateSpec(
    name="mainland_yellow",
    width=440, height=140,
    bg_color=(230, 190, 30),      # 黃色
    text_color=(0, 0, 0),         # 黑色文字
    font_size=80,
    char_spacing=6,
    margin_x=30, margin_y=25,
    border_width=4,
    border_color=(0, 0, 0),
)

# 新能源車牌（綠牌）
MAINLAND_GREEN = PlateSpec(
    name="mainland_green",
    width=440, height=140,
    bg_color=(40, 160, 80),       # 綠色
    text_color=(255, 255, 255),   # 白色文字
    font_size=80,
    char_spacing=6,
    margin_x=30, margin_y=25,
    border_width=4,
    border_color=(255, 255, 255),
)

# 白色警車牌
MAINLAND_WHITE = PlateSpec(
    name="mainland_white",
    width=440, height=140,
    bg_color=(250, 250, 250),     # 白色
    text_color=(0, 0, 0),         # 黑色文字
    font_size=80,
    char_spacing=6,
    margin_x=30, margin_y=25,
    border_width=4,
    border_color=(0, 0, 0),
)

# 雙層黃牌（大車）
MAINLAND_DOUBLE_YELLOW = PlateSpec(
    name="mainland_double_yellow",
    width=220, height=280,        # 上下兩層
    bg_color=(230, 190, 30),
    text_color=(0, 0, 0),
    font_size=80,
    char_spacing=6,
    margin_x=15, margin_y=15,
    border_width=3,
    border_color=(0, 0, 0),
    is_double=True,
    top_height=130,
    bottom_height=130,
    separator_color=(0, 0, 0),
)

# ============================================================
# 粵Z兩地牌規格
# ============================================================

# 粵Z單層（黑底白字，類似港澳風格）
GANGAO_SINGLE = PlateSpec(
    name="gangao_single",
    width=440, height=140,
    bg_color=(20, 20, 20),        # 黑色底
    text_color=(255, 255, 255),   # 白色字
    font_size=75,
    char_spacing=6,
    margin_x=30, margin_y=25,
    border_width=4,
    border_color=(255, 255, 255),
)

# 粵Z雙層
GANGAO_DOUBLE = PlateSpec(
    name="gangao_double",
    width=220, height=280,
    bg_color=(20, 20, 20),
    text_color=(255, 255, 255),
    font_size=70,
    char_spacing=5,
    margin_x=20, margin_y=20,
    border_width=4,
    border_color=(255, 255, 255),
    is_double=True,
    top_height=130,
    bottom_height=130,
    separator_color=(255, 255, 255),
)

# ============================================================
# 香港車牌規格
# ============================================================

# 香港前牌（白底黑字）- 單行
HK_FRONT = PlateSpec(
    name="hk_front",
    width=370, height=130,        # 香港標準比例
    bg_color=(245, 245, 245),     # 白色
    text_color=(20, 20, 20),      # 黑色
    font_size=75,
    char_spacing=8,
    margin_x=25, margin_y=22,
    border_width=4,
    border_color=(30, 30, 30),
)

# 香港後牌（黃底黑字）- 單行
HK_REAR = PlateSpec(
    name="hk_rear",
    width=370, height=130,
    bg_color=(240, 200, 20),      # 黃色
    text_color=(20, 20, 20),      # 黑色
    font_size=75,
    char_spacing=8,
    margin_x=25, margin_y=22,
    border_width=4,
    border_color=(30, 30, 30),
)

# 香港前牌（白底黑字）- 雙行方形
HK_FRONT_DOUBLE = PlateSpec(
    name="hk_front_double",
    width=220, height=260,        # 方形雙行
    bg_color=(245, 245, 245),     # 白色
    text_color=(20, 20, 20),      # 黑色
    font_size=70,
    char_spacing=6,
    margin_x=20, margin_y=20,
    border_width=4,
    border_color=(30, 30, 30),
    is_double=True,
    top_height=120,
    bottom_height=120,
    separator_color=None,         # 無分隔線
)

# 香港後牌（黃底黑字）- 雙行方形
HK_REAR_DOUBLE = PlateSpec(
    name="hk_rear_double",
    width=220, height=260,        # 方形雙行
    bg_color=(240, 200, 20),      # 黃色
    text_color=(20, 20, 20),      # 黑色
    font_size=70,
    char_spacing=6,
    margin_x=20, margin_y=20,
    border_width=4,
    border_color=(30, 30, 30),
    is_double=True,
    top_height=120,
    bottom_height=120,
    separator_color=None,         # 無分隔線
)

# ============================================================
# 澳門車牌規格
# ============================================================

# 澳門（黑底白字）
MO_PLATE = PlateSpec(
    name="mo_plate",
    width=370, height=130,
    bg_color=(15, 15, 15),        # 黑色
    text_color=(245, 245, 245),   # 白色
    font_size=70,
    char_spacing=6,
    margin_x=25, margin_y=22,
    border_width=4,
    border_color=(245, 245, 245),
)


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
    """生成粵Z兩地牌"""
    suffix = random.choice(GANGAO_SUFFIX)
    is_double = random.random() < 0.3

    if is_double:
        # 雙層: 粵Z+2位 / 3位+港/澳
        top_chars = ''.join(random.choices(MAINLAND_ALPHA + string.digits, k=2))
        bottom_chars = ''.join(random.choices(MAINLAND_ALPHA + string.digits, k=3))
        return f"粤Z{top_chars}\n{bottom_chars}{suffix}", GANGAO_DOUBLE
    else:
        # 單層: 粵Z+4位英數+港/澳
        mid = ''.join(random.choices(MAINLAND_ALPHA + string.digits, k=4))
        return f"粤Z{mid}{suffix}", GANGAO_SINGLE


def generate_hk_plate() -> Tuple[str, PlateSpec]:
    """生成香港車牌"""
    # 香港車牌格式: 1-2個字母 + 1-4個數字 (如 AB 1234, X 567)
    spec = random.choice([HK_FRONT, HK_REAR])

    num_letters = random.choices([1, 2], weights=[20, 80], k=1)[0]
    letters = ''.join(random.choices(string.ascii_uppercase, k=num_letters))

    num_digits = random.randint(1, 4)
    digits = ''.join(random.choices(string.digits, k=num_digits))

    plate_text = f"{letters} {digits}"
    return plate_text, spec


def generate_mo_plate() -> Tuple[str, PlateSpec]:
    """生成澳門車牌"""
    # 澳門車牌格式: M/MA + 兩位數字 + 兩位數字 (如 MG 12-34, MA 56-78)
    prefix = random.choice(["MG", "MA", "MK"])
    num1 = ''.join(random.choices(string.digits, k=2))
    num2 = ''.join(random.choices(string.digits, k=2))
    plate_text = f"{prefix} {num1} {num2}"
    return plate_text, MO_PLATE


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
