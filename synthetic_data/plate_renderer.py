"""
車牌渲染器 - 使用 Pillow 渲染四地車牌
"""
import os
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from typing import Tuple, Optional
from .plate_specs import PlateSpec, generate_plate


# 字體路徑
FONT_PATHS = {
    "chinese_plate": os.path.join(os.path.dirname(__file__), "..", "fonts", "platech.ttf"),
    "droid_fallback": "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
}

#: 尾字需用紅色渲染的字符（僅人民警察牌的「警」；
#: 2017 版《外交車輛管理辦法》已把「使」「領」由紅色改為白色置於末位）
_RED_SUFFIX_CHARS = frozenset({"警"})

#: 分隔圓點字符（內地「粵A·12345」、粵Z「粵Z·1234港」中間的那一點）。
#: 字體自帶的 U+00B7 基線偏低，故改用繪圖自畫，置於垂直中央。
_DOT_CHARS = frozenset({"·", "•", "・"})

#: 官方規格（GA 36-2018）：間隔符直徑 10mm，前後各留 12mm，合計佔位 34mm。
#: 牌面以 1px = 1mm 繪製，內地牌 font_size=120 對應字高 86.4px ≈ 90mm，
#: 故換算成字號比例：佔位 34/120、半徑 5/120。
DOT_WIDTH_RATIO = 34 / 120      # 圓點總佔位（含前後間距）
DOT_RADIUS_RATIO = 5 / 120      # 圓點半徑


def get_font(size: int, font_name: str = "chinese_plate") -> ImageFont.FreeTypeFont:
    """獲取字體"""
    path = FONT_PATHS.get(font_name, FONT_PATHS["chinese_plate"])
    if not os.path.exists(path):
        path = FONT_PATHS["droid_fallback"]
    return ImageFont.truetype(path, size)


def _make_gradient(w: int, h: int,
                   top: Tuple[int, int, int],
                   bottom: Tuple[int, int, int]) -> Image.Image:
    """
    產生上→下線性漸變底色（新能源小型車的「漸變綠」）。

    用 numpy 逐行插值再貼成整張圖，比逐像素 PIL 快兩個數量級。
    """
    rows = np.linspace(np.array(top, dtype=np.float32),
                       np.array(bottom, dtype=np.float32),
                       h).astype(np.uint8)              # (h, 3)
    canvas = np.repeat(rows[:, None, :], w, axis=1)     # (h, w, 3)
    return Image.fromarray(canvas, mode="RGB")


def _make_split(w: int, h: int,
                top: Tuple[int, int, int],
                bottom: Tuple[int, int, int]) -> Image.Image:
    """
    產生上下雙色拼接底色（大型新能源車的「黃綠雙拼」）。

    上半為黃、下半為綠，貼合 GA 36-2018 對大型新能源號牌的配色規定。
    """
    half = h // 2
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    canvas[:half] = np.array(top, dtype=np.uint8)
    canvas[half:] = np.array(bottom, dtype=np.uint8)
    return Image.fromarray(canvas, mode="RGB")


def render_plate(
    text: str,
    spec: PlateSpec,
    noise_level: float = 0.0,
    blur_level: float = 0.0,
    brightness_offset: float = 0.0,
    perspective: Optional[Tuple[float, float]] = None,
) -> np.ndarray:
    """
    渲染車牌圖像

    Args:
        text: 車牌文字（雙層用 \n 分隔）
        spec: 車牌規格
        noise_level: 噪聲強度 (0-1)
        blur_level: 模糊強度 (0-5)
        brightness_offset: 亮度偏移 (-50 to 50)
        perspective: 透視變形 (dx, dy) 像素

    Returns:
        numpy array (H, W, 3) BGR 格式（OpenCV 兼容）
    """
    # 創建圖像（支持漸變 / 上下雙色 / 純色三種底色）
    if spec.bg_gradient:
        top, bottom = spec.bg_gradient
        img = _make_gradient(spec.width, spec.height, top, bottom)
    elif spec.bg_split:
        top, bottom = spec.bg_split
        img = _make_split(spec.width, spec.height, top, bottom)
    else:
        img = Image.new('RGB', (spec.width, spec.height), spec.bg_color)
    draw = ImageDraw.Draw(img)

    # 繪製邊框
    if spec.border_width > 0 and spec.border_color:
        for i in range(spec.border_width):
            draw.rectangle(
                [i, i, spec.width - 1 - i, spec.height - 1 - i],
                outline=spec.border_color
            )

    # 處理雙層車牌
    if spec.is_double and '\n' in text:
        lines = text.split('\n')
        _render_double_plate(draw, lines, spec)
    else:
        _render_single_plate(draw, text, spec)

    # 應用後處理
    img = _apply_post_processing(img, noise_level, blur_level, brightness_offset, perspective)

    # 轉換為 BGR numpy array
    img_rgb = np.array(img)
    img_bgr = img_rgb[:, :, ::-1].copy()
    return img_bgr


def _char_width(font, ch: str, font_size: int) -> int:
    """單個字符的繪製寬度。分隔圓點用固定寬度（字體自帶的過寬且偏低）。"""
    if ch in _DOT_CHARS:
        return max(6, int(font_size * DOT_WIDTH_RATIO))
    bbox = font.getbbox(ch)
    return bbox[2] - bbox[0]


def _draw_dot(draw: ImageDraw.Draw, cx: int, cy: int, font_size: int,
              color) -> None:
    """在 (cx, cy) 畫間隔符圓點。官方直徑 10mm（字號 120 時半徑 5px）。"""
    r = max(2, int(round(font_size * DOT_RADIUS_RATIO)))
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)


def _render_single_plate(draw: ImageDraw.Draw, text: str, spec: PlateSpec):
    """渲染單層車牌 - 自動縮放字體以適應板面"""
    font_size = spec.font_size
    font = get_font(font_size)

    # 自動縮放：如果文字超出板面，逐步縮小字體
    usable_w = spec.width - 2 * spec.margin_x
    usable_h = spec.height - 2 * spec.margin_y

    while font_size > 20:
        total_width = sum(_char_width(font, ch, font_size) for ch in text)
        total_width += spec.char_spacing * max(0, len(text) - 1)

        if total_width <= usable_w:
            break
        font_size -= 2
        font = get_font(font_size)

    # 重新計算總寬度
    char_widths = []
    total_width = 0
    for ch in text:
        w = _char_width(font, ch, font_size)
        char_widths.append(w)
        total_width += w
    total_width += spec.char_spacing * max(0, len(text) - 1)

    # 起始位置（居中）
    x = (spec.width - total_width) // 2
    y = (spec.height - font_size) // 2

    # 繪製每個字符
    for i, ch in enumerate(text):
        # 尾字／前綴特殊顏色：警（紅尾字）；軍車武警消防（紅前綴）
        color = spec.text_color
        if spec.suffix_color and i == len(text) - 1 and ch in _RED_SUFFIX_CHARS:
            color = spec.suffix_color
        elif spec.prefix_color and i < spec.prefix_len:
            color = spec.prefix_color

        if ch in _DOT_CHARS:
            # 分隔圓點：畫在垂直中央，不用字體字形
            _draw_dot(draw, x + char_widths[i] // 2, spec.height // 2,
                      font_size, color)
        else:
            bbox = font.getbbox(ch)
            ch_h = bbox[3] - bbox[1]
            ch_y = y + (font_size - ch_h) // 2 - bbox[1]
            draw.text((x, ch_y), ch, fill=color, font=font)
        x += char_widths[i] + spec.char_spacing


def _render_double_plate(draw: ImageDraw.Draw, lines: list, spec: PlateSpec):
    """渲染雙層車牌 - 自動縮放字體"""
    # 繪製分隔線
    if spec.separator_color:
        y_sep = spec.top_height
        draw.line(
            [(spec.border_width, y_sep), (spec.width - spec.border_width, y_sep)],
            fill=spec.separator_color,
            width=2
        )

    for layer_idx, line_text in enumerate(lines[:2]):
        if layer_idx == 0:
            y_offset = 0
            layer_height = spec.top_height
        else:
            y_offset = spec.top_height
            layer_height = spec.bottom_height

        font_size = spec.font_size
        font = get_font(font_size)
        usable_w = spec.width - 2 * spec.margin_x

        # 自動縮放
        while font_size > 16:
            total_width = sum(_char_width(font, ch, font_size)
                              for ch in line_text)
            total_width += spec.char_spacing * max(0, len(line_text) - 1)
            if total_width <= usable_w:
                break
            font_size -= 2
            font = get_font(font_size)

        # 計算文字寬度
        char_widths = []
        total_width = 0
        for ch in line_text:
            w = _char_width(font, ch, font_size)
            char_widths.append(w)
            total_width += w
        total_width += spec.char_spacing * max(0, len(line_text) - 1)

        # 起始位置
        x = (spec.width - total_width) // 2
        y = y_offset + (layer_height - font_size) // 2

        # 繪製字符
        for i, ch in enumerate(line_text):
            if ch in _DOT_CHARS:
                # 分隔圓點：置於該行垂直中央
                _draw_dot(draw, x + char_widths[i] // 2,
                          y_offset + layer_height // 2,
                          font_size, spec.text_color)
            else:
                bbox = font.getbbox(ch)
                ch_h = bbox[3] - bbox[1]
                ch_y = y + (font_size - ch_h) // 2 - bbox[1]
                draw.text((x, ch_y), ch, fill=spec.text_color, font=font)
            x += char_widths[i] + spec.char_spacing


def _apply_post_processing(
    img: Image.Image,
    noise_level: float,
    blur_level: float,
    brightness_offset: float,
    perspective: Optional[Tuple[float, float]],
) -> Image.Image:
    """應用後處理效果"""
    # 亮度調整
    if brightness_offset != 0:
        from PIL import ImageEnhance
        factor = 1.0 + brightness_offset / 100.0
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(factor)

    # 高斯模糊
    if blur_level > 0:
        img = img.filter(ImageFilter.GaussianBlur(radius=blur_level))

    # 添加噪聲
    if noise_level > 0:
        arr = np.array(img).astype(np.float32)
        noise = np.random.normal(0, noise_level * 30, arr.shape)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)

    # 透視變形
    if perspective:
        img = _apply_perspective(img, perspective)

    return img


def _apply_perspective(img: Image.Image, params: Tuple[float, float]) -> Image.Image:
    """應用輕微透視變形"""
    dx, dy = params
    w, h = img.size

    # 定義四點變換
    coeffs = [
        1.0, dx / w, 0,
        dy / h, 1.0, 0,
        0, 0, 1.0
    ]
    # 使用簡化的仿射變換
    img = img.transform(
        img.size,
        Image.AFFINE,
        (1.0 + dx / 1000, dx / w, 0, dy / h, 1.0 + dy / 1000, 0),
        Image.BICUBIC
    )
    return img


def render_random_plate(
    region: str = "random",
    noise_level: float = None,
    blur_level: float = None,
) -> Tuple[np.ndarray, str, str, str]:
    """
    渲染隨機車牌

    Returns:
        (image_bgr, plate_text, spec_name, region)
    """
    text, spec, region = generate_plate(region)

    # 隨機增強參數（如果未指定）
    if noise_level is None:
        noise_level = random.uniform(0, 0.3)
    if blur_level is None:
        blur_level = random.uniform(0, 1.5)

    brightness = random.uniform(-20, 20)
    perspective = (random.uniform(-3, 3), random.uniform(-2, 2)) if random.random() < 0.3 else None

    img = render_plate(text, spec, noise_level, blur_level, brightness, perspective)
    return img, text, spec.name, region


if __name__ == "__main__":
    import cv2

    # 測試渲染
    os.makedirs("test_output", exist_ok=True)

    for i in range(20):
        img, text, spec_name, region = render_random_plate()
        filename = f"test_output/test_{i:03d}_{region}_{spec_name}.jpg"
        cv2.imwrite(filename, img)
        print(f"[{i:02d}] [{region:8s}] [{spec_name:25s}] {text} -> {filename}")

    print("\nDone! Check test_output/")
