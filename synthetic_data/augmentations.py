"""
數據增強模塊 - 模擬真實場景的車牌圖像
"""
import cv2
import numpy as np
import random
from typing import Tuple, Optional


def augment_plate(
    img: np.ndarray,
    level: str = "medium",
) -> np.ndarray:
    """
    對車牌圖像進行增強

    Args:
        img: BGR 格式的車牌圖像
        level: 增強強度 ("light", "medium", "heavy")

    Returns:
        增強後的圖像
    """
    if level == "light":
        prob = 0.3
    elif level == "medium":
        prob = 0.5
    else:
        prob = 0.7

    # 隨機選擇增強組合
    if random.random() < prob:
        img = add_gaussian_noise(img, sigma=random.uniform(5, 20))

    if random.random() < prob:
        img = add_motion_blur(img, size=random.randint(3, 9))

    if random.random() < prob:
        img = adjust_brightness_contrast(
            img,
            brightness=random.uniform(-30, 30),
            contrast=random.uniform(0.8, 1.3)
        )

    if random.random() < prob * 0.5:
        img = add_shadow(img)

    if random.random() < prob * 0.3:
        img = add_rain_drops(img, num_drops=random.randint(3, 10))

    if random.random() < prob * 0.5:
        img = add_lens_flare(img)

    if random.random() < prob * 0.3:
        img = add_jpeg_compression(img, quality=random.randint(40, 80))

    return img


def add_gaussian_noise(img: np.ndarray, sigma: float = 10.0) -> np.ndarray:
    """添加高斯噪聲"""
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    noisy = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return noisy


def add_motion_blur(img: np.ndarray, size: int = 5, angle: float = None) -> np.ndarray:
    """添加運動模糊"""
    if angle is None:
        angle = random.uniform(0, 180)

    kernel = np.zeros((size, size), dtype=np.float32)
    kernel[size // 2, :] = 1.0

    # 旋轉核
    M = cv2.getRotationMatrix2D((size / 2, size / 2), angle, 1)
    kernel = cv2.warpAffine(kernel, M, (size, size))
    kernel = kernel / kernel.sum()

    blurred = cv2.filter2D(img, -1, kernel)
    return blurred


def adjust_brightness_contrast(
    img: np.ndarray,
    brightness: float = 0,
    contrast: float = 1.0
) -> np.ndarray:
    """調整亮度和對比度"""
    result = img.astype(np.float32)
    result = result * contrast + brightness
    return np.clip(result, 0, 255).astype(np.uint8)


def add_shadow(img: np.ndarray) -> np.ndarray:
    """添加陰影效果"""
    h, w = img.shape[:2]

    # 隨機選擇陰影方向
    direction = random.choice(['left', 'right', 'top', 'bottom', 'diagonal'])

    mask = np.zeros((h, w), dtype=np.float32)

    if direction == 'left':
        mask[:, :w // 2] = 1.0
    elif direction == 'right':
        mask[:, w // 2:] = 1.0
    elif direction == 'top':
        mask[:h // 2, :] = 1.0
    elif direction == 'bottom':
        mask[h // 2:, :] = 1.0
    else:
        # 對角線
        for i in range(h):
            boundary = int(w * i / h)
            if random.random() < 0.5:
                mask[i, :boundary] = 1.0
            else:
                mask[i, boundary:] = 1.0

    # 高斯模糊使過渡平滑
    mask = cv2.GaussianBlur(mask, (21, 21), 0)
    mask = mask[:, :, np.newaxis]

    # 應用陰影
    shadow_strength = random.uniform(0.3, 0.6)
    result = img.astype(np.float32) * (1.0 - mask * shadow_strength)
    return np.clip(result, 0, 255).astype(np.uint8)


def add_rain_drops(img: np.ndarray, num_drops: int = 5) -> np.ndarray:
    """添加雨滴遮擋效果"""
    h, w = img.shape[:2]
    result = img.copy()

    for _ in range(num_drops):
        # 隨機位置
        cx = random.randint(0, w)
        cy = random.randint(0, h)
        radius = random.randint(2, 8)

        # 繪製半透明圓形
        overlay = result.copy()
        cv2.circle(overlay, (cx, cy), radius, (200, 200, 200), -1)
        alpha = random.uniform(0.3, 0.6)
        cv2.addWeighted(overlay, alpha, result, 1 - alpha, 0, result)

    return result


def add_lens_flare(img: np.ndarray) -> np.ndarray:
    """添加鏡頭光暈"""
    h, w = img.shape[:2]
    result = img.copy().astype(np.float32)

    # 隨機光源位置
    cx = random.randint(0, w)
    cy = random.randint(0, h)

    # 創建光暈
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    max_dist = np.sqrt(w ** 2 + h ** 2)
    flare = np.exp(-dist / (max_dist * 0.3)) * 100

    # 應用光暈
    for c in range(3):
        result[:, :, c] = np.clip(result[:, :, c] + flare, 0, 255)

    return result.astype(np.uint8)


def add_jpeg_compression(img: np.ndarray, quality: int = 70) -> np.ndarray:
    """模擬 JPEG 壓縮偽影"""
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encoded = cv2.imencode('.jpg', img, encode_param)
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    return decoded


def add_weather_effect(img: np.ndarray, weather: str = None) -> np.ndarray:
    """添加天氣效果"""
    if weather is None:
        weather = random.choice(['clear', 'rain', 'fog', 'night'])

    if weather == 'rain':
        img = add_rain_drops(img, num_drops=random.randint(5, 15))
        img = add_motion_blur(img, size=random.randint(3, 7), angle=80)

    elif weather == 'fog':
        # 添加霧氣
        fog = np.ones_like(img, dtype=np.float32) * random.uniform(150, 200)
        alpha = random.uniform(0.3, 0.6)
        img = cv2.addWeighted(fog.astype(np.uint8), alpha, img, 1 - alpha, 0)

    elif weather == 'night':
        # 夜間低光
        img = adjust_brightness_contrast(img, brightness=-50, contrast=0.7)
        img = add_gaussian_noise(img, sigma=15)

    return img


def add_occlusion(img: np.ndarray, occlusion_type: str = None) -> np.ndarray:
    """添加遮擋效果"""
    if occlusion_type is None:
        occlusion_type = random.choice(['none', 'dirt', 'scratch', 'sticker'])

    h, w = img.shape[:2]

    if occlusion_type == 'dirt':
        # 污漬
        for _ in range(random.randint(3, 8)):
            cx = random.randint(0, w)
            cy = random.randint(0, h)
            radius = random.randint(5, 15)
            color = tuple(random.randint(60, 120) for _ in range(3))
            cv2.circle(img, (cx, cy), radius, color, -1)

    elif occlusion_type == 'scratch':
        # 劃痕
        for _ in range(random.randint(1, 3)):
            pt1 = (random.randint(0, w), random.randint(0, h))
            pt2 = (random.randint(0, w), random.randint(0, h))
            color = tuple(random.randint(180, 255) for _ in range(3))
            cv2.line(img, pt1, pt2, color, random.randint(1, 3))

    elif occlusion_type == 'sticker':
        # 貼紙遮擋
        x1 = random.randint(0, w // 2)
        y1 = random.randint(0, h // 2)
        x2 = x1 + random.randint(w // 4, w // 2)
        y2 = y1 + random.randint(h // 4, h // 2)
        color = tuple(random.randint(0, 255) for _ in range(3))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)

    return img


def simulate_capture_condition(
    img: np.ndarray,
    condition: str = None,
) -> np.ndarray:
    """
    模擬真實拍攝條件

    condition: "daylight", "night", "rainy", "foggy", "backlit"
    """
    if condition is None:
        condition = random.choice(['daylight', 'night', 'rainy', 'foggy', 'backlit'])

    if condition == 'daylight':
        img = adjust_brightness_contrast(img, brightness=random.uniform(-10, 10))

    elif condition == 'night':
        img = adjust_brightness_contrast(img, brightness=-40, contrast=0.8)
        img = add_gaussian_noise(img, sigma=12)

    elif condition == 'rainy':
        img = add_rain_drops(img, num_drops=random.randint(8, 20))
        img = add_motion_blur(img, size=random.randint(3, 6))
        img = adjust_brightness_contrast(img, brightness=-15)

    elif condition == 'foggy':
        fog = np.ones_like(img, dtype=np.float32) * 180
        alpha = random.uniform(0.4, 0.7)
        img = cv2.addWeighted(fog.astype(np.uint8), alpha, img, 1 - alpha, 0)

    elif condition == 'backlit':
        # 逆光
        h, w = img.shape[:2]
        gradient = np.linspace(1.0, 0.5, w).reshape(1, -1, 1)
        img = (img.astype(np.float32) * gradient).astype(np.uint8)
        img = adjust_brightness_contrast(img, contrast=0.7)

    return img


if __name__ == "__main__":
    import os

    # 測試增強效果
    test_img = np.ones((140, 440, 3), dtype=np.uint8) * 100
    os.makedirs("augment_test", exist_ok=True)

    effects = [
        ("original", test_img),
        ("gaussian_noise", add_gaussian_noise(test_img, sigma=20)),
        ("motion_blur", add_motion_blur(test_img, size=7)),
        ("shadow", add_shadow(test_img)),
        ("rain", add_rain_drops(test_img, 10)),
        ("flare", add_lens_flare(test_img)),
        ("jpeg_50", add_jpeg_compression(test_img, 50)),
        ("night", simulate_capture_condition(test_img, "night")),
        ("rainy", simulate_capture_condition(test_img, "rainy")),
        ("backlit", simulate_capture_condition(test_img, "backlit")),
    ]

    for name, img in effects:
        cv2.imwrite(f"augment_test/{name}.jpg", img)
        print(f"Saved: augment_test/{name}.jpg")
