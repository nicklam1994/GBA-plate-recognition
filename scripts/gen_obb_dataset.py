#!/usr/bin/env python3
"""
生成 YOLO-OBB 訓練數據（車牌貼進真實場景，四角精確）。

為什麼需要這支腳本：
    現有的 generate_4region_15k.py 只產生「單獨的車牌圖」，用於 OCR 訓練。
    而 OBB 檢測要學的是「車牌在整個場景中的位置與傾斜角」，必須有背景，
    並且要精確知道車牌貼上去之後的四個角點。

    四角不需要標註 —— 貼圖時的仿射變換矩陣是我們自己算的，
    把車牌原始四角套進同一個變換就得到精確答案，誤差為零。
    這也是合成數據相對真實數據的核心優勢。

輸出（YOLO-OBB 格式）：
    images/train/*.jpg
    labels/train/*.txt        class cx cy w h angle  （全部歸一化，angle 為弧度）
    data.yaml

用法：
    ./venv/bin/python scripts/gen_obb_dataset.py --n 16000
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from synthetic_data.augmentations import augment_plate  # noqa: E402
from synthetic_data.plate_renderer import render_plate  # noqa: E402
from synthetic_data.plate_specs import (  # noqa: E402
    GANGAO_SINGLE, HK_FRONT, HK_FRONT_DOUBLE, HK_REAR, HK_REAR_DOUBLE,
    MAINLAND_BLUE, MAINLAND_GREEN, MAINLAND_WHITE, MAINLAND_YELLOW,
    MO_PLATE, MO_WIDE,
)

#: 類別順序與 gba_plate/base.py 的 Region 枚舉一致
CLASS_NAMES = ["mainland", "gangao", "hk", "mo"]
CLS_IDX = {n: i for i, n in enumerate(CLASS_NAMES)}

#: 各省簡稱（簡體）+ 字母池（排除 I/O，GA36 規定）
PROVINCES = "京津冀晋蒙辽吉黑沪苏浙皖闽赣鲁豫鄂湘粤桂琼渝川贵云藏陕甘青宁新"
ALPHA_NO_IO = "ABCDEFGHJKLMNPQRSTUVWXYZ"          # 去掉 I, O
ALPHANUM_NO_IO = ALPHA_NO_IO + "0123456789"
HK_ALPHA = "".join(c for c in ALPHA_NO_IO if c != "Q")   # 香港另禁 Q
MO_ALPHA = "".join(c for c in HK_ALPHA if c != "V")      # 澳門再禁 V


def rand_mainland_text(rng: random.Random) -> tuple[str, object]:
    """內地車牌：省 + 市字母 + 5 位（或新能源 6 位）。"""
    prov = rng.choice(PROVINCES)
    city = rng.choice(ALPHA_NO_IO)
    r = rng.random()
    if r < 0.25:                                  # 新能源（綠牌，8 位）
        body = rng.choice("DABCEF") + "".join(
            rng.choice(ALPHANUM_NO_IO) for _ in range(5))
        return f"{prov}{city}{body}", (MAINLAND_GREEN, "green")
    if r < 0.35:                                  # 大型車（黃牌）
        return f"{prov}{city}" + "".join(rng.choice(ALPHANUM_NO_IO) for _ in range(5)), \
               (MAINLAND_YELLOW, "yellow")
    if r < 0.40:                                  # 警用（白牌）
        return f"{prov}{city}" + "".join(rng.choice("0123456789") for _ in range(4)) + "警", \
               (MAINLAND_WHITE, "white")
    return f"{prov}{city}" + "".join(rng.choice(ALPHANUM_NO_IO) for _ in range(5)), \
           (MAINLAND_BLUE, "blue")


def rand_gangao_text(rng: random.Random) -> str:
    """粵Z 兩地牌：粤Z + 3~5 位英數 + 港/澳（永遠單行）。"""
    body = "".join(rng.choice(ALPHANUM_NO_IO) for _ in range(rng.randint(3, 4)))
    return f"粤Z{body}{rng.choice('港澳')}"


def rand_hk_text(rng: random.Random, double: bool) -> str:
    """香港牌：1-2 字母 + 1-4 數字，數字不以 0 開頭。"""
    letters = "".join(rng.choice(HK_ALPHA) for _ in range(rng.randint(1, 2)))
    digits = str(rng.randint(1, 9)) + "".join(
        rng.choice("0123456789") for _ in range(rng.randint(0, 3)))
    return f"{letters}\n{digits}" if double else f"{letters}{digits}"


def rand_mo_text(rng: random.Random, double: bool) -> str:
    """澳門牌：M + 字母（禁 I/O/Q/V）+ 2+2 數字。"""
    alpha = rng.choice(MO_ALPHA)
    a, b = rng.randint(0, 99), rng.randint(0, 99)
    return f"M{alpha}\n{a:02d}{b:02d}" if double else f"M{alpha}{a:02d}{b:02d}"


def make_plate(rng: random.Random) -> tuple[np.ndarray, int]:
    """產生一張車牌（已做輕度增強），回傳 (BGR 圖, class_id)。"""
    region = rng.choices(CLASS_NAMES, weights=[30, 15, 35, 20])[0]
    if region == "mainland":
        text, spec_pair = rand_mainland_text(rng)
        spec = spec_pair[0]
    elif region == "gangao":
        text, spec = rand_gangao_text(rng), GANGAO_SINGLE
    elif region == "hk":
        double = rng.random() < 0.30
        text = rand_hk_text(rng, double)
        spec = (HK_FRONT_DOUBLE if double else HK_FRONT) if rng.random() < 0.5 \
            else (HK_REAR_DOUBLE if double else HK_REAR)
    else:
        double = rng.random() < 0.30
        text = rand_mo_text(rng, double)
        # 澳門雙行必須用 MO_WIDE（330×175 方牌）；
        # 短牌 MO_SHORT_L/S 是單行尺寸，塞兩行字會讓 raqm 排版失敗。
        spec = MO_WIDE if double else MO_PLATE

    img = render_plate(text, spec)
    if img is None or img.size == 0:
        img = np.full((110, 380, 3), 40, dtype=np.uint8)
    # 輕度退化：讓模型不要只認「乾淨的合成圖」
    img = augment_plate(img, level=rng.choice(["light", "light", "medium"]))
    return img, CLS_IDX[region]


#: 背景圖自帶車牌的偽標註快取（避免每張都重跑檢測）
_BG_LABELS: dict[str, list[tuple[int, float, float, float, float]]] = {}


def load_bg_labels(cache_path: Path) -> dict:
    """載入背景圖偽標註快取。"""
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))
    return {}


def build_bg_labels(bgs: list[Path], cache_path: Path, device=None) -> dict:
    """
    用四類 detect 模型替背景圖產生偽標註。

    為什麼需要：背景取自真實行車記錄儀圖，畫面裡本來就有車牌。
    如果只標合成上去的牌，原圖那些真牌就成了「未標註的假陰性」，
    模型會學到「這些東西不是車牌」。把真牌一併標上才算誠實。
    """
    if cache_path.exists():
        data = json.loads(cache_path.read_text(encoding="utf-8"))
        if len(data) >= len(bgs) * 0.9:
            print(f"沿用偽標註快取（{len(data)} 張）")
            return data

    from ultralytics import YOLO

    wpath = ROOT / "weights/gba_4class_yolov8s_best.pt"
    model = YOLO(str(wpath))
    print(f"為 {len(bgs)} 張背景圖產生偽標註…")
    out: dict[str, list] = {}
    for i, p in enumerate(bgs, 1):
        img = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        r = model.predict(img, imgsz=640, conf=0.35, device=device, verbose=False)[0]
        H, W = img.shape[:2]
        rows = []
        if r.boxes is not None and len(r.boxes):
            for (x1, y1, x2, y2), cls in zip(r.boxes.xyxy.cpu().numpy(),
                                            r.boxes.cls.cpu().numpy().astype(int)):
                rows.append([int(cls),
                             float((x1 + x2) / 2 / W), float((y1 + y2) / 2 / H),
                             float((x2 - x1) / W), float((y2 - y1) / H)])
        out[p.name] = rows
        if i % 100 == 0:
            print(f"  {i}/{len(bgs)}", flush=True)
    cache_path.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print(f"偽標註完成，已快取 → {cache_path.name}")
    return out


def pick_background(rng: random.Random, bgs: list[Path]):
    """
    取一張背景，回傳 (圖, 檔名, 裁切參數)。

    只取下半部（車身、路面）—— 先前上下各半，導致車牌被貼到天空與樹林上。
    裁切參數用來把背景自帶車牌的偽標註做同樣的座標變換。
    """
    p = rng.choice(bgs)
    img = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return np.full((720, 1280, 3), 60, dtype=np.uint8), p.name, (0, 1.0, 720, 1280)
    h, w = img.shape[:2]
    band = int(h * rng.uniform(0.42, 0.58))
    y0 = h - band
    crop = img[y0:]
    out = cv2.resize(crop, (1280, 720), interpolation=cv2.INTER_AREA)
    # (裁切起點 y, 原圖高, 輸出高, 輸出寬)
    return out, p.name, (y0, h, 720, 1280)


def remap_bg_labels(rows: list, crop) -> list[str]:
    """
    把背景自帶車牌的偽標註，換算到裁切縮放後的座標。

    偽標註是相對「原圖」的歸一化值；背景只取了下半 band 再縮放到 1280×720。
    垂直方向需要平移 + 伸縮，水平方向沒裁切所以 cx / bw 不變。

    不做這一步，原圖裡的車牌就變成未標註的假陰性，
    模型會學到「這些東西不是車牌」。
    """
    y0, oh, out_h, out_w = crop
    band = oh - y0                       # 裁切後的高度（原圖像素）
    if band <= 0:
        return []

    res = []
    for cls, cx, cy, bw, bh in rows:
        cy_px = cy * oh                  # 原圖像素座標
        bh_px = bh * oh

        top, bot = cy_px - bh_px / 2, cy_px + bh_px / 2
        # 車牌要完整落在裁切範圍內，否則丟棄（避免出現半截牌的錯誤標籤）
        if top < y0 or bot > oh:
            continue

        new_cy = (cy_px - y0) / band
        new_bh = bh_px / band
        if not (0.0 < new_cy < 1.0) or new_bh <= 0.002:
            continue
        res.append(f"{cls} {cx:.6f} {new_cy:.6f} {bw:.6f} {new_bh:.6f} 0.000000")
    return res


def composite(bg: np.ndarray, plate: np.ndarray, rng: random.Random):
    """
    把車牌以隨機旋轉角、縮放、位置貼到背景上。

    Returns:
        (貼好的圖, 四角座標 (4,2) float32, 裁切範圍)
    """
    H, W = bg.shape[:2]
    ph, pw = plate.shape[:2]

    # 目標寬度：車牌在畫面中的大小。上限從 320 收到 230 ——
    # 320px 寬的牌在 1280 畫面裡大得不像話，真實行車畫面很少這樣。
    target_w = rng.randint(45, 230)
    scale = target_w / pw
    plate_r = cv2.resize(plate, (target_w, max(2, int(ph * scale))),
                         interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
    ph2, pw2 = plate_r.shape[:2]

    # 傾斜角：多數接近正拍，少數大角度（真實場景分佈）
    angle = rng.gauss(0, 9)
    angle = max(-32.0, min(32.0, angle))
    if rng.random() < 0.12:                    # 12% 強制大角度，強化斜拍樣本
        angle = rng.choice([-1, 1]) * rng.uniform(15, 32)

    # 旋轉後畫布
    diag = int(np.ceil(np.hypot(pw2, ph2))) + 8
    canvas = np.zeros((diag, diag, 3), dtype=np.uint8)
    ox, oy = (diag - pw2) // 2, (diag - ph2) // 2
    canvas[oy:oy + ph2, ox:ox + pw2] = plate_r

    M = cv2.getRotationMatrix2D((diag / 2, diag / 2), angle, 1.0)
    rot = cv2.warpAffine(canvas, M, (diag, diag), flags=cv2.INTER_LINEAR,
                         borderValue=(0, 0, 0))

    # 貼圖位置：水平可全幅，垂直限制在 35%~95%（車身/保險桿高度）。
    # 車牌不會長在天空，限制範圍能讓樣本分佈貼近真實。
    cx = rng.randint(diag // 2, max(diag // 2 + 1, W - diag // 2))
    y_lo = int(H * 0.35) + diag // 2
    y_hi = int(H * 0.95) - diag // 2
    if y_hi <= y_lo:
        y_lo, y_hi = diag // 2, H - diag // 2
    cy = rng.randint(y_lo, max(y_lo + 1, y_hi))
    x0, y0 = cx - diag // 2, cy - diag // 2
    x1, y1 = x0 + diag, y0 + diag
    sx0, sy0 = max(0, x0), max(0, y0)
    sx1, sy1 = min(W, x1), min(H, y1)
    sub = rot[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0]

    # 用非黑區域做遮罩貼上（保留車牌邊界，避免黑框污染背景）
    mask = (sub.max(axis=2) > 12).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    roi = bg[sy0:sy1, sx0:sx1]
    roi[mask > 0] = sub[mask > 0]

    # 四角：原始矩形四角 → 旋轉 → 平移到背景座標
    rect = np.array([[ox, oy], [ox + pw2, oy], [ox + pw2, oy + ph2], [ox, oy + ph2]],
                    dtype=np.float32)
    ones = np.ones((4, 1), dtype=np.float32)
    pts = np.hstack([rect, ones]) @ M.T
    pts += np.array([x0, y0], dtype=np.float32)
    return bg, pts, (sx0, sy0, sx1, sy1)


def to_obb_label(pts: np.ndarray, cls: int, W: int, H: int) -> str | None:
    """四角 → YOLO-OBB 標籤 (class cx cy w h angle)，歸一化。"""
    rect = cv2.minAreaRect(pts.astype(np.float32))
    (cx, cy), (w, h), ang = rect
    if w < 6 or h < 4:
        return None
    # 統一成長邊水平：w 為長邊，角度落在 [-90, 0)
    if w < h:
        w, h = h, w
        ang += 90.0
    if ang >= 90:
        ang -= 180
    if ang < -90:
        ang += 180
    return (f"{cls} {cx / W:.6f} {cy / H:.6f} {w / W:.6f} {h / H:.6f} "
            f"{np.deg2rad(ang):.6f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=16000, help="生成張數")
    ap.add_argument("--out", default="dataset_obb")
    ap.add_argument("--bg-dir", default="test_data/real/hk", help="背景圖來源")
    ap.add_argument("--val-ratio", type=float, default=0.12)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--device", default=None, help="推論裝置（偽標註用）")
    args = ap.parse_args()

    out = ROOT / args.out
    bgs = sorted(ROOT.glob(f"{args.bg_dir}/*.jpg")) + \
          sorted(ROOT.glob(f"{args.bg_dir}/*.png"))
    if not bgs:
        raise SystemExit(f"找不到背景圖：{args.bg_dir}")
    print(f"背景圖 {len(bgs)} 張")

    rng = random.Random(args.seed)
    n_val = int(args.n * args.val_ratio)
    splits = {"train": args.n - n_val, "val": n_val}
    for split in splits:
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    # 背景自帶車牌的偽標註（不快取就要每張重跑檢測）
    cache = out / "bg_pseudo_labels.json"
    bg_labels = build_bg_labels(bgs, cache, device=args.device)
    n_bg_total = sum(len(v) for v in bg_labels.values())
    print(f"背景自帶車牌 {n_bg_total} 個（會一併標註，避免假陰性）")

    made = {"train": 0, "val": 0}
    angles_seen: list[float] = []
    idx = 0
    for split, count in splits.items():
        for k in range(count):
            bg, bg_name, crop = pick_background(rng, bgs)
            # 先把背景原有車牌的偽標註寫進去
            labels = remap_bg_labels(bg_labels.get(bg_name, []), crop)
            # 15% 機率一張圖貼多塊牌（真實場景常見：前後牌、雙牌車）
            n_plates = 2 if rng.random() < 0.15 else 1
            for _ in range(n_plates):
                plate, cls = make_plate(rng)
                bg, pts, _ = composite(bg, plate, rng)
                lab = to_obb_label(pts, cls, bg.shape[1], bg.shape[0])
                if lab:
                    labels.append(lab)
                    angles_seen.append(float(lab.split()[-1]))
            if not labels:
                continue
            name = f"obb_{split}_{idx:06d}"
            idx += 1
            cv2.imwrite(str(out / "images" / split / f"{name}.jpg"), bg,
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
            (out / "labels" / split / f"{name}.txt").write_text(
                "\n".join(labels) + "\n", encoding="utf-8")
            made[split] += 1
            if (k + 1) % 2000 == 0:
                print(f"  {split} {k+1}/{count}", flush=True)

    yaml = (f"# GBA 四地區車牌 OBB 數據集（合成，四角由變換矩陣精確計算）\n"
            f"path: {out}\n"
            f"train: images/train\nval: images/val\n\n"
            f"nc: 4\nnames:\n"
            + "".join(f"  {i}: {n}\n" for i, n in enumerate(CLASS_NAMES)))
    (out / "data.yaml").write_text(yaml, encoding="utf-8")

    stats = {
        "train": made["train"], "val": made["val"],
        "angle_deg": {
            "mean": round(float(np.degrees(np.mean(angles_seen))), 2),
            "abs_mean": round(float(np.degrees(np.mean(np.abs(angles_seen)))), 2),
            "p95_abs": round(float(np.degrees(np.percentile(np.abs(angles_seen), 95))), 2),
        },
    }
    (out / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1),
                                    encoding="utf-8")
    print(f"\n完成：train {made['train']} / val {made['val']}")
    print(f"傾斜角 |mean| {stats['angle_deg']['abs_mean']}° "
          f"p95 {stats['angle_deg']['p95_abs']}°")
    print(f"輸出 {out}")


if __name__ == "__main__":
    main()
