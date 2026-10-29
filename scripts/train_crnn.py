#!/usr/bin/env python3
"""
訓練 CRNN 車牌識別模型（CTC + 顏色分類）。

相比現有權重的改進：
    現有: cfg=small [8,8,16,16,M,32,32,M,48,48,M,64,128] → 180k 參數，46.2% 整串準確
    新版: cfg=big   [32,32,64,64,M,128,128,M,196,196,M,256,256] → 約 8M 參數
    數據: CCPD 10 萬張（零人工標註，檔名自帶真值）

顏色標籤由 HSV 自動判定，無需人工：
    藍/綠/黃/白/黑 五類，規則見 detect_color()

用法：
    python scripts/train_crnn.py --data crnn_dataset --epochs 30 --batch 64
"""
from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(ROOT))

from plate_recognition.plateNet import myNet_ocr_color  # noqa: E402
from gba_plate.preprocess.double_line import (  # noqa: E402
    looks_double_line,
    split_merge_double,
)

INPUT_W, INPUT_H = 256, 48   # 寬度上限提高以容納 4.73:1 的香港長牌（等比縮放 + padding）
PAD_VALUE = 128
MEAN_VALUE, STD_VALUE = 0.588, 0.193


def fit_aspect(img, out_h=INPUT_H, out_w=INPUT_W, pad=PAD_VALUE):
    """等比縮放到固定高度，寬度不足處灰階填充（左對齊），避免車牌變形。"""
    if img is None or img.size == 0:
        return np.full((out_h, out_w, 3), pad, dtype=np.uint8)
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return np.full((out_h, out_w, 3), pad, dtype=np.uint8)
    scale = out_h / float(h)
    new_w = max(1, min(int(round(w * scale)), out_w))
    resized = cv2.resize(img, (new_w, out_h), interpolation=cv2.INTER_LINEAR)
    if new_w == out_w:
        return resized
    canvas = np.full((out_h, out_w, 3), pad, dtype=np.uint8)
    canvas[:, :new_w] = resized
    return canvas

#: cfg 選項
CFG_SMALL = [8, 8, 16, 16, 'M', 32, 32, 'M', 48, 48, 'M', 64, 128]
CFG_MEDIUM = [16, 16, 32, 32, 'M', 64, 64, 'M', 96, 96, 'M', 128, 256]
CFG_BIG = [32, 32, 64, 64, 'M', 128, 128, 'M', 196, 196, 'M', 256, 256]

CHARSET = (
    "京沪津渝冀晋蒙辽吉黑苏浙皖闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新"
    "学警港澳挂使领民航危"
    "0123456789"
    "ABCDEFGHJKLMNPQRSTUVWXYZ"
    "险品"
)
#: 0 = CTC blank，字符從 1 開始
IDX2CHAR = {i + 1: c for i, c in enumerate(CHARSET)}
NUM_CLASSES = len(CHARSET) + 1  # 78

COLOR_NAMES = ["黑", "藍", "綠", "白", "黃"]


def detect_color(bgr: np.ndarray) -> int:
    """
    判定車牌底色（自動標註顏色標籤）。

    Returns:
        0=黑 1=藍 2=綠 3=白 4=黃

    兩個關鍵修正（舊版用整圖 HSV 平均值，粵Z 只有 3/7 判對）：

    1. H 通道是圓形數據，不能取平均。0° 與 180° 都是紅，平均會落到 90°（青色）。
    2. 黑底白字牌的白字筆劃很粗，中央區域可能被白字佔滿，
       用「多數像素」必然判成白底。改看「最暗的 20% 像素」= 底板本身。

    實測（7 個粵Z 樣本）：3/7 → 6/7 判黑。
    """
    h, w = bgr.shape[:2]
    c = bgr[int(h * 0.2):int(h * 0.8), int(w * 0.15):int(w * 0.85)]
    if c.size == 0:
        c = bgr
    hsv = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)
    H, S, V = [x.reshape(-1) for x in cv2.split(hsv)]
    if V.size == 0:
        return 3

    n = V.size
    # 彩色底（面積大，用佔比即可）
    if float(((H >= 15) & (H <= 35) & (S > 90) & (V > 90)).sum()) / n >= 0.30:
        return 4                      # 黃
    if float(((H >= 40) & (H <= 90) & (S > 70) & (V > 70)).sum()) / n >= 0.30:
        return 2                      # 綠
    if float(((H >= 95) & (H <= 130) & (S > 70) & (V > 70)).sum()) / n >= 0.30:
        return 1                      # 藍

    # 無彩色（黑底 / 白底）：看底板亮度，而非整體平均
    v_lo = float(np.percentile(V, 20))     # 暗的那批 = 底板
    v_hi = float(np.percentile(V, 80))
    s_lo = float(np.percentile(S, 20))
    if v_lo < 110:
        return 0                      # 底板暗 → 黑底
    if v_lo > 165 and s_lo < 60:
        return 3                      # 底板亮且低飽和 → 白底
    return 3 if v_hi > 180 else 0


class PlateDataset(Dataset):
    def __init__(self, list_file: Path, train: bool = True):
        self.samples: list[tuple[Path, list[int], int]] = []
        with open(list_file, encoding="utf-8") as fh:
            for line in fh:
                parts = line.strip().split("\t")
                if len(parts) != 2:
                    continue
                path, idxs = parts
                labels = [int(v) for v in idxs.split()]
                if not labels or len(labels) > 9:
                    continue
                self.samples.append((Path(path), labels, 0))
        self.train = train
        print(f"  {list_file.name}: {len(self.samples)} 樣本")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, labels, _ = self.samples[i]
        img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            img = np.full((INPUT_H, INPUT_W, 3), 128, dtype=np.uint8)

        color = detect_color(img)

        # 與推理端 alpr.py 一致：雙行牌先拼成單行再送 OCR
        # （否則模型學的是「兩行擠在一張圖」，實際用的是「拼好的寬圖」）
        if looks_double_line(img):
            img = split_merge_double(img)

        # 輕度增強（車牌文字不能翻轉、不能大幅扭曲）
        if self.train:
            if random.random() < 0.3:
                img = cv2.convertScaleAbs(img, alpha=random.uniform(0.8, 1.2),
                                          beta=random.randint(-20, 20))
            if random.random() < 0.2:
                img = cv2.GaussianBlur(img, (3, 3), random.uniform(0, 1.0))

        img = fit_aspect(img, INPUT_H, INPUT_W)
        img = np.reshape(img, (INPUT_H, INPUT_W, 3)).astype(np.float32)
        img = (img / 255.0 - MEAN_VALUE) / STD_VALUE
        img = img.transpose(2, 0, 1)

        return torch.from_numpy(np.ascontiguousarray(img)), labels, color


def collate(batch):
    imgs, labels, colors = zip(*batch)
    imgs = torch.stack(imgs)
    colors = torch.tensor(colors, dtype=torch.long)
    target_lengths = torch.tensor([len(l) for l in labels], dtype=torch.long)
    flat = torch.tensor([v for l in labels for v in l], dtype=torch.long)
    return imgs, flat, target_lengths, colors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="crnn_dataset")
    ap.add_argument("--cfg", default="big", choices=["small", "medium", "big"])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--device", default="0")
    ap.add_argument("--out", default="weights/plate_rec_color_big.pth")
    ap.add_argument("--init", default=None,
                    help="從現有權重初始化（cfg 必須一致，用於微調）")
    ap.add_argument("--color-weight", type=float, default=0.2)
    args = ap.parse_args()

    data_dir = Path(args.data)
    cfg = {"small": CFG_SMALL, "medium": CFG_MEDIUM, "big": CFG_BIG}[args.cfg]

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    print("=" * 66)
    print("訓練 CRNN 車牌識別")
    print("=" * 66)
    print(f"cfg:      {args.cfg} ({len(cfg)} layers)")
    print(f"checkpoint: {data_dir/'train.txt'}  {data_dir/'val.txt'}")
    print(f"device:   {device}")
    print(f"epochs={args.epochs} batch={args.batch} lr={args.lr}")
    print()

    print("載入數據...")
    train_ds = PlateDataset(data_dir / "train.txt", train=True)
    val_ds = PlateDataset(data_dir / "val.txt", train=False)

    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                          num_workers=args.workers, collate_fn=collate,
                          pin_memory=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                        num_workers=args.workers, collate_fn=collate, pin_memory=True)

    print("建立模型...")
    # export=True 與 CRNNPlateOCR 推論時一致：返回 raw logits (T, B, C)，不做 log_softmax
    model = myNet_ocr_color(cfg=cfg, num_classes=NUM_CLASSES, color_num=5,
                            export=True).to(device)

    # 微調模式：載入現有權重
    if args.init:
        init_path = Path(args.init)
        if not init_path.is_absolute():
            init_path = ROOT / init_path
        if not init_path.exists():
            raise SystemExit(f"--init 權重不存在: {init_path}")
        ckpt = torch.load(init_path, map_location=device, weights_only=False)
        init_cfg = ckpt.get("cfg")
        if init_cfg != cfg:
            raise SystemExit(
                f"--init 的 cfg 與 --cfg 不一致，無法載入。\n"
                f"  init cfg: {init_cfg}\n"
                f"  目標 cfg: {cfg}\n"
                f"提示：微調時請用 --cfg small；換大模型時不要加 --init（從頭訓練）。"
            )
        model.load_state_dict(ckpt["state_dict"])
        print(f"  ✅ 已載入初始權重: {init_path.name}（微調模式）")

    n_param = sum(p.numel() for p in model.parameters())
    print(f"  參數量: {n_param:,} ({n_param*4/1024/1024:.1f} MB fp32)")

    ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True)
    color_loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc = 0.0
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        total_loss = 0.0
        n_batch = 0

        for imgs, targets, target_lens, colors in train_dl:
            imgs = imgs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            target_lens = target_lens.to(device, non_blocking=True)
            colors = colors.to(device, non_blocking=True)

            char_logits, color_logits = model(imgs)
            # 實測 export=True 輸出為 (B, T, C)（前人註釋 [w,b,c] 有誤）
            log_probs = char_logits.permute(1, 0, 2).log_softmax(2)  # → (T, B, C)
            T = log_probs.size(0)
            B = log_probs.size(1)
            input_lens = torch.full((B,), T, dtype=torch.long, device=device)

            loss_ctc = ctc_loss(log_probs, targets, input_lens, target_lens)
            loss_color = color_loss(color_logits, colors) if color_logits is not None else 0.0
            loss = loss_ctc + args.color_weight * loss_color

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            total_loss += float(loss)
            n_batch += 1

        # 驗證
        model.eval()
        exact = 0
        total = 0
        with torch.no_grad():
            for imgs, targets, target_lens, colors in val_dl:
                imgs = imgs.to(device, non_blocking=True)
                char_logits, _ = model(imgs)
                # (B, T, C) → argmax → (B, T)
                preds = char_logits.argmax(dim=-1).cpu().numpy()

                offset = 0
                for b in range(preds.shape[0]):
                    L = int(target_lens[b])
                    gt_ids = targets[offset:offset + L].tolist()
                    offset += L
                    gt = "".join(IDX2CHAR.get(v, "") for v in gt_ids)

                    # CTC 貪心解碼
                    prev = -1
                    chars = []
                    for v in preds[b]:
                        v = int(v)
                        if v == 0 or v == prev:
                            prev = v
                            continue
                        chars.append(IDX2CHAR.get(v, ""))
                        prev = v
                    pred_text = "".join(chars)

                    total += 1
                    if pred_text == gt:
                        exact += 1

        acc = exact / max(total, 1)
        dt = time.time() - t0
        print(f"Epoch {epoch:>3}/{args.epochs}  loss={total_loss/max(n_batch,1):.4f}  "
              f"val_exact={acc*100:.2f}% ({exact}/{total})  {dt:.0f}s", flush=True)

        if acc > best_acc:
            best_acc = acc
            torch.save({"cfg": cfg, "state_dict": model.state_dict()}, out_path)
            print(f"  ✅ 新最佳 {acc*100:.2f}% → {out_path}")

        scheduler.step()

    print()
    print("=" * 66)
    print(f"訓練完成，最佳驗證準確率: {best_acc*100:.2f}%")
    print(f"權重: {out_path}  ({out_path.stat().st_size/1024/1024:.1f} MB)" if out_path.exists() else "")
    print("=" * 66)


if __name__ == "__main__":
    main()
