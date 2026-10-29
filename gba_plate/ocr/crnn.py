"""
CRNN 車牌識別器（CTC）。

網絡結構複用 plate_recognition/plateNet.py 的 myNet_ocr_color：
    cfg 實際權重: [8, 8, 16, 16, 'M', 32, 32, 'M', 48, 48, 'M', 64, 128]
    參數量: 180,435（0.69 MB fp32）—— 比 LPRNet 小 2.5 倍

前處理（與原項目一致，勿改）：
    resize 168×48 → (img/255 - 0.588) / 0.193 → CHW
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

from gba_plate.base import BaseOCR, OcrResult, PlateColor
from gba_plate.ocr.vocab import BASE_VOCAB_V78, VOCAB_V79

#: 識別輸入尺寸。固定高度、寬度上限 —— 保持寬高比，避免長牌壓扁／方牌拉長。
#: 香港長牌 520×110 (4.73:1)、方牌 330×170 (1.94:1)、內地 440×140 (3.14:1)，
#: 最寬的長牌在 H=48 下需 227px，故上限取 256（CNN stride 4 → 64 時間步，
#: CTC 對 8 字符標籤需 ≥17 步，餘裕充足）。
INPUT_W, INPUT_H = 256, 48
INPUT_SIZE = (INPUT_W, INPUT_H)
#: padding 填充值（與車牌平均底色接近的灰）
PAD_VALUE = 128
#: 正規化參數（原項目實測值，勿改）
MEAN_VALUE = 0.588
STD_VALUE = 0.193


def fit_aspect(img: np.ndarray, out_h: int = INPUT_H, out_w: int = INPUT_W,
               pad: int = PAD_VALUE) -> np.ndarray:
    """
    等比縮放到固定高度，寬度不足處以灰階填充（左對齊）。

    取代直接 cv2.resize 到固定尺寸 —— 後者會把 4.73:1 的香港長牌壓扁、
    把 1.94:1 的方牌拉長 80%，字符形狀失真直接損害識別率。
    """
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

#: 顏色索引 → PlateColor（順序與權重的顏色頭一致）
_COLOR_MAP = [
    PlateColor.BLACK,   # 0 黑（使領館 / 粵Z / 澳門）
    PlateColor.BLUE,    # 1 藍（內地小型車）
    PlateColor.GREEN,   # 2 綠（內地新能源）
    PlateColor.WHITE,   # 3 白（內地軍警 / 香港前牌）
    PlateColor.YELLOW,  # 4 黃（內地大型車 / 香港後牌）
]

#: 字母與數字（香港/澳門車牌只用這些）
_ALNUM = frozenset("ABCDEFGHJKLMNPQRSTUVWXYZ0123456789")

#: 按地區限制可輸出字符集 —— 解決「香港車牌被識別成皖A123」的分佈偏移問題。
#: 檢測器已判定地區，OCR 卻自由輸出漢字，兩者矛盾。約束後模型只能在合法字符裡選次優解。
REGION_CHARSET: dict[str, frozenset[str]] = {
    # 香港：1-2 字母 + 1-4 數字，無漢字
    "hk": _ALNUM,
    # 澳門：M + 字母 + 數字，無漢字
    "mo": _ALNUM,
    # 粵Z 兩地牌：粵 Z + 字母數字 + 港/澳 後綴
    "gangao": frozenset("粵Z港澳") | _ALNUM,
    # 內地：不限制（省份、警、學、領等都要）
    "mainland": None,  # type: ignore[dict-item]
}


def ctc_greedy_decode(
    indices: np.ndarray,
    probs: np.ndarray,
    vocab: str,
    allowed: frozenset[str] | None = None,
) -> tuple[str, list[float]]:
    """
    CTC 貪心解碼。

    規則：跳過 blank（索引 0），折疊連續重複字符。

    Args:
        indices: 每帧 argmax 的類別索引
        probs: 每帧對應的最大概率
        vocab: 字符表（索引 0 必須是 blank）
        allowed: 允許的字符集合；None 表示不限制（用於按地區約束輸出）

    Returns:
        (解碼文字, 每字符置信度列表)
    """
    chars: list[str] = []
    char_probs: list[float] = []
    prev = -1

    for idx, prob in zip(indices.tolist(), probs.tolist()):
        if idx == 0:          # blank
            prev = idx
            continue
        if idx == prev:       # 連續重複 → CTC 折疊
            prev = idx
            continue
        if 0 < idx < len(vocab):
            ch = vocab[idx]
            if allowed is None or ch in allowed:
                chars.append(ch)
                char_probs.append(float(prob))
        prev = idx

    return "".join(chars), char_probs


class CRNNPlateOCR(BaseOCR):
    """CRNN + CTC 識別器，支持顏色分類。"""

    def __init__(
        self,
        weights: str | Path = "weights/plate_rec_color.pth",
        device: str | torch.device | None = None,
        use_color: bool = True,
    ) -> None:
        """
        Args:
            weights: 權重路徑（.pth checkpoint，含 cfg 與 state_dict）
            device: torch 設備，None 表示自動選擇
            use_color: 是否同時做顏色分類
        """
        self.weights = Path(weights)
        if not self.weights.exists():
            raise FileNotFoundError(f"識別權重不存在: {self.weights}")

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.use_color = use_color
        self.model = self._load_model()

        # 依權重實際類別數選擇詞表（78 或 79）
        n_cls = self._num_classes
        self.vocab = VOCAB_V79 if n_cls == len(VOCAB_V79) else BASE_VOCAB_V78
        if n_cls != len(self.vocab):
            raise ValueError(
                f"權重類別數 {n_cls} 與詞表 {len(self.vocab)} 不符"
            )

    # ------------------------------------------------------------------ 內部

    def _load_model(self):
        # 延遲導入，避免 gba_plate 在無 torch 環境下 import 失敗
        from plate_recognition.plateNet import myNet_ocr_color

        checkpoint = torch.load(
            self.weights, map_location=self.device, weights_only=False
        )
        cfg = checkpoint["cfg"]
        state_dict = checkpoint["state_dict"]
        color_classes = 5 if self.use_color else 0

        # 從 state_dict 推斷類別數（newCnn.weight 的 out_channels）
        self._num_classes = state_dict["newCnn.weight"].shape[0]

        model = myNet_ocr_color(
            num_classes=self._num_classes,
            export=True,
            cfg=cfg,
            color_num=color_classes,
        )
        model.load_state_dict(state_dict, strict=False)
        model.to(self.device)
        model.eval()
        return model

    def _preprocess(self, plate_crop: np.ndarray) -> torch.Tensor:
        img = fit_aspect(plate_crop, INPUT_H, INPUT_W)
        img = img.astype(np.float32)
        img = (img / 255.0 - MEAN_VALUE) / STD_VALUE
        img = img.transpose(2, 0, 1)                  # HWC → CHW
        tensor = torch.from_numpy(np.ascontiguousarray(img)).to(self.device)
        return tensor.unsqueeze(0)                     # (1, 3, H, W)

    # ------------------------------------------------------------------ 對外

    def predict(
        self,
        plate_crop: np.ndarray,
        region: str | None = None,
    ) -> OcrResult | None:
        """
        Args:
            plate_crop: 已校正的車牌小圖（BGR）
            region: 地區字串（"hk"/"mo"/"gangao"/"mainland"）；
                    給定時會限制可輸出字符集，避免分佈偏移
                    （例如香港車牌不會輸出漢字省份）
        """
        if plate_crop is None or plate_crop.size == 0:
            return None

        tensor = self._preprocess(plate_crop)

        allowed = REGION_CHARSET.get(region) if region else None

        with torch.no_grad():
            if self.use_color:
                char_logits, color_logits = self.model(tensor)
                color_probs = torch.softmax(color_logits, dim=-1)
                color_conf, color_idx = torch.max(color_probs, dim=-1)
                color = _COLOR_MAP[int(color_idx.item())]
                color_conf = float(color_conf.item())
            else:
                char_logits = self.model(tensor)
                color, color_conf = None, None

            # 地區約束：非法字符的概率轉移到 CTC blank（而非直接 -inf）。
            # 直接用 -inf 會讓模型被迫在每個時間步選字母，產生長串亂碼；
            # 轉移到 blank 則等價於「模型想輸出漢字時當作跳過」。
            if allowed:
                idx_map = {ch: i for i, ch in enumerate(self.vocab)}
                keep = {idx_map[ch] for ch in allowed if ch in idx_map}
                keep.add(0)  # blank 永遠合法

                illegal = [i for i in range(len(self.vocab)) if i not in keep]
                if illegal:
                    char_logits = char_logits.clone()
                    illegal_logits = char_logits[..., illegal]
                    transfer = torch.logsumexp(illegal_logits, dim=-1)
                    char_logits[..., 0] = torch.logaddexp(char_logits[..., 0], transfer)
                    char_logits[..., illegal] = float("-inf")

            probs = torch.softmax(char_logits, dim=-1)
            max_probs, indices = probs.max(dim=-1)
            indices = indices.view(-1).cpu().numpy()
            max_probs = max_probs.view(-1).cpu().numpy()

        text, char_probs = ctc_greedy_decode(indices, max_probs, self.vocab, allowed)
        if not text:
            return None

        return OcrResult(
            text=text,
            confidence=float(np.mean(char_probs)) if char_probs else 0.0,
            char_confidences=char_probs,
            color=color,
            color_confidence=color_conf,
        )
