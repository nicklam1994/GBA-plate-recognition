"""
字符詞表管理。

79 類融合方案（三輪調研結論）：
    基類 78 類 = we0091234 的權重（31 省份 + 學警港澳挂使領民航危險品 + 0-9 + 24 字母無 I/O）
    新增 1 類   = '-'（澳門車牌 `MA-12-34` 的連字符，參考 HuKai97 的做法）

設計要點：
- 既有權重（78 類）保持不變，向後兼容
- 新增 '-' 需要重訓識別模型；未重訓時允許用 78 類權重推理
- 提供 char→index 與 index→char 雙向映射
"""

from __future__ import annotations

#: 原始 78 類（與 weights/plate_rec_color.pth 完全一致，勿改順序）
BASE_VOCAB_V78 = (
    "#"                                          # index 0 = CTC blank
    "京沪津渝冀晋蒙辽吉黑苏浙皖闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新"  # 31 省份
    "学警港澳挂使领民航危"                        # 10 特殊標識
    "0123456789"                                 # 10 數字
    "ABCDEFGHJKLMNPQRSTUVWXYZ"                   # 24 字母（無 I/O）
    "险品"                                       # 2 補充
)

#: 擴展後的 79 類（增加連字符）
VOCAB_V79 = BASE_VOCAB_V78 + "-"

#: 港澳特殊字符集合（用於地區推斷與校驗）
HK_MO_CHARS = frozenset("港澳")

#: 省份簡稱集合
PROVINCES = frozenset("京沪津渝冀晋蒙辽吉黑苏浙皖闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新")


def build_vocab(extended: bool = False) -> str:
    """
    取得字符表。

    Args:
        extended: True 回傳 79 類（含 '-'），False 回傳原始 78 類

    Returns:
        字符表字串，索引即為 CTC 類別編號
    """
    return VOCAB_V79 if extended else BASE_VOCAB_V78


def vocab_size(extended: bool = False) -> int:
    """字符表大小"""
    return len(build_vocab(extended))


def build_index_map(vocab: str | None = None) -> dict[str, int]:
    """字符 → 索引"""
    vocab = vocab if vocab is not None else BASE_VOCAB_V78
    return {ch: idx for idx, ch in enumerate(vocab)}


def blank_index() -> int:
    """CTC blank 索引，固定為 0"""
    return 0


def is_province(ch: str) -> bool:
    """是否為省級行政區簡稱"""
    return ch in PROVINCES


def has_gangao_suffix(text: str) -> bool:
    """車牌是否以「港」或「澳」結尾（粵Z 兩地牌特徵）"""
    return bool(text) and text[-1] in HK_MO_CHARS


if __name__ == "__main__":
    print(f"78 類: {vocab_size(False)} -> {BASE_VOCAB_V78[:20]}...")
    print(f"79 類: {vocab_size(True)}  -> 末尾新增加 '{VOCAB_V79[-1]}'")
