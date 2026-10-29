"""
前端 i18n 資源一致性測試。

不跑瀏覽器，只做靜態檢查：
  1. index.html 用到的每個 data-i18n key，在 i18n.js 三種語言都要有定義
  2. i18n.js 三種語言的 key 集合要一致（防止漏翻譯）
  3. web_ui.py 傳給前端的 lang 值必須是支援的語言碼
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
I18N_JS = ROOT / "static" / "js" / "i18n.js"
INDEX_HTML = ROOT / "templates" / "index.html"
WEB_UI = ROOT / "web_ui.py"


@pytest.fixture(scope="module")
def i18n_source() -> str:
    assert I18N_JS.exists(), f"i18n.js 不存在: {I18N_JS}"
    return I18N_JS.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def html_source() -> str:
    assert INDEX_HTML.exists()
    return INDEX_HTML.read_text(encoding="utf-8")


def _extract_lang_blocks(js: str) -> dict[str, set[str]]:
    """
    從 i18n.js 抽出每個語言的 key 集合。

    注意：檔案裡有兩個地方出現語言碼——`LANGS`（按鈕標籤）與 `MESSAGES`（翻譯）。
    必須從 `var MESSAGES` 之後開始搜尋，否則會抓到 LANGS 的小區塊。
    """
    # 定位 MESSAGES 起點
    msg_pos = js.find("var MESSAGES")
    if msg_pos < 0:
        raise ValueError("找不到 var MESSAGES 定義")
    scope = js[msg_pos:]

    def _extract_object(s: str, start: int) -> str:
        """從 start（指向 '{'）做括號配對，回傳完整物件字串"""
        depth = 0
        for i in range(start, len(s)):
            if s[i] == "{":
                depth += 1
            elif s[i] == "}":
                depth -= 1
                if depth == 0:
                    return s[start : i + 1]
        return ""

    blocks: dict[str, set[str]] = {}
    for lang in ("zh-TW", "zh-CN", "en"):
        m = re.search(r'"' + re.escape(lang) + r'"\s*:\s*\{', scope)
        if not m:
            continue
        body = _extract_object(scope, m.end() - 1)
        keys = set(re.findall(r'"([a-zA-Z0-9_.]+)"\s*:', body))
        blocks[lang] = keys
    return blocks


class TestI18nJs:
    def test_file_exists(self):
        assert I18N_JS.exists(), "static/js/i18n.js 缺失（曾因路徑錯誤導致 404）"

    def test_all_langs_present(self, i18n_source):
        blocks = _extract_lang_blocks(i18n_source)
        assert set(blocks) == {"zh-TW", "zh-CN", "en"}, f"實際: {set(blocks)}"

    def test_key_sets_identical(self, i18n_source):
        """三語言必須有完全相同的 key 集合"""
        blocks = _extract_lang_blocks(i18n_source)
        assert len(blocks) == 3

        base = blocks["zh-TW"]
        for lang in ("zh-CN", "en"):
            missing = base - blocks[lang]
            extra = blocks[lang] - base
            assert not missing, f"{lang} 缺少 key: {sorted(missing)}"
            assert not extra, f"{lang} 多餘 key: {sorted(extra)}"

    def test_default_lang_is_traditional(self, i18n_source):
        """繁體為預設（項目定位）"""
        assert re.search(r'DEFAULT_LANG\s*=\s*"zh-TW"', i18n_source)

    def test_storage_key_defined(self, i18n_source):
        assert re.search(r'STORAGE_KEY\s*=\s*"gba_plate_lang"', i18n_source)

    def test_has_public_api(self, i18n_source):
        for name in ("t:", "setLang:", "onChange:", "apply:"):
            assert name in i18n_source, f"缺少 API {name}"


class TestHtmlKeys:
    def test_html_references_i18n(self, html_source):
        assert "/static/js/i18n.js" in html_source, "index.html 未載入 i18n.js"

    def test_all_data_i18n_keys_defined(self, html_source, i18n_source):
        """HTML 用到的每個 key 都要在 i18n.js 有定義"""
        used = set(re.findall(r'data-i18n(?:-placeholder|-title)?="([^"]+)"', html_source))
        assert used, "HTML 沒有任何 data-i18n 屬性"

        defined = set()
        for keys in _extract_lang_blocks(i18n_source).values():
            defined |= keys

        missing = used - defined
        assert not missing, f"HTML 使用但 i18n.js 未定義: {sorted(missing)}"

    def test_lang_switch_buttons(self, html_source):
        """三個語言切換按鈕都要在"""
        for lang in ("zh-TW", "zh-CN", "en"):
            assert f'data-lang-btn="{lang}"' in html_source, f"缺語言按鈕 {lang}"

    def test_no_hardcoded_ui_text(self, html_source):
        """
        關鍵 UI 文字必須透過 data-i18n 綁定，不能裸寫在標籤內。

        span 內保留預設繁體文字是刻意的（JS 失效時當 fallback），
        但該 span 必須帶 data-i18n 屬性。
        """
        targets = [
            "啟動 WebCam",
            "開始識別",
            "上傳圖片",
            "連接狀態",
            "性能統計",
            "識別結果",
            "上傳測試",
            "停止",
        ]
        naked = []
        for text in targets:
            if text not in html_source:
                continue
            # 必須存在 "data-i18n=...">文字< 的綁定形式
            bound = re.search(
                r'data-i18n="[^"]+"\s*>\s*' + re.escape(text) + r'\s*<',
                html_source,
            )
            if not bound:
                naked.append(text)
        assert not naked, f"以下文字未透過 data-i18n 綁定: {naked}"


class TestWebUiLang:
    def test_accepts_lang_param(self):
        src = WEB_UI.read_text(encoding="utf-8")
        assert 'data.get("lang"' in src, "web_ui.py 未讀取 lang 參數"

    def test_uses_localize(self):
        src = WEB_UI.read_text(encoding="utf-8")
        assert "localize_plate_dict" in src
        assert "DEFAULT_LANG" in src

    def test_default_lang_consistent(self, i18n_source):
        """前後端預設語言要一致"""
        from gba_plate.i18n import DEFAULT_LANG as BACKEND_DEFAULT

        m = re.search(r'DEFAULT_LANG\s*=\s*"([^"]+)"', i18n_source)
        assert m, "前端未宣告 DEFAULT_LANG"
        assert m.group(1) == BACKEND_DEFAULT, (
            f"預設語言不一致：前端 {m.group(1)} vs 後端 {BACKEND_DEFAULT}"
        )
