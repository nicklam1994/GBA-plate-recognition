/**
 * GBA Plate Recognition — 前端 i18n
 *
 * 支援 zh-TW（繁體，預設）/ zh-CN（簡體）/ en（英文）
 * 選擇存 localStorage，重新載入後保持。
 *
 * 用法：
 *   <script src="/static/js/i18n.js"></script>
 *   <span data-i18n="btn.start"></span>
 *   <input data-i18n-placeholder="upload.hint">
 *   I18N.t("stat.fps")
 *   I18N.setLang("zh-CN")
 *   I18N.lang          // 目前語言
 *   I18N.onChange(fn)  // 語言切換回調
 */
(function (global) {
    "use strict";

    var STORAGE_KEY = "gba_plate_lang";
    var DEFAULT_LANG = "zh-TW";

    var LANGS = {
        "zh-TW": { label: "繁", name: "繁體中文" },
        "zh-CN": { label: "简", name: "简体中文" },
        "en": { label: "EN", name: "English" }
    };

    var MESSAGES = {
        // ==================== 繁體中文 ====================
        "zh-TW": {
            "app.title": "四地車牌識別系統",
            "app.subtitle": "內地 · 粵Z兩地牌 · 香港 · 澳門 ｜ WebCam 實時測試",
            "app.docTitle": "四地車牌識別 - WebCam 測試",

            "btn.startCam": "啟動 WebCam",
            "btn.stop": "停止",
            "btn.startDetect": "開始識別",
            "btn.pauseDetect": "暫停識別",
            "btn.upload": "上傳圖片",

            "panel.status": "連接狀態",
            "panel.stats": "性能統計",
            "panel.results": "識別結果",
            "panel.upload": "上傳測試",

            "stat.fps": "FPS",
            "stat.latency": "延遲 (ms)",
            "stat.plates": "檢測車牌",
            "stat.frames": "處理幀數",

            "status.disconnected": "未連接",
            "status.connected": "已連接",
            "status.camOn": "WebCam 已啟動",
            "status.closed": "已斷開",
            "status.connecting": "連接中…",
            "status.modelNotReady": "模型未載入",

            "result.empty": "暫無識別結果",
            "result.recognize": "辨識",
            "result.detect": "檢測",
            "result.double": "雙行",
            "result.corrected": "已糾正",
            "result.invalid": "格式不符",

            "upload.hint": "點擊上傳車牌圖片",
            "upload.orDrag": "或拖拽到此處",

            "region.mainland": "內地",
            "region.gangao": "粵Z",
            "region.hk": "香港",
            "region.mo": "澳門",
            "region.unknown": "未知",

            "err.camDenied": "無法訪問 WebCam",
            "err.decode": "無法解碼圖片"
        },

        // ==================== 简体中文 ====================
        "zh-CN": {
            "app.title": "四地车牌识别系统",
            "app.subtitle": "内地 · 粤Z两地牌 · 香港 · 澳门 ｜ WebCam 实时测试",
            "app.docTitle": "四地车牌识别 - WebCam 测试",

            "btn.startCam": "启动 WebCam",
            "btn.stop": "停止",
            "btn.startDetect": "开始识别",
            "btn.pauseDetect": "暂停识别",
            "btn.upload": "上传图片",

            "panel.status": "连接状态",
            "panel.stats": "性能统计",
            "panel.results": "识别结果",
            "panel.upload": "上传测试",

            "stat.fps": "FPS",
            "stat.latency": "延迟 (ms)",
            "stat.plates": "检测车牌",
            "stat.frames": "处理帧数",

            "status.disconnected": "未连接",
            "status.connected": "已连接",
            "status.camOn": "WebCam 已启动",
            "status.closed": "已断开",
            "status.connecting": "连接中…",
            "status.modelNotReady": "模型未载入",

            "result.empty": "暂无识别结果",
            "result.recognize": "识别",
            "result.detect": "检测",
            "result.double": "双行",
            "result.corrected": "已纠正",
            "result.invalid": "格式不符",

            "upload.hint": "点击上传车牌图片",
            "upload.orDrag": "或拖拽到此处",

            "region.mainland": "内地",
            "region.gangao": "粤Z",
            "region.hk": "香港",
            "region.mo": "澳门",
            "region.unknown": "未知",

            "err.camDenied": "无法访问 WebCam",
            "err.decode": "无法解码图片"
        },

        // ==================== English ====================
        "en": {
            "app.title": "GBA Plate Recognition",
            "app.subtitle": "Mainland · Guangdong-Z Two-Plate · Hong Kong · Macau ｜ Live WebCam Test",
            "app.docTitle": "GBA Plate Recognition - WebCam Test",

            "btn.startCam": "Start WebCam",
            "btn.stop": "Stop",
            "btn.startDetect": "Start Detection",
            "btn.pauseDetect": "Pause Detection",
            "btn.upload": "Upload Image",

            "panel.status": "Connection",
            "panel.stats": "Performance",
            "panel.results": "Results",
            "panel.upload": "Upload Test",

            "stat.fps": "FPS",
            "stat.latency": "Latency (ms)",
            "stat.plates": "Plates",
            "stat.frames": "Frames",

            "status.disconnected": "Disconnected",
            "status.connected": "Connected",
            "status.camOn": "WebCam active",
            "status.closed": "Disconnected",
            "status.connecting": "Connecting…",
            "status.modelNotReady": "Model not loaded",

            "result.empty": "No results yet",
            "result.recognize": "OCR",
            "result.detect": "Detect",
            "result.double": "2-line",
            "result.corrected": "corrected",
            "result.invalid": "invalid format",

            "upload.hint": "Click to upload a plate image",
            "upload.orDrag": "or drag it here",

            "region.mainland": "Mainland",
            "region.gangao": "GD-Z",
            "region.hk": "Hong Kong",
            "region.mo": "Macau",
            "region.unknown": "Unknown",

            "err.camDenied": "Cannot access WebCam",
            "err.decode": "Cannot decode image"
        }
    };

    var listeners = [];
    var current = DEFAULT_LANG;

    function detectInitial() {
        try {
            var saved = localStorage.getItem(STORAGE_KEY);
            if (saved && MESSAGES[saved]) return saved;
        } catch (e) {
            /* localStorage 不可用時忽略 */
        }
        // 依瀏覽器語言猜測
        var nav = (navigator.language || "zh-TW").toLowerCase();
        if (nav.indexOf("zh-cn") === 0 || nav.indexOf("zh-hans") === 0) return "zh-CN";
        if (nav.indexOf("zh") === 0) return "zh-TW";
        return "zh-TW";  // 非中文語系仍預設繁體（符合項目定位）
    }

    function t(key, fallback) {
        var pack = MESSAGES[current] || MESSAGES[DEFAULT_LANG];
        if (pack && Object.prototype.hasOwnProperty.call(pack, key)) return pack[key];
        var base = MESSAGES[DEFAULT_LANG];
        if (base && Object.prototype.hasOwnProperty.call(base, key)) return base[key];
        return fallback !== undefined ? fallback : key;
    }

    function applyTo(root) {
        root = root || document;

        // 一般文字
        root.querySelectorAll("[data-i18n]").forEach(function (el) {
            var key = el.getAttribute("data-i18n");
            el.textContent = t(key, el.textContent);
        });

        // placeholder
        root.querySelectorAll("[data-i18n-placeholder]").forEach(function (el) {
            el.setAttribute("placeholder", t(el.getAttribute("data-i18n-placeholder")));
        });

        // title 屬性
        root.querySelectorAll("[data-i18n-title]").forEach(function (el) {
            el.setAttribute("title", t(el.getAttribute("data-i18n-title")));
        });

        // 頁面標題
        document.title = t("app.docTitle");

        // html lang
        document.documentElement.setAttribute(
            "lang",
            current === "en" ? "en" : current
        );

        // 更新語言切換按鈕的選中狀態
        root.querySelectorAll("[data-lang-btn]").forEach(function (btn) {
            var on = btn.getAttribute("data-lang-btn") === current;
            btn.classList.toggle("active", on);
            btn.setAttribute("aria-pressed", on ? "true" : "false");
        });
    }

    function setLang(lang, silent) {
        if (!MESSAGES[lang]) return current;
        current = lang;
        try {
            localStorage.setItem(STORAGE_KEY, lang);
        } catch (e) {
            /* 忽略 */
        }
        applyTo(document);
        if (!silent) {
            listeners.forEach(function (fn) {
                try {
                    fn(lang);
                } catch (e) {
                    console.error("i18n listener error", e);
                }
            });
        }
        return current;
    }

    function onChange(fn) {
        if (typeof fn === "function") listeners.push(fn);
    }

    function init() {
        current = detectInitial();
        // 等 DOM ready 後套用
        if (document.readyState === "loading") {
            document.addEventListener("DOMContentLoaded", function () {
                applyTo(document);
            });
        } else {
            applyTo(document);
        }
    }

    global.I18N = {
        t: t,
        setLang: setLang,
        onChange: onChange,
        apply: applyTo,
        init: init,
        get lang() {
            return current;
        },
        LANGS: LANGS
    };

    init();
})(window);
