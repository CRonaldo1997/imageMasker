"""工具门面：懒加载单例。

节点通过 get_xxx_tool() 取工具；测试可向 _cache 注入假工具，
不影响生产路径。VLM 客户端在首次取用时才构造，避免 import 期
因缺少 API Key 直接失败。

批量模式下多个 case 线程会并发首次取用工具：用锁保护「检查-创建」
整个过程，否则可能创建出两份实例（两份 OCR 引擎 = 双倍内存，
两份 VLM 客户端 = 双倍连接与降级状态不一致）。
"""
from __future__ import annotations

import threading
from typing import Dict

from src.config import Settings, get_settings
from src.tools.ocr_tool import OcrTool
from src.tools.qa_tool import QaTool
from src.tools.redaction_tool import RedactionTool
from src.tools.vlm_tool import VlmTool

_cache: Dict[str, object] = {}
# 可重入锁：get_qa_tool 持有锁时会再调 get_ocr_tool / get_vlm_tool，
# 用 RLock 避免自锁。
_cache_lock = threading.RLock()


def _settings() -> Settings:
    return get_settings()


def get_ocr_tool() -> OcrTool:
    with _cache_lock:
        if "ocr" not in _cache:
            _cache["ocr"] = OcrTool(_settings())
        return _cache["ocr"]  # type: ignore[return-value]


def get_vlm_tool() -> VlmTool:
    with _cache_lock:
        if "vlm" not in _cache:
            _cache["vlm"] = VlmTool(_settings())
        return _cache["vlm"]  # type: ignore[return-value]


def get_redaction_tool() -> RedactionTool:
    with _cache_lock:
        if "redact" not in _cache:
            _cache["redact"] = RedactionTool(_settings())
        return _cache["redact"]  # type: ignore[return-value]


def get_qa_tool() -> QaTool:
    with _cache_lock:
        if "qa" not in _cache:
            # 质检复用同一个 OCR 工具（重 OCR 同引擎同参数）与同一个
            # VLM 工具（语义质检复用模型客户端）
            _cache["qa"] = QaTool(_settings(), get_ocr_tool(), get_vlm_tool())
        return _cache["qa"]  # type: ignore[return-value]


def reset_tools() -> None:
    """清空工具缓存（测试用）。"""
    with _cache_lock:
        _cache.clear()


__all__ = [
    "get_ocr_tool",
    "get_vlm_tool",
    "get_redaction_tool",
    "get_qa_tool",
    "reset_tools",
    "OcrTool",
    "VlmTool",
    "RedactionTool",
    "QaTool",
]
