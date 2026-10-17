"""OCR 节点：提取文字块和坐标。

职责边界：只识别文本和坐标；不做敏感判断；不修改原图；
失败时落地为明确的失败状态，不吞异常。
"""
from __future__ import annotations

import logging
from pathlib import Path

from src.graph_state import RedactionState
from src.nodes.common import already_failed, fail
from src.tools import get_ocr_tool

logger = logging.getLogger(__name__)


def ocr_node(state: RedactionState) -> RedactionState:
    if already_failed(state):
        return {}
    if not state.get("image_path") and not state.get("original_image"):
        return fail("ocr", "image_path 与 original_image 不能同时为空")

    try:
        result = get_ocr_tool().extract(
            image_path=state.get("image_path"),
            image_bytes=state.get("original_image"),
        )
    except Exception as exc:  # 工具异常 → 明确失败状态
        return fail("ocr", f"{type(exc).__name__}: {exc}")

    updates: RedactionState = {"ocr_result": result}
    # 入口只给了路径时，先补齐原图字节，供 VLM / 脱敏 / 质检使用
    if not state.get("original_image") and state.get("image_path"):
        updates["original_image"] = Path(state["image_path"]).read_bytes()
    # 方向矫正：后续 VLM/脱敏/质检全部在转正图上进行，
    # 原图字节替换为转正图，rotation 供 finalize 逆旋转还原
    if result.aligned_image:
        updates["original_image"] = result.aligned_image
        updates["rotation"] = result.rotation
    return updates
