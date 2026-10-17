"""脱敏节点：根据 VLM 输出的敏感区域，在原图上执行遮罩/打码。

约束：保留原图；方式可配置；不修改 OCR 坐标；输出记录被脱敏的块。
"""
from __future__ import annotations

import logging

from src.graph_state import RedactionState
from src.nodes.common import already_failed, fail, require
from src.tools import get_redaction_tool

logger = logging.getLogger(__name__)


def redact_node(state: RedactionState) -> RedactionState:
    if already_failed(state):
        return {}
    missing = require(state, "redact", "vlm_analysis", "original_image")
    if missing:
        return missing

    try:
        result = get_redaction_tool().apply(
            image=state["original_image"],
            vlm_analysis=state["vlm_analysis"],
            output_path=state.get("output_path"),
        )
    except Exception as exc:
        return fail("redact", f"{type(exc).__name__}: {exc}")

    return {"redaction_result": result}
