"""VLM 节点：判断哪些文字块属于敏感信息。

首次执行 feedback 为空；回退执行时结合质检意见修正。
规则通道（身份证/手机号/银行卡/邮箱正则）与 VLM 结果取并集，
作为确定性兜底——固定格式敏感号宁可多脱，不可漏。
"""
from __future__ import annotations

import logging

from src.config import get_settings
from src.graph_state import RedactionState
from src.nodes.common import already_failed, fail, require
from src.schemas.vlm_analysis import RedactionMethod, VlmAnalysis
from src.sensitive_rules import match_blocks, merge_sensitive
from src.tools import get_vlm_tool
from src.tools.vlm_tool import StructuredOutputError

logger = logging.getLogger(__name__)


def vlm_node(state: RedactionState) -> RedactionState:
    if already_failed(state):
        return {}
    missing = require(state, "vlm", "ocr_result", "original_image")
    if missing:
        return missing

    feedback = state.get("feedback")
    try:
        analysis = get_vlm_tool().analyze(
            image=state["original_image"],
            ocr_result=state["ocr_result"],
            previous_redaction=state.get("redaction_result"),
            feedback=feedback,
        )
    except StructuredOutputError as exc:
        # LLM 多次重试仍无法产出合法结构：不让整图直接判死，退化为
        # 规则通道（身份证/手机号等固定格式仍会被脱敏），姓名等语义
        # 类敏感信息交由后续 VLM 质检发现并触发回退。
        logger.error("VLM 结构化输出连续失败，退化为规则通道：%s", exc)
        analysis = VlmAnalysis(sensitive_blocks=[], model="rule-only")
    except Exception as exc:
        return fail("vlm", f"{type(exc).__name__}: {exc}")

    # 规则通道并集（VLM 为主，正则兜底）
    rule_blocks = match_blocks(
        state["ocr_result"], RedactionMethod(get_settings().default_method)
    )
    analysis.sensitive_blocks = merge_sensitive(
        analysis.sensitive_blocks, rule_blocks
    )

    retry_count = state.get("retry_count", 0)
    if feedback is not None:
        retry_count += 1

    return {
        "vlm_analysis": analysis,
        "retry_count": retry_count,
        "feedback": None,
    }
