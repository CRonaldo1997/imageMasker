"""质检节点：重 OCR 闭环验证脱敏结果。

qa_enabled=False 时直接产出通过报告，不执行任何质检逻辑。
质检未通过时把结构化反馈写入 feedback，供回退 VLM 修正。
"""
from __future__ import annotations

import logging

from src.graph_state import RedactionState
from src.nodes.common import already_failed, fail, require
from src.schemas.qa_report import QaReport
from src.tools import get_qa_tool

logger = logging.getLogger(__name__)


def qa_node(state: RedactionState) -> RedactionState:
    if already_failed(state):
        return {}

    if not state.get("qa_enabled", True):
        return {
            "qa_report": QaReport(
                status="passed",
                reason="质检已禁用",
                details={},
                feedback=None,
            ),
            "status": "passed",
        }

    missing = require(state, "qa", "redaction_result", "vlm_analysis")
    if missing:
        return missing

    # 历史质检意见（上一轮 QA 写入 qa_report.feedback，供 VLM 质检参考）
    previous_report = state.get("qa_report")
    history_feedback = previous_report.feedback if previous_report else None

    try:
        report = get_qa_tool().verify(
            original_image=state["original_image"],
            redacted_image=state["redaction_result"].redacted_image,
            vlm_analysis=state["vlm_analysis"],
            ocr_result=state.get("ocr_result"),
            redaction_result=state["redaction_result"],
            feedback=history_feedback,
        )
    except Exception as exc:
        return fail("qa", f"{type(exc).__name__}: {exc}")

    # 未通过时必须给出非空 feedback：vlm_node 以 feedback 是否为空判断
    # 是否进入下一轮并自增 retry_count。若 VLM 未返回 feedback 而留空，
    # retry_count 永不增长，route_after_qa 会无限回退到 vlm，
    # 直到撞上 recursion_limit 才报错。
    if report.status == "passed":
        feedback = None
    else:
        feedback = report.feedback or (
            f"质检未通过（{report.reason or '未给出原因'}），"
            "请核对残留块与误伤块并修正敏感块列表。"
        )
        # VLM 质检复判出的完整敏感块集合，直接作为下一轮修正依据
        if report.revised_block_ids:
            feedback += (
                "复判建议脱敏块："
                + "、".join(report.revised_block_ids)
                + "。"
            )

    return {
        "qa_report": report,
        "status": "passed" if report.status == "passed" else "processing",
        "feedback": feedback,
    }
