"""LangGraph 编排：确定性流水线，LLM 只保留在 VLM 和 QA 节点内部。

流程图：
    START → ocr → vlm → redact → qa → route_after_qa
                                        ├─ end      → finalize → END
                                        ├─ retry_vlm → vlm
                                        └─ fallback  → fallback → finalize → END

开启方向矫正时，整条管线（OCR/VLM/脱敏/质检）都在转正图上进行，
finalize 是唯一逆旋转的地方：把脱敏图转回原方向后输出。

不使用 ReAct：流程顺序、回退次数由图控制，不由模型自主调度。
recursion_limit 按 max_retries 计算而非硬编码——调大重试次数不会撞顶。
"""
from __future__ import annotations

import logging

from langgraph.graph import END, START, StateGraph

from src.graph_state import RedactionState
from src.nodes import (
    fallback_node,
    finalize_node,
    ocr_node,
    qa_node,
    redact_node,
    vlm_node,
)

logger = logging.getLogger(__name__)


def route_after_qa(state: RedactionState) -> str:
    """质检后的条件路由。每个分支都有出口，不会死循环。"""
    if not state.get("qa_enabled", True):
        return "end"

    report = state.get("qa_report")
    if report is None:  # 质检未执行（如上游失败），直接结束
        return "end"

    if report.status == "passed":
        return "end"

    if state.get("retry_count", 0) >= state.get("max_retries", 3):
        return "fallback"

    return "retry_vlm"


def compute_recursion_limit(max_retries: int) -> int:
    """ocr(1) + (max_retries+1) 轮 (vlm+redact+qa)(3) + fallback(1) + finalize(1) + 余量。"""
    return 3 * (max_retries + 1) + 6


def build_graph():
    builder = StateGraph(RedactionState)

    builder.add_node("ocr", ocr_node)
    builder.add_node("vlm", vlm_node)
    builder.add_node("redact", redact_node)
    builder.add_node("qa", qa_node)
    builder.add_node("fallback", fallback_node)
    builder.add_node("finalize", finalize_node)

    builder.add_edge(START, "ocr")
    builder.add_edge("ocr", "vlm")
    builder.add_edge("vlm", "redact")
    builder.add_edge("redact", "qa")

    builder.add_conditional_edges(
        "qa",
        route_after_qa,
        {
            "end": "finalize",
            "retry_vlm": "vlm",
            "fallback": "fallback",
        },
    )
    builder.add_edge("fallback", "finalize")
    builder.add_edge("finalize", END)

    return builder.compile()
