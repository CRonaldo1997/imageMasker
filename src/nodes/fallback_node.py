"""兜底节点：超过最大重试次数时强制结束。

保留当前脱敏结果，标记失败原因，交由下游自行判断是否可用。
"""
from __future__ import annotations

import logging

from src.graph_state import RedactionState

logger = logging.getLogger(__name__)


def fallback_node(state: RedactionState) -> RedactionState:
    max_retries = state.get("max_retries", 3)
    logger.warning("质检重试 %d 次仍未通过，保留当前脱敏结果并强制结束", max_retries)
    return {
        "status": "max_retry",
        "feedback": (
            f"质检重试 {max_retries} 次仍未通过，"
            f"已保留当前脱敏结果并强制结束。"
        ),
    }
