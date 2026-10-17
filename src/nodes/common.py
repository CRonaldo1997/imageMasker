"""节点公共辅助：失败状态落地与输入守卫。

约定：
- 节点捕获工具异常后返回 {"status": "failed", "error": ...}，不裸抛、
  不吞异常——错误信息完整落在状态里，调用方可见；
- 任一节点置 failed 后，后续节点检测到即短路返回 {}，图自然走到 END；
- 节点只返回需要更新的字段，不修改整个 state。
"""
from __future__ import annotations

import logging
from typing import Dict, Optional

from src.graph_state import RedactionState

logger = logging.getLogger(__name__)


def fail(node: str, message: str) -> Dict:
    """生成明确的失败状态更新。"""
    logger.error("[%s] %s", node, message)
    return {"status": "failed", "error": f"[{node}] {message}"}


def already_failed(state: RedactionState) -> bool:
    """上游已失败时，本节点短路。"""
    return state.get("status") == "failed"


def require(state: RedactionState, node: str, *keys: str) -> Optional[Dict]:
    """检查必需字段，缺失则返回失败状态更新；齐全则返回 None。"""
    for key in keys:
        if state.get(key) is None:
            return fail(node, f"缺少必需状态字段 {key}")
    return None
