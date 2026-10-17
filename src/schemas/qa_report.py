"""质检报告的结构化定义。"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class QaReport(BaseModel):
    """质检结论：status 取 passed / failed。"""

    status: str = Field(description="passed / failed")
    reason: str = ""
    details: Dict[str, Any] = Field(default_factory=dict)
    feedback: Optional[str] = Field(
        default=None, description="未通过时给 VLM 的修正意见；通过时为 None"
    )
    residual_texts: List[str] = Field(
        default_factory=list, description="脱敏图中仍可识别的敏感文本/块 ID"
    )
    residual_patterns: List[str] = Field(
        default_factory=list, description="正则兜底在脱敏图中命中的模式名"
    )
    damaged_texts: List[str] = Field(
        default_factory=list, description="被误遮挡的非敏感文本/块 ID"
    )
    revised_block_ids: List[str] = Field(
        default_factory=list, description="VLM 质检复判后应脱敏的块 ID（回退修正用）"
    )
