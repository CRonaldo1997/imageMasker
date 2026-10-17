"""VLM 质检结论的结构化定义。"""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class QaVerdict(BaseModel):
    """VLM 质检输出：当前脱敏图是否合格、原因、修正意见。

    revised_block_ids：VLM 复判后认为应当脱敏的块 ID 列表，
    质检不通过时回退 VLM 节点作为修正依据。
    """

    status: str = Field(description="passed / failed")
    reason: str = ""
    feedback: Optional[str] = Field(default=None, description="未通过时给 VLM 的修正意见")
    residual_block_ids: List[str] = Field(
        default_factory=list, description="脱敏后仍可识别的敏感块"
    )
    damaged_block_ids: List[str] = Field(
        default_factory=list, description="被误遮挡的非敏感块"
    )
    revised_block_ids: List[str] = Field(
        default_factory=list, description="VLM 复判后应脱敏的块 ID"
    )
