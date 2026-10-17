"""VLM 判断结果的结构化定义。"""
from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class RedactionMethod(str, Enum):
    """脱敏方式。white_mask 为默认：白底重绘假数据，信息论上彻底移除。"""

    WHITE_MASK = "white_mask"
    BLUR = "blur"
    MOSAIC = "mosaic"


class SensitiveKind(str, Enum):
    """敏感信息类型（当前仅支持这 5 类）。"""

    NAME = "name"
    MOBILE = "mobile"
    ID_CARD = "id_card"
    MEDICAL_CARD = "medical_card"
    ADDRESS = "address"


class SensitiveBlock(BaseModel):
    """一个被判为敏感的文字块。

    bbox 必须由 VLM 从 OCR 结果拷贝，供脱敏节点定位——
    脱敏节点只拿 vlm_analysis，拿不到 ocr_result。
    value 是块文本中敏感值的精确子串（如块文本"姓名：张三"的
    value 为"张三"），脱敏时只替换 value、保留标签等业务上下文。
    """

    block_id: str
    text: str = ""
    bbox: List[int] = Field(default_factory=list)
    points: List[List[int]] = Field(
        default_factory=list, description="四边形顶点 [[x, y] x4]，从 OCR 拷贝"
    )
    angle: float = Field(default=0.0, description="文字行方向角（度），从 OCR 拷贝")
    reason: str = ""
    method: RedactionMethod = RedactionMethod.WHITE_MASK
    source: str = "vlm"  # vlm / rule
    value: str = Field(default="", description="块文本中的敏感值原文，必须是 text 的子串")
    kind: str = Field(default="", description="name/mobile/id_card/medical_card/address")

    @field_validator("method", mode="before")
    @classmethod
    def _coerce_method(cls, value):
        """LLM 可能给出非法/空的脱敏方式，统一兜底为 white_mask。

        枚举字段默认会因非法值直接抛 ValidationError，导致整张图处理
        中断；这里前置兜底，保持管线容错。
        """
        if isinstance(value, RedactionMethod):
            return value
        try:
            return RedactionMethod(value)
        except (ValueError, TypeError):
            return RedactionMethod.WHITE_MASK


class VlmAnalysis(BaseModel):
    """VLM 节点输出：哪些块敏感、怎么脱、为什么。"""

    sensitive_blocks: List[SensitiveBlock] = Field(default_factory=list)
    model: str = ""
    summary: Optional[str] = None
