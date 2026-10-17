"""脱敏结果的结构化定义。"""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class RedactedBlock(BaseModel):
    """一条脱敏命中记录（审计用，不落敏感原文到日志）。"""

    block_id: str
    text: str = ""
    bbox: List[int] = Field(default_factory=list)
    points: List[List[int]] = Field(default_factory=list)
    angle: float = Field(default=0.0)
    method: str = "white_mask"
    source: str = "vlm"  # vlm / rule / qr


class RedactionResult(BaseModel):
    """脱敏后的图片及命中记录。原图始终保留，本工具不修改原图。"""

    redacted_image: bytes = Field(description="脱敏后图片字节（PNG）")
    output_path: Optional[str] = Field(default=None, description="落盘路径")
    image_format: str = "PNG"
    width: int = 0
    height: int = 0
    redacted_blocks: List[RedactedBlock] = Field(default_factory=list)
