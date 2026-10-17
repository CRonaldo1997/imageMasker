"""OCR 结果的结构化定义。"""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class OcrTextBlock(BaseModel):
    """单个文字块：文本 + 坐标 + 置信度。

    bbox 是四边形外接矩形 [x1, y1, x2, y2]（向后兼容的粗略坐标）；
    points 是检测器输出的四边形顶点 [[x, y] x4]（精确边界，
    文字倾斜/竖排时比外接矩形准）；
    angle 是文字行方向角（度，[-90, 90)）：0 为水平，±90 为竖排，
    由四边形长边方向计算，驱动脱敏时的文字旋转渲染。
    """

    block_id: str = Field(description="块唯一 ID，如 b0")
    text: str = Field(description="识别出的文本")
    bbox: List[int] = Field(
        default_factory=list, description="外接矩形 [x1, y1, x2, y2]"
    )
    points: List[List[int]] = Field(
        default_factory=list, description="四边形顶点 [[x, y] x4]，无则退化用 bbox"
    )
    angle: float = Field(default=0.0, description="文字行方向角（度），0 水平，±90 竖排")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class OcrResult(BaseModel):
    """OCR 工具输出。

    开启方向矫正时，blocks/width/height 均为**转正图坐标系**：
    rotation 记录原图需**逆时针**旋转的度数（0/90/180/270），
    aligned_image 是转正图字节（供下游节点替换原图使用，
    最后由 finalize 节点统一逆旋转回原方向）。语义见
    docs/ubuntu-runbook.md 与 tests/test_orientation.py。
    """

    engine: str = Field(description="OCR 引擎名称")
    width: int = 0
    height: int = 0
    blocks: List[OcrTextBlock] = Field(default_factory=list)
    image_path: Optional[str] = None
    rotation: int = Field(
        default=0, description="原图需逆时针旋转的度数（0/90/180/270）"
    )
    aligned_image: Optional[bytes] = Field(
        default=None, description="转正图字节（未矫正时为 None）"
    )
