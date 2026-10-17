"""RedactionState：LangGraph 全节点共享的状态定义。

节点只能返回需要更新的字段，不能直接修改整个状态。
error 为编程说明书之外的补充字段：节点失败时用于落地明确的错误状态，
否则“失败时返回明确错误状态”这一要求无处安放。
"""
from __future__ import annotations

from typing import Optional

from typing_extensions import TypedDict

from src.schemas.ocr_result import OcrResult
from src.schemas.qa_report import QaReport
from src.schemas.redaction_result import RedactionResult
from src.schemas.vlm_analysis import VlmAnalysis


class RedactionState(TypedDict, total=False):
    image_path: str
    original_image: bytes
    rotation: int
    output_path: Optional[str]
    ocr_result: Optional[OcrResult]
    vlm_analysis: Optional[VlmAnalysis]
    redaction_result: Optional[RedactionResult]
    qa_report: Optional[QaReport]
    qa_enabled: bool
    retry_count: int
    max_retries: int
    feedback: Optional[str]
    status: str
    error: Optional[str]
