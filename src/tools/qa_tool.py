"""质检工具：确定性预检 + VLM 语义质检。

两层判定：
1. 确定性预检（本地、零成本）：重 OCR 残留 + 正则兜底（剔除假数据）
   + 几何误伤。发现硬残留直接判 failed，不消耗 VLM 调用；
2. VLM 语义质检：输入原图 + 当前 mask 图 + OCR/敏感块 JSON +
   历史质检意见，输出结构化结论（残留/误伤/修正块列表）。
   负责抓确定性通道查不了的语义漏判（如漏掉的姓名）。
"""
from __future__ import annotations

import logging
from typing import List, Optional

from src.config import Settings
from src.schemas.ocr_result import OcrResult
from src.schemas.qa_report import QaReport
from src.schemas.redaction_result import RedactionResult
from src.schemas.vlm_analysis import VlmAnalysis
from src.sensitive_rules import SENSITIVE_PATTERNS
from src.tools.ocr_tool import OcrTool
from src.tools.vlm_tool import VlmTool

logger = logging.getLogger(__name__)

_PUNCT = "，。、；：？！（）()[]【】{}<>《》“”\"'’‘-—_/|.:：,;!?~`+*#@$%^& \t\r\n"


def normalize(text: str) -> str:
    """去空白与标点，降低 OCR 噪声对匹配的干扰。"""
    return "".join(ch for ch in text if ch not in _PUNCT)


class QaTool:
    """质检工具。重 OCR 复用首轮 OCR 工具（同引擎同参数）；
    VLM 质检复用 VLM 工具（同一模型客户端）。"""

    def __init__(
        self,
        settings: Settings,
        ocr_tool: OcrTool,
        vlm_tool: Optional[VlmTool] = None,
    ) -> None:
        self._settings = settings
        self._ocr = ocr_tool
        self._vlm = vlm_tool

    def verify(
        self,
        original_image: bytes,
        redacted_image: bytes,
        vlm_analysis: VlmAnalysis,
        ocr_result: Optional[OcrResult] = None,
        redaction_result: Optional[RedactionResult] = None,
        feedback: Optional[str] = None,
    ) -> QaReport:
        """质检脱敏结果，输出结构化报告。"""
        re_ocr = self._ocr.extract(image_bytes=redacted_image)
        re_texts = [b.text for b in re_ocr.blocks]
        drawn_texts = (
            [rb.text for rb in redaction_result.redacted_blocks if rb.text]
            if redaction_result
            else []
        )

        # ---- 第一层：确定性预检 ----
        residual_texts = self._find_residuals(vlm_analysis, re_texts)
        residual_patterns = self._find_patterns(re_texts, drawn_texts)
        damaged = self._find_collateral(vlm_analysis, ocr_result, redaction_result)
        hints = {
            "re_ocr_blocks": re_texts,
            "residual_block_ids": residual_texts,
            "pattern_hits": residual_patterns,
            "damaged_block_ids": damaged,
        }

        if residual_texts or residual_patterns:
            # 硬残留：确定性证据充分，直接判失败
            return self._report(
                status="failed",
                reason=self._build_reason(residual_texts, residual_patterns, damaged),
                feedback=self._build_feedback(residual_texts, residual_patterns, damaged),
                residual_texts=residual_texts,
                residual_patterns=residual_patterns,
                damaged_texts=damaged,
                details=hints,
            )

        # ---- 第二层：VLM 语义质检 ----
        if self._vlm is None:
            return self._report(
                status="failed" if damaged else "passed",
                reason=self._build_reason(residual_texts, residual_patterns, damaged)
                or "确定性预检未检出残留与误伤（未配置 VLM 质检）",
                feedback=self._build_feedback(residual_texts, residual_patterns, damaged),
                residual_texts=residual_texts,
                residual_patterns=residual_patterns,
                damaged_texts=damaged,
                details=hints,
            )

        try:
            verdict = self._vlm.qa_verify(
                original_image=original_image,
                redacted_image=redacted_image,
                vlm_analysis=vlm_analysis,
                ocr_result=ocr_result,
                feedback=feedback,
                deterministic_hints=hints,
                redaction_result=redaction_result,
            )
        except Exception as exc:
            logger.error("VLM 质检失败：%s", exc)
            return self._report(
                status="failed",
                reason=f"VLM 质检调用失败：{exc}",
                feedback="质检调用失败，请检查 VLM 配置后重试",
                residual_texts=residual_texts,
                residual_patterns=residual_patterns,
                damaged_texts=damaged,
                details=hints,
            )

        return self._report(
            status="failed" if verdict.status != "passed" else "passed",
            reason=verdict.reason,
            feedback=verdict.feedback,
            residual_texts=verdict.residual_block_ids,
            residual_patterns=residual_patterns,
            damaged_texts=verdict.damaged_block_ids or damaged,
            details={**hints, "revised_block_ids": verdict.revised_block_ids},
            revised_block_ids=verdict.revised_block_ids,
        )

    # ------------------------------------------------------------------ 第一层检测

    def _find_residuals(
        self, vlm_analysis: VlmAnalysis, re_texts: List[str]
    ) -> List[str]:
        """敏感值在重 OCR 结果中仍可被识别 → 残留。

        判定规则（刻意不用相似度：票据号/医保号等长数字串常共享
        日期前缀，相似度会互相误报）：
        1. 重 OCR 文本完整包含目标文本；
        2. 重 OCR 文本是目标文本的子串（遮罩不彻底导致部分读出）；
        3. 目标中的长数字段完整出现在重 OCR 全文中。
        """
        residuals: List[str] = []
        min_len = self._settings.qa_min_match_len
        min_digits = self._settings.qa_min_digits
        full_text = "".join(normalize(t) for t in re_texts)
        for sb in vlm_analysis.sensitive_blocks:
            target = normalize(sb.value or sb.text)
            if not target:
                continue
            hit = False
            for text in re_texts:
                candidate = normalize(text)
                if not candidate:
                    continue
                if target in candidate:
                    hit = True
                    break
                if len(candidate) >= min(2, min_len) and candidate in target:
                    hit = True
                    break
            if not hit:
                for digits in _digit_runs(target, min_digits):
                    if digits in full_text:
                        hit = True
                        break
            if hit:
                residuals.append(sb.block_id)
        return residuals

    def _find_patterns(
        self, re_texts: List[str], drawn_texts: Optional[List[str]] = None
    ) -> List[str]:
        """正则兜底：脱敏图中出现固定格式敏感号模式。

        假数据（等长合法格式）不算残留：命中值若与任一已绘制
        假数据相同或高度相似，则跳过。
        """
        full = " ".join(re_texts)
        drawn = drawn_texts or []
        hits: List[str] = []
        for name, (pattern, _kind) in SENSITIVE_PATTERNS.items():
            for match in pattern.finditer(full):
                value = match.group(0)
                if self._looks_like_drawn_fake(value, drawn):
                    continue
                hits.append(name)
                break  # 每个模式只报一次
        return hits

    @staticmethod
    def _looks_like_drawn_fake(value: str, drawn: List[str]) -> bool:
        for text in drawn:
            if value in text:
                return True
            if _similarity(value, text) >= 0.6:
                return True
        return False

    @staticmethod
    def _find_collateral(
        vlm_analysis: VlmAnalysis,
        ocr_result: Optional[OcrResult],
        redaction_result: Optional[RedactionResult],
    ) -> List[str]:
        """几何误伤：非敏感块的中心点被任一遮罩区域（四边形/矩形）覆盖。

        用中心点而非外接矩形重叠比：竖排图中相邻列的外接矩形容易
        “重叠”，但实际遮罩是多边形、不会盖到邻居；中心点被盖才
        说明真的误伤。
        """
        if ocr_result is None or redaction_result is None:
            return []
        sensitive_ids = {sb.block_id for sb in vlm_analysis.sensitive_blocks}
        damaged: List[str] = []
        for block in ocr_result.blocks:
            if block.block_id in sensitive_ids:
                continue
            cx = (block.bbox[0] + block.bbox[2]) / 2
            cy = (block.bbox[1] + block.bbox[3]) / 2
            for rb in redaction_result.redacted_blocks:
                if _point_covered(cx, cy, rb.points or _bbox_to_points(rb.bbox)):
                    damaged.append(block.block_id)
                    break
        return damaged

    # ------------------------------------------------------------------ 报告

    @staticmethod
    def _build_reason(
        residual: List[str], patterns: List[str], damaged: List[str]
    ) -> str:
        parts = []
        if residual:
            parts.append(f"残留敏感块：{residual}")
        if patterns:
            parts.append(f"正则命中：{patterns}")
        if damaged:
            parts.append(f"误伤非敏感块：{damaged}")
        return "；".join(parts)

    @staticmethod
    def _build_feedback(
        residual: List[str], patterns: List[str], damaged: List[str]
    ) -> Optional[str]:
        """给 VLM 的修正意见。只引用块 ID，不堆敏感原文。"""
        actions = []
        if residual:
            actions.append(
                f"块 {residual} 脱敏后仍可识别，请将其纳入敏感列表并采用足够强的脱敏方式"
            )
        if patterns:
            actions.append(
                f"脱敏图中仍检出敏感格式（{patterns}），请检查对应文字块是否遗漏"
            )
        if damaged:
            actions.append(f"块 {damaged} 是非敏感内容却被遮挡，请移出敏感列表")
        return "质检未通过：" + "；".join(actions) + "。" if actions else None

    @staticmethod
    def _report(
        status: str,
        reason: str,
        feedback: Optional[str],
        residual_texts: List[str],
        residual_patterns: List[str],
        damaged_texts: List[str],
        details: dict,
        revised_block_ids: Optional[List[str]] = None,
    ) -> QaReport:
        return QaReport(
            status=status,
            reason=reason,
            details=details,
            feedback=feedback if status != "passed" else None,
            residual_texts=residual_texts,
            residual_patterns=residual_patterns,
            damaged_texts=damaged_texts,
            revised_block_ids=revised_block_ids or [],
        )


# ---------------------------------------------------------------- 辅助


def _digit_runs(text: str, min_len: int) -> List[str]:
    import re

    return [m for m in re.findall(r"\d+", text) if len(m) >= min_len]


def _similarity(a: str, b: str) -> float:
    from difflib import SequenceMatcher

    return SequenceMatcher(None, a, b).ratio()


def _bbox_to_points(bbox: List[int]) -> List[List[int]]:
    """外接矩形转 4 顶点。"""
    if len(bbox) < 4:
        return []
    x1, y1, x2, y2 = bbox
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def _point_covered(x: float, y: float, points: List[List[int]]) -> bool:
    """射线法判断点是否在多边形内。"""
    if len(points) < 3:
        return False
    inside = False
    n = len(points)
    j = n - 1
    for i in range(n):
        xi, yi = points[i]
        xj, yj = points[j]
        if (yi > y) != (yj > y):
            x_cross = (xj - xi) * (y - yi) / (yj - yi) + xi
            if x < x_cross:
                inside = not inside
        j = i
    return inside
