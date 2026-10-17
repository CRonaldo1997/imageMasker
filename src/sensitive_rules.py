"""敏感信息规则通道：用正则做确定性兜底。

规则与 VLM 判断取并集（宁可多脱，不可漏）。

只保留**格式唯一性强**的类型：
- 身份证号：18 位 + 校验位，不会与票据号/流水号混淆；
- 手机号：1[3-9] 开头 11 位，特征强。

刻意不收固定电话：0+区号+7/8 位的数字串与票据号码/流水号
同构（如票据号 0137126549 会被误判为固话），误伤率高，
交由 VLM 结合上下文判断。姓名/医保卡号/地址同理无稳定正则，
完全靠 VLM。
"""
from __future__ import annotations

import re
from typing import Dict, List, Pattern

from src.schemas.ocr_result import OcrResult
from src.schemas.vlm_analysis import RedactionMethod, SensitiveBlock

# 模式名 → (正则, kind)
SENSITIVE_PATTERNS: Dict[str, tuple[Pattern[str], str]] = {
    "身份证号": (re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)"), "id_card"),
    "手机号": (re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)"), "mobile"),
}


def scan_text(text: str) -> List[str]:
    """返回文本中命中的敏感模式名列表（质检兜底扫描用）。"""
    if not text:
        return []
    return [name for name, (pattern, _) in SENSITIVE_PATTERNS.items() if pattern.search(text)]


def match_blocks(
    ocr_result: OcrResult, method: RedactionMethod = RedactionMethod.WHITE_MASK
) -> List[SensitiveBlock]:
    """对 OCR 文字块做正则匹配，产出规则通道的敏感块（带 kind）。"""
    blocks: List[SensitiveBlock] = []
    for block in ocr_result.blocks:
        for name, (pattern, kind) in SENSITIVE_PATTERNS.items():
            matched = pattern.search(block.text)
            if matched:
                blocks.append(
                    SensitiveBlock(
                        block_id=block.block_id,
                        text=block.text,
                        bbox=list(block.bbox),
                        points=[list(p) for p in block.points],
                        angle=block.angle,
                        reason=f"规则命中：{name}",
                        method=method,
                        source="rule",
                        value=matched.group(0),  # 只替换命中的号码本身，保留标签
                        kind=kind,
                    )
                )
                break  # 一个块只记一次
    return blocks


def merge_sensitive(
    vlm_blocks: List[SensitiveBlock], rule_blocks: List[SensitiveBlock]
) -> List[SensitiveBlock]:
    """VLM 与规则通道取并集，按 block_id 去重，VLM 判断优先保留。"""
    merged: "dict[str, SensitiveBlock]" = {}
    for block in vlm_blocks:
        merged[block.block_id] = block
    for block in rule_blocks:
        if block.block_id not in merged:
            merged[block.block_id] = block
    return list(merged.values())
