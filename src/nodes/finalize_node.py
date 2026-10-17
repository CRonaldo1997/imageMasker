"""finalize 节点：出口前把脱敏图逆旋转回原方向。

整条管线（OCR / VLM / 脱敏 / 质检）都在方向矫正后的转正图上进行，
本节点是唯一逆旋转的地方：把脱敏图按 rotation 转回原方向并落盘，
保证输出与原图方向一致。未矫正（rotation=0）时无操作。
"""
from __future__ import annotations

import logging
from io import BytesIO
from pathlib import Path
from typing import Optional

from PIL import Image

from src.config import PROJECT_ROOT
from src.graph_state import RedactionState
from src.schemas.redaction_result import RedactionResult
from src.tools.redaction_tool import encode_image, format_for_suffix

logger = logging.getLogger(__name__)


def finalize_node(state: RedactionState) -> RedactionState:
    result: Optional[RedactionResult] = state.get("redaction_result")
    rotation = state.get("rotation", 0)
    if result is None or not rotation:
        return {}

    img = Image.open(BytesIO(result.redacted_image)).convert("RGB")
    # 转正是 rotate(+rotation)（逆时针），还原即逆操作 rotate(-rotation)
    restored = img.rotate(-rotation, expand=True)

    # 按目标扩展名决定格式（.jpg→JPEG、.tif→TIFF、.webp→WEBP …）
    fmt = format_for_suffix(Path(result.output_path).suffix) if result.output_path else "PNG"
    result.redacted_image = encode_image(restored, fmt)
    result.width, result.height = restored.size

    # 覆盖写回原输出路径（转正版本已被替换为原方向版本）
    if result.output_path:
        path = Path(result.output_path)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(result.redacted_image)

    logger.info(
        "finalize：脱敏图已逆旋转 %d° 回原方向（%dx%d）",
        rotation, result.width, result.height,
    )
    return {"redaction_result": result}
