"""方向矫正测试：转正坐标系输出、finalize 逆旋转还原、降级路径。

方案：整条管线在转正图上进行，finalize 节点统一逆旋转回原方向。
RapidOrientation 返回值 = 需**逆时针**旋转的度数（用户实测确认，
OCR 输出无法反推转向——四边形角度被 OCR 栈归一化，已用特征点
像素测试固化该语义）。
"""
from __future__ import annotations

from io import BytesIO

import pytest
from PIL import Image

from src.config import Settings
from src.nodes.finalize_node import finalize_node
from src.tools.ocr_tool import OcrTool

SAMPLE = "data/folder1/caseA/sub/vertical.png"


@pytest.fixture
def correct_settings(tmp_path):
    return Settings(output_dir=str(tmp_path / "out"), ocr_orientation_correct=True)


# --------------------------------------------------------------------- OCR 转正输出


def test_extract_returns_aligned_coordinates(correct_settings):
    """开启矫正：OCR 在转正图上跑，坐标是转正图坐标系，附带转正图字节。"""
    tool = OcrTool(correct_settings)
    result = tool.extract(image_path=SAMPLE)

    # 转正图尺寸（原图 560x900 竖排 → 转正 900x560 横排）
    assert result.width == 900 and result.height == 560
    assert result.rotation == 90
    assert result.aligned_image is not None

    aligned = Image.open(BytesIO(result.aligned_image))
    assert aligned.size == (900, 560)

    # 转正后块更全（原图直跑 23 块）
    assert len(result.blocks) > 23
    # 敏感行齐全
    texts = [b.text for b in result.blocks]
    assert any("姓名" in t for t in texts)
    assert any("身份证" in t for t in texts)
    # 坐标在转正图范围内
    for b in result.blocks:
        assert 0 <= b.bbox[0] < b.bbox[2] <= 900
        assert 0 <= b.bbox[1] < b.bbox[3] <= 560


def test_alignment_uses_counterclockwise_rotation(correct_settings, tmp_path):
    """固化转向语义：RapidOrientation 返回 90 = 需逆时针旋转 90°。

    用特征点像素验证（OCR 输出分不清顺/逆时针，不能用识别效果判断）。
    逆时针 90° 时，原图左上角特征点应出现在转正图左下角。
    """
    import numpy as np

    img = Image.new("RGB", (400, 100), "white")
    img.putpixel((5, 5), (255, 0, 0))  # 左上角红点
    src = tmp_path / "feature.png"
    img.save(src)

    class MockOrientation:
        def __call__(self, arr):
            return "90", 0.01

    tool = OcrTool(correct_settings)
    tool._orientation_engine = MockOrientation()
    tool._np = np
    result = tool.extract(image_path=str(src))

    aligned = Image.open(BytesIO(result.aligned_image))
    assert aligned.size == (100, 400)  # 转正后竖图变横图
    # 逆时针 90°：(5,5) → (y, W-1-x) = (5, 394)
    assert aligned.getpixel((5, 394)) == (255, 0, 0)
    # 若是错误的顺时针方向，红点会出现在右上角 (94, 5)
    assert aligned.getpixel((94, 5)) != (255, 0, 0)


def test_extract_horizontal_image_not_rotated(correct_settings):
    """水平图：检测为 0°，不旋转，无 aligned_image。"""
    tool = OcrTool(correct_settings)
    result = tool.extract(image_path="data/sample_bill.png")
    assert result.rotation == 0
    assert result.aligned_image is None
    assert len(result.blocks) == 26


def test_extract_off_by_default(tmp_path):
    """显式关闭：原图直跑，转正图相关字段为空（显式传 False，避免被 .env 影响）。"""
    tool = OcrTool(
        Settings(output_dir=str(tmp_path / "out"), ocr_orientation_correct=False)
    )
    result = tool.extract(image_path=SAMPLE)
    assert result.rotation == 0
    assert result.aligned_image is None
    assert len(result.blocks) == 23


def test_orientation_engine_failure_degrades(correct_settings):
    """引擎不可用：降级原图直跑，不抛异常。"""
    tool = OcrTool(correct_settings)
    tool._orientation_failed = True
    result = tool.extract(image_path=SAMPLE)
    assert result.rotation == 0
    assert len(result.blocks) == 23


# --------------------------------------------------------------------- finalize 逆旋转


def _redaction_result(redacted_image: bytes, width: int, height: int):
    from src.schemas.redaction_result import RedactionResult

    return RedactionResult(
        redacted_image=redacted_image,
        output_path=None,
        width=width,
        height=height,
    )


def test_finalize_restores_orientation():
    """转正图（逆时针 90° 得到）经 finalize 逆操作后回到原方向，像素往返一致。"""
    original = Image.open(SAMPLE).convert("RGB")
    original_bytes = BytesIO()
    original.save(original_bytes, format="PNG")

    # 模拟管线：按修正后的语义转正（逆时针 90° = rotate(+90)）
    aligned = original.rotate(90, expand=True)
    aligned_bytes = BytesIO()
    aligned.save(aligned_bytes, format="PNG")

    state = {
        "rotation": 90,
        "redaction_result": _redaction_result(aligned_bytes.getvalue(), 900, 560),
    }
    updates = finalize_node(state)
    result = updates["redaction_result"]

    restored = Image.open(BytesIO(result.redacted_image))
    assert restored.size == original.size  # (560, 900)
    # 像素级往返：转正再逆旋转 == 原图
    assert list(restored.getdata()) == list(original.getdata())


def test_finalize_no_rotation_noop():
    """rotation=0：无操作。"""
    result = _redaction_result(b"img", 100, 100)
    assert finalize_node({"rotation": 0, "redaction_result": result}) == {}
    assert finalize_node({"rotation": 90, "redaction_result": None}) == {}


def test_finalize_writes_output_file(tmp_path):
    """finalize 把还原后的图写回 output_path。"""
    original = Image.open(SAMPLE).convert("RGB")
    aligned = original.rotate(90, expand=True)
    buf = BytesIO()
    aligned.save(buf, format="PNG")

    out = tmp_path / "masked.png"
    state = {
        "rotation": 90,
        "redaction_result": _redaction_result(buf.getvalue(), 900, 560),
    }
    state["redaction_result"].output_path = str(out)
    finalize_node(state)

    saved = Image.open(out)
    assert saved.size == original.size
