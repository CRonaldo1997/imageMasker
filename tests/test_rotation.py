"""旋转/竖排文字处理测试：角度计算、多边形覆盖、假数据旋转渲染、误伤判定。"""
from __future__ import annotations

from io import BytesIO

import pytest
from PIL import Image, ImageDraw

from src.config import Settings
from src.schemas.ocr_result import OcrResult, OcrTextBlock
from src.schemas.redaction_result import RedactedBlock, RedactionResult
from src.schemas.vlm_analysis import SensitiveBlock, VlmAnalysis
from src.tools.ocr_tool import _quad_angle, _quad_points, _quad_to_xyxy
from src.tools.qa_tool import QaTool, _point_covered
from src.tools.redaction_tool import RedactionTool
from tests.fakes import StubOcrEngine


@pytest.fixture
def settings(tmp_path):
    return Settings(output_dir=str(tmp_path / "outputs"))


@pytest.fixture
def redaction_tool(settings):
    return RedactionTool(settings)


# --------------------------------------------------------------------- 角度


def test_quad_angle_horizontal():
    quad = [[10, 20], [200, 22], [200, 60], [10, 58]]  # 横长条
    assert abs(_quad_angle(quad)) < 1


def test_quad_angle_vertical():
    quad = [[43, 37], [74, 37], [74, 160], [43, 160]]  # 竖长条
    assert abs(_quad_angle(quad) - 90) < 1


def test_quad_angle_inverted():
    quad = [[200, 60], [10, 58], [10, 20], [200, 22]]  # 上边指向左
    assert abs(abs(_quad_angle(quad)) - 180) < 1


def test_quad_angle_slight_tilt():
    quad = [[29, 18], [506, 21], [506, 53], [29, 51]]
    assert 0 < _quad_angle(quad) < 1


def test_quad_to_xyxy_and_points():
    quad = [[43, 37], [74, 37], [74, 160], [43, 160]]
    assert _quad_to_xyxy(quad) == [43, 37, 74, 160]
    assert _quad_points(quad) == [[43, 37], [74, 37], [74, 160], [43, 160]]


# --------------------------------------------------------------------- 覆盖与渲染


def _vertical_image() -> bytes:
    """白底上一行横排黑字，整体旋转 90° 后的竖排测试图。"""
    img = Image.new("RGB", (400, 80), "white")
    draw = ImageDraw.Draw(img)
    draw.rectangle((20, 25, 380, 55), fill="black")  # 模拟一行文字
    rotated = img.rotate(-90, expand=True)  # 顺时针 90°
    buf = BytesIO()
    rotated.save(buf, format="PNG")
    return buf.getvalue()


def test_vertical_mask_polygon_and_rotated_fake(redaction_tool):
    if not redaction_tool._font_path:
        pytest.skip("未找到中文字体")
    # 竖排敏感块：四边形竖条 + 行方向 90°
    analysis = VlmAnalysis(
        sensitive_blocks=[
            SensitiveBlock(
                block_id="b0",
                text="姓名：张三",
                value="张三",
                kind="name",
                bbox=[23, 20, 55, 380],
                points=[[23, 20], [55, 20], [55, 380], [23, 380]],
                angle=90,
            )
        ]
    )
    result = redaction_tool.apply(_vertical_image(), analysis)
    record = result.redacted_blocks[0]
    assert record.angle == 90
    assert record.text.startswith("姓名：")
    assert "张三" not in record.text
    assert len(record.text) == 5  # 等长

    img = Image.open(BytesIO(result.redacted_image))
    region = img.crop((20, 17, 58, 383)).convert("L")
    assert min(region.getdata()) < 128  # 竖条内有黑色文字（假数据已渲染）
    # 条外相邻列不受影响（未被多边形误盖）
    assert img.convert("L").getpixel((10, 200)) == 255


def test_vertical_mask_without_font_falls_back_to_polygon(settings):
    tool = RedactionTool(settings)
    tool._font_path = None  # 模拟无字体环境
    analysis = VlmAnalysis(
        sensitive_blocks=[
            SensitiveBlock(
                block_id="b0", text="姓名：张三", value="张三", kind="name",
                bbox=[23, 20, 55, 380],
                points=[[23, 20], [55, 20], [55, 380], [23, 380]],
                angle=90,
            )
        ]
    )
    result = tool.apply(_vertical_image(), analysis)
    img = Image.open(BytesIO(result.redacted_image))
    # 多边形白底覆盖生效
    assert img.getpixel((39, 200)) == (255, 255, 255)
    assert result.redacted_blocks[0].text == "姓名：孙礼"  # 仍记录假数据


def test_horizontal_mask_unchanged(redaction_tool):
    """水平块（angle≈0）行为与之前一致。"""
    buf = BytesIO()
    Image.open(BytesIO(_vertical_image())).rotate(90, expand=True).save(buf, format="PNG")
    img_bytes = buf.getvalue()  # 还原成横排
    analysis = VlmAnalysis(
        sensitive_blocks=[
            SensitiveBlock(block_id="b0", text="张三", bbox=[20, 25, 380, 55], angle=0)
        ]
    )
    result = redaction_tool.apply(img_bytes, analysis)
    out = Image.open(BytesIO(result.redacted_image))
    assert out.getpixel((200, 40)) == (255, 255, 255)


# --------------------------------------------------------------------- 误伤


def test_collateral_vertical_columns_not_damaged(settings):
    """竖排图中相邻列中心点不在遮罩多边形内 → 不误伤。"""
    ocr = OcrResult(
        engine="stub",
        blocks=[
            OcrTextBlock(block_id="b0", text="敏感", bbox=[23, 20, 55, 380],
                         points=[[23, 20], [55, 20], [55, 380], [23, 380]], angle=90),
            OcrTextBlock(block_id="b1", text="普通列", bbox=[100, 20, 130, 380],
                         points=[[100, 20], [130, 20], [130, 380], [100, 380]], angle=90),
        ],
    )
    analysis = VlmAnalysis(sensitive_blocks=[
        SensitiveBlock(block_id="b0", text="敏感", bbox=[23, 20, 55, 380],
                       points=[[23, 20], [55, 20], [55, 380], [23, 380]], angle=90),
    ])
    redaction = RedactionResult(
        redacted_image=b"",
        redacted_blocks=[
            RedactedBlock(block_id="b0", bbox=[23, 20, 55, 380],
                          points=[[23, 20], [55, 20], [55, 380], [23, 380]], angle=90),
        ],
    )
    qa = QaTool(settings, StubOcrEngine([]))
    damaged = qa._find_collateral(analysis, ocr, redaction)
    assert damaged == []


def test_point_covered_basic():
    square = [[0, 0], [10, 0], [10, 10], [0, 10]]
    assert _point_covered(5, 5, square)
    assert not _point_covered(15, 5, square)
    assert not _point_covered(-1, 5, square)
