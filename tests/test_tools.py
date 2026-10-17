"""工具级测试：真实脱敏像素变化、质检闭环判定、规则通道。"""
from __future__ import annotations

from io import BytesIO
from typing import Optional

import pytest
from PIL import Image, ImageDraw

from src.config import Settings
from src.schemas.ocr_result import OcrResult, OcrTextBlock
from src.schemas.redaction_result import RedactedBlock, RedactionResult
from src.schemas.vlm_analysis import RedactionMethod, SensitiveBlock, VlmAnalysis
from src.sensitive_rules import match_blocks, merge_sensitive, scan_text
from src.tools.qa_tool import QaTool, normalize
from src.tools.redaction_tool import RedactionTool
from tests.fakes import StubOcrEngine


def make_image_bytes(color="white", size=(400, 200)) -> bytes:
    img = Image.new("RGB", size, color)
    draw = ImageDraw.Draw(img)
    draw.rectangle((50, 50, 150, 80), fill="black")
    draw.rectangle((50, 100, 150, 130), fill="black")
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def settings(tmp_path):
    return Settings(output_dir=str(tmp_path / "outputs"))


@pytest.fixture
def redaction_tool(settings):
    return RedactionTool(settings)


# --------------------------------------------------------------------- 脱敏


def test_white_mask_erases_region(redaction_tool):
    analysis = VlmAnalysis(
        sensitive_blocks=[
            SensitiveBlock(block_id="b0", text="张三", bbox=[50, 50, 150, 80])
        ]
    )
    result = redaction_tool.apply(make_image_bytes(), analysis)

    img = Image.open(BytesIO(result.redacted_image))
    assert img.getpixel((100, 65)) == (255, 255, 255)  # 敏感区被遮白
    assert img.getpixel((100, 115)) == (0, 0, 0)  # 非敏感区保留
    assert result.width == 400 and result.height == 200
    assert len(result.redacted_blocks) == 1
    assert result.redacted_blocks[0].block_id == "b0"
    assert result.output_path and result.output_path.endswith(".png")


def test_white_mask_redraws_fake_data(redaction_tool):
    """白底覆盖后在原 bbox 重绘假数据：区域内有黑字（非纯白），
    且假数据替代了敏感值、保留了标签。"""
    if not redaction_tool._font_path:
        pytest.skip("未找到中文字体")
    analysis = VlmAnalysis(
        sensitive_blocks=[
            SensitiveBlock(
                block_id="b0",
                text="姓名：张三",
                value="张三",
                kind="name",
                bbox=[50, 50, 350, 80],
            )
        ]
    )
    result = redaction_tool.apply(make_image_bytes(), analysis)
    record = result.redacted_blocks[0]
    assert record.text.startswith("姓名：")  # 标签保留
    assert "张三" not in record.text  # 敏感值被替换
    assert len(record.text.split("：")[1]) == 2  # 假姓名等长

    img = Image.open(BytesIO(result.redacted_image))
    region = img.crop((47, 47, 353, 83))
    assert min(region.convert("L").getdata()) < 128  # 区域内有黑色文字
    assert img.getpixel((100, 115)) == (0, 0, 0)  # 相邻非敏感行不受影响


def test_same_value_maps_to_same_fake(redaction_tool):
    """相同敏感值在不同块中映射到相同假数据。"""
    blocks = [
        SensitiveBlock(block_id=f"b{i}", text="姓名：张三", value="张三", kind="name",
                       bbox=[50, 50 + i * 60, 350, 80 + i * 60])
        for i in range(2)
    ]
    result = redaction_tool.apply(make_image_bytes(), VlmAnalysis(sensitive_blocks=blocks))
    texts = [r.text for r in result.redacted_blocks]
    assert texts[0] == texts[1]
    assert "张三" not in texts[0]


def test_jpeg_output_matches_extension(redaction_tool, tmp_path):
    """.jpg 输出内容必须是 JPEG（扩展名与格式一致）。"""
    analysis = VlmAnalysis(
        sensitive_blocks=[SensitiveBlock(block_id="b0", text="张三", bbox=[50, 50, 150, 80])]
    )
    out = tmp_path / "sub" / "ticket.jpg"
    result = redaction_tool.apply(make_image_bytes(), analysis, output_path=str(out))
    assert out.exists()
    data = out.read_bytes()
    assert data[:2] == b"\xff\xd8"  # JPEG 魔数
    # RedactionResult 中的字节与落盘一致
    assert result.redacted_image[:2] == b"\xff\xd8"
    assert result.image_format == "JPG"


def test_png_output_stays_png(redaction_tool, tmp_path):
    """.png 输出仍是 PNG。"""
    analysis = VlmAnalysis(
        sensitive_blocks=[SensitiveBlock(block_id="b0", text="张三", bbox=[50, 50, 150, 80])]
    )
    out = tmp_path / "ticket.png"
    redaction_tool.apply(make_image_bytes(), analysis, output_path=str(out))
    assert out.read_bytes()[:4] == b"\x89PNG"


@pytest.mark.parametrize(
    "suffix, magics",
    [
        (".tif", (b"II*\x00", b"MM\x00*")),
        (".bmp", (b"BM",)),
        (".webp", (b"RIFF",)),
    ],
)
def test_other_extensions_encode_matching_format(redaction_tool, tmp_path, suffix, magics):
    """bmp/tif/webp 输出也必须是该格式。

    历史 bug：只对 .jpg 做转码，其它扩展名把 PNG 字节原样写进去，
    文件头与扩展名不符（踩坑 #15 的 .tif/.bmp/.webp 版本）。
    """
    analysis = VlmAnalysis(
        sensitive_blocks=[SensitiveBlock(block_id="b0", text="张三", bbox=[50, 50, 150, 80])]
    )
    out = tmp_path / f"ticket{suffix}"
    result = redaction_tool.apply(make_image_bytes(), analysis, output_path=str(out))

    data = out.read_bytes()
    assert any(data.startswith(m) for m in magics), data[:8]
    assert result.redacted_image == data
    with Image.open(BytesIO(data)) as img:
        assert img.size == (400, 200)


def test_fake_data_disabled_falls_back_to_plain_mask(settings, redaction_tool):
    settings.fake_data_enabled = False
    analysis = VlmAnalysis(
        sensitive_blocks=[
            SensitiveBlock(block_id="b0", text="姓名：张三", value="张三", kind="name",
                           bbox=[50, 50, 350, 80])
        ]
    )
    result = redaction_tool.apply(make_image_bytes(), analysis)
    assert result.redacted_blocks[0].text == ""  # 纯覆盖，不重绘


def test_original_image_not_modified(redaction_tool):
    raw = make_image_bytes()
    analysis = VlmAnalysis(
        sensitive_blocks=[SensitiveBlock(block_id="b0", bbox=[50, 50, 150, 80])]
    )
    redaction_tool.apply(raw, analysis)
    # 原图字节不变
    assert Image.open(BytesIO(raw)).getpixel((100, 65)) == (0, 0, 0)


def test_blur_and_mosaic(redaction_tool, settings):
    raw = make_image_bytes()
    analysis = VlmAnalysis(
        sensitive_blocks=[
            SensitiveBlock(block_id="b0", bbox=[50, 50, 150, 80]),
            SensitiveBlock(
                block_id="b1", bbox=[50, 100, 150, 130], method=RedactionMethod.BLUR
            ),
        ]
    )
    settings.blur_radius = 12
    result = redaction_tool.apply(raw, analysis)
    img = Image.open(BytesIO(result.redacted_image))
    assert img.getpixel((100, 65)) == (255, 255, 255)
    # 模糊后黑块中心变灰（既不是纯黑也不是纯白）
    r, g, b = img.getpixel((100, 115))
    assert 0 < r < 255


def test_block_without_bbox_is_skipped(redaction_tool):
    analysis = VlmAnalysis(
        sensitive_blocks=[SensitiveBlock(block_id="b0", text="无坐标", bbox=[])]
    )
    result = redaction_tool.apply(make_image_bytes(), analysis)
    assert result.redacted_blocks == []


# --------------------------------------------------------------------- 质检


def make_ocr_result() -> OcrResult:
    return OcrResult(
        engine="stub",
        width=400,
        height=200,
        blocks=[
            OcrTextBlock(block_id="b0", text="张三", bbox=[50, 50, 150, 80]),
            OcrTextBlock(block_id="b1", text="血常规", bbox=[50, 100, 150, 130]),
        ],
    )


def make_redaction_result(bboxes) -> RedactionResult:
    return RedactionResult(
        redacted_image=b"",
        width=400,
        height=200,
        redacted_blocks=[
            RedactedBlock(block_id=f"b{i}", bbox=list(bbox)) for i, bbox in enumerate(bboxes)
        ],
    )


def make_redaction_result_with_texts(texts) -> RedactionResult:
    return RedactionResult(
        redacted_image=b"",
        width=400,
        height=200,
        redacted_blocks=[
            RedactedBlock(block_id=f"b{i}", text=t, bbox=[50, 50 + i * 40, 350, 80 + i * 40])
            for i, t in enumerate(texts)
        ],
    )


def test_qa_detects_residual(settings):
    analysis = VlmAnalysis(
        sensitive_blocks=[
            SensitiveBlock(block_id="b0", text="姓名：张三", value="张三", bbox=[50, 50, 150, 80])
        ]
    )
    qa = QaTool(settings, StubOcrEngine(["张三"]))
    report = qa.verify(
        original_image=b"",
        redacted_image=b"",
        vlm_analysis=analysis,
        ocr_result=make_ocr_result(),
        redaction_result=make_redaction_result([[50, 50, 150, 80]]),
    )
    assert report.status == "failed"
    assert report.residual_texts == ["b0"]
    assert report.feedback and "b0" in report.feedback


def test_qa_passes_when_clean(settings):
    analysis = VlmAnalysis(
        sensitive_blocks=[SensitiveBlock(block_id="b0", text="张三", bbox=[50, 50, 150, 80])]
    )
    qa = QaTool(settings, StubOcrEngine(["医院名称", "血常规"]))
    report = qa.verify(
        original_image=b"",
        redacted_image=b"",
        vlm_analysis=analysis,
        ocr_result=make_ocr_result(),
        redaction_result=make_redaction_result([[50, 50, 150, 80]]),
    )
    assert report.status == "passed"


def test_qa_detects_pattern_residual(settings):
    analysis = VlmAnalysis(
        sensitive_blocks=[SensitiveBlock(block_id="b0", text="张三", bbox=[50, 50, 150, 80])]
    )
    qa = QaTool(settings, StubOcrEngine(["13800138000"]))
    report = qa.verify(
        original_image=b"",
        redacted_image=b"",
        vlm_analysis=analysis,
        ocr_result=make_ocr_result(),
        redaction_result=make_redaction_result([[50, 50, 150, 80]]),
    )
    assert report.status == "failed"
    assert "手机号" in report.residual_patterns


def test_qa_detects_collateral_damage(settings):
    # 遮罩覆盖了非敏感块 b1 的大部分 → 误伤
    analysis = VlmAnalysis(
        sensitive_blocks=[SensitiveBlock(block_id="b0", text="张三", bbox=[50, 50, 150, 180])]
    )
    qa = QaTool(settings, StubOcrEngine(["医院名称"]))
    report = qa.verify(
        original_image=b"",
        redacted_image=b"",
        vlm_analysis=analysis,
        ocr_result=make_ocr_result(),
        redaction_result=make_redaction_result([[50, 50, 150, 180]]),
    )
    assert report.status == "failed"
    assert report.damaged_texts == ["b1"]


def test_qa_normalize():
    assert normalize("张 三，") == "张三"


def test_qa_vlm_verdict_failed(settings):
    """确定性预检干净时，由 VLM 语义质检判定（如漏判姓名）。"""
    from src.schemas.qa_verdict import QaVerdict
    from tests.fakes import FakeVlmTool

    vlm = FakeVlmTool(
        qa_verdict=QaVerdict(
            status="failed",
            reason="mask 图中仍可识别患者姓名",
            feedback="块 ['b1'] 疑似患者姓名，请纳入敏感列表",
            residual_block_ids=["b1"],
            revised_block_ids=["b0", "b1"],
        )
    )
    analysis = VlmAnalysis(
        sensitive_blocks=[SensitiveBlock(block_id="b0", text="身份证号：320123199001011234", value="320123199001011234", kind="id_card", bbox=[50, 50, 350, 80])]
    )
    qa = QaTool(settings, StubOcrEngine(["医院名称"]), vlm)
    report = qa.verify(
        original_image=b"",
        redacted_image=b"",
        vlm_analysis=analysis,
        ocr_result=None,
        redaction_result=None,
    )
    assert report.status == "failed"
    assert report.residual_texts == ["b1"]
    assert report.revised_block_ids == ["b0", "b1"]
    assert report.feedback and "b1" in report.feedback


def test_qa_vlm_verdict_passed(settings):
    from tests.fakes import FakeVlmTool

    vlm = FakeVlmTool()
    analysis = VlmAnalysis(
        sensitive_blocks=[SensitiveBlock(block_id="b0", text="张三", value="张三", kind="name", bbox=[50, 50, 150, 80])]
    )
    qa = QaTool(settings, StubOcrEngine(["医院名称"]), vlm)
    report = qa.verify(
        original_image=b"",
        redacted_image=b"",
        vlm_analysis=analysis,
        ocr_result=None,
        redaction_result=None,
    )
    assert report.status == "passed"
    assert vlm.qa_calls == 1


def test_qa_no_false_positive_on_shared_digit_prefix(settings):
    """回归：票据号与医保号共享日期前缀，不得因相似误报残留。"""
    ocr = OcrResult(
        engine="stub",
        width=400,
        height=200,
        blocks=[
            OcrTextBlock(block_id="b0", text="医保卡号：YB2026101600123", bbox=[50, 50, 350, 80]),
            OcrTextBlock(block_id="b1", text="票据号：20261016001", bbox=[50, 100, 350, 130]),
        ],
    )
    analysis = VlmAnalysis(
        sensitive_blocks=[
            SensitiveBlock(block_id="b0", text="医保卡号：YB2026101600123", bbox=[50, 50, 350, 80])
        ]
    )
    qa = QaTool(settings, StubOcrEngine(["票据号：20261016001", "合计335.00"]))
    report = qa.verify(
        original_image=b"",
        redacted_image=b"",
        vlm_analysis=analysis,
        ocr_result=ocr,
        redaction_result=make_redaction_result([[50, 50, 350, 80]]),
    )
    assert report.status == "passed"


def test_qa_fake_data_not_counted_as_residual(settings):
    """假数据（等长合法格式）不得被正则兜底误报为残留。"""
    from src.fake_data import FakeDataMapper

    mapper = FakeDataMapper()
    fake_id = mapper.map("320123199001011234", "id_card")
    fake_mobile = mapper.map("13407982134", "mobile")
    analysis = VlmAnalysis(
        sensitive_blocks=[
            SensitiveBlock(
                block_id="b0", text="身份证号：320123199001011234",
                value="320123199001011234", kind="id_card", bbox=[50, 50, 350, 80],
            ),
            SensitiveBlock(
                block_id="b1", text="手机号：13407982134",
                value="13407982134", kind="mobile", bbox=[50, 100, 350, 130],
            ),
        ]
    )
    # 脱敏图重 OCR 读到的是假数据
    drawn = [
        f"身份证号：{fake_id}",
        f"手机号：{fake_mobile}",
    ]
    qa = QaTool(settings, StubOcrEngine(drawn))
    report = qa.verify(
        original_image=b"",
        redacted_image=b"",
        vlm_analysis=analysis,
        ocr_result=None,
        redaction_result=make_redaction_result_with_texts(drawn),
    )
    assert report.status == "passed"
    assert report.residual_patterns == []


# ---------------------------------------------------------------- 部署健壮性


def _png(width=400, height=200, fmt="PNG") -> bytes:
    img = Image.new("RGB", (width, height), "white")
    buf = BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def test_ensure_utf8_stdio_switches_ascii_only():
    """LANG=C 的服务器必须切 UTF-8；中文 Windows 控制台不能被乱动。"""
    import sys
    from types import SimpleNamespace

    import main

    switched = []

    def make_stream(encoding):
        return SimpleNamespace(
            encoding=encoding,
            reconfigure=lambda **kw: switched.append(kw),
        )

    real_stdout, real_stderr = sys.stdout, sys.stderr
    try:
        # LANG=C / POSIX：两个流都要切
        switched.clear()
        sys.stdout, sys.stderr = make_stream("ascii"), make_stream("ANSI_X3.4-1968")
        main.ensure_utf8_stdio()
        assert switched == [
            {"encoding": "utf-8", "errors": "replace"},
            {"encoding": "utf-8", "errors": "replace"},
        ]

        # 中文 Windows（cp936）/ 已是 UTF-8：不动
        switched.clear()
        sys.stdout, sys.stderr = make_stream("cp936"), make_stream("utf-8")
        main.ensure_utf8_stdio()
        assert switched == []
    finally:
        sys.stdout, sys.stderr = real_stdout, real_stderr


def test_sensitive_block_invalid_method_coerced():
    """LLM 给出非法脱敏方式时兜底为 white_mask，而不是解析失败。"""
    assert SensitiveBlock(block_id="b0", method="bogus").method is RedactionMethod.WHITE_MASK
    assert SensitiveBlock(block_id="b0", method=None).method is RedactionMethod.WHITE_MASK
    assert SensitiveBlock(block_id="b0", method="blur").method is RedactionMethod.BLUR


def test_vlm_sanitize_malformed_bbox_falls_back_to_ocr():
    """畸形 bbox 长度不得触发 IndexError，直接采用 OCR 坐标。"""
    from src.tools.vlm_tool import VlmTool

    ocr = OcrResult(
        engine="stub",
        blocks=[OcrTextBlock(block_id="b0", text="姓名：张三", bbox=[10, 20, 30, 40])],
    )
    analysis = VlmAnalysis(
        sensitive_blocks=[
            SensitiveBlock(block_id="b0", text="姓名：张三", value="张三", kind="name", bbox=[1, 2])
        ]
    )
    VlmTool._sanitize(analysis, ocr)
    assert analysis.sensitive_blocks[0].bbox == [10, 20, 30, 40]


def test_ocr_detect_orientation_accepts_float(settings):
    """RapidOrientation 返回 float 时不得因 int(str(...)) 报错而永久禁用矫正。"""
    from types import SimpleNamespace

    from src.tools.ocr_tool import OcrTool

    tool = OcrTool(settings)
    tool._local.orientation_engine = lambda arr: (90.0, 0.0)
    tool._local.np = SimpleNamespace(array=lambda img: img)

    assert tool._detect_orientation(Image.new("RGB", (20, 10))) == 90
    assert getattr(tool._local, "orientation_failed", False) is False


def test_qa_detects_digit_run_residual(settings):
    """长数字段完整出现在脱敏图中 → 残留。"""
    analysis = VlmAnalysis(
        sensitive_blocks=[
            SensitiveBlock(block_id="b0", text="身份证号：320123199001011234", bbox=[50, 50, 350, 80])
        ]
    )
    qa = QaTool(settings, StubOcrEngine(["证件号码320123199001011234"]))
    report = qa.verify(
        original_image=b"",
        redacted_image=b"",
        vlm_analysis=analysis,
        ocr_result=None,
        redaction_result=None,
    )
    assert report.status == "failed"
    assert report.residual_texts == ["b0"]


# --------------------------------------------------------------------- 规则


def test_rules_match_ids():
    ocr = OcrResult(
        engine="stub",
        blocks=[
            OcrTextBlock(block_id="b0", text="身份证号320123199001011234"),
            OcrTextBlock(block_id="b1", text="手机号13800138000"),
            OcrTextBlock(block_id="b2", text="科室：内科门诊"),
        ],
    )
    hits = match_blocks(ocr)
    assert {b.block_id for b in hits} == {"b0", "b1"}
    assert all(b.source == "rule" for b in hits)


def test_rules_do_not_match_invoice_numbers():
    """回归：票据号码等数字串不得被规则通道误判为敏感（真实票据案例）。"""
    ocr = OcrResult(
        engine="stub",
        blocks=[
            OcrTextBlock(block_id="b0", text="票据号码：0137126549"),
            OcrTextBlock(block_id="b1", text="业务流水号：23556020240310082249"),
            OcrTextBlock(block_id="b2", text="校验码：6ca80a"),
        ],
    )
    assert match_blocks(ocr) == []


def test_rules_do_not_match_landline_like_numbers():
    """回归：固定电话已从规则通道移除（与票据号同构，误报率高），交 VLM 判断。"""
    ocr = OcrResult(
        engine="stub",
        blocks=[
            OcrTextBlock(block_id="b0", text="电话：028-12345678"),
            OcrTextBlock(block_id="b1", text="Tel：07551234567"),
        ],
    )
    assert match_blocks(ocr) == []
    # scan_text（质检兜底）同样不再命中固话
    assert scan_text("028-12345678") == []


def test_merge_dedup_prefers_vlm():
    from src.schemas.vlm_analysis import SensitiveBlock as SB

    vlm = [SB(block_id="b0", reason="vlm")]
    rule = [SB(block_id="b0", reason="rule"), SB(block_id="b1", reason="rule")]
    merged = merge_sensitive(vlm, rule)
    assert len(merged) == 2
    assert merged[0].source == "vlm"


def test_scan_text():
    assert scan_text("联系方式 13800138000") == ["手机号"]
    assert scan_text("无敏感信息") == []


# --------------------------------------------------------------------- 性能
def test_qa_reocr_skips_orientation_and_caps_resolution(settings):
    """质检重 OCR：不重复做方向检测，且按 QA_REOCR_MAX_SIDE 压分辨率。

    两者都只影响耗时，不影响坐标（遮罩坐标来自首轮 OCR）。
    """
    stub = StubOcrEngine(["医院名称"])
    qa = QaTool(settings, stub)
    big = _png(2000, 1000)

    qa.verify(
        original_image=b"",
        redacted_image=big,
        vlm_analysis=VlmAnalysis(sensitive_blocks=[]),
        ocr_result=None,
        redaction_result=make_redaction_result([]),
    )
    assert stub.kwargs.get("detect_orientation") is False
    assert stub.last_image_bytes != big  # 已被缩小

    settings.qa_reocr_max_side = 0  # 关闭缩放
    qa.verify(
        original_image=b"",
        redacted_image=big,
        vlm_analysis=VlmAnalysis(sensitive_blocks=[]),
        ocr_result=None,
        redaction_result=make_redaction_result([]),
    )
    assert stub.last_image_bytes == big


def test_cap_side_edge_cases():
    from src.tools.qa_tool import _cap_side

    small = _png(100, 60)
    assert _cap_side(small, 1280) == small          # 未超限：原样返回
    assert _cap_side(small, 0) == small             # 关闭：原样返回
    assert _cap_side(b"not-an-image", 1280) == b"not-an-image"  # 解码失败：原样返回
    capped = _cap_side(_png(2000, 1000), 500)
    with Image.open(BytesIO(capped)) as img:
        assert img.size == (500, 250)               # 等比


def test_vlm_prepare_image_passes_through_jpeg_and_reports_mime(tmp_path):
    """阈值内的 JPEG 原样透传（省解码+重编码，也不把 JPEG 膨胀成 PNG）。"""
    from src.tools.vlm_tool import VlmTool

    tool = VlmTool(Settings(output_dir=str(tmp_path)))
    jpeg = _png(300, 200, fmt="JPEG")
    payload, mime = tool._prepare_image(jpeg, max_side=1024)

    assert mime == "image/jpeg"
    import base64 as _b64

    assert _b64.b64decode(payload) == jpeg
    assert tool._image_url(jpeg, max_side=1024).startswith("data:image/jpeg;base64,")


def test_vlm_qa_mask_image_uses_smaller_side(tmp_path):
    """质检的「脱敏图」按 VLM_QA_MASK_IMAGE_SIDE 发送，载荷显著小于原图。"""
    from src.tools.vlm_tool import VlmTool

    tool = VlmTool(
        Settings(output_dir=str(tmp_path), vlm_qa_mask_image_side=64)
    )
    big = _png(2000, 1000)
    original_url = tool._image_url(big)                       # 默认 1024
    mask_url = tool._image_url(big, tool._qa_mask_max_side())  # 质检遮罩图

    assert len(mask_url) < len(original_url) / 4
    assert tool._qa_mask_max_side() == 64
    # 0 / 负数 = 跟随原图上限
    tool2 = VlmTool(Settings(output_dir=str(tmp_path), vlm_qa_mask_image_side=0))
    assert tool2._qa_mask_max_side() == tool2._settings.vlm_max_image_side


class _RecordingVlm:
    """记录 qa_verify 入参的假 VLM（用于验证并发质检的语义）。"""

    def __init__(self, verdict=None, exc: Optional[Exception] = None):
        self.verdict = verdict
        self.exc = exc
        self.calls: list = []

    def qa_verify(self, **kwargs):
        self.calls.append(kwargs)
        if self.exc:
            raise self.exc
        return self.verdict


def _residual_analysis() -> VlmAnalysis:
    return VlmAnalysis(
        sensitive_blocks=[
            SensitiveBlock(
                block_id="b0", text="身份证号：320123199001011234",
                value="320123199001011234", kind="id_card", bbox=[50, 50, 350, 80],
            )
        ]
    )


def test_qa_overlap_precheck_and_vlm_run_together(settings):
    """默认并发：确定性预检与 VLM 同时跑，VLM 不带 hints；
    硬残留仍然权威——VLM 说 passed 也判 failed。"""
    from src.schemas.qa_verdict import QaVerdict

    vlm = _RecordingVlm(verdict=QaVerdict(status="passed", reason="vlm ok"))
    stub = StubOcrEngine(["身份证号320123199001011234"])  # 重 OCR 仍读到原值
    qa = QaTool(settings, stub, vlm)

    report = qa.verify(
        b"", b"", _residual_analysis(),
        ocr_result=None, redaction_result=make_redaction_result([[50, 50, 350, 80]]),
    )

    assert report.status == "failed"
    assert report.residual_texts == ["b0"]
    assert len(vlm.calls) == 1                       # VLM 确实被调用了
    assert vlm.calls[0]["deterministic_hints"] is None  # 并发模式不带 hints


def test_qa_overlap_uses_vlm_verdict_when_precheck_clean(settings):
    from src.schemas.qa_verdict import QaVerdict

    vlm = _RecordingVlm(
        verdict=QaVerdict(
            status="failed", reason="漏了姓名", feedback="请补 b9",
            residual_block_ids=["b9"], revised_block_ids=["b9"],
        )
    )
    stub = StubOcrEngine(["医院名称", "血常规"])
    qa = QaTool(settings, stub, vlm)

    report = qa.verify(
        b"", b"", VlmAnalysis(sensitive_blocks=[]),
        ocr_result=None, redaction_result=make_redaction_result([]),
    )

    assert report.status == "failed"
    assert report.reason == "漏了姓名"
    assert report.residual_texts == ["b9"]
    assert report.revised_block_ids == ["b9"]


def test_qa_overlap_vlm_exception_becomes_failed_report(settings):
    vlm = _RecordingVlm(exc=RuntimeError("502 bad gateway"))
    stub = StubOcrEngine(["医院名称"])
    qa = QaTool(settings, stub, vlm)

    report = qa.verify(
        b"", b"", VlmAnalysis(sensitive_blocks=[]),
        ocr_result=None, redaction_result=make_redaction_result([]),
    )

    assert report.status == "failed"
    assert "502 bad gateway" in report.reason


def test_qa_sequential_mode_keeps_deterministic_hints(settings):
    """QA_PRECHECK_BACKGROUND=false：回到串行 + 带 hints 的旧行为。"""
    from src.schemas.qa_verdict import QaVerdict

    settings.qa_precheck_background = False
    vlm = _RecordingVlm(verdict=QaVerdict(status="passed", reason="ok"))
    stub = StubOcrEngine(["医院名称"])
    qa = QaTool(settings, stub, vlm)

    report = qa.verify(
        b"", b"", VlmAnalysis(sensitive_blocks=[]),
        ocr_result=None, redaction_result=make_redaction_result([]),
    )

    assert report.status == "passed"
    assert vlm.calls[0]["deterministic_hints"]["re_ocr_blocks"] == ["医院名称"]


# --------------------------------------------------------------------- 引擎池


def test_ocr_engine_pool_parallel_and_capped(settings):
    """引擎按需创建、用完归还；并发数不超过 OCR_ENGINE_INSTANCES。"""
    import threading
    import time

    from src.tools.ocr_tool import OcrTool

    settings.ocr_engine_instances = 2
    tool = OcrTool(settings)
    created: list = []
    tool._new_engine = lambda: created.append(object()) or object()

    seen: list = []
    lock = threading.Lock()

    def worker():
        with tool._borrow_engine() as engine:
            with lock:
                seen.append(engine)
            time.sleep(0.05)

    threads = [threading.Thread(target=worker) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(created) == 2            # 上限生效：只创建 2 个实例
    assert len(set(seen)) == 2          # 6 个线程复用这 2 个实例
    assert tool._pool.qsize() == 2      # 全部归还


def test_ocr_engine_pool_single_instance_serializes_without_deadlock(settings):
    """OCR_ENGINE_INSTANCES=1：完全串行（旧行为），但不能死锁。"""
    import threading

    from src.tools.ocr_tool import OcrTool

    settings.ocr_engine_instances = 1
    tool = OcrTool(settings)
    created: list = []
    tool._new_engine = lambda: created.append(object()) or object()

    done: list = []
    lock = threading.Lock()

    def worker(i):
        with tool._borrow_engine():
            pass
        with lock:
            done.append(i)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)
        assert not t.is_alive()

    assert len(created) == 1
    assert sorted(done) == [0, 1, 2, 3]


def test_ocr_engine_pool_releases_slot_when_creation_fails(settings):
    """引擎创建失败要把名额还回去，否则后续调用全部堵死。"""
    from src.tools.ocr_tool import OcrTool

    settings.ocr_engine_instances = 1
    tool = OcrTool(settings)
    calls = {"n": 0}

    def flaky_new():
        calls["n"] += 1
        if calls["n"] == 1:
            raise ImportError("no rapidocr")
        return object()

    tool._new_engine = flaky_new
    with pytest.raises(ImportError):
        with tool._borrow_engine():
            pass
    # 第二次应当能重新创建（名额已释放）
    with tool._borrow_engine():
        pass
    assert calls["n"] == 2
