"""LangGraph 全流程测试：通过 / 重试 / 回退 / 质检关闭 / 失败传播。"""
from __future__ import annotations

from pathlib import Path

import pytest

from src import tools
from src.config import Settings
from src.graph import build_graph, compute_recursion_limit, route_after_qa
from src.graph_state import RedactionState
from src.nodes import fallback_node, ocr_node
from tests.fakes import (
    FakeOcrTool,
    FakeQaTool,
    FakeRedactionTool,
    FakeVlmTool,
    make_ocr_result,
)


def initial_state(**overrides) -> RedactionState:
    state: RedactionState = {
        "image_path": "data/sample_bill.png",
        "original_image": b"fake-bytes",
        "ocr_result": None,
        "vlm_analysis": None,
        "redaction_result": None,
        "qa_report": None,
        "qa_enabled": True,
        "retry_count": 0,
        "max_retries": 3,
        "feedback": None,
        "status": "processing",
    }
    state.update(overrides)
    return state


@pytest.fixture(autouse=True)
def clean_tools():
    tools.reset_tools()
    yield
    tools.reset_tools()


def _png_bytes(tmp_path, size=(64, 32)) -> bytes:
    """生成一张最小合法 PNG，避免测试依赖仓库内示例图。"""
    from PIL import Image

    path = tmp_path / "sample.png"
    Image.new("RGB", size, "white").save(path)
    return path.read_bytes()


def inject(ocr=None, vlm=None, redact=None, qa=None):
    if ocr is not None:
        tools._cache["ocr"] = ocr
    if vlm is not None:
        tools._cache["vlm"] = vlm
    if redact is not None:
        tools._cache["redact"] = redact
    if qa is not None:
        tools._cache["qa"] = qa


# --------------------------------------------------------------------- 流程


def test_qa_pass_first_try():
    inject(
        ocr=FakeOcrTool(),
        vlm=FakeVlmTool(),
        redact=FakeRedactionTool(),
        qa=FakeQaTool(fail_times=0),
    )
    result = build_graph().invoke(initial_state(), config={"recursion_limit": 20})

    assert result["status"] == "passed"
    assert result["retry_count"] == 0
    assert result["qa_report"].status == "passed"
    assert result["redaction_result"].redacted_blocks  # 命中了敏感块
    # 规则通道把身份证块 b2 也并了进来
    assert {sb.block_id for sb in result["vlm_analysis"].sensitive_blocks} == {"b0", "b2"}


def test_qa_retry_then_pass():
    inject(
        ocr=FakeOcrTool(),
        vlm=FakeVlmTool(),
        redact=FakeRedactionTool(),
        qa=FakeQaTool(fail_times=1),
    )
    result = build_graph().invoke(initial_state(), config={"recursion_limit": 20})

    assert result["status"] == "passed"
    assert result["retry_count"] == 1
    # 第二次 VLM 调用收到了质检反馈
    assert result["vlm_analysis"] is not None
    # VLM 按反馈修正后补了 b2（fake 语义）
    assert {sb.block_id for sb in result["vlm_analysis"].sensitive_blocks} == {"b0", "b2"}


def test_max_retry_fallback():
    inject(
        ocr=FakeOcrTool(),
        vlm=FakeVlmTool(),
        redact=FakeRedactionTool(),
        qa=FakeQaTool(fail_times=99),
    )
    result = build_graph().invoke(initial_state(), config={"recursion_limit": 20})

    assert result["status"] == "max_retry"
    assert result["retry_count"] == 3
    assert "重试" in result["feedback"]
    assert result["redaction_result"] is not None  # 保留当前脱敏结果
    assert result["qa_report"].status == "failed"


def test_qa_disabled():
    inject(
        ocr=FakeOcrTool(),
        vlm=FakeVlmTool(),
        redact=FakeRedactionTool(),
        qa=FakeQaTool(fail_times=99),
    )
    result = build_graph().invoke(
        initial_state(qa_enabled=False), config={"recursion_limit": 20}
    )

    assert result["qa_report"].status == "passed"
    assert result["qa_report"].reason == "质检已禁用"
    assert result["status"] == "passed"
    assert result["retry_count"] == 0


def test_ocr_failure_propagates_as_failed_status():
    inject(
        ocr=FakeOcrTool(exc=RuntimeError("engine crash")),
        vlm=FakeVlmTool(),
        redact=FakeRedactionTool(),
        qa=FakeQaTool(),
    )
    result = build_graph().invoke(initial_state(), config={"recursion_limit": 20})

    assert result["status"] == "failed"
    assert "engine crash" in result["error"]
    assert result["vlm_analysis"] is None
    assert result["qa_report"] is None


def test_vlm_failure_propagates():
    inject(
        ocr=FakeOcrTool(),
        vlm=FakeVlmTool(exc=ValueError("vlm timeout")),
        redact=FakeRedactionTool(),
        qa=FakeQaTool(),
    )
    result = build_graph().invoke(initial_state(), config={"recursion_limit": 20})

    assert result["status"] == "failed"
    assert "vlm timeout" in result["error"]


def test_recursion_limit_covers_worst_case():
    # 最坏路径：ocr + (max_retries+1) 轮 vlm/redact/qa + fallback
    worst = 1 + 3 * (3 + 1) + 1
    assert compute_recursion_limit(3) > worst


# --------------------------------------------------------------------- 路由


def test_route_after_qa_branches():
    base = initial_state()
    assert route_after_qa(base) == "end"  # report 为 None

    state = dict(base, qa_enabled=False, qa_report=object())
    assert route_after_qa(state) == "end"

    from src.schemas.qa_report import QaReport

    passed = QaReport(status="passed", reason="ok")
    assert route_after_qa(dict(base, qa_report=passed)) == "end"

    failed = QaReport(status="failed", reason="residual", feedback="fix b0")
    assert route_after_qa(dict(base, qa_report=failed, retry_count=0)) == "retry_vlm"
    assert route_after_qa(dict(base, qa_report=failed, retry_count=3)) == "fallback"


def test_fallback_node_message():
    result = fallback_node(initial_state(max_retries=3))
    assert result["status"] == "max_retry"
    assert "3" in result["feedback"]


def test_ocr_node_fills_original_image_bytes(tmp_path):
    image = tmp_path / "ticket.png"
    image.write_bytes(b"raw-bytes")
    inject(ocr=FakeOcrTool())
    result = ocr_node(initial_state(image_path=str(image), original_image=b""))
    assert result["ocr_result"].engine == "fake"
    assert result["original_image"] == b"raw-bytes"


def test_make_ocr_result_shape():
    result = make_ocr_result()
    assert [b.block_id for b in result.blocks] == ["b0", "b1", "b2"]


# --------------------------------------------------------------------- VLM 兼容性降级


def test_vlm_extra_body_degradation(tmp_path):
    """extra_body 不被服务端接受时（如 vLLM 版本不兼容），自动去掉重试。"""
    from src.config import Settings
    from src.schemas.ocr_result import OcrResult, OcrTextBlock
    from src.tools.vlm_tool import VlmTool

    settings = Settings(output_dir=str(tmp_path), vlm_enable_thinking=False)
    tool = VlmTool(settings)

    calls = {"n": 0}

    class FakeStructured:
        def invoke(self, messages):
            calls["n"] += 1
            if calls["n"] == 1 and tool._extra_body:
                raise RuntimeError("400: unexpected keyword chat_template_kwargs")
            from src.schemas.vlm_analysis import VlmAnalysis

            return VlmAnalysis(sensitive_blocks=[], model="fake")

    class FakeModel:
        def with_structured_output(self, schema, method=None):
            # 所有结构化输出必须走 ToolStrategy（function_calling）
            assert method == "function_calling"
            return FakeStructured()

    # 降级时会重建模型客户端，patch 构造方法避免真实网络调用
    tool._build_model = lambda: FakeModel()
    tool._model = FakeModel()
    result = tool.analyze(
        image=_png_bytes(tmp_path),
        ocr_result=OcrResult(engine="stub", blocks=[OcrTextBlock(block_id="b0", text="姓名：张三")]),
    )
    assert calls["n"] == 2  # 第一次带 extra_body 失败，第二次去掉后成功
    assert tool._extra_body == {}  # 已降级
    assert tool._extra_body_disabled is True
    # analyze 会用配置的模型名覆写（符合代码逻辑）
    assert result.model == settings.vlm_model


def test_vlm_extra_body_not_disabled_when_thinking_on(tmp_path):
    """vlm_enable_thinking=True 时本就不带 extra_body，无需降级。"""
    from src.config import Settings
    from src.tools.vlm_tool import VlmTool

    tool = VlmTool(Settings(output_dir=str(tmp_path), vlm_enable_thinking=True))
    assert tool._extra_body == {}


def test_vlm_structured_output_uses_tool_strategy(tmp_path):
    """所有 LLM 结构化输出必须显式走 ToolStrategy（function_calling）。"""
    from src.config import Settings
    from src.schemas.ocr_result import OcrResult, OcrTextBlock
    from src.schemas.qa_verdict import QaVerdict
    from src.schemas.vlm_analysis import VlmAnalysis
    from src.tools.vlm_tool import VlmTool

    tool = VlmTool(Settings(output_dir=str(tmp_path), vlm_enable_thinking=True))
    seen = []

    class FakeStructured:
        def __init__(self, schema):
            self._schema = schema

        def invoke(self, messages):
            if self._schema is QaVerdict:
                return QaVerdict(status="passed", reason="ok")
            return VlmAnalysis(sensitive_blocks=[], model="fake")

    class FakeModel:
        def with_structured_output(self, schema, method=None):
            seen.append(method)
            return FakeStructured(schema)

    tool._build_model = lambda: FakeModel()
    tool._model = FakeModel()

    img = _png_bytes(tmp_path)
    ocr = OcrResult(engine="stub", blocks=[OcrTextBlock(block_id="b0", text="姓名：张三")])
    analysis = tool.analyze(image=img, ocr_result=ocr)
    tool.qa_verify(
        original_image=img, redacted_image=img, vlm_analysis=analysis, ocr_result=ocr
    )

    # 敏感判断与质检两条路径都必须走 ToolStrategy
    assert seen == ["function_calling", "function_calling"]


@pytest.mark.parametrize(
    "raw, expected",
    [
        ('{"status": "passed"}', {"status": "passed"}),
        ('[{"status": "passed"}]', {"status": "passed"}),
        # 多元素数组（模型偶发把对象包成 list 且附多份）
        ('[{"status": "passed"}, {"status": "failed"}]', {"status": "passed"}),
        # 代码块包裹
        ('```json\n{"status": "passed"}\n```', {"status": "passed"}),
        # 前后夹带说明文字
        ('说明文字 {"status": "passed"} 后缀', {"status": "passed"}),
        # 多个对象拼接（曾导致 JSONDecodeError: Extra data）
        ('{"a": 1}{"b": 2}', {"a": 1}),
    ],
)
def test_extract_json_robust(raw, expected):
    from src.tools.vlm_tool import VlmTool

    assert VlmTool._extract_json(raw) == expected


def test_extract_json_empty_array_raises():
    from src.tools.vlm_tool import VlmTool

    with pytest.raises(ValueError):
        VlmTool._extract_json("[]")


def test_qa_failure_without_feedback_still_terminates():
    """质检失败但未给出 feedback 时，仍要自增 retry_count 并最终 max_retry，
    不能无限回退到 vlm 撞 recursion_limit。"""
    from src.schemas.qa_report import QaReport

    class NoFeedbackQa:
        def __init__(self):
            self.calls = 0

        def verify(self, **kwargs):
            self.calls += 1
            return QaReport(status="failed", reason="漏检", feedback=None)

    qa = NoFeedbackQa()
    inject(ocr=FakeOcrTool(), vlm=FakeVlmTool(), redact=FakeRedactionTool(), qa=qa)

    max_retries = 2
    result = build_graph().invoke(
        initial_state(max_retries=max_retries),
        config={"recursion_limit": compute_recursion_limit(max_retries)},
    )

    assert result["status"] == "max_retry"
    assert qa.calls == max_retries + 1  # 初始 1 轮 + 重试轮


# --------------------------------------------------------------------- 结构化输出重试


def test_structured_output_retries_then_succeeds(tmp_path):
    """坏结构化字段前两次失败，第 3 次成功，不得直接 fail。"""
    from langchain_core.messages import HumanMessage

    from src.schemas.qa_verdict import QaVerdict
    from src.tools.vlm_tool import VlmTool

    class FlakyStructured:
        def __init__(self):
            self.calls = 0

        def invoke(self, messages):
            self.calls += 1
            if self.calls < 3:
                QaVerdict.model_validate({})  # 缺必填 status → ValidationError
            return QaVerdict(status="passed", reason="ok")

    flaky = FlakyStructured()

    class FakeModel:
        def with_structured_output(self, schema, method=None):
            assert method == "function_calling"
            return flaky

    tool = VlmTool(Settings(output_dir=str(tmp_path), vlm_enable_thinking=True))
    tool._model = FakeModel()

    result = tool._invoke_structured([HumanMessage(content="x")], QaVerdict)
    assert result.status == "passed"
    assert flaky.calls == 3


def test_structured_output_exhausts_retries_raises(tmp_path):
    """连续失败且 JSON 兜底也失败时，抛 StructuredOutputError 供节点降级。"""
    from types import SimpleNamespace

    from langchain_core.messages import HumanMessage

    from src.schemas.qa_verdict import QaVerdict
    from src.tools.vlm_tool import (
        STRUCTURED_OUTPUT_ATTEMPTS,
        StructuredOutputError,
        VlmTool,
    )

    class BadStructured:
        def __init__(self):
            self.calls = 0

        def invoke(self, messages):
            self.calls += 1
            QaVerdict.model_validate({})

    bad = BadStructured()

    class FakeModel:
        def with_structured_output(self, schema, method=None):
            return bad

        def invoke(self, messages):
            return SimpleNamespace(content="not-a-json")

    tool = VlmTool(Settings(output_dir=str(tmp_path), vlm_enable_thinking=True))
    tool._model = FakeModel()

    with pytest.raises(StructuredOutputError):
        tool._invoke_structured([HumanMessage(content="x")], QaVerdict)
    assert bad.calls == STRUCTURED_OUTPUT_ATTEMPTS


def test_vlm_structured_failure_degrades_to_rules():
    """结构化彻底失败时，vlm 节点退化为规则通道，而不是整图 failed。"""
    from src.tools.vlm_tool import StructuredOutputError

    class BadVlm(FakeVlmTool):
        def analyze(self, **kwargs):
            raise StructuredOutputError("boom")

    inject(ocr=FakeOcrTool(), vlm=BadVlm(), redact=FakeRedactionTool(), qa=FakeQaTool())

    result = build_graph().invoke(
        initial_state(), config={"recursion_limit": compute_recursion_limit(3)}
    )

    assert result["status"] == "passed"
    assert result["vlm_analysis"].model == "rule-only"
    # 规则通道仍然脱掉了身份证号块 b2
    assert any(sb.block_id == "b2" for sb in result["vlm_analysis"].sensitive_blocks)


# --------------------------------------------------------------------- 回退消息要素


def test_analyze_retry_messages_include_mask_feedback_and_ocr(tmp_path):
    """回退 VLM 时必须带：原图 + 当前 mask 图 + OCR(block_id/text/bbox) + 质检意见。"""
    from src.schemas.ocr_result import OcrResult, OcrTextBlock
    from src.schemas.redaction_result import RedactionResult
    from src.tools.vlm_tool import VlmTool

    tool = VlmTool(Settings(output_dir=str(tmp_path), vlm_enable_thinking=True))
    png = _png_bytes(tmp_path)
    ocr = OcrResult(
        engine="stub",
        blocks=[OcrTextBlock(block_id="b0", text="姓名：张三", bbox=[1, 2, 3, 4])],
    )
    rr = RedactionResult(redacted_image=png, output_path=None)

    msgs = tool._build_analyze_messages(
        image=png, ocr_result=ocr, previous_redaction=rr, feedback="块 b0 仍残留"
    )
    content = msgs[1].content
    images = [c for c in content if c["type"] == "image_url"]
    texts = " ".join(c["text"] for c in content if c["type"] == "text")

    assert len(images) == 2                      # 原图 + 当前 mask 图
    assert "block_id" in texts and "b0" in texts  # OCR 结果 JSON
    assert "块 b0 仍残留" in texts                 # 质检意见


def test_qa_messages_include_original_mask_and_prompt(tmp_path):
    """质检必须带：质检提示词 + 原图 + mask 图 + OCR/敏感块 JSON。"""
    from src.schemas.ocr_result import OcrResult, OcrTextBlock
    from src.schemas.redaction_result import RedactedBlock, RedactionResult
    from src.schemas.vlm_analysis import SensitiveBlock, VlmAnalysis
    from src.tools.vlm_tool import VlmTool

    tool = VlmTool(Settings(output_dir=str(tmp_path), vlm_enable_thinking=True))
    png = _png_bytes(tmp_path)
    ocr = OcrResult(
        engine="stub",
        blocks=[OcrTextBlock(block_id="b0", text="姓名：张三", bbox=[1, 2, 3, 4])],
    )
    rr = RedactionResult(
        redacted_image=png,
        output_path=None,
        redacted_blocks=[RedactedBlock(block_id="b0", text="姓名：李四", bbox=[1, 2, 3, 4])],
    )
    analysis = VlmAnalysis(
        sensitive_blocks=[
            SensitiveBlock(
                block_id="b0", text="姓名：张三", value="张三", kind="name", bbox=[1, 2, 3, 4]
            )
        ]
    )

    msgs = tool._build_qa_messages(
        png, png, analysis, ocr, None, {"x": 1}, rr
    )
    assert "质检" in msgs[0].content
    images = [c for c in msgs[1].content if c["type"] == "image_url"]
    texts = " ".join(c["text"] for c in msgs[1].content if c["type"] == "text")
    assert len(images) == 2          # 原图 + mask 图
    assert "block_id" in texts and "张三" in texts
