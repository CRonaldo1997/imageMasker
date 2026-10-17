"""测试替身：不依赖网络与 OCR 引擎，驱动 LangGraph 全流程。"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from src.schemas.ocr_result import OcrResult, OcrTextBlock
from src.schemas.qa_report import QaReport
from src.schemas.qa_verdict import QaVerdict
from src.schemas.redaction_result import RedactedBlock, RedactionResult
from src.schemas.vlm_analysis import RedactionMethod, SensitiveBlock, VlmAnalysis


def make_ocr_result() -> OcrResult:
    return OcrResult(
        engine="fake",
        width=400,
        height=200,
        blocks=[
            OcrTextBlock(block_id="b0", text="张三", bbox=[50, 50, 150, 80], confidence=0.99),
            OcrTextBlock(block_id="b1", text="血常规", bbox=[50, 100, 150, 130], confidence=0.98),
            OcrTextBlock(
                block_id="b2", text="身份证号320123199001011234", bbox=[50, 150, 380, 180],
                confidence=0.97,
            ),
        ],
    )


class FakeOcrTool:
    """可配置抛出异常或返回固定结果。"""

    def __init__(self, exc: Optional[Exception] = None, result: Optional[OcrResult] = None):
        self.exc = exc
        self.result = result or make_ocr_result()
        self.calls = 0

    def extract(self, image_path=None, image_bytes=None) -> OcrResult:
        self.calls += 1
        if self.exc:
            raise self.exc
        return self.result


class FakeVlmTool:
    """首轮只标 b0；收到 feedback 后补上 b2（模拟按质检意见修正）。

    同时提供 qa_verify，模拟 VLM 语义质检（默认通过）。
    """

    def __init__(self, exc: Optional[Exception] = None, qa_verdict=None):
        self.exc = exc
        self.qa_verdict = qa_verdict
        self.calls = 0
        self.feedbacks: List[Optional[str]] = []
        self.qa_calls = 0

    def analyze(self, image, ocr_result, previous_redaction=None, feedback=None):
        self.calls += 1
        self.feedbacks.append(feedback)
        if self.exc:
            raise self.exc
        ids = ["b0"] if feedback is None else ["b0", "b2"]
        bbox_map = {b.block_id: list(b.bbox) for b in ocr_result.blocks}
        return VlmAnalysis(
            sensitive_blocks=[
                SensitiveBlock(
                    block_id=bid,
                    bbox=bbox_map[bid],
                    reason="fake",
                    method=RedactionMethod.WHITE_MASK,
                )
                for bid in ids
            ],
            model="fake-vlm",
        )

    def qa_verify(
        self,
        original_image,
        redacted_image,
        vlm_analysis,
        ocr_result=None,
        feedback=None,
        deterministic_hints=None,
        redaction_result=None,
    ):
        self.qa_calls += 1
        if self.exc:
            raise self.exc
        if self.qa_verdict is not None:
            return self.qa_verdict
        return QaVerdict(
            status="passed",
            reason="fake qa passed",
            revised_block_ids=[sb.block_id for sb in vlm_analysis.sensitive_blocks],
        )


class FakeRedactionTool:
    def __init__(self):
        self.calls = 0

    def apply(self, image, vlm_analysis, output_path=None) -> RedactionResult:
        self.calls += 1
        saved = output_path or "outputs/fake.png"
        Path(saved).parent.mkdir(parents=True, exist_ok=True)
        Path(saved).write_bytes(b"fake-redacted")
        return RedactionResult(
            redacted_image=b"fake-redacted",
            output_path=saved,
            width=400,
            height=200,
            redacted_blocks=[
                RedactedBlock(
                    block_id=sb.block_id,
                    text=sb.text,
                    bbox=sb.bbox,
                    method=sb.method.value,
                    source=sb.source,
                )
                for sb in vlm_analysis.sensitive_blocks
            ],
        )


class FakeQaTool:
    """前 fail_times 次质检失败，之后通过。"""

    def __init__(self, fail_times: int = 0, exc: Optional[Exception] = None):
        self.fail_times = fail_times
        self.exc = exc
        self.calls = 0

    def verify(
        self,
        original_image,
        redacted_image,
        vlm_analysis,
        ocr_result=None,
        redaction_result=None,
        feedback=None,
    ) -> QaReport:
        self.calls += 1
        if self.exc:
            raise self.exc
        if self.calls <= self.fail_times:
            return QaReport(
                status="failed",
                reason="残留敏感块：['b0']",
                feedback="块 ['b0'] 脱敏后仍可识别，请加强脱敏",
            )
        return QaReport(status="passed", reason="未检出残留敏感信息与误伤")


class StubOcrEngine:
    """给 QaTool 注入的桩引擎：重 OCR 返回固定文本块。"""

    def __init__(self, texts: List[str]):
        self.texts = texts
        self.kwargs: dict = {}
        self.last_image_bytes: Optional[bytes] = None

    def extract(self, image_bytes=None, image_path=None, **kwargs) -> OcrResult:
        self.kwargs = kwargs
        self.last_image_bytes = image_bytes
        return OcrResult(
            engine="stub",
            width=400,
            height=200,
            blocks=[
                OcrTextBlock(block_id=f"r{i}", text=t, bbox=[50, 50 + i * 40, 150, 80 + i * 40])
                for i, t in enumerate(self.texts)
            ],
        )
