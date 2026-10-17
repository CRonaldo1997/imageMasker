"""本地端到端验证：真实 RapidOCR + 真实脱敏 + 真实质检闭环。

VLM 用假工具（开发阶段未配 API Key 时也能验证除 VLM 外的全链路）。
直接运行：python tests/e2e_local.py
"""
from __future__ import annotations

import logging
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import tools  # noqa: E402
from src.graph import build_graph, compute_recursion_limit  # noqa: E402
from src.graph_state import RedactionState  # noqa: E402
from src.schemas.qa_verdict import QaVerdict  # noqa: E402
from src.schemas.vlm_analysis import RedactionMethod, SensitiveBlock, VlmAnalysis  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

IMAGE = Path(__file__).resolve().parent.parent / "data" / "sample_bill.png"

# 敏感词 → (kind, 值抽取正则)：模拟真实 VLM 的 value/kind 输出
SENSITIVE_RULES = (
    ("姓名", "name", re.compile(r"姓名[：:]\s*(.+)$")),
    ("身份证", "id_card", re.compile(r"身份证号[：:]\s*(.+)$")),
    ("手机", "mobile", re.compile(r"手机号[：:]\s*(.+)$")),
    ("医保", "medical_card", re.compile(r"医保卡号[：:]\s*(.+)$")),
    ("地址", "address", re.compile(r"地址[：:]\s*(.+)$")),
)


class HeuristicVlmTool:
    """按文本启发式模拟真实 VLM 的敏感判断（仅用于无 API Key 时的本地验证）。"""

    def analyze(self, image, ocr_result, previous_redaction=None, feedback=None):
        seen, blocks = set(), []
        for b in ocr_result.blocks:
            for keyword, kind, value_re in SENSITIVE_RULES:
                if keyword not in b.text:
                    continue
                matched = value_re.search(b.text)
                if not matched:
                    continue
                value = matched.group(1).strip()
                if b.block_id in seen or not value:
                    continue
                seen.add(b.block_id)
                blocks.append(
                    SensitiveBlock(
                        block_id=b.block_id,
                        text=b.text,
                        bbox=list(b.bbox),
                        reason=f"命中敏感词 {keyword}",
                        method=RedactionMethod.WHITE_MASK,
                        value=value,
                        kind=kind,
                    )
                )
        return VlmAnalysis(sensitive_blocks=blocks, model="heuristic-vlm")

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
        """本地模拟质检：确定性预检有硬残留即失败，否则通过。"""
        hints = deterministic_hints or {}
        if hints.get("residual_block_ids") or hints.get("pattern_hits"):
            return QaVerdict(
                status="failed",
                reason=f"预检残留：{hints}",
                feedback="存在残留敏感信息，请加强脱敏",
            )
        return QaVerdict(
            status="passed",
            reason="启发式质检通过",
            revised_block_ids=[sb.block_id for sb in vlm_analysis.sensitive_blocks],
        )


def main() -> int:
    if not IMAGE.exists():
        print(f"示例图不存在，请先运行：python main.py --make-sample")
        return 1

    # 只替换 VLM，OCR / 脱敏 / 质检全部走真实实现
    tools._cache["vlm"] = HeuristicVlmTool()

    state: RedactionState = {
        "image_path": str(IMAGE),
        "original_image": IMAGE.read_bytes(),
        "qa_enabled": True,
        "retry_count": 0,
        "max_retries": 3,
        "status": "processing",
    }
    result = build_graph().invoke(
        state, config={"recursion_limit": compute_recursion_limit(3)}
    )

    print("\n================ 端到端结果 ================")
    print("status        :", result.get("status"))
    print("retry_count   :", result.get("retry_count"))
    ocr = result.get("ocr_result")
    if ocr:
        print(f"ocr blocks    : {len(ocr.blocks)} ({ocr.engine})")
        for b in ocr.blocks:
            print(f"   {b.block_id}: {b.text!r} @ {b.bbox}")
    analysis = result.get("vlm_analysis")
    if analysis:
        print("sensitive     :", [sb.block_id for sb in analysis.sensitive_blocks])
    redaction = result.get("redaction_result")
    if redaction:
        print("redacted      :", redaction.output_path)
        print("hits          :", [(r.block_id, r.source) for r in redaction.redacted_blocks])
    report = result.get("qa_report")
    if report:
        print("qa            :", report.status, "-", report.reason)

    # 独立复核：对脱敏图重跑 OCR，确认敏感文本不再出现
    if redaction:
        from src.tools import get_ocr_tool

        re_ocr = get_ocr_tool().extract(
            image_bytes=Path(redaction.output_path).read_bytes()
        )
        texts = [b.text for b in re_ocr.blocks]
        leaked = [t for t in texts if "张三" in t or "320123199001011234" in t]
        print("re-ocr leaked :", leaked if leaked else "无（脱敏有效）")
        if leaked:
            print("端到端验证失败：脱敏图中仍可识别敏感信息")
            return 1

    ok = result.get("status") == "passed"
    print("结论          :", "通过" if ok else "未通过")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
