"""真实 VLM 端到端验证：ModelScope VLM + 真实 OCR + 脱敏 + VLM 质检。

需要 .env 中配置有效的 VLM_API_KEY。直接运行：
    python tests/e2e_remote.py
"""
from __future__ import annotations

import io
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from src import tools  # noqa: E402
from src.config import get_settings  # noqa: E402
from src.graph import build_graph, compute_recursion_limit  # noqa: E402
from src.graph_state import RedactionState  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

IMAGE = Path(__file__).resolve().parent.parent / "data" / "sample_bill.png"


def main() -> int:
    settings = get_settings()
    if not settings.vlm_api_key:
        print("未配置 VLM_API_KEY，请先写好 .env")
        return 1
    if not IMAGE.exists():
        print("示例图不存在，请先运行：python main.py --make-sample")
        return 1

    state: RedactionState = {
        "image_path": str(IMAGE),
        "original_image": IMAGE.read_bytes(),
        "qa_enabled": True,
        "retry_count": 0,
        "max_retries": 3,
        "status": "processing",
    }
    # 全部真实实现：OCR(rapid) + VLM(modelscope) + 脱敏(假数据) + 质检(VLM)
    result = build_graph().invoke(
        state, config={"recursion_limit": compute_recursion_limit(3)}
    )

    print("\n================ 真实 VLM 端到端 ================")
    print("status      :", result.get("status"))
    print("retry_count :", result.get("retry_count"))
    if result.get("error"):
        print("error       :", result["error"])

    analysis = result.get("vlm_analysis")
    if analysis:
        print("sensitive   :")
        for sb in analysis.sensitive_blocks:
            print(f"   {sb.block_id} [{sb.kind}/{sb.source}] {sb.value!r}")

    redaction = result.get("redaction_result")
    if redaction:
        print("redacted    :", redaction.output_path)
        for rb in redaction.redacted_blocks:
            print(f"   {rb.block_id}: {rb.text!r}")

    report = result.get("qa_report")
    if report:
        print("qa status   :", report.status)
        print("qa reason   :", report.reason)
        print("qa feedback :", report.feedback)
        print("revised     :", report.revised_block_ids)

    ok = result.get("status") == "passed"
    print("结论        :", "通过" if ok else "未通过")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
