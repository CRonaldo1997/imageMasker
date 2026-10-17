"""调用入口：医疗票据图片脱敏 CLI。

用法：
    python main.py data/ticket.png                # 单图脱敏
    python main.py data/ticket.png --no-qa        # 关闭质检
    python main.py data/ticket.png --method blur  # 指定脱敏方式
    python main.py --make-sample                  # 生成示例票据图

退出码：0 质检通过；1 失败（含 max_retry）。
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

from src.config import PROJECT_ROOT
from src.graph import build_graph, compute_recursion_limit
from src.graph_state import RedactionState
from src.image_preprocess import DEFAULT_MAX_IMAGE_BYTES, maybe_shrink

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger("main")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="医疗票据图片隐私脱敏工具")
    parser.add_argument("image", nargs="?", help="待脱敏图片路径，或包含多个 case 的文件夹")
    parser.add_argument("--no-qa", action="store_true", help="关闭质检")
    parser.add_argument("--max-retries", type=int, default=3, help="质检最大重试次数")
    parser.add_argument(
        "--method",
        choices=["white_mask", "blur", "mosaic"],
        default=None,
        help="脱敏方式（默认取配置 DEFAULT_METHOD）",
    )
    parser.add_argument(
        "--output", default=None, help="批量模式输出目录（默认 <输入文件夹名>_masked）"
    )
    parser.add_argument(
        "--workers", type=int, default=3, help="批量模式并行 case 数（默认 3）"
    )
    parser.add_argument(
        "--skip-existing", action="store_true", help="批量模式跳过输出已存在的图片"
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="批量模式忽略断点进度，强制全量重跑（默认启用断点续传）",
    )
    parser.add_argument(
        "--max-image-mb",
        type=float,
        default=DEFAULT_MAX_IMAGE_BYTES / 1024 / 1024,
        help="超过该大小（MB）的图片先等比缩小再处理；0 或负数表示不缩小（默认 5）",
    )
    parser.add_argument(
        "--no-shrink",
        action="store_true",
        help="关闭大图缩小，按原始分辨率处理",
    )
    parser.add_argument("--make-sample", action="store_true", help="生成示例票据图后退出")
    parser.add_argument("--sample-path", default="data/sample_bill.png", help="示例图输出路径")
    return parser.parse_args()


def shrink_limit_bytes(args: argparse.Namespace) -> int:
    """把 CLI 的大图阈值换算成字节数；0 表示不缩小。"""
    if args.no_shrink or args.max_image_mb is None or args.max_image_mb <= 0:
        return 0
    return int(args.max_image_mb * 1024 * 1024)


def run_single(
    graph,
    image_path: str,
    qa_enabled: bool = True,
    max_retries: int = 3,
    output_path: Optional[str] = None,
    shrink_max_bytes: int = DEFAULT_MAX_IMAGE_BYTES,
) -> dict:
    """对单张图片执行脱敏流水线，返回最终状态。graph 由调用方提供（批量可复用）。"""
    image_bytes, _ = maybe_shrink(Path(image_path).read_bytes(), shrink_max_bytes)

    initial_state: RedactionState = {
        "image_path": image_path,
        "original_image": image_bytes,
        "qa_enabled": qa_enabled,
        "retry_count": 0,
        "max_retries": max_retries,
        "status": "processing",
    }
    if output_path:
        initial_state["output_path"] = output_path

    # recursion_limit 按 max_retries 计算，不能省略
    config = {"recursion_limit": compute_recursion_limit(max_retries)}
    result = graph.invoke(initial_state, config=config)

    _print_summary(result)
    return result


def run(
    image_path: str,
    qa_enabled: bool = True,
    max_retries: int = 3,
    shrink_max_bytes: int = DEFAULT_MAX_IMAGE_BYTES,
) -> dict:
    """单图 CLI 入口。"""
    return run_single(
        build_graph(), image_path, qa_enabled, max_retries,
        shrink_max_bytes=shrink_max_bytes,
    )


def _print_summary(state: RedactionState) -> None:
    status = state.get("status")
    logger.info("================ 运行结果 ================")
    logger.info("状态: %s", status)
    if state.get("error"):
        logger.error("错误: %s", state["error"])

    ocr_result = state.get("ocr_result")
    if ocr_result:
        logger.info("OCR[%s]: %d 个文字块", ocr_result.engine, len(ocr_result.blocks))

    analysis = state.get("vlm_analysis")
    if analysis:
        sources: dict[str, int] = {}
        for sb in analysis.sensitive_blocks:
            sources[sb.source] = sources.get(sb.source, 0) + 1
        logger.info("VLM 判断敏感块: %d 个 %s", len(analysis.sensitive_blocks), sources)

    redaction = state.get("redaction_result")
    if redaction:
        logger.info("脱敏完成: %s（命中 %d 处）", redaction.output_path, len(redaction.redacted_blocks))
    if state.get("rotation"):
        logger.info("方向矫正: 原图旋转 %d° 处理，输出已还原为原方向", state["rotation"])

    report = state.get("qa_report")
    if report:
        logger.info("质检: %s - %s", report.status, report.reason)
        if report.feedback:
            logger.info("质检反馈: %s", report.feedback)

    logger.info("重试次数: %s / %s", state.get("retry_count", 0), state.get("max_retries", 3))


def make_sample(path: str) -> Path:
    """生成一张带敏感信息的示例票据图，用于本地验证。"""
    from PIL import Image, ImageDraw, ImageFont

    target = Path(path)
    if not target.is_absolute():
        target = PROJECT_ROOT / target
    target.parent.mkdir(parents=True, exist_ok=True)

    font = _load_cjk_font(22)
    font_large = _load_cjk_font(30)

    img = Image.new("RGB", (900, 560), "white")
    draw = ImageDraw.Draw(img)
    draw.rectangle((0, 0, 900, 60), fill=(222, 235, 247))
    draw.text((30, 15), "XX大学附属第一医院  门诊收费票据", font=font_large, fill="black")

    lines = [
        "姓名：张三          性别：男        年龄：45",
        "身份证号：320123199001011234",
        "手机号：13800138000",
        "医保卡号：YB2026101600123",
        "科室：内科门诊      就诊日期：2026-10-16",
        "地址：江苏省南京市鼓楼区中山路100号",
        "项目                  数量    金额（元）",
        "血常规                  1       25.00",
        "尿常规                  1       30.00",
        "CT检查（胸部）          1      280.00",
        "合计                              335.00",
        "收款员：05           票据号：20261016001",
    ]
    y = 90
    for i, line in enumerate(lines):
        draw.text((40, y), line, font=font, fill="black")
        y += 36

    img.save(target)
    logger.info("示例票据已生成: %s", target)
    return target


def _load_cjk_font(size: int):
    from PIL import ImageFont

    candidates = [
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/simsun.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/System/Library/Fonts/PingFang.ttc",
    ]
    for candidate in candidates:
        try:
            if Path(candidate).exists():
                return ImageFont.truetype(candidate, size)
        except Exception:
            continue
    raise RuntimeError(
        "未找到中文字体，示例图生成失败；请手动准备测试图片。"
        "Linux 可安装：apt install fonts-noto-cjk"
    )


def main() -> int:
    args = parse_args()
    shrink_max_bytes = shrink_limit_bytes(args)

    if args.make_sample:
        make_sample(args.sample_path)
        return 0

    if not args.image:
        logger.error("请提供图片路径、case 文件夹，或使用 --make-sample 生成示例图")
        return 1
    target = Path(args.image)
    if not target.exists():
        logger.error("路径不存在: %s", args.image)
        return 1

    if shrink_max_bytes > 0:
        logger.info("大图预处理：超过 %.1fMB 的图片将等比缩小后处理", shrink_max_bytes / 1024 / 1024)
    else:
        logger.info("大图预处理：已关闭（按原始分辨率处理）")

    if args.method:
        import os

        os.environ["DEFAULT_METHOD"] = args.method

    if target.is_dir():
        # 批量模式：folder1（含多个 case 文件夹）→ folder1_masked
        from src.batch import run_batch

        report = run_batch(
            str(target),
            output_root=args.output,
            qa_enabled=not args.no_qa,
            max_retries=args.max_retries,
            workers=args.workers,
            skip_existing=args.skip_existing,
            resume=not args.no_resume,
            shrink_max_bytes=shrink_max_bytes,
        )
        bad = {
            k: v
            for k, v in report["summary"].items()
            if k in ("failed", "max_retry", "error")
        }
        return 1 if bad else 0

    result = run(
        str(target),
        qa_enabled=not args.no_qa,
        max_retries=args.max_retries,
        shrink_max_bytes=shrink_max_bytes,
    )
    return 0 if result.get("status") == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
