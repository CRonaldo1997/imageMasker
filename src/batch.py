"""批量处理：递归 glob case 文件夹，输出保持相同目录结构。

目录约定：
    输入  folder1/
            ├── caseA/            # case 是 folder1 的直接子文件夹
            │   ├── img1.png      # case 内部可任意嵌套
            │   └── sub/img2.jpg
            └── caseB/...
    输出  folder1_masked/         # 默认 <输入名>_masked，与输入同级
            ├── caseA/
            │   ├── img1_masked.png
            │   └── sub/img2_masked.jpg
            └── caseB/...
    汇总  folder1_masked/batch_report.json

图片发现用 glob 递归（**/*）；并行粒度是 case（case 内图片按顺序处理），
避免同 case 内并发打爆 VLM API。

断点续传：
    每处理完一张图立即把状态增量写入 <output_root>/batch_progress.json
    （临时文件 + 原子替换，写一半断电也不会损坏）；重启时读取该文件，
    状态为 passed 的图片直接跳过，failed / max_retry / 未记录的重新处理。
    --no-resume 可忽略进度强制全量重跑。
"""
from __future__ import annotations

import glob
import json
import logging
import os
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from tqdm import tqdm

from src.graph import build_graph, compute_recursion_limit
from src.graph_state import RedactionState
from src.image_preprocess import DEFAULT_MAX_IMAGE_BYTES, maybe_shrink

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

PROGRESS_FILENAME = "batch_progress.json"
PROGRESS_VERSION = 1


@dataclass
class ImageResult:
    image: str
    status: str  # passed / failed / max_retry
    output: Optional[str] = None
    error: Optional[str] = None
    qa_reason: Optional[str] = None
    duration_sec: float = 0.0  # 该图实际处理耗时（跳过为 0）
    shrunk: bool = False  # 入口是否做了大图等比缩小


@dataclass
class CaseResult:
    case: str
    status: str = "pending"  # passed / failed / max_retry / empty / error
    images: List[ImageResult] = field(default_factory=list)
    duration_sec: float = 0.0  # 该 case 实际处理耗时（不含跳过图，empty 为 0）


def find_cases(input_root: Path) -> List[Path]:
    """folder1 的直接子文件夹即 case。"""
    return sorted(p for p in input_root.iterdir() if p.is_dir())


def find_images(case_dir: Path) -> List[Path]:
    """glob 递归收集 case 内所有图片（大小写不敏感的扩展名）。"""
    pattern = str(case_dir / "**" / "*")
    files = glob.glob(pattern, recursive=True)
    return sorted(
        Path(f) for f in files if Path(f).is_file() and Path(f).suffix.lower() in IMAGE_EXTENSIONS
    )


def map_output_path(image_path: Path, input_root: Path, output_root: Path) -> Path:
    """输入图片 → 输出路径：完全镜像输入的相对路径与文件名。

    输出根目录已带 _masked 标识（folder1 → folder1_masked），
    内部文件不再加后缀，保持与原目录结构一一对应。
    """
    return output_root / image_path.relative_to(input_root)


def _make_progress_bar(total: int):
    """终端进度条；输出被重定向到文件（nohup/CI）时自动关闭，避免刷日志。"""
    return tqdm(
        total=total,
        desc="脱敏进度",
        unit="img",
        file=sys.stdout,
        dynamic_ncols=True,
        leave=True,
        disable=total <= 0 or not sys.stdout.isatty(),
    )


def run_batch(
    input_root: str,
    output_root: Optional[str] = None,
    qa_enabled: bool = True,
    max_retries: int = 3,
    workers: int = 3,
    skip_existing: bool = False,
    resume: bool = True,
    shrink_max_bytes: int = DEFAULT_MAX_IMAGE_BYTES,
) -> dict:
    """批量脱敏入口。返回汇总报告（同时写入 batch_report.json）。

    resume=True（默认）：读取断点进度文件，passed 的图片跳过，
    实现断点续传；中途中断后重跑同一命令即可接着处理。
    shrink_max_bytes：超过该字节数的图片在入口等比缩小（<=0 表示不缩小）。
    """
    root = Path(input_root).resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"输入不是文件夹：{input_root}")

    out_root = (
        Path(output_root).resolve()
        if output_root
        else root.parent / f"{root.name}_masked"
    )
    out_root.mkdir(parents=True, exist_ok=True)

    cases = find_cases(root)
    logger.info("输入 %s 下发现 %d 个 case，输出目录 %s", root, len(cases), out_root)
    if not cases:
        return _build_report(root, out_root, [], 0.0, 0, False)

    # 图片清单只扫一次：进度条需要总数，各 case 复用同一份结果
    case_images = {case_dir: find_images(case_dir) for case_dir in cases}
    total_images = sum(len(images) for images in case_images.values())
    logger.info(
        "共发现 %d 张图片（%d 个 case，并行 %d）",
        total_images, len(cases), max(1, workers),
    )
    bar = _make_progress_bar(total_images)

    # ---- 断点进度 ----
    progress: Dict[str, dict] = {}
    resumed_from_progress = False
    if resume:
        loaded = _load_progress(out_root, root)
        if loaded:
            progress = loaded
            resumed_from_progress = True
            done = sum(1 for v in progress.values() if v.get("status") == "passed")
            logger.info("检测到断点进度：%d 张图片已完成，将跳过", done)
    lock = threading.Lock()

    graph = build_graph()
    started = time.time()
    results: List[CaseResult] = []

    try:
        if workers <= 1:
            for case_dir in cases:
                results.append(
                    _process_case(
                        graph, case_dir, root, out_root, qa_enabled, max_retries,
                        skip_existing, progress, lock,
                        images=case_images[case_dir],
                        bar=bar,
                        shrink_max_bytes=shrink_max_bytes,
                    )
                )
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {
                    pool.submit(
                        _process_case,
                        graph,
                        case_dir,
                        root,
                        out_root,
                        qa_enabled,
                        max_retries,
                        skip_existing,
                        progress,
                        lock,
                        images=case_images[case_dir],
                        bar=bar,
                        shrink_max_bytes=shrink_max_bytes,
                    ): case_dir
                    for case_dir in cases
                }
                for future in as_completed(futures):
                    case_dir = futures[future]
                    try:
                        results.append(future.result())
                    except Exception as exc:  # case 级异常不中断整批
                        logger.exception("case %s 处理异常", case_dir.name)
                        results.append(
                            CaseResult(case=case_dir.name, status="error", images=[
                                ImageResult(image=str(case_dir), status="failed", error=str(exc))
                            ])
                        )
            results.sort(key=lambda r: r.case)
    finally:
        bar.close()

    duration = time.time() - started
    skipped = sum(
        1
        for case in results
        for img in case.images
        if img.status == "skipped"
    )
    report = _build_report(root, out_root, results, duration, skipped, resumed_from_progress)
    _log_summary(report)
    return report


def _process_case(
    graph,
    case_dir: Path,
    input_root: Path,
    output_root: Path,
    qa_enabled: bool,
    max_retries: int,
    skip_existing: bool,
    progress: Dict[str, dict],
    lock: threading.Lock,
    images: Optional[List[Path]] = None,
    bar=None,
    shrink_max_bytes: int = DEFAULT_MAX_IMAGE_BYTES,
) -> CaseResult:
    images = images if images is not None else find_images(case_dir)
    result = CaseResult(case=case_dir.name)
    if not images:
        result.status = "empty"
        logger.info("case %s：无图片，跳过", case_dir.name)
        return result

    for image_path in images:
        output_path = map_output_path(image_path, input_root, output_root)
        # 进度 key 统一用 POSIX 分隔符，保证跨平台可移植
        key = image_path.relative_to(input_root).as_posix()

        # 断点续传：已成功的图片直接跳过
        record = progress.get(key)
        if record and record.get("status") == "passed":
            logger.info("断点跳过（已完成）：%s", image_path)
            result.images.append(
                ImageResult(
                    image=str(image_path),
                    status="skipped",
                    output=record.get("output") or str(output_path),
                    qa_reason=record.get("qa_reason"),
                )
            )
            _tick(bar)
            continue

        if skip_existing and output_path.exists():
            logger.info("跳过已存在输出：%s", output_path)
            result.images.append(
                ImageResult(image=str(image_path), status="skipped", output=str(output_path))
            )
            _tick(bar)
            continue

        image_result = _process_image(
            graph, image_path, output_path, qa_enabled, max_retries, shrink_max_bytes
        )
        result.images.append(image_result)
        _tick(bar)

        # 每处理完一张立即增量落盘（原子替换，中断不丢已完成进度）
        with lock:
            progress[key] = {
                "status": image_result.status,
                "output": image_result.output,
                "error": image_result.error,
                "qa_reason": image_result.qa_reason,
                "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            _save_progress(output_root, input_root, progress)

    statuses = {ir.status for ir in result.images}
    if "failed" in statuses:
        result.status = "failed"
    elif "max_retry" in statuses:
        result.status = "max_retry"
    elif statuses == {"skipped"}:
        result.status = "skipped"
    else:
        result.status = "passed"
    # case 耗时 = 实际处理的图片耗时之和（跳过的图不计入）
    result.duration_sec = round(
        sum(ir.duration_sec for ir in result.images if ir.status != "skipped"), 3
    )
    return result


def _tick(bar) -> None:
    """推进度条：跳过的图片也计数，否则进度条永远走不满。"""
    if bar is not None:
        bar.update(1)


def _process_image(
    graph,
    image_path: Path,
    output_path: Path,
    qa_enabled: bool,
    max_retries: int,
    shrink_max_bytes: int = DEFAULT_MAX_IMAGE_BYTES,
) -> ImageResult:
    logger.info("处理 %s → %s", image_path, output_path)
    started = time.time()
    try:
        # 大图在入口统一等比缩小：OCR/VLM/脱敏/质检全链路共用缩小后的字节，
        # 坐标天然自洽，无需回换算。
        image_bytes, _shrink_info = maybe_shrink(
            image_path.read_bytes(), shrink_max_bytes
        )
        initial_state: RedactionState = {
            "image_path": str(image_path),
            "original_image": image_bytes,
            "output_path": str(output_path),
            "qa_enabled": qa_enabled,
            "retry_count": 0,
            "max_retries": max_retries,
            "status": "processing",
        }
        result = graph.invoke(
            initial_state, config={"recursion_limit": compute_recursion_limit(max_retries)}
        )
    except Exception as exc:
        logger.exception("图片 %s 处理异常", image_path)
        return ImageResult(
            image=str(image_path), status="failed", error=str(exc),
            duration_sec=round(time.time() - started, 3),
        )

    redaction = result.get("redaction_result")
    report = result.get("qa_report")
    return ImageResult(
        image=str(image_path),
        status=result.get("status", "failed"),
        output=str(redaction.output_path) if redaction else None,
        error=result.get("error"),
        qa_reason=report.reason if report else None,
        duration_sec=round(time.time() - started, 3),
        shrunk=_shrink_info is not None,
    )


def _progress_path(output_root: Path) -> Path:
    return output_root / PROGRESS_FILENAME


def _load_progress(output_root: Path, input_root: Path) -> Dict[str, dict]:
    """加载断点进度。文件损坏或与本次输入不匹配时按空进度处理（不阻断）。"""
    path = _progress_path(output_root)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("version") != PROGRESS_VERSION:
            logger.warning("进度文件版本不匹配，忽略断点：%s", path)
            return {}
        if data.get("input_root") != str(input_root):
            logger.warning(
                "进度文件对应输入 %s 与本次 %s 不一致，忽略断点",
                data.get("input_root"), input_root,
            )
            return {}
        images = data.get("images", {})
        if not isinstance(images, dict):
            logger.warning("进度文件 images 字段异常，忽略断点")
            return {}
        return images
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("进度文件损坏，按无断点处理：%s（%s）", path, exc)
        return {}


def _save_progress(output_root: Path, input_root: Path, progress: Dict[str, dict]) -> None:
    """原子写入：先写临时文件再 os.replace，写一半断电也不会损坏旧进度。"""
    path = _progress_path(output_root)
    payload = {
        "version": PROGRESS_VERSION,
        "input_root": str(input_root),
        "output_root": str(output_root),
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "images": progress,
    }
    fd, tmp_name = tempfile.mkstemp(dir=str(output_root), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=1)
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _build_report(
    input_root: Path,
    output_root: Path,
    results: List[CaseResult],
    duration: float,
    skipped: int = 0,
    resumed: bool = False,
) -> dict:
    summary: Dict[str, int] = {}
    for case in results:
        summary[case.status] = summary.get(case.status, 0) + 1

    # ---- 统计指标 ----
    processed_cases = [c for c in results if c.status != "empty"]           # 实际处理的 case
    processed_images = [ir for c in results for ir in c.images if ir.status != "skipped"]  # 实际处理的图
    case_duration = sum(c.duration_sec for c in processed_cases)
    image_duration = sum(ir.duration_sec for ir in processed_images)
    statistics = {
        "cases_processed": len(processed_cases),                            # 处理了多少 case
        "images_processed": len(processed_images),                          # 累计处理了多少图
        "shrunk_images": sum(1 for ir in processed_images if ir.shrunk),    # 其中做了大图缩小的张数
        "avg_case_duration_sec": round(case_duration / len(processed_cases), 2) if processed_cases else 0.0,   # 平均每个 case 处理时长
        "avg_images_per_case": round(len(processed_images) / len(processed_cases), 2) if processed_cases else 0.0,  # 平均每个 case 图像数
        "avg_image_duration_sec": round(image_duration / len(processed_images), 2) if processed_images else 0.0,  # 平均每张图处理时长
    }

    report = {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "duration_sec": round(duration, 1),
        "resumed": resumed,
        "resumed_skipped": skipped,
        "summary": summary,
        "statistics": statistics,
        "cases": [asdict(case) for case in results],
    }
    report_path = output_root / "batch_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    report["report_path"] = str(report_path)
    return report


def _log_summary(report: dict) -> None:
    logger.info("================ 批量处理汇总 ================")
    logger.info("输入：%s", report["input_root"])
    logger.info("输出：%s", report["output_root"])
    logger.info(
        "耗时：%.1fs，case 状态分布：%s",
        report["duration_sec"], report["summary"],
    )
    if report.get("resumed_skipped"):
        logger.info("断点续传：跳过已完成图片 %d 张", report["resumed_skipped"])
    stats = report.get("statistics", {})
    if stats:
        logger.info("---------------- 统计指标 ----------------")
        logger.info("处理 case 数：%d", stats["cases_processed"])
        logger.info("累计处理图片：%d 张", stats["images_processed"])
        if stats.get("shrunk_images"):
            logger.info("其中大图缩小：%d 张", stats["shrunk_images"])
        logger.info("平均每个 case 处理时长：%.2fs", stats["avg_case_duration_sec"])
        logger.info("平均每个 case 图像数：%.2f 张", stats["avg_images_per_case"])
        logger.info("平均每张图处理时长：%.2fs", stats["avg_image_duration_sec"])
    logger.info("报告：%s", report.get("report_path"))
    for case in report["cases"]:
        if case["status"] not in ("passed", "skipped"):
            logger.warning("需关注：case %s 状态 %s", case["case"], case["status"])
