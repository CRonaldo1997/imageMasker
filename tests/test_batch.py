"""批量处理测试：glob 递归、目录结构映射、汇总报告。"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src import tools
from src.batch import (
    find_cases,
    find_images,
    map_output_path,
    run_batch,
)


@pytest.fixture(autouse=True)
def clean_tools():
    tools.reset_tools()
    yield
    tools.reset_tools()


@pytest.fixture
def sample_tree(tmp_path) -> Path:
    """构造 folder1/caseA/(img + 嵌套子目录)、folder1/caseB/img。"""
    root = tmp_path / "folder1"
    (root / "caseA" / "sub").mkdir(parents=True)
    (root / "caseB").mkdir(parents=True)
    (root / "not_a_case.txt").write_text("x", encoding="utf-8")
    for path in (
        root / "caseA" / "a1.png",
        root / "caseA" / "a2.JPG",
        root / "caseA" / "sub" / "a3.png",
        root / "caseB" / "b1.png",
        root / "caseB" / "notes.txt",
    ):
        path.write_bytes(b"fake-image")
    return root


# --------------------------------------------------------------------- 发现


def test_find_cases_only_direct_subdirs(sample_tree):
    cases = find_cases(sample_tree)
    assert [c.name for c in cases] == ["caseA", "caseB"]


def test_find_images_recursive_and_case_scoped(sample_tree):
    case_a = sample_tree / "caseA"
    images = find_images(case_a)
    names = [p.name for p in images]
    assert names == ["a1.png", "a2.JPG", "a3.png"]  # 大小写不敏感，不含 txt
    # case 之间不串味
    assert find_images(sample_tree / "caseB") == [sample_tree / "caseB" / "b1.png"]


# --------------------------------------------------------------------- 映射


def test_map_output_path_keeps_structure(sample_tree, tmp_path):
    out_root = tmp_path / "folder1_masked"
    src = sample_tree / "caseA" / "sub" / "a3.png"
    mapped = map_output_path(src, sample_tree, out_root)
    assert mapped == out_root / "caseA" / "sub" / "a3.png"  # 文件名与输入一致


def test_map_output_path_preserves_extension_case(sample_tree, tmp_path):
    src = sample_tree / "caseA" / "a2.JPG"
    mapped = map_output_path(src, sample_tree, tmp_path / "out")
    assert mapped.name == "a2.JPG"  # 扩展名原样保留


# --------------------------------------------------------------------- 批量


def _inject_fakes(qa_fail_times: int = 0):
    from tests.fakes import FakeOcrTool, FakeQaTool, FakeRedactionTool, FakeVlmTool

    tools._cache["ocr"] = FakeOcrTool()
    tools._cache["vlm"] = FakeVlmTool()
    tools._cache["redact"] = FakeRedactionTool()
    tools._cache["qa"] = FakeQaTool(fail_times=qa_fail_times)


def test_run_batch_full_flow(sample_tree, tmp_path):
    """全流程：输出目录结构、batch_report.json、状态汇总。"""
    _inject_fakes()
    out_root = tmp_path / "folder1_masked"
    report = run_batch(str(sample_tree), output_root=str(out_root), workers=1)

    assert report["summary"] == {"passed": 2}
    assert (out_root / "batch_report.json").exists()

    # 输出保持 case 名与 case 内相对路径，文件名与输入一致
    for rel in (
        "caseA/a1.png",
        "caseA/a2.JPG",
        "caseA/sub/a3.png",
        "caseB/b1.png",
    ):
        assert (out_root / rel).exists(), rel

    # 报告内容落盘且可解析
    on_disk = json.loads((out_root / "batch_report.json").read_text(encoding="utf-8"))
    assert on_disk["summary"] == {"passed": 2}
    assert len(on_disk["cases"]) == 2
    assert all(c["status"] == "passed" for c in on_disk["cases"])


def test_run_batch_parallel_workers(sample_tree, tmp_path):
    _inject_fakes()
    report = run_batch(
        str(sample_tree), output_root=str(tmp_path / "out_par"), workers=3
    )
    assert report["summary"] == {"passed": 2}


def test_run_batch_statistics(sample_tree, tmp_path):
    """统计指标：case 数、图片数、case/图片平均耗时与平均图像数。"""
    _inject_fakes()
    report = run_batch(str(sample_tree), output_root=str(tmp_path / "out_stat"), workers=1)

    stats = report["statistics"]
    assert stats["cases_processed"] == 2                       # 2 个非空 case
    assert stats["images_processed"] == 4                      # 累计 4 张图
    assert stats["avg_images_per_case"] == 2.0                 # 4 / 2
    assert stats["avg_case_duration_sec"] >= 0
    assert stats["avg_image_duration_sec"] >= 0
    # case 级耗时 = 该 case 内实际处理图片耗时之和
    for case in report["cases"]:
        assert case["duration_sec"] == pytest.approx(
            sum(img["duration_sec"] for img in case["images"]), abs=0.01
        )


def test_run_batch_statistics_skip_existing(sample_tree, tmp_path):
    """跳过的图不计入统计。"""
    _inject_fakes()
    out_root = tmp_path / "out_stat_skip"
    run_batch(str(sample_tree), output_root=str(out_root), workers=1)

    report = run_batch(
        str(sample_tree), output_root=str(out_root), workers=1, skip_existing=True
    )
    stats = report["statistics"]
    assert stats["cases_processed"] == 2
    assert stats["images_processed"] == 0                      # 全部跳过
    assert stats["avg_images_per_case"] == 0.0
    assert stats["avg_image_duration_sec"] == 0.0


def test_run_batch_statistics_empty_case(tmp_path):
    """空 case 不计入 case 数，避免除零。"""
    _inject_fakes()
    root = tmp_path / "folder_empty"
    (root / "caseX").mkdir(parents=True)
    report = run_batch(str(root), output_root=str(tmp_path / "out_empty_stat"), workers=1)
    stats = report["statistics"]
    assert stats["cases_processed"] == 0
    assert stats["images_processed"] == 0
    assert stats["avg_case_duration_sec"] == 0.0


def test_run_batch_qa_failure_marked(sample_tree, tmp_path):
    _inject_fakes(qa_fail_times=99)
    report = run_batch(str(sample_tree), output_root=str(tmp_path / "out_fail"), workers=1)
    # max_retries 耗尽 → 两个 case 均 max_retry
    assert report["summary"] == {"max_retry": 2}
    # 脱敏结果仍保留
    assert (tmp_path / "out_fail" / "caseA" / "a1.png").exists()


def test_run_batch_skip_existing(sample_tree, tmp_path):
    _inject_fakes()
    out_root = tmp_path / "out_skip"
    run_batch(str(sample_tree), output_root=str(out_root), workers=1)

    # 第一次已生成全部输出；第二次带 skip_existing 应全部跳过
    report = run_batch(
        str(sample_tree), output_root=str(out_root), workers=1, skip_existing=True
    )
    for case in report["cases"]:
        statuses = {img["status"] for img in case["images"]}
        assert statuses == {"skipped"}, case["case"]


def test_run_batch_default_output_dir_name(sample_tree):
    """未指定输出目录时，默认 <输入名>_masked 且与输入同级。"""
    _inject_fakes()
    report = run_batch(str(sample_tree), workers=1)
    expected = sample_tree.parent / "folder1_masked"
    assert Path(report["output_root"]) == expected
    assert expected.exists()


def test_run_batch_empty_case(tmp_path):
    root = tmp_path / "folder_empty"
    (root / "caseX").mkdir(parents=True)
    report = run_batch(str(root), output_root=str(tmp_path / "out_empty"), workers=1)
    assert report["summary"] == {"empty": 1}


# --------------------------------------------------------------------- 断点续传


def test_progress_file_written_incrementally(sample_tree, tmp_path):
    """每张图处理完立即落盘进度文件。"""
    _inject_fakes()
    out_root = tmp_path / "out_prog"
    run_batch(str(sample_tree), output_root=str(out_root), workers=1)
    progress_file = out_root / "batch_progress.json"
    assert progress_file.exists()
    import json

    data = json.loads(progress_file.read_text(encoding="utf-8"))
    assert data["version"] == 1
    assert data["input_root"] == str(sample_tree.resolve())
    assert set(data["images"]) == {
        "caseA/a1.png", "caseA/a2.JPG", "caseA/sub/a3.png", "caseB/b1.png",
    }
    assert all(v["status"] == "passed" for v in data["images"].values())


def test_resume_skips_passed_images(sample_tree, tmp_path):
    """中断后重跑：进度中 passed 的图片跳过，未完成的继续处理。"""
    import json

    _inject_fakes()
    out_root = tmp_path / "out_resume"
    out_root.mkdir(parents=True)
    # 预置断点：caseA/a1.png 已完成，其余未处理
    (out_root / "batch_progress.json").write_text(
        json.dumps(
            {
                "version": 1,
                "input_root": str(sample_tree.resolve()),
                "images": {
                    "caseA/a1.png": {
                        "status": "passed",
                        "output": str(out_root / "caseA" / "a1.png"),
                    }
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    report = run_batch(str(sample_tree), output_root=str(out_root), workers=1)
    assert report["resumed"] is True
    assert report["resumed_skipped"] == 1
    case_a = next(c for c in report["cases"] if c["case"] == "caseA")
    statuses = {Path(img["image"]).name: img["status"] for img in case_a["images"]}
    assert statuses["a1.png"] == "skipped"
    assert statuses["a2.JPG"] == "passed"
    assert statuses["a3.png"] == "passed"
    # 最终报告全部 passed（skipped 也算完成）
    assert report["summary"] == {"passed": 2}


def test_resume_retries_failed_images(sample_tree, tmp_path):
    """进度中 failed 的图片会被重试（默认不跳过失败项）。"""
    import json

    _inject_fakes()
    out_root = tmp_path / "out_retry"
    out_root.mkdir(parents=True)
    (out_root / "batch_progress.json").write_text(
        json.dumps(
            {
                "version": 1,
                "input_root": str(sample_tree.resolve()),
                "images": {
                    "caseA/a1.png": {"status": "failed", "error": "timeout"},
                    "caseA/a2.JPG": {"status": "passed"},
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    report = run_batch(str(sample_tree), output_root=str(out_root), workers=1)
    case_a = next(c for c in report["cases"] if c["case"] == "caseA")
    statuses = {Path(img["image"]).name: img["status"] for img in case_a["images"]}
    assert statuses["a1.png"] == "passed"  # 失败项被重试并成功
    assert statuses["a2.JPG"] == "skipped"


def test_no_resume_ignores_progress(sample_tree, tmp_path):
    """--no-resume 忽略断点强制全量重跑。"""
    import json

    _inject_fakes()
    out_root = tmp_path / "out_no_resume"
    out_root.mkdir(parents=True)
    (out_root / "batch_progress.json").write_text(
        json.dumps(
            {
                "version": 1,
                "input_root": str(sample_tree.resolve()),
                "images": {"caseA/a1.png": {"status": "passed"}},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    report = run_batch(str(sample_tree), output_root=str(out_root), workers=1, resume=False)
    assert report["resumed"] is False
    assert report["resumed_skipped"] == 0
    assert report["summary"] == {"passed": 2}


def test_corrupted_progress_falls_back(sample_tree, tmp_path):
    """进度文件损坏时不崩溃，按无断点全量处理。"""
    _inject_fakes()
    out_root = tmp_path / "out_corrupt"
    out_root.mkdir(parents=True)
    (out_root / "batch_progress.json").write_text("{broken json", encoding="utf-8")

    report = run_batch(str(sample_tree), output_root=str(out_root), workers=1)
    assert report["summary"] == {"passed": 2}
    assert report["resumed_skipped"] == 0


def test_progress_bar_tracks_all_images(sample_tree, tmp_path, monkeypatch):
    """进度条总数 = 全部图片数；跳过的图也计数；结束时必须关闭。"""
    from src import batch as batch_mod

    class FakeBar:
        def __init__(self):
            self.count = 0
            self.closed = False

        def update(self, n=1):
            self.count += n

        def close(self):
            self.closed = True

    totals = []
    bar = FakeBar()
    monkeypatch.setattr(
        batch_mod, "_make_progress_bar",
        lambda total: (totals.append(total), bar)[1],
    )
    _inject_fakes()

    out_root = tmp_path / "out_bar"
    run_batch(str(sample_tree), output_root=str(out_root), workers=1)
    assert totals == [4]           # 2 个 case 共 4 张图
    assert bar.count == 4
    assert bar.closed is True

    # 第二轮全部走断点跳过，进度条同样要走满
    bar2 = FakeBar()
    monkeypatch.setattr(batch_mod, "_make_progress_bar", lambda total: bar2)
    run_batch(str(sample_tree), output_root=str(out_root), workers=1)
    assert bar2.count == 4
    assert bar2.closed is True


def test_run_batch_shrinks_large_images(tmp_path):
    """批量入口按阈值缩小大图，并在统计里体现张数。"""
    import os

    from PIL import Image

    _inject_fakes()
    root = tmp_path / "folder_shrink"
    (root / "caseA").mkdir(parents=True)
    img = Image.frombytes("RGB", (1200, 1200), os.urandom(1200 * 1200 * 3))
    img.save(root / "caseA" / "big.png")
    assert (root / "caseA" / "big.png").stat().st_size > 200_000

    report = run_batch(
        str(root), output_root=str(tmp_path / "out_shrink"),
        workers=1, shrink_max_bytes=200_000,
    )
    assert report["summary"] == {"passed": 1}
    assert report["statistics"]["shrunk_images"] == 1
    image_result = report["cases"][0]["images"][0]
    assert image_result["shrunk"] is True


def test_run_batch_can_disable_shrink(tmp_path):
    """shrink_max_bytes=0 时保持原图（--no-shrink 的等价行为）。"""
    import os

    from PIL import Image

    _inject_fakes()
    root = tmp_path / "folder_noshrink"
    (root / "caseA").mkdir(parents=True)
    img = Image.frombytes("RGB", (1200, 1200), os.urandom(1200 * 1200 * 3))
    img.save(root / "caseA" / "big.png")

    report = run_batch(
        str(root), output_root=str(tmp_path / "out_noshrink"),
        workers=1, shrink_max_bytes=0,
    )
    assert report["statistics"]["shrunk_images"] == 0
    assert report["cases"][0]["images"][0]["shrunk"] is False


def test_progress_input_mismatch_ignored(tmp_path):
    """进度属于另一个输入目录时忽略断点。"""
    import json

    _inject_fakes()
    root = tmp_path / "folder1"
    (root / "caseA").mkdir(parents=True)
    (root / "caseA" / "a1.png").write_bytes(b"x")
    out_root = tmp_path / "out_mismatch"
    out_root.mkdir(parents=True)
    (out_root / "batch_progress.json").write_text(
        json.dumps(
            {
                "version": 1,
                "input_root": "/some/other/folder",
                "images": {"caseA/a1.png": {"status": "passed"}},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    report = run_batch(str(root), output_root=str(out_root), workers=1)
    assert report["resumed_skipped"] == 0
    assert report["summary"] == {"passed": 1}
