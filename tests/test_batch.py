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


def test_find_images_with_glob_metachars_in_case_name(tmp_path):
    """case 目录名带 [ ] * ? 时也必须能发现图片。

    glob.glob(case_dir/**/*) 会把整个路径当模式解析，"case[1]" 这种
    目录会静默匹配不到图片（变成 empty case）；rglob 只通配最后一段。
    """
    case = tmp_path / "case[1]"
    (case / "sub").mkdir(parents=True)
    (case / "a.png").write_bytes(b"x")
    (case / "sub" / "b.jpg").write_bytes(b"x")

    assert [p.name for p in find_images(case)] == ["a.png", "b.jpg"]


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


def test_tool_singleton_is_thread_safe():
    """并发首次取用必须返回同一实例。

    锁要覆盖「检查-创建」整个过程，否则多个 case 线程可能各创建一份
    OCR 引擎（双倍内存）或 VLM 客户端（降级状态不一致）。
    """
    import threading

    from src import tools

    tools.reset_tools()
    try:
        seen: list = []
        lock = threading.Lock()

        def grab() -> None:
            tool = tools.get_ocr_tool()
            with lock:
                seen.append(id(tool))

        threads = [threading.Thread(target=grab) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(set(seen)) == 1
        assert tools.get_qa_tool() is tools._cache["qa"]  # qa 门面内部复用同一实例
    finally:
        tools.reset_tools()


# --------------------------------------------------------------------- 中断与续跑


#: 子进程脚本：注入假工具后跑批，处理完第 N 张图立刻 SIGKILL 自己，
#: 模拟「跑到一半进程被杀」的最坏情况（连 finally 都没机会执行）。
_INTERRUPT_CHILD = '''
import os, signal, sys
from pathlib import Path

sys.path.insert(0, {repo!r})
sys.path.insert(0, str(Path({repo!r}) / "tests"))

from src import tools
from src.batch import _save_progress, run_batch
from tests.fakes import FakeOcrTool, FakeQaTool, FakeRedactionTool, FakeVlmTool

tools._cache["ocr"] = FakeOcrTool()
tools._cache["vlm"] = FakeVlmTool()
tools._cache["redact"] = FakeRedactionTool()
tools._cache["qa"] = FakeQaTool()

counter = {{"n": 0}}

def killing_save_progress(*args, **kwargs):
    counter["n"] += 1
    _save_progress(*args, **kwargs)
    if counter["n"] >= {kill_after}:
        # 进度已落盘、报告还没写：此刻强杀进程
        hard_kill = getattr(signal, "SIGKILL", signal.SIGTERM)
        os.kill(os.getpid(), hard_kill)

import src.batch as batch_module
batch_module._save_progress = killing_save_progress

run_batch({root!r}, output_root={out!r}, workers={workers!r})
print("SHOULD_NOT_REACH_HERE")
'''


def _write_interrupt_child(tmp_path: Path, repo: Path, kill_after: int, workers: int) -> Path:
    root = tmp_path / "folder_kill"
    out = tmp_path / "out_kill"
    for c in range(2):
        case = root / f"case{c}"
        case.mkdir(parents=True)
        for i in range(3):
            (case / f"img{i}.png").write_bytes(b"fake-image")
    script = tmp_path / "child_kill.py"
    script.write_text(
        _INTERRUPT_CHILD.format(
            repo=str(repo), root=str(root), out=str(out),
            kill_after=kill_after, workers=workers,
        ),
        encoding="utf-8",
    )
    return script, root, out


def test_kill_mid_run_then_resume_completes(tmp_path):
    """SIGKILL 杀掉跑批进程 → 重跑同一条命令必须只补剩下的图。

    覆盖最恶劣的中断：子进程在第 N 张图进度落盘后被 kill -9，
    finally 没执行、batch_report.json 没写、tqdm 没关闭。
    """
    import subprocess
    import sys

    repo = Path(__file__).resolve().parent.parent
    script, root, out_root = _write_interrupt_child(tmp_path, repo, kill_after=3, workers=1)

    proc = subprocess.run(
        [sys.executable, str(script)], capture_output=True, text=True, timeout=300
    )
    # POSIX 上是 -9（SIGKILL），Windows 上 TerminateProcess 也是立刻死、
    # 不跑 finally；两者共同特征是：非 0 退出且没打印"跑完了"
    assert proc.returncode != 0, f"子进程应被强杀，实际 {proc.returncode}"
    assert "SHOULD_NOT_REACH_HERE" not in proc.stdout

    # 被杀时的现场：进度文件存在且只有已完成的那几张，报告不存在
    progress_file = out_root / "batch_progress.json"
    assert progress_file.exists()
    saved = json.loads(progress_file.read_text(encoding="utf-8"))
    assert len(saved["images"]) == 3
    assert all(v["status"] == "passed" for v in saved["images"].values())
    assert not (out_root / "batch_report.json").exists()

    # 续跑：同样的命令（默认 resume=True）
    _inject_fakes()
    report = run_batch(str(root), output_root=str(out_root), workers=1)

    assert report["resumed"] is True
    assert report["resumed_skipped"] == 3          # 只跳过已完成的 3 张
    # case0 的 3 张在中断前已全部完成 → 整个 case 记为 skipped；
    # case1 的 3 张是这次补的 → passed。没有 failed/max_retry。
    assert report["summary"] == {"skipped": 1, "passed": 1}
    assert report["statistics"]["images_processed"] == 3  # 只补剩下的 3 张

    # 6 张图全部有产物，进度文件记录 6 条 passed
    for case in report["cases"]:
        for img in case["images"]:
            assert Path(img["output"]).exists()
    final = json.loads(progress_file.read_text(encoding="utf-8"))
    assert len(final["images"]) == 6
    assert all(v["status"] == "passed" for v in final["images"].values())
    assert (out_root / "batch_report.json").exists()


#: 子进程脚本：并行跑批中在第 N 张图落盘后抛 KeyboardInterrupt（等价于 Ctrl+C），
#: 打印实际耗时——用来验证「中断后不会把剩下所有 case 都跑完才退出」。
_CTRLC_CHILD = '''
import sys, time
from pathlib import Path

sys.path.insert(0, {repo!r})
sys.path.insert(0, str(Path({repo!r}) / "tests"))

from src import tools
from src.batch import _save_progress, run_batch
from tests.fakes import FakeOcrTool, FakeQaTool, FakeRedactionTool, FakeVlmTool

class SlowVlm(FakeVlmTool):
    def analyze(self, *a, **kw):
        time.sleep(0.4)
        return super().analyze(*a, **kw)
    def qa_verify(self, *a, **kw):
        time.sleep(0.4)
        return super().qa_verify(*a, **kw)

tools._cache["ocr"] = FakeOcrTool()
tools._cache["vlm"] = SlowVlm()
tools._cache["redact"] = FakeRedactionTool()
tools._cache["qa"] = FakeQaTool()

counter = {{"n": 0}}

def interrupting_save_progress(*args, **kwargs):
    counter["n"] += 1
    _save_progress(*args, **kwargs)
    if counter["n"] >= {kill_after}:
        raise KeyboardInterrupt("simulated ctrl-c")

import src.batch as batch_module
batch_module._save_progress = interrupting_save_progress

started = time.time()
try:
    run_batch({root!r}, output_root={out!r}, workers={workers!r})
except KeyboardInterrupt:
    print(f"INTERRUPTED_AFTER {{time.time() - started:.1f}}s")
    sys.exit(130)
sys.exit(0)
'''


def test_ctrl_c_mid_run_stops_promptly_and_keeps_progress(tmp_path):
    """并行跑批中按 Ctrl+C：必须尽快退出、进度不丢、不写报告。

    历史问题：ThreadPoolExecutor 的 with 语句会等所有已提交 case 跑完，
    100 个 case 时 Ctrl+C 要等几小时；现在取消排队中的 case 只让在跑的收尾。
    """
    import subprocess
    import sys

    repo = Path(__file__).resolve().parent.parent
    root = tmp_path / "folder_ctrlc"
    out_root = tmp_path / "out_ctrlc"
    for c in range(10):  # 10 个 case，一次只跑 2 个
        case = root / f"case{c}"
        case.mkdir(parents=True)
        for i in range(2):
            (case / f"img{i}.png").write_bytes(b"fake-image")

    script = tmp_path / "child_ctrlc.py"
    script.write_text(
        _CTRLC_CHILD.format(
            repo=str(repo), root=str(root), out=str(out_root),
            kill_after=2, workers=2,
        ),
        encoding="utf-8",
    )

    proc = subprocess.run(
        [sys.executable, str(script)], capture_output=True, text=True, timeout=300
    )
    assert proc.returncode == 130, f"应按 Ctrl+C 语义退出，实际 {proc.returncode}"
    assert "INTERRUPTED_AFTER" in proc.stdout
    elapsed = float(proc.stdout.split("INTERRUPTED_AFTER")[1].split("s")[0])
    # 全部跑完约 16s（10 case / 2 并发 × 2 图 × 0.8s）；取消后应远小于此
    assert elapsed < 8.0, f"中断后仍耗时 {elapsed}s，说明没取消排队中的 case"

    # 中断现场：进度只包含已落盘的那几张，报告没写
    progress_file = out_root / "batch_progress.json"
    saved = json.loads(progress_file.read_text(encoding="utf-8"))
    assert 0 < len(saved["images"]) < 20
    assert not (out_root / "batch_report.json").exists()

    # 续跑把剩下的补完
    _inject_fakes()
    report = run_batch(str(root), output_root=str(out_root), workers=2)
    assert report["summary"].get("failed", 0) == 0
    assert report["statistics"]["images_processed"] + report["resumed_skipped"] == 20
    assert len(json.loads(progress_file.read_text(encoding="utf-8"))["images"]) == 20


#: 子进程脚本：慢速跑批，等外部发来 SIGTERM（systemctl stop / kill）后死掉。
_SIGTERM_CHILD = '''
import sys, time
from pathlib import Path

sys.path.insert(0, {repo!r})
sys.path.insert(0, str(Path({repo!r}) / "tests"))

from src import tools
from src.batch import run_batch
from tests.fakes import FakeOcrTool, FakeQaTool, FakeRedactionTool, FakeVlmTool

class SlowVlm(FakeVlmTool):
    def analyze(self, *a, **kw):
        time.sleep(0.5)
        return super().analyze(*a, **kw)
    def qa_verify(self, *a, **kw):
        time.sleep(0.5)
        return super().qa_verify(*a, **kw)

tools._cache["ocr"] = FakeOcrTool()
tools._cache["vlm"] = SlowVlm()
tools._cache["redact"] = FakeRedactionTool()
tools._cache["qa"] = FakeQaTool()

run_batch({root!r}, output_root={out!r}, workers={workers!r})
print("SHOULD_NOT_REACH_HERE")
'''


def test_sigterm_mid_run_keeps_progress_and_resumes(tmp_path):
    """systemd stop / kill 发来 SIGTERM：进度文件必须完好，续跑能补完。

    SIGTERM 的默认动作是立刻终止（不跑 finally），所以安全性完全依赖
    「每张图原子写进度」这一条；顺带验证残留的 *.tmp 会被下次启动清掉。
    """
    import subprocess
    import sys
    import time

    repo = Path(__file__).resolve().parent.parent
    root = tmp_path / "folder_term"
    out_root = tmp_path / "out_term"
    for c in range(6):
        case = root / f"case{c}"
        case.mkdir(parents=True)
        for i in range(2):
            (case / f"img{i}.png").write_bytes(b"fake-image")

    script = tmp_path / "child_term.py"
    script.write_text(
        _SIGTERM_CHILD.format(
            repo=str(repo), root=str(root), out=str(out_root), workers=2
        ),
        encoding="utf-8",
    )

    proc = subprocess.Popen(
        [sys.executable, str(script)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    time.sleep(2.5)  # 让它处理几张
    proc.terminate()
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
        pytest.fail("SIGTERM 之后 30 秒仍未退出")
    assert "SHOULD_NOT_REACH_HERE" not in (proc.stdout.read() or "")

    # 中断现场：进度文件是合法 JSON、有已完成的记录、报告没写
    progress_file = out_root / "batch_progress.json"
    assert progress_file.exists()
    saved = json.loads(progress_file.read_text(encoding="utf-8"))
    assert 0 < len(saved["images"]) < 12
    assert not (out_root / "batch_report.json").exists()

    # 人为放一个残留临时文件，验证下次启动会清掉
    (out_root / "tmpstale.tmp").write_bytes(b"half-written")

    _inject_fakes()
    report = run_batch(str(root), output_root=str(out_root), workers=2)
    assert report["resumed"] is True
    assert report["summary"].get("failed", 0) == 0
    assert report["resumed_skipped"] == len(saved["images"])
    assert (
        report["statistics"]["images_processed"] + report["resumed_skipped"] == 12
    )
    assert not list(out_root.glob("*.tmp"))  # 残留临时文件已清掉
    assert len(json.loads(progress_file.read_text(encoding="utf-8"))["images"]) == 12


def test_batch_runs_when_stdout_is_not_a_tty(tmp_path):
    """nohup / systemd 下 stdout 不是终端：进度条必须自动关闭、不能刷屏报错。"""
    import subprocess
    import sys

    repo = Path(__file__).resolve().parent.parent
    root = tmp_path / "folder_pipe"
    (root / "case0").mkdir(parents=True)
    (root / "case0" / "img.png").write_bytes(b"fake-image")
    out_root = tmp_path / "out_pipe"
    log = tmp_path / "run.log"

    script = tmp_path / "child_pipe.py"
    script.write_text(
        "import sys\n"
        f"sys.path.insert(0, {str(repo)!r})\n"
        f"sys.path.insert(0, {str(repo / 'tests')!r})\n"
        "from src import tools\n"
        "from src.batch import run_batch\n"
        "from tests.fakes import FakeOcrTool, FakeQaTool, FakeRedactionTool, FakeVlmTool\n"
        'tools._cache.update({"ocr": FakeOcrTool(), "vlm": FakeVlmTool(), '
        '"redact": FakeRedactionTool(), "qa": FakeQaTool()})\n'
        f"report = run_batch({str(root)!r}, output_root={str(out_root)!r}, workers=1)\n"
        'print("SUMMARY", report["summary"])\n',
        encoding="utf-8",
    )

    with open(log, "w", encoding="utf-8") as fh:  # stdout 重定向到文件 = 非 TTY
        proc = subprocess.run(
            [sys.executable, str(script)], stdout=fh, stderr=subprocess.STDOUT,
            timeout=120,
        )
    assert proc.returncode == 0

    content = log.read_text(encoding="utf-8")
    assert "SUMMARY {'passed': 1}" in content
    assert "脱敏进度" not in content  # 进度条在非 TTY 下关闭，不写进日志
    assert "UnicodeEncodeError" not in content
    assert "Traceback" not in content


def test_resume_reprocesses_when_output_missing(tmp_path):
    """进度说 passed 但产物被删 → 必须重新处理，不能静默跳过。"""
    _inject_fakes()
    root = tmp_path / "folder_missing"
    for c in range(2):
        case = root / f"case{c}"
        case.mkdir(parents=True)
        (case / "img.png").write_bytes(b"fake-image")
    out_root = tmp_path / "out_missing"

    first = run_batch(str(root), output_root=str(out_root), workers=1)
    assert first["summary"] == {"passed": 2}

    # 人为删掉一张产物（模拟误删/清理/磁盘故障）
    victim = out_root / "case0" / "img.png"
    victim.unlink()

    report = run_batch(str(root), output_root=str(out_root), workers=1)
    assert report["resumed"] is True
    assert report["resumed_skipped"] == 1           # 只有产物还在的那张被跳过
    assert report["statistics"]["images_processed"] == 1
    assert victim.exists()                          # 被删的重新生成了


def test_output_inside_input_is_rejected(tmp_path):
    """输出目录落在输入目录内必须直接报错（否则会覆盖原图）。"""
    root = tmp_path / "folder_in"
    (root / "case0").mkdir(parents=True)
    (root / "case0" / "img.png").write_bytes(b"fake-image")

    with pytest.raises(ValueError, match="输出目录不能位于输入目录"):
        run_batch(str(root), output_root=str(root / "inside"), workers=1)

    with pytest.raises(ValueError, match="输出目录不能位于输入目录"):
        run_batch(str(root), output_root=str(root), workers=1)


def test_large_messy_dataset_completes(tmp_path):
    """脏大数据集：大小写扩展名/深层嵌套/空格中文名/非图片文件/损坏文件混排。

    必须跑完并正确分类：非图片文件被忽略、损坏文件只记自己 failed、
    其余全部 passed；再跑一遍应只剩损坏的那张重试，其余全跳过（幂等）。
    """
    from PIL import UnidentifiedImageError

    from tests.fakes import FakeOcrTool

    root = tmp_path / "messy"

    # case 1：正常 + 大小写扩展名 + 深层嵌套 + 空格/中文文件名
    c1 = root / "case 正常"
    (c1 / "deep" / "deeper").mkdir(parents=True)
    for name in (
        "a.png", "b.JPG", "c.jpeg", "带 空格.png", "中文名.png",
    ):
        (c1 / name).write_bytes(b"fake-image")
    (c1 / "deep" / "deeper" / "d.webp").write_bytes(b"fake-image")
    # 非图片文件必须被忽略（不计入进度、不报错）
    (c1 / "notes.txt").write_text("x", encoding="utf-8")
    (c1 / "data.json").write_text("{}", encoding="utf-8")
    (c1 / "deep" / ".hidden.png").write_bytes(b"fake-image")  # 隐藏文件也要处理

    # case 2：单个 case 内 50 张（case 内串行，验证不按图片数爆炸）
    c2 = root / "case_bulk"
    c2.mkdir(parents=True)
    for i in range(50):
        (c2 / f"img{i:03d}.png").write_bytes(b"fake-image")

    # case 3：只有非图片 → empty
    c3 = root / "case_noimg"
    c3.mkdir(parents=True)
    (c3 / "readme.md").write_text("x", encoding="utf-8")

    # case 4：一张损坏 + 一张正常
    c4 = root / "case_corrupt"
    c4.mkdir(parents=True)
    (c4 / "broken.png").write_bytes(b"not-an-image")
    (c4 / "ok.png").write_bytes(b"fake-image")

    class PickyOcr(FakeOcrTool):
        """对损坏文件抛 OCR 解码异常（真实引擎就是这么失败的）。"""

        def extract(self, image_path=None, image_bytes=None):
            if image_path and "broken" in str(image_path):
                raise UnidentifiedImageError("cannot identify image file")
            return super().extract(image_path=image_path, image_bytes=image_bytes)

    from tests.fakes import FakeQaTool, FakeRedactionTool, FakeVlmTool

    tools._cache["ocr"] = PickyOcr()
    tools._cache["vlm"] = FakeVlmTool()
    tools._cache["redact"] = FakeRedactionTool()
    tools._cache["qa"] = FakeQaTool()

    out_root = tmp_path / "out_messy"
    report = run_batch(str(root), output_root=str(out_root), workers=4)

    # case 正常(7 张) + case_bulk(50 张) + case_corrupt(2 张) = 59 张被发现；
    # case_corrupt 里有 1 张损坏 → 该 case 记 failed；case_noimg 记 empty
    assert report["summary"] == {"passed": 2, "failed": 1, "empty": 1}
    assert report["statistics"]["images_processed"] == 59

    progress = json.loads(
        (out_root / "batch_progress.json").read_text(encoding="utf-8")
    )
    assert len(progress["images"]) == 59
    statuses = [v["status"] for v in progress["images"].values()]
    assert statuses.count("passed") == 58
    assert statuses.count("failed") == 1
    # 损坏文件带明确错误原因，不是空壳
    broken = progress["images"]["case_corrupt/broken.png"]
    assert "UnidentifiedImageError" in (broken["error"] or "")

    # 幂等：再跑一遍（这次用普通假 OCR，损坏文件能被"修好"），
    # 58 张跳过、只有上一轮失败的那张重试；两个全跳过的 case 记 skipped。
    _inject_fakes()
    again = run_batch(str(root), output_root=str(out_root), workers=4)
    assert again["resumed"] is True
    assert again["resumed_skipped"] == 58
    assert again["statistics"]["images_processed"] == 1
    assert again["summary"] == {"empty": 1, "skipped": 2, "passed": 1}
    assert again["summary"].get("failed", 0) == 0


def test_batch_concurrent_stress(tmp_path):
    """并发跑批压力：6 case × 4 图 × 3 worker。

    质检后台预检线程 + OCR 引擎池 + 进度文件写入都跑在同一批里，
    报告、进度文件、输出文件必须全部自洽（无丢图、无串写、无异常）。
    """
    _inject_fakes()
    root = tmp_path / "folder_stress"
    for c in range(6):
        case = root / f"case{c}"
        case.mkdir(parents=True)
        for i in range(4):
            (case / f"img{i}.png").write_bytes(b"fake-image")
    out_root = tmp_path / "out_stress"

    report = run_batch(str(root), output_root=str(out_root), workers=3)

    assert report["summary"] == {"passed": 6}
    assert report["statistics"]["images_processed"] == 24
    assert [c["case"] for c in report["cases"]] == [
        f"case{i}" for i in range(6)
    ]  # 并发下也按 case 名排序，报告稳定可 diff

    progress = json.loads(
        (out_root / "batch_progress.json").read_text(encoding="utf-8")
    )
    assert len(progress["images"]) == 24
    assert all(v["status"] == "passed" for v in progress["images"].values())

    for case in report["cases"]:
        for img in case["images"]:
            assert Path(img["output"]).exists()


def test_case_exception_does_not_abort_sequential_run(sample_tree, tmp_path, monkeypatch):
    """顺序模式（--workers 1）下单个 case 异常也必须产出报告并继续跑完。

    历史 bug：异常直接冒泡出 run_batch，整批中断且没有 batch_report.json。
    """
    import src.batch as batch_module

    _inject_fakes()
    real_process_case = batch_module._process_case

    def flaky_process_case(graph, case_dir, *args, **kwargs):
        if case_dir.name == "caseA":
            raise OSError("disk full")
        return real_process_case(graph, case_dir, *args, **kwargs)

    monkeypatch.setattr(batch_module, "_process_case", flaky_process_case)
    out_root = tmp_path / "out_flaky"
    report = run_batch(str(sample_tree), output_root=str(out_root), workers=1)

    assert (out_root / "batch_report.json").exists()
    assert report["summary"] == {"error": 1, "passed": 1}
    assert (out_root / "caseB" / "b1.png").exists()


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
    # 预置断点：caseA/a1.png 已完成（产物也要在，否则按缺失重新处理）
    done = out_root / "caseA" / "a1.png"
    done.parent.mkdir(parents=True)
    done.write_bytes(b"masked")
    (out_root / "batch_progress.json").write_text(
        json.dumps(
            {
                "version": 1,
                "input_root": str(sample_tree.resolve()),
                "images": {
                    "caseA/a1.png": {
                        "status": "passed",
                        "output": str(done),
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
    # a2.JPG  passed 的产物也要在，否则会被判定为缺失而重新处理
    done = out_root / "caseA" / "a2.JPG"
    done.parent.mkdir(parents=True)
    done.write_bytes(b"masked")
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
